# `braingen_CondDiffuser_BraTS_v1` — Model Card, Provenance, and Deployment Notes

**Status:** integration in progress. The model is trained and evaluated; the website entry
is not yet deployed. This document is the reference for what the model is, what it can
honestly claim, and what has to happen to put it on the site.

**The one architectural fact to know before reading anything else:** the backend has **two
rendering modes**, selected by the `CONDDIFF_MODE` environment variable, and **`live` is the
code default**. In live mode a request builds the 85.3 M-parameter U-Net, loads the checkpoint
and runs the DDIM loop, so the image the visitor sees was sampled *for them*, from the real
model, conditioned on real anatomy. That is what this deployment does: **on an EC2 GPU host
(g4dn.xlarge class), synchronously, inside the existing `/generate` flow — with no frontend
change and no job queue** (§8.10).

`gallery` — serve PNGs rendered ahead of time on a Great Lakes GPU at the paper's full 200 DDIM
steps — is fully implemented and is **not** dead code. It is what the four **Lightsail** deploy
scripts pin their container to, because that box (`"power": "micro"` = 0.25 vCPU / 1 GB) cannot
run live at all: it would OOM on a 325 MB fp32 checkpoint and then be health-check-killed,
taking the five working GAN models down with it (§8.0.1). Read the pairing as: **live is the
product; gallery is the safety interlock on the one target that cannot serve the product.**

Map of §8, because it is long: **§8.0** the two modes and why live is the default · **§8.0.1**
the arithmetic that disqualifies the CPU box · **§8.1–8.4** the cluster-side work (export the
conditioning bank, prove the shared core did not drift, optionally render a gallery, transfer)
· **§8.5–8.7** where each file goes and how to verify from outside the container · **§8.8** what
live on *Lightsail* would have cost, kept because it is the argument for not doing that ·
**§8.9** the six registration points · **§8.10 the GPU host runbook — the one to actually
follow.** §5.4 is the shared module that makes the two modes provably the same code.

**Audience:** the SOCR lab (Achu, Prof. Dinov) and whoever maintains this backend next.

**Author of the model and this integration:** Alex Liu (undergraduate researcher).
**Research repository:** `Summer2026` (the paper's code — atlas registration, preprocessing,
training, sampling, evaluation).
**Target repository:** `SOCR/Brain-Image-Generator` (this repo — the deployed site).

---

## 0. At a glance

| | |
|---|---|
| Website model name | `braingen_CondDiffuser_BraTS_v1 (2D)` |
| Kind | Conditional denoising diffusion probabilistic model (DDPM), ε-prediction |
| Backbone | `diffusers.UNet2DModel`, 2-D, 256 × 256 |
| Parameters | ≈ 85.3 M (see §2.2 — confirm on the cluster before sizing the deployment) |
| Inputs | 9 channels: noisy FLAIR + patient T1 + tumour mask + 6 atlas lobe maps |
| Output | 1 channel: predicted noise → a synthetic **FLAIR** slice |
| Training data | BraTS 2023 adult glioma (GLI), ≈ 1 252 patients, 163 974 axial slices |
| Sampler | DDIM, `clip_sample=True`, 200 steps, deterministic (η = 0) |
| Published metrics | PSNR 17.09 ± 2.84 dB · SSIM 0.402 ± 0.127 · background-SNR 66.6 ± 35.5 |
| Checkpoint | `models/diffusion_ema.pt` — a bare EMA `state_dict`. **Measured on Great Lakes 2026-09-08: 341,171,155 bytes (325 MiB), 85,261,185 parameters**, loads into `build_model()` with `strict=True`. **Too big to ever commit** (§8.5) |
| Rendering modes | `CONDDIFF_MODE=live` is the **code default** and is what the GPU host runs: real DDIM sampling per request. `CONDDIFF_MODE=gallery` serves images pre-rendered on a cluster GPU at 200 steps, and is what the four Lightsail deploy scripts pin their CPU container to. §8.0 |
| Live host | **EC2 GPU, g4dn.xlarge class** (1 × T4, 16 GB VRAM, 4 vCPU, 16 GB RAM). ≈ **1 s/sample** vs ≈ 2 min on a full multi-core laptop CPU — which is why the existing synchronous `/generate` needs no rework. §8.0.2, §8.10 |
| Parameter space | **57 cells** — Tumour(2) × Lobe(6) × Level(3) × Size(3), collapsed because lobe and size are meaningless without a tumour. **36 of them are actually offerable**: only 11 of the 18 (lobe, level) conditioning cells exist in the anatomy (§6.1), and `GET /conddiff-cells` tells the UI which to grey out |
| Gallery payload (gallery mode only) | ~10–15 MB of PNGs + one `manifest.json`. **No checkpoint, no torch, no diffusers on the server.** |
| Shared correctness core | `conddiff_core.py`, byte-identical in both repos; its SHA-256 is stamped into the gallery manifest and reported by `/conddiff-selftest`. §5.4 |
| What is new on the site | **Atlas-guided lobe placement.** No other model on the site can be told *where* to put a lesion. |
| The catch | It is **conditional**. It needs a real patient's T1 and a registered atlas as input. The website has no concept of either, so both must ship with the model — or, in gallery mode, be consumed on the cluster and never shipped at all. See §6 and §7. |

---

## 1. What the model is, in one paragraph

Every other generator on this site invents a brain from noise. This one does not. It is a
**conditional** diffusion model: you hand it a real patient's T1-weighted anatomy, a tumour
segmentation mask, and a six-channel probabilistic lobe atlas, and it renders the **FLAIR
contrast** that those inputs imply. The anatomy in the output comes from the conditioning;
the FLAIR appearance, the tissue contrast, and the lesion rendering come from the model.
That distinction is the whole scientific point (§4) and it is also the main thing the
website has to be honest about (§7, §9).

---

## 2. Architecture

### 2.1 Definition (single source of truth)

`Summer2026/src/model.py::build_model()`. Its docstring says *"DO NOT change it, or the
deployed checkpoint `models/diffusion_ema.pt` will no longer load"* — that is not a
formality. The exact call is:

```python
UNet2DModel(
    sample_size=256,
    in_channels=9,                              # noisy FLAIR + 8 conditioning channels
    out_channels=1,                             # predicted noise (epsilon)
    layers_per_block=2,
    block_out_channels=(128, 256, 384, 512),    # resolutions 256 -> 128 -> 64 -> 32
    down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
    up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
)
```

Self-attention lives only at the two deepest levels (32 × 32 and, through the mid block,
the same). At 256 × 256 full-resolution attention would be 65 536 tokens attending to each
other and would not fit on any GPU.

The same file also defines `build_text_model()` (a `UNet2DConditionModel` with
cross-attention for captions). **That is a separate, in-progress model and is not what is
being deployed.** Do not copy it into this repo; there is no checkpoint for it here.

### 2.2 Parameter count and checkpoint size — verify, do not assume

≈ **85.3 M parameters**, originally from an analytic layer-by-layer enumeration of the
architecture above, because the checkpoint is not present on the development laptop
(`Summer2026/models/` contains only a zero-byte `.gitkeep`).

**That estimate has since been measured and it holds.** On Great Lakes, 2026-09-08:
**85,261,185 parameters**, **341,171,155 bytes on disk (325 MiB)**, and the file loads into
`build_model()` with `strict=True` — so the architecture match is *proven*, not assumed, which
is the single fact live mode most depends on. (Both "341 MB" and "325 MB" appear in this
document and in the code comments: they are the same file, counted in decimal MB and in MiB.
Nothing anywhere is describing two different checkpoints.)

The commands that produced those numbers, kept because they are the ones to re-run if the
checkpoint is ever retrained:

```bash
ls -lh models/diffusion_ema.pt
python -c "import torch; sd = torch.load('models/diffusion_ema.pt', map_location='cpu'); \
           print(type(sd), len(sd), sum(v.numel() for v in sd.values()))"
```

Two things that matter about the checkpoint's *format*: it is a **bare `state_dict` of the
EMA weights** (`scripts/train.py` calls `ema.copy_to(model.parameters())` and then
`torch.save(model.state_dict(), ...)`), so there is no `["model"]` or `["epoch"]` wrapper to
unwrap; and it should be loaded with `strict=True`. Every existing loader in
`backend/inference/` uses `strict=False`, which on a key-name mismatch loads **nothing** and
produces a randomly-initialised network that emits noise with no error and no log line. A
partially-loaded diffusion U-Net still produces brain-shaped output, so a visual check will
not catch it. Use `strict=True` and let it raise.

---

## 3. Training data and training setup

### 3.1 Data

| Item | Value |
|---|---|
| Source | BraTS 2023, adult glioma (GLI) challenge training data |
| Patients used | ≈ **1 252** |
| Native volumes | 240 × 240 × 155, 1 mm isotropic, skull-stripped, already co-registered to **SRI24** |
| Modalities | `t1n` (T1), `t1c` (T1ce), `t2w` (T2), **`t2f` (FLAIR — the generation target)**, `seg` |
| Segmentation labels | 1 = necrotic core, 2 = edema, 3 = enhancing tumour |
| Slices kept | **163 974** (a slice is kept if it has ≥ 1 000 brain voxels) |
| Tumour / tumour-free | 79 001 / 84 973 |
| Split | **70 / 15 / 15 by patient**, `random.seed(42)` — see §7 |
| Anatomical prior | ICBM452 probabilistic lobular atlas, 6 lobes |

Because BraTS patients are *already* co-registered to SRI24, the atlas is registered into
SRI24 **once** (ANTs SyN, ICBM452 T1 → SRI24 `spgr.nii.gz`) and the transform is reused for
every patient. A real bug was solved here and is worth knowing about: ANTs/ITK misread the
ICBM Analyze `.hdr/.img` orientation (an anterior–posterior flip) while `nibabel` read it
correctly, so the pipeline converts Analyze → NIfTI with
`nibabel.as_closest_canonical` *before* handing anything to ANTs.

### 3.2 The 9-channel slice format

One preprocessed slice is a `(9, 256, 256)` `float16` `.npy`:

| ch | contents | range |
|---|---|---|
| 0 | FLAIR — **the generation target** | [0, 1] |
| 1 | patient T1 | [0, 1] |
| 2 | tumour mask | **{0, 1, 2, 3}** — raw BraTS labels, *not* normalised |
| 3–8 | atlas frontal, parietal, temporal, occipital, cerebellum, insula | [0, 1] |

Three details that are easy to get wrong and expensive to debug:

- **Channel 2 is the one channel that is not in [0, 1].** FLAIR and T1 go through the
  percentile normaliser; the segmentation goes in raw. Rescaling it would put the model far
  off-distribution.
- **Normalisation is per-volume, over brain voxels only, at percentiles 1 and 99**, then
  clipped to [0, 1] with background forced back to exactly 0. It is applied to FLAIR and T1
  with the *same* function. Re-normalising per slice would change every slice's brightness.
- **The atlas is scaled by one global scalar** — `atlas = atlas / atlas.max()` over all six
  lobes and all 155 slices at once, not per lobe. Every downstream threshold (notably
  `PROB_TH = 0.4`) is calibrated against that scaling.

240 → 256 is achieved by **zero-padding, not resizing** (`top = left = 8`), so anatomy is
never resampled. Real data occupies `[8:248, 8:248]`; the outer 8-pixel frame is exactly zero.

### 3.3 Training

| Item | Value |
|---|---|
| Objective | MSE between predicted and true noise (ε-prediction) |
| Forward process | `betas = linspace(1e-4, 0.02, T=1000)`, `alpha_bar = cumprod(1 - betas)` |
| Target scaling | FLAIR [0, 1] → [−1, 1]; **conditioning channels are not rescaled** |
| Optimiser / lr | AdamW / 1e-4 |
| Batch size | 32, mixed precision (batch 64 OOMs even at 96 GB) |
| EMA | decay 0.999 — **the EMA weights are what we sample from** |
| Hardware | U-M Great Lakes, NVIDIA RTX PRO 6000 Blackwell 96 GB (sm_120, needs PyTorch cu128) |

That "conditioning is not rescaled" line is load-bearing. `train.py` does
`target = target * 2 - 1` and then `cond = cond.to(DEVICE)` with no arithmetic. Applying
`2*x - 1` to the conditioning at inference time is a silent, plausible-looking bug.

### 3.4 Sampling

DDIM, `DDIMScheduler(num_train_timesteps=1000, clip_sample=True)`, `set_timesteps(200)`,
deterministic (η = 0). Every unspecified argument is a diffusers default that happens to
match training exactly (`beta_start=0.0001`, `beta_end=0.02`, `beta_schedule="linear"`,
`prediction_type="epsilon"`). Passing `beta_schedule="scaled_linear"` or
`prediction_type="v_prediction"` produces noise, not brains.

DDIM was chosen over plain 1000-step DDPM for stability, and the improvement is measured.
Under DDPM, six samples on one fixed conditioning had brain-mean intensities of
0.17 / 0.17 / 0.29 / 0.59 / 0.63 / 0.72 against a real 0.58. Under DDIM on a held-out slice
with real mean 0.413, six samples read 0.477 / 0.584 / 0.426 / 0.474 / 0.415 / 0.491.

---

## 4. The scientific point

### 4.1 The published metrics, and how to read them

Computed by `Summer2026/scripts/eval_metrics.py` over **100 randomly sampled held-out
validation slices**, one sample per slice, 200 DDIM steps, seed 0:

| Metric | Value | What it measures |
|---|---|---|
| **PSNR** | **17.09 ± 2.84 dB** | pixel closeness to the real FLAIR, computed **over brain voxels only** (`real > 0.05`) |
| **SSIM** | **0.402 ± 0.127** | structural similarity to the real FLAIR, whole image, `data_range=1.0` |
| **background-SNR** | **66.6 ± 35.5** | mean brain signal ÷ std of the generated background where the real background is exactly 0 |

**These numbers must be presented with their caveat or they are misleading in the wrong
direction.** PSNR and SSIM are *full-reference* metrics: they measure agreement with one
particular real slice, so they reward pixel-copying and penalise diversity — which is the
opposite of what a generative model is for. A model that memorised its training set would
score better. Moderate PSNR and SSIM are therefore partly by design, not purely a deficiency.
The honest framing is: *these are the fidelity numbers; the appropriate metrics for the
actual goal are distributional (FID) and downstream-utility (does synthetic data improve a
segmentation model?), and neither has been run yet.*

One further caveat we should state before a reviewer does: whole-image SSIM here is **not**
inflated by the matching black background, because the real background has zero variance,
which actually depresses SSIM's contrast term. A brain-masked SSIM is the honest
tissue-structure number and is planned, not done.

### 4.2 The informational floor — the paper's contribution

This is the finding worth putting in front of Prof. Dinov.

Every configuration conditioned on **mask + atlas only** — small model, large model,
150-patient subset, full dataset — plateaued at a training ε-prediction MSE of
**≈ 0.006**. Adding more capacity did not help. Training longer did not help. That pattern
is the signature of an *informational* limit rather than a capacity or optimisation limit:
ε-prediction MSE is lower-bounded by the conditional variance of the target given the
conditioning, so weak conditioning implies high residual variance implies a high floor.

Adding the **patient's own T1** as a ninth channel lowered the plateau to **≈ 0.0042**, a
~30 % reduction, and produced visibly sharper samples with clean black backgrounds.
Representative per-epoch trajectory (run 52090423): 0.0266 (ep 0) → 0.0072 (ep 10) →
0.0055 (ep 40) → 0.0044 (ep 60) → **0.0042 (ep 79)**.

The claim is therefore not "our model is better." It is: *"the loss won't go below X" is an
information statement about the conditioning, not a training failure — and it predicts,
correctly, that supplying more informative conditioning lowers the floor.* That is a
falsifiable prediction that was tested and held.

The direct consequence for the website: **the anatomy in a generated image is a function of
the conditioning, not of the model's imagination.** That is exactly why the model works,
and exactly why the site has to be careful about what it claims (§7).

---

## 5. Atlas-guided lobe placement — the controllable axis

This is the parameter that makes the model worth adding to a site that already has four
generators.

### 5.1 The mechanism

The model's channel 2 is a tumour mask. The paper's "mask owns anatomy" observation is that
the model **faithfully renders a lesion wherever the mask says one is**. So if we *draw* a
mask, we control placement. The reference implementation is
`Summer2026/scripts/augment_tumor.py` (written for Prof. Dinov's synthetic-augmentation
suggestion), and it does exactly what the website feature needs:

1. Pick a tumour-free validation slice that contains the target lobe.
2. Synthesise a lesion mask **inside that lobe**, guided by the atlas.
3. Swap it into channel 2 of the conditioning.
4. Run the DDIM loop.
5. Out comes a FLAIR with a tumour in the chosen lobe — **plus the mask, which is ground
   truth because we drew it.**

That last point is the one to emphasise: this produces a *labelled* pair for free. No
annotator, no inter-rater disagreement. It is the reason the technique is interesting for
data augmentation and not just for a demo.

### 5.2 Why the atlas is what makes it scientifically meaningful

Drawing a circle in a random place would also "control placement", and would also be
meaningless — it could straddle the ventricles, cross the midline, or sit half outside the
brain. The atlas is what makes the placement *anatomically valid*:

`synth_mask()` scores every pixel by three multiplied terms —

```
score = atlas_lobe_probability  ×  compactness_around_a_seed  ×  smooth_random_field
score[not in lobe] = 0.0        # hard containment constraint
```

- the **atlas** term makes the lesion hug the lobe's true shape rather than being a blob;
- the **compactness** term (a Gaussian falloff scaled to the equivalent-circle radius of the
  target area) keeps it one connected mass instead of scattered islands;
- the **smoothed random field** (σ = 6 px, amplitude ±0.45) gives it an organic, irregular
  outline rather than a perfect ellipse.

The seed point is drawn **proportional to atlas probability**, not uniformly, so lesions
nucleate in the lobe's core rather than at its ragged margin. Lesion area is a **fraction of
the lobe's area in that slice** (default 15–35 %), so a big lobe automatically gets a big
tumour. The enhancing core is the top 30 % of the lesion by the same score field, which puts
it at the lesion's centre of mass rather than making it a concentric circle. Only labels
2 (edema) and 3 (enhancing core) are emitted — synthetic lesions do not include label 1
(necrotic).

`Summer2026/scripts/check_atlas_fit.py` exists specifically to verify this, without running
the diffusion model at all, and reports three numbers per lobe: **containment** (must be
100 %, because of the hard constraint line above), **coverage** (should land in the 15–35 %
band), and **circularity** (1.00 would be a perfect disc; lower means organic — the earlier
fixed-circle version scored ~1). Re-run it whenever any shape constant changes.

### 5.3 How the four website parameters map onto this

| UI control | Values | What it does mechanically |
|---|---|---|
| **Tumour** | With / Without Tumor | whether a synthetic mask is placed in channel 2 at all |
| **Lobe** | Frontal, Parietal, Temporal, Occipital, Cerebellum, Insula | selects which atlas channel guides `synth_mask` (lobe *k* is stack channel `3 + k`; **this order is fixed by the preprocessing and must not be reordered**) |
| **Slice Location** | Inferior / Middle / Superior | selects which conditioning slice is used, by its z index — `inferior` z < 60, `superior` z > 100, else `middle` |
| **Tumour Size** | Small / Moderate / Large | scales the lesion's target area fraction |

Slice Location and Tumour Size reuse the paper's own published vocabulary
(`scripts/06_captions_v2.py`): sizes are **small < 500 px ≤ moderate ≤ 2000 px < large**,
where 1 px = 1 mm² at BraTS's 1 mm resolution (so ~500 mm² is a 25 mm-diameter lesion and
~2000 mm² a 50 mm one). Using the same thresholds means the website's label means the same
thing as the paper's caption. Whatever is implemented, the **realised** lesion area should be
recorded alongside the requested size so the two can be checked against each other.

### 5.4 `conddiff_core.py` — one file, two repositories, byte-identical

Everything in §5 is a *shape* computation: which pixels are in the lobe, how big the lesion is,
where its core sits. Those computations happen in two different places — on Great Lakes when
the gallery is rendered (§8.3), and inside the web container when live mode runs (§8.8) — and
**the site is only honest if both places compute the same thing.** If the renderer used
`PROB_TH = 0.4` and the server used `0.45`, or if one used `scipy.ndimage.gaussian_filter` and
the other a numpy reimplementation with a different boundary mode, the site would advertise a
parameter it did not actually honour, silently and permanently.

So everything the two sides must agree on lives in a single module that exists **byte-identically
at two paths**:

```
Summer2026/src/conddiff_core.py                            <- the cluster renderer imports this
Brain-Image-Generator/backend/inference/conddiff_core.py   <- the backend imports this
```

It contains: the constants (`LOBES`, `ATLAS0`, `PROB_TH`, `MIN_LOBE`, `AREA_FRAC_*`,
`CORE_FRAC`, `MIN_AREA`, `COMPACT`, `NOISE_SIGMA`, `NOISE_AMP`, the size and level vocabulary,
and the four UI-string maps); the scipy-free numpy reimplementations `_gaussian_blur` and
`_largest_connected_component`; the mask synthesiser `synth_mask` / `synth_mask_sized`;
`params_to_spec`; `cell_id` and `iter_cells`; and `build_cond`, `render_flair`, `to_png_bytes`.
It imports **only the standard library and numpy** — `torch` is imported lazily inside
`build_cond` and `render_flair`, `PIL` lazily inside `to_png_bytes`, so importing the core
itself costs nothing and gallery mode never touches either.

What deliberately stayed *out* of it: paths, environment variables, device selection, the
conditioning-bank CSV, the gallery index, checkpoint loading, Supabase publishing, and
`MIN_BRAIN` (which is deployment-tunable via `CONDDIFF_MIN_BRAIN` and is *allowed* to differ
between the cluster and the server). Deployment decisions must not be able to change what an
image looks like.

Three mechanisms keep the two copies honest, and all three are needed:

1. `scripts/verify_conddiff_port.py` compares the two files by SHA-256 before a render (§8.2);
2. the renderer stamps `core_sha256` — the hash of the file *it actually imported* — into
   `manifest.json`;
3. `/conddiff-selftest` reports that hash back, so it can be compared against the container's
   own copy at any time, months later, without a shell (§8.7).

`cell_id(spec)` deserves its own note, because it is the join key between the two sides:
it is a pure function of the four choices, lowercase ASCII, and no component contains an
underscore, so it round-trips by splitting — `with_frontal_middle_moderate`, `without_middle`.
`iter_cells()` yields all 57 in a fixed order using the exact frontend dropdown strings, so the
renderer and the backend cannot disagree about how many cells exist or what they are called.

---

## 6. What has to ship with the model

> **Mode note.** Everything in §6 describes the *conditioning bank*, which is built on Great
> Lakes in both modes but only **shipped to the web server in live mode**. In gallery mode the
> bank is consumed on the cluster by the renderer and never leaves it: the server receives PNGs
> only. The sizing arithmetic below still matters, because the bank still has to be built, and
> because it is what a Phase 2 live deployment would have to transfer.

The model cannot generate without conditioning, and the website has no way to obtain it:

- The **patient T1** is a real scan. There is no way to synthesise one server-side.
- The **six atlas channels** are the output of a one-time ANTs SyN registration, normalised
  by a single global scalar (§3.2). Re-deriving them anywhere else would silently change
  every lesion size, because `PROB_TH = 0.4` is calibrated against that specific scaling.
  They must be **copied verbatim** from the preprocessing output, never recomputed.

So the deployment ships a small **conditioning bank**: a set of preprocessed held-out
validation slices, each carrying T1 + mask + the 6 atlas maps. Sizes, for planning:

| Variant | Per slice | Notes |
|---|---|---|
| 9 channels, float16 | 1.125 MiB | exactly as preprocessing writes them |
| 8 channels (FLAIR dropped), float16 | 1.00 MiB | the model never reads channel 0; dropping it also removes any chance of the site publishing a real patient's FLAIR |
| 9 channels, float32 | 2.25 MiB | pointless — the cast at load time is free |

**Recommendation: ship 8 channels, float16, with channel 0 dropped**, plus a manifest CSV
recording `file, patient, z, level, lobes_available, brain_px, lobe_px` so the server can
select a slice without opening every array.

### 6.1 Open question that must be answered before the bank is built

The bank has to cover 6 lobes × 3 slice levels = 18 cells, and **it is not yet known that
all 18 are fillable**. (Those 18 are the *conditioning* cells. The user-visible space is the 57
cells of §8.0.2, which is the same 18 crossed with the three tumour sizes plus the three
no-tumour cells; an unfillable conditioning cell therefore knocks out three gallery cells at
once, and `manifest.json["empty_cells"]` must list them so the UI can disable them.) Some are anatomically doubtful rather than merely tight:

- the **cerebellum** does not exist at superior slice levels (z > 100), so
  (cerebellum, Superior) may have zero admissible slices — no threshold relaxation fixes that;
- `MIN_BRAIN = 15000` was written specifically to *reject* thin inferior slices, which is
  where the cerebellum does live, so (cerebellum, Inferior) is squeezed from both sides;
- the **insula** is small, and after the *global* atlas normalisation its probability map may
  never exceed 0.4 by enough area to clear `MIN_LOBE = 500` at any level.

**Do this first, before writing any web integration code:** run the slice-selection pass over
the validation split and print a 6 × 3 candidate-count table. If a cell is empty, the site
must either disable that combination in the dropdown or return a clear, visible error.
It must **not** silently substitute a different slice level and label it as the one the user
asked for — that would make the site display a value it did not honour.

---

## 7. Provenance of the conditioning slices — and why it is defensible

**Where they come from:** `data/processed/slices/val/` only. The split is at the **patient**
level, 70 / 15 / 15, `random.seed(42)`, performed before any slice is written. Patient-level
splitting (rather than slice-level) is deliberate: adjacent axial slices of the same brain
are near-duplicates, so a slice-level split would leak.

**Why validation and not training:** every conditioning slice the website shows comes from a
patient the model never saw during training. If the bank were drawn from the training split,
the site would be displaying anatomy the model had memorised, which would quietly undercut
the held-out claim in the paper. Drawing from val is what keeps the deployed site consistent
with the paper's evaluation protocol.

**What must be stated plainly, because it is the thing most likely to be misread:**

> The anatomy in every generated image comes from a real, held-out BraTS patient's T1 scan.
> The FLAIR contrast and the lesion are generated by the model. These images are **not**
> brains invented from noise, unlike the other generators on this site.

That sentence, or something like it, belongs in the model's description in the UI — not only
in a database column. Two concrete reasons:

1. **The site's other four models do invent anatomy from noise.** A user comparing five
   entries in one dropdown will reasonably assume all five work the same way. "Synthetic
   FLAIR of a real brain" is a materially different claim from "synthetic brain image."
2. **The playground path is unauthenticated.** `/dashboard/playground` skips the session
   check entirely, and in playground mode nothing is written to the database at all — so any
   disclosure that lives only in a `parameters_used` column does not exist on the path most
   people will actually use.

**What gallery mode changes here, and what it does not.** It changes one real thing: in gallery
mode **no patient data ever reaches the web server**. The conditioning slices — real held-out
T1 anatomy — are read on Great Lakes by the renderer and stay there; what ships is a rendered
FLAIR PNG and a JSON row recording the de-identified BraTS subject id and z index of the slice
that conditioned it. There is no `.npy` of a real scan on a public-facing host to leak or to
mis-serve. It does **not** change the underlying question: the published image is still a
per-patient derived rendering, and the manifest still names the subject it came from.

**Unresolved and blocking:** whether the BraTS / TCIA data use agreement permits publishing
per-patient derived renderings to a public, unauthenticated endpoint. This is a one-sentence
question with no good improvised answer, and it should be answered in writing before the
bank is exported. If the answer is no, the fallback worth pricing now is to condition on a
**template** T1 (the SRI24 `spgr` volume already used in the registration step) instead of a
patient's. That is off-distribution for the model and would need its own quality check, but
it is a real option and it is much cheaper to evaluate before the bank exists than after.

A related recommendation: **consider dropping the "Without Tumor" cells.** Conditioned on a
real patient's T1, that patient's real (empty) mask, and the real atlas, with only the FLAIR
channel withheld, the model is performing a near-reconstruction of that held-out patient's
actual FLAIR scan. That is the least defensible artefact this pipeline can produce. A
minimum-size lesion is a better default than none.

---

## 8. Deployment

### 8.0 Two rendering modes — and why `live` is the default

`backend/inference/conddiff_inference.py` reads one environment variable, `CONDDIFF_MODE`,
and runs one of two completely separate data paths:

| `CONDDIFF_MODE` | What a request does | What must be on disk | Imports torch? |
|---|---|---|---|
| **`live`** (the **code default**, and what the GPU host runs) | build the 85.3 M-parameter U-Net, load the checkpoint, run the DDIM loop at request time | `inference/conddiff_bundle/manifest.json` + `inference/conddiff_bundle/cond_slices/*.npy`, plus the checkpoint at `inference/model/diffusion_ema.pt` (or the `.safetensors` export, or the bundle's own copy) | yes |
| `gallery` (what the four **Lightsail** deploy scripts pin their CPU container to) | dictionary lookup → pick one of *K* pre-rendered seed variants → read a PNG off disk | `inference/conddiff_gallery/manifest.json` + `inference/conddiff_gallery/<cell_id>/*.png` | **no** |

**Why live is the default, in one sentence:** *a generator that hands back a pre-rendered image
is a lookup table wearing a model's name.* Two visitors who pick the same four dropdown values
get the same file; a third visitor clicking Generate twice gets the same file (§9.2). The
scientific claim this site entry exists to demonstrate — that the model **samples** a FLAIR
conditioned on **real anatomy**, and that *where* the lesion goes is a controlled input — is a
claim about sampling. The default therefore has to be the mode that actually samples, and the
code says so at `conddiff_inference.py:309`:

```python
_MODE_RAW = os.environ.get("CONDDIFF_MODE", "live")
```

The variable is parsed at module scope, case-insensitively, and **cannot raise** — that matters
because `image_generation.py` imports this module at the top of the file, next to the four GAN
modules, so an exception here would take the five working models down with it. An unset,
misspelled, or nonsense value **falls back to `live`** and prints why
(`conddiff_inference.py:311-314`). The fallback simply follows the configured default rather
than second-guessing it; what keeps that safe on the one host that cannot serve live is not the
fallback but the **interlock in the deploy scripts** (§8.6), which always emit an explicit
`CONDDIFF_MODE` into the deployment they submit.

> **The residual risk, stated rather than hidden.** All four Lightsail scripts read
> `CONDDIFF_MODE` from the environment/`.env` and default it to `gallery` — e.g.
> `deploy.sh:120` emits `"CONDDIFF_MODE": "${CONDDIFF_MODE:-gallery}"`. A *typo'd* value in
> `.env` (`galery`) is not empty, so it passes through the script, reaches the container, fails
> the `("gallery", "live")` membership test, and lands on **live** — on the box that cannot run
> it. The mitigation is `/conddiff-selftest`, whose first field is `mode` (§8.7): read it after
> every deploy rather than assuming.

`CONDDIFF_STEPS` is read in both modes but is **meaningless in gallery mode** — the images were
rendered at whatever step count the cluster used, and `/conddiff-selftest` reports that number,
not this one.

#### 8.0.1 The arithmetic — why a GPU host, and why the CPU box is disqualified

This is not a preference, and it is not about money first. It is one number about the model and
five about the deployment.

**The number about the model.** One U-Net forward pass at 256 × 256 is **≈ 950 GFLOP** (§9.1),
and DDIM needs **50–200 of them per image**. *Every other model on this site is one forward
pass* — the existing `VanillaGenerator` is ~3.9 GFLOP. So this model, at even 50 steps, is on
the order of **10⁴ times** the compute of every other entry in the dropdown. Nothing about the
host that was chosen for five GANs generalises to it, and that is the whole argument in one
line.

**Measured, not modelled:** ≈ **1 s/sample on a GPU** versus ≈ **2 min on a full multi-core
laptop CPU**. The laptop is many times the deployed container's 0.25 vCPU, so 2 min is a
*floor* for CPU, not an estimate of the deployed box (§9.1 models that separately, and much
worse).

**The five deployment facts that disqualify the CPU box:**

| Fact | Value | Source |
|---|---|---|
| Deployed instance | `"power": "micro"` = **0.25 vCPU, 1 GB RAM** | `backend/lightsail-config.json` (a **record of intent** — no deploy script reads it, §8.6) |
| Compute per U-Net forward pass at 256 × 256 | **≈ 950 GFLOP** (≈ 190 TFLOP for 200 DDIM steps) | analytic enumeration, §9.1 |
| Checkpoint on disk | **325 MiB / 341 MB** fp32 — alone, before torch or activations (→ ~1.3–1.6 GB peak RSS, §8.8) | §2.2, measured 2026-09-08 |
| Health check | `timeoutSeconds: 5`, `intervalSeconds: 30`, `unhealthyThreshold: 2` → the container is declared unhealthy and **replaced after ~65 s** of unanswered `/health` | `backend/lightsail-config.json` |
| `api.py:81` | `async def generate_images(...)` calling **synchronous** `image_generation.generate_image(...)` directly | `backend/api.py` |

The last two combine into the killer. Because `/generate` is an `async def` wrapping
synchronous work, the generation runs *on uvicorn's event loop thread*. Nothing else is served
while it runs — including `/health`. So any generation longer than ~65 s guarantees that
Lightsail tears the container down **mid-request, every time** — and it would take the **five
working GAN models down with it**, because they share the container. On 0.25 vCPU a 200-step
sample is ~9.6 hours (§9.1), if it did not OOM at 1 GB first, which it does.

Live CPU diffusion here is not "slow". It is a deployment that deletes itself, and takes the
rest of the site with it. Hence: **live runs on an EC2 GPU host (g4dn.xlarge class); the
Lightsail CPU container keeps serving the five GANs and pins itself to `gallery`.**

#### 8.0.1.1 Moving to a GPU host cannot change the GAN models' behaviour — verified by reading them

The obvious worry about putting the shared container on a GPU box is that the five existing
models silently start using the GPU and stop producing the images they produce today. **They
cannot.** Every GAN module resolves its device with the same two lines, in this order:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')          # <-- immediately overwrites the line above
```

so the `cuda` line is **dead code** in all four:

| Module | The `cuda` line | The line that overrides it |
|---|---|---|
| `inference/cgan_inference.py` | 150 | **151** |
| `inference/cgan_inference_v2.py` | 108 | **109** |
| `inference/wgan_inference.py` | 151 (already commented out) | **152** |
| `inference/gan_3d_inference_v1.py` | 61 | **62** |

And every weight load is pinned to CPU regardless of host —
`torch.load(model_path, map_location=torch.device('cpu'))` at `cgan_inference.py:174`,
`cgan_inference_v2.py:134`, `wgan_inference.py:168-169`, `gan_3d_inference_v1.py:88`.

**Therefore the host change is not a risk to the five production models. The only thing that
could move their numerics is a change to the torch *version*** — which is exactly why §8.10.5
tries `torch==2.0.1+cu118` first: the same version already in production, only the CUDA build.

> **One exception, noted and deliberately not fixed.** `cgan_inference.py:120-121` calls
> `model.to("cuda")` and `noisy_sample.to("cuda")` **unconditionally**. That is the legacy
> `braingen_diffuser_BraTS_v1 (2D)` path, which has therefore never worked on the CPU-only
> container (§9.3 — it also needs `diffusers`, which was not installed until now). On a GPU host
> those two lines would stop raising, which is a *change of behaviour in a model nobody has
> validated*. It is out of scope here: do not treat it as fixed, and do not advertise that model
> as working because the container gained a GPU.

#### 8.0.2 Gallery mode's way out: the parameter space is finite

Every conditioning input comes from a fixed shipped bank, so a visitor cannot supply anything
of their own. The only things they can vary are four dropdowns:

> Tumour (2) × Lobe (6) × Slice Location (3) × Tumour Size (3)

and *Lobe* and *Tumour Size* are meaningless when Tumour = "Without Tumor". So the entire
user-visible space is

> 6 × 3 × 3 = **54** with-tumour cells + **3** without-tumour cells = **57 cells**

enumerated by `conddiff_core.iter_cells()`, with `conddiff_core.cell_id()` naming each one
(`with_frontal_middle_moderate`, `without_middle`). Render *K* seed variants per cell **once**,
on a Great Lakes GPU, at the paper's full **200 DDIM steps**, ship the PNGs, serve them.

At *K* = 4 that is 228 FLAIR PNGs plus 216 mask PNGs — **on the order of 10–15 MB total**
(estimate; measure it after the first render). Against a 341 MB checkpoint plus a conditioning
bank the container could not use anyway, that is the difference between a deployment that fits
on the existing Lightsail instance and one that does not exist.

That finiteness is exactly what makes gallery mode possible **and** exactly what makes it a
lookup table (§9.2) — which is why it is the interlock and not the product.

#### 8.0.3 Gallery mode is a real fallback, not a stub — say so explicitly

Gallery mode is **not** dead code and **not** a placeholder, and if the GPU host is ever retired
this is what the site falls back to. The instinct is to read "pre-rendered" as "cheaper,
therefore worse"; on image quality it is the reverse:

- **Step count.** The gallery is rendered at **200 steps — the paper's value**, the same one
  `eval_metrics.py` used to produce the published PSNR / SSIM numbers. Live mode defaults to
  `CONDDIFF_STEPS=50`, and 50 is a *placeholder that has never been quality-checked against
  200* by either repo. On a GPU that gap is closable rather than fatal: DDIM cost is linear in
  the step count, so if the measured ≈ 1 s/sample is at 50 steps then 200 steps is ≈ 4 s —
  still inside the health-check window. **Measure it, then raise `CONDDIFF_STEPS` to 200**
  (§8.10.6). Until that is done, gallery mode is the mode showing the paper's images.
- **Same model, same code.** Every gallery pixel came out of this same checkpoint through
  `conddiff_core.py`, the byte-identical shared module (§5.4). It is not a mock, not a
  placeholder, and not a hand-picked figure lifted from the paper.
- **Same honesty guarantees.** The renderer measures the lesion it actually drew
  (`lesion_px`, `core_px`, `lobe_px`, `coverage`, `size_realized`, `size_in_band`) and writes
  those numbers into `manifest.json` per variant; `inference()` copies them into the database
  record. Any field the manifest does not carry is **omitted and named in `not_recorded`** —
  nothing is ever synthesised to fill a gap, because a plausible-looking invented lesion area
  is exactly the kind of quiet lie this design exists to prevent.
- **It imports nothing heavy at runtime.** Gallery mode never imports `torch`, never imports
  `diffusers`, never opens the checkpoint, never reads a `.npy`. (`diffusers` and `safetensors`
  *are* now pinned in the shared requirements — §8.6 explains why they are installed on both
  targets rather than only the GPU one — but gallery mode never executes an import of either.)

What it genuinely costs is in §9.2: the set is finite, so two users who pick identical
parameters draw from the same *K* variants, and changing any shape constant means re-rendering
on the cluster rather than redeploying a container. And the UI must still disclose that the
image was rendered in advance (§10, decision 3). Serving a file made three weeks ago from a
button labelled "Generate", disclosed only in a database column that no page renders and that
is not written at all in playground mode, is not disclosure.

#### 8.0.4 Why there is **no** async job queue — and why that is not a shortcut

Earlier drafts of this document (and `conddiff_inference.py`'s own module docstring, which is
stale on this point) say live mode "requires an async job pattern": `202 Accepted` + a job id,
a `GET /jobs/{id}` status endpoint, a worker or `run_in_executor` so the event loop stays free,
and a frontend that polls. **That requirement was never a property of diffusion. It was a
property of CPU latency**, and it disappears with the CPU.

The chain is short enough to check line by line:

1. `api.py:81` is `async def generate_images(...)` calling **synchronous**
   `image_generation.generate_image(...)`, so the request occupies uvicorn's event loop for its
   whole duration and `/health` cannot answer meanwhile.
2. The health check kills the container after **~65 s** of that (5 s timeout × 30 s interval ×
   threshold 2).
3. A CPU sample is **minutes** → the kill fires every time → you must get the work off the
   event loop → job queue.
4. A GPU sample is **~1–2 s** → the loop is blocked for ~1–2 s → `/health` answers on the very
   next poll → **nothing to fix.**

So the GPU decision does not merely make live mode *faster*; it removes an entire scoped
sub-project — a new endpoint, a job store, frontend polling, and three further timeouts that
are not ours to configure (Vercel function duration, the Lightsail load-balancer idle timeout,
the browser `fetch` with no `AbortController`; §9.1). **No frontend change and no queue** is
therefore a consequence of the host choice, not a corner cut.

Two conditions keep it true, and both are checkable rather than hoped-for:

- **`MAX_IMAGES = 1`** (`conddiff_inference.py:432`, enforced by `image_generation.py`). The UI
  offers up to 5 images and the backend loops sequentially, so without the clamp a request is
  5 × the latency. Do not raise it without redoing this arithmetic.
- **The device must actually be `cuda`.** A CPU-only torch wheel on a GPU host still *works* —
  every request succeeds, at ~2 min instead of ~1 s, with correct images and no error anywhere
  — and then every number above is wrong and the health check starts killing the container.
  That is why `selftest()` resolves the device eagerly in live mode and reports `device`,
  `cuda_available`, `torch_version`, `cuda_built` and `gpu_name` (§8.7, §8.10.6). **Check those
  five fields before timing anything.**

---

### 8.1 Runbook, step 1 — export the conditioning bank (Great Lakes)

Everything in §8.1–8.3 runs on Great Lakes, because the checkpoint and the validation slices
exist **only** there.

`Summer2026/scripts/export_for_website.py` (793 lines, written) does the conditioning-bank
half: it scans `data/processed/slices/val/`, applies the three admissibility gates, selects
slices per (lobe, level) cell, writes `cond_slices/*.npy` plus `manifest.json`, copies and
SHA-256s the checkpoint, and verifies that the checkpoint loads with `strict=True`.

```bash
ssh <uniqname>@greatlakes.arc-ts.umich.edu
cd ~/Summer2026
module load python
source ~/envs/brainmri/bin/activate        # required even for --dry-run: the script's
                                           # constants are imported from augment_tumor, which
                                           # imports torch / diffusers / scipy / matplotlib

# 1. cheap reconnaissance: scan every 10th slice, write nothing, print the coverage table
python scripts/export_for_website.py --dry-run --stride 10

# 2. the real export
python scripts/export_for_website.py \
    --out /scratch/dinov_root/dinov1/<uniqname>/web_bundle \
    --per-cell 4
```

Read the printed **6 × 3 coverage table** before going further. An empty cell is a dropdown
combination the site must *disable*, never silently substitute (§6.1); the script records them
in `manifest.json["empty_cells"]`, and the renderer in §8.3 must propagate that list so
`/conddiff-selftest` can report them as `cells_missing`.

**Known gap, flagged rather than hidden:** `export_for_website.py` still does
`from augment_tumor import LOBES, ATLAS0, PROB_TH, MIN_BRAIN, MIN_LOBE`, i.e. it takes its
constants from the **scipy-dependent** research script rather than from `src/conddiff_core.py`.
The values agree today, but only because a human keeps them agreeing — which is precisely the
divergence §5.4 exists to eliminate. Repointing that import at `src.conddiff_core` is a
one-line change and should be made before the first production render.

---

### 8.2 Runbook, step 2 — prove the two copies of the core agree

**`Summer2026/scripts/verify_conddiff_port.py` — not yet written.** It is the gate between
"the renderer ran" and "the renderer ran the code the server ships", and the gallery's entire
honesty claim rests on it, so its contract is recorded here:

1. **Byte-identity.** SHA-256 of `Summer2026/src/conddiff_core.py` must equal SHA-256 of
   `Brain-Image-Generator/backend/inference/conddiff_core.py`. Anything else is a hard fail.
   At the time of writing both read
   `87785659fcbf23415166fe1f682f8e67984b4f5fcda00b355d1fe2007f5127ed` (795 lines) — but compute
   it, do not trust this document; the hash the manifest carries is the one that matters.
2. **The numpy reimplementations still match scipy.** `conddiff_core._gaussian_blur` and
   `_largest_connected_component` replace `scipy.ndimage.gaussian_filter` and
   `scipy.ndimage.label` so the backend never needs scipy (§8.6). The cluster *has* scipy, so it
   is the only place the two implementations can be compared: assert max-absolute-difference on
   random fields (scipy defaults `truncate=4.0`, `mode='reflect'`) and label agreement on random
   binary masks (4-connectivity).
3. **The cell space round-trips.** `iter_cells()` yields 57 cells; every one survives
   `params_to_spec(...)` → `cell_id(...)` back to the same id; junk input lands somewhere in the
   space rather than raising.

Run it on the cluster immediately before rendering, and again on the laptop after the transfer:

```bash
python scripts/verify_conddiff_port.py \
    --backend-core /path/to/Brain-Image-Generator/backend/inference/conddiff_core.py
```

If the two files ever diverge, **do not render**. Copy one over the other, re-run, then render.

---

### 8.3 Runbook, step 3 — render the gallery on the GPU

**`Summer2026/scripts/render_gallery.py` + `scripts/render_gallery.sbatch` — not yet written.**
Contract:

- import `src/conddiff_core.py` (never `augment_tumor`), walk `iter_cells()`, and draw *K*
  distinct seeds for each of the 57 cells;
- for each (cell, seed): choose a conditioning slice from the exported bank with the same
  selection logic the live path uses, call `synth_mask_sized`, `build_cond`, run **200** DDIM
  steps, and write `v<NN>_flair.png` — plus `v<NN>_seg.png` for with-tumour cells only, since
  an all-zero mask is not an image;
- write `manifest.json` with `"schema": "conddiff_gallery/1"` (the exact string
  `conddiff_inference.GALLERY_SCHEMA` gates on — a mismatch is meant to fail loudly at startup),
  plus `model_version`, `rendered_utc`, `core_sha256` (the §8.2 hash, computed on the file the
  renderer actually imported), `renderer: {ddim_steps, device, torch, diffusers}`, `cells`, and
  `empty_cells`;
- per variant, record the **realised measurements**: `seed`, `ddim_steps`, `bank_file`,
  `bank_patient`, `bank_z`, `bank_brain_px`, `lesion_px`, `core_px`, `lobe_px`, `coverage`,
  `centre_axis0`, `centre_axis1`, `tries`, `size_requested`, `size_realized`, `size_in_band`.
  These are what the site later tells the user about their image. The schema block in
  `conddiff_inference.py` SECTION 6B is the authoritative field list; read it, do not
  reconstruct it from here.

The sbatch follows this repo's existing idiom (`scripts/train.sbatch`, `train_3d_*.sbatch`):

```bash
#!/bin/bash
#SBATCH --job-name=render-gallery
#SBATCH --account=dinov_owned1
#SBATCH --partition=gpu-rtx6000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=outputs/render_gallery_%j.log

module load python
source ~/envs/brainmri/bin/activate
cd ~/Summer2026
python -u scripts/render_gallery.py \
    --bundle /scratch/dinov_root/dinov1/<uniqname>/web_bundle \
    --out    /scratch/dinov_root/dinov1/<uniqname>/conddiff_gallery \
    --variants 4 --steps 200
```

```bash
sbatch scripts/render_gallery.sbatch
squeue -u $USER
tail -f outputs/render_gallery_<jobid>.log
```

**Sizing the job.** 57 cells × 4 variants = 228 samples × 200 steps = **45 600 U-Net forward
passes**. On the RTX 6000 that is minutes-to-an-hour of GPU work, not days; the four-hour wall
time above is slack, not an estimate. Batch one cell's seeds together if it matters.

---

### 8.4 Runbook, step 4 — bring the bundle down (`sftp`, not `scp`)

Use **`sftp`**. `scp` on Great Lakes is unreliable because the U-M login banner is printed into
the same channel `scp` uses for its own protocol and corrupts the transfer; `sftp` negotiates a
proper subsystem after login, so the banner is harmless.

**Protect the shared repo before dropping anything into it** — §8.5 explains why this is not
optional:

```bash
cd C:/Users/alexl/code/Research/socr/Brain-Image-Generator
printf 'backend/inference/conddiff_gallery/\nbackend/inference/conddiff_bundle/\nbackend/inference/model/*.pt\nbackend/inference/model/*.pth\nbackend/inference/model/*.safetensors\n' >> .git/info/exclude
git status --porcelain          # re-run after every file drop; it must say nothing about them
```

`.git/info/exclude` is local-only and modifies no tracked file, which is exactly what is wanted
in a repository this work must never commit to.

```bash
sftp <uniqname>@greatlakes.arc-ts.umich.edu
sftp> lcd C:/Users/alexl/code/Research/socr/Brain-Image-Generator/backend/inference
sftp> cd /scratch/dinov_root/dinov1/<uniqname>
sftp> get -r conddiff_gallery
sftp> bye
```

Fetch the **folder**, not its contents: `get -r .` is not a portable sftp idiom, and pulling the
contents would scatter `manifest.json` and the per-cell directories next to the `.py` files
instead of inside the folder the backend looks for.

Then verify the transfer locally, before anything is built:

```bash
cd C:/Users/alexl/code/Research/socr/Brain-Image-Generator/backend/inference
python -c "import json; m=json.load(open('conddiff_gallery/manifest.json')); print(m['schema'], len(m['cells']), 'cells', sum(len(c['variants']) for c in m['cells'].values()), 'variants')"
python -c "import glob; print(len(glob.glob('conddiff_gallery/*/*.png')), 'png files actually on disk')"
python -c "import hashlib; print(hashlib.sha256(open('conddiff_core.py','rb').read()).hexdigest())"
```

The first line only reads the manifest against itself, so on its own it cannot notice a PNG
that sftp silently dropped — that is what the second is for. The third must equal the
`core_sha256` recorded in the manifest.

---

### 8.5 Runbook, step 5 — where each file goes, and the one file that can never be committed

**Gallery deployment (this is the whole payload):**

```
backend/inference/conddiff_gallery/
├── manifest.json                        # schema "conddiff_gallery/1" + provenance + per-variant stats
├── with_frontal_middle_moderate/
│   ├── v00_flair.png   v00_seg.png
│   └── v01_flair.png   v01_seg.png      # ... K variants
├── without_middle/
│   └── v00_flair.png                    # no seg: an all-zero mask is not an image
└── ...                                  # the 57 cells, minus any listed in empty_cells
```

**Live deployment (the default — `CONDDIFF_MODE` defaults to `live`):**

```
backend/inference/conddiff_bundle/                # the bundle folder, dropped in whole
├── manifest.json                          # the index -- JSON, not CSV; keys: slices[], supported_cells, empty_cells
├── cond_slices/*.npy                      # 9-channel float16 conditioning slices
└── diffusion_ema.pt                       # the exporter's own copy of the checkpoint
backend/inference/model/diffusion_ema.pt          # OPTIONAL and preferred; or diffusion_ema_fp16.safetensors
```

> **The three-way path mismatch that used to be here is fixed — do not "reconcile" it again.**
> An earlier revision of this document, of `conddiff_inference.py` and of `backend/config.py`
> all described a `conddiff_bank/manifest.csv` layout with the slices at the bank root. Nothing
> ever produced that layout. The backend now reads exactly what
> `scripts/export_for_website.py` emits, so the bundle is dropped in whole and renamed, with no
> file shuffling and no change to the exporter:
>
> | what reads it | constant | value |
> |---|---|---|
> | `conddiff_inference.py:343` | `BANK_DIR` | `inference/conddiff_bundle` |
> | `conddiff_inference.py:344` | `MANIFEST_PATH` | `inference/conddiff_bundle/manifest.json` |
> | `conddiff_inference.py:345` | `SLICE_SUBDIR` | `cond_slices` |
> | `conddiff_inference.py:367` | `CKPT_CANDIDATES` | `model/diffusion_ema_fp16.safetensors`, `model/diffusion_ema.safetensors`, `model/diffusion_ema.pt`, `conddiff_bundle/diffusion_ema.pt` |
> | `api.py` `/check-models` | (re-hardcoded copies) | the same four paths plus both manifests |
>
> `export_for_website.py:148-150` defines `SLICE_SUBDIR = "cond_slices"`, `CKPT_NAME =
> "diffusion_ema.pt"`, `MANIFEST = "manifest.json"`, and its closing epilogue prints the same
> `backend/inference/conddiff_bundle/` destination. If you are ever tempted to make the
> exporter emit a CSV, re-read this table first: the CSV expectation was the bug.

#### The weight-shipping constraint, stated plainly

**The checkpoint cannot be committed, whatever `.gitignore` says.**

- `models/diffusion_ema.pt` is **≈ 341 MB** — 85.3 M fp32 parameters, a bare EMA `state_dict`.
- **GitHub rejects any single file over 100 MB.** That is a hard server-side limit on the push,
  not a warning. `.gitignore` does not change it, and neither does asking.
- An **fp16 `.safetensors` export is still ≈ 171 MB** — half of 341, and still 71 MB over the
  limit. Halving does not solve it. Nothing short of quantisation or pruning would, and neither
  is something to do to a checkpoint that produced published numbers.
- The five existing GAN `.pth` files *are* small enough to live in the repo, which is exactly
  why the habit around them misleads here. `backend/inference/model/` is empty in git; a human
  drops the weights in before `docker build`, `COPY . .` bakes them into the image, and
  `.dockerignore` does not exclude `inference/`. This model must use that same out-of-band path
  — and unlike the GANs it has **no soft failure**: a stray `git add -A` stages a 341 MB file
  into `github.com/SOCR/Brain-Image-Generator`, the push is rejected, and unwinding it needs an
  interactive rebase on the lab's shared repository.

**In gallery mode the server never needs the checkpoint at all**, and that is a real advantage
worth stating in its own sentence: no 341 MB file in the Docker image, no 341 MB of fp32 weights
in a 1 GB container's RSS, no `torch.load` pickle surface, and no chance of the largest file in
this project ending up in the lab's shared repo. The entire deliverable is ~10–15 MB of PNGs
and a JSON index.

---

### 8.6 Runbook, step 6 — requirements, the two build targets, and `lightsail-config.json`

#### 8.6.0 One pin list, two torch builds

The backend is now built for **two targets that differ in exactly one line** — which build of
torch they install:

```
backend/requirements-common.txt    every pin that is identical on both targets
backend/requirements.txt      (CPU)  -r requirements-common.txt  +  torch==2.0.1+cpu
backend/requirements-gpu.txt  (GPU)  -r requirements-common.txt  +  torch==2.0.1+cu118
```

The alternative — copy the list and change one line — was rejected for a specific reason, not
for tidiness: the five GAN models are pinned to `numpy==1.24.3` / `Pillow==10.0.0` for torch
2.0.1 compatibility and every one of them loads weights with `strict=False`
(`cgan_inference.py:174` and siblings), so a dependency that drifted on only one target would
**not raise** — it would quietly return randomly-initialised generators that still emit
brain-shaped output. One shared file makes that class of bug unrepresentable. The full argument
is in `requirements-common.txt`'s own header; read it there rather than trusting this summary.

Two mechanics that are easy to get wrong:

- `-r requirements-common.txt` is resolved **relative to the file that contains it**, not to the
  process working directory — so `pip install -r requirements.txt` works from anywhere, but
  **each Dockerfile must `COPY` both files before running pip**, or pip dies on a missing include.
- `requirements-common.txt` is **not installable on its own**: it deliberately contains no
  torch, and nothing in it requires torch (see the `diffusers` note below), so installing it
  alone silently produces an environment with no torch and no error.

`diffusers==0.38.0` and `safetensors==0.8.0` live in the **shared** file, not the GPU one, even
though only live mode imports them. `CONDDIFF_MODE` is an environment variable, not a build
flag: someone can set `CONDDIFF_MODE=live` on a CPU container, and the honest failure there is
a slow image, not `ModuleNotFoundError: diffusers` from inside a request handler. It also means
the smoke test (§8.10.5) answers the diffusers/torch question **once, for both images**.

Two pins that are forced rather than chosen, both verified rather than assumed:

- **`diffusers==0.38.0`** because that is the version that *wrote* `diffusion_ema.pt` (measured
  on Great Lakes, 2026-09-08). diffusers renamed the attention block's parameters across
  versions (`query/key/value/proj_attn` → `to_q/to_k/to_v/to_out.0`); `strict=True` (§2.2) turns
  a skew into a loud failure instead of a randomly-initialised network, but only if the pin is
  right in the first place.
- **`safetensors==0.8.0`** because diffusers 0.38.0's metadata declares
  `Requires-Dist: safetensors>=0.8.0-rc.0`. The earlier `0.4.5` pin did not make the set merely
  old — it made it **unsatisfiable**: pip exits `ResolutionImpossible`, `RUN pip install` fails,
  `docker build` fails, and `deploy-to-lightsail.sh` (`set -e`) aborts at Step 1, blocking every
  redeploy of the CPU container including ones that have nothing to do with this model.
  Verified 2026-09-09 with `pip install --dry-run --python-version 3.10 …`; it still co-resolves
  with `supabase==2.0.3`, which holds httpx `<0.25`.

#### 8.6.1 What each mode actually needs at runtime

**Gallery mode needs no new Python dependencies. None.** `conddiff_inference.py` imports only
the standard library, `numpy` (already pinned at 1.24.3) and `conddiff_core`, which itself
imports only the standard library and numpy. `torch` is imported lazily inside `_device()`,
`_load_state_dict()`, `_get_model()` and `conddiff_core.build_cond` / `render_flair`;
`diffusers` lazily inside `_require_diffusers()`; `PIL` lazily inside `to_png_bytes`, which
gallery mode never calls because it serves bytes that are already PNG. **`scipy` is not required
in either mode** — the two `scipy.ndimage` calls are reimplemented in ~40 lines of numpy in the
shared core. That is the single biggest reason the shared image does not have to be rebuilt
against a fresh resolver run that could disturb `numpy==1.24.3` / `Pillow==10.0.0` for the five
GAN models that currently work.

#### 8.6.2 `lightsail-config.json`, and the trap that it is not what deploys

What *does* need editing is the deployment JSON — and since the code default changed, this is
no longer cosmetic. `conddiff_inference.py:309` now reads

```python
_MODE_RAW = os.environ.get("CONDDIFF_MODE", "live")
```

so **an unset variable means live**, not gallery. `backend/lightsail-config.json` deploys to a
`micro` container (0.25 vCPU / 1 GB), which live mode cannot survive: the fp32 checkpoint alone
is 325 MB, `api.py:81` is an `async def` wrapping synchronous work so a long request blocks the
uvicorn event loop, and the healthCheck below (5 s timeout, 30 s interval, threshold 2) then
replaces the container after ~65 s — with no Python traceback anywhere. The Lightsail file
therefore has to pin the mode *down*:

```json
"environment": {
  "SUPABASE_URL": "",
  "SUPABASE_KEY": "",
  "FRONTEND_URL": "",
  "CONDDIFF_MODE": "gallery"
}
```

This is set in the file (JSON cannot carry a comment, which is why the reason is written here).

**But setting it in `lightsail-config.json` alone does nothing, and that trap is worth stating
plainly.** No deploy script reads that file. All four — `deploy-to-lightsail.sh`, `deploy.sh`,
`deploy-to-lightsail.bat`, `deploy.bat` — *generate* `deployment.json` inline with a heredoc /
`echo` block and hand it to `aws lightsail create-container-service-deployment
--cli-input-json`. Before this was fixed, that generated block listed only `SUPABASE_URL`,
`SUPABASE_KEY` and `FRONTEND_URL`, so the deployed container received **no** `CONDDIFF_MODE`
and therefore ran in **live** mode, whatever `lightsail-config.json` said. Each of the four
scripts now defaults `CONDDIFF_MODE` to `gallery` (overridable from `.env`) and emits it into
the JSON it generates; `lightsail-config.json` remains the human-readable record of intent.

Read the pairing as: **the Lightsail deploy scripts are the legacy CPU/micro target and pin
`CONDDIFF_MODE=gallery` in the deployment they actually submit; the GPU host is the live target
and sets `CONDDIFF_MODE=live` explicitly — better than leaving it unset, because an operator
reading the deployment config should be able to tell which mode production is in without
reading Python.**

---

### 8.7 Runbook, step 7 — verify from outside the container

Two endpoints, in this order.

**`/check-models` answers only "does the file exist".**

```bash
curl -s https://<backend-host>/check-models | python -m json.tool
```

> **Closed.** `backend/api.py`'s `check_model_files()` now lists the five GAN `.pth` files, the
> four live-mode conddiff checkpoint/bundle candidates (`CONDDIFF_CHECKPOINT_PT`,
> `CONDDIFF_CHECKPOINT_SAFETENSORS`, `CONDDIFF_CHECKPOINT_IN_BUNDLE`,
> `CONDDIFF_BUNDLE_MANIFEST`) **and** `CONDDIFF_GALLERY_MANIFEST`. Whichever mode is deployed,
> the assets the other mode would have used are *correctly* reported missing, so read the
> block that matches `mode` in `/conddiff-selftest` rather than expecting the whole table to be
> green. The rule in that function's comment still stands: these paths must stay in lockstep
> with `conddiff_inference.py`'s own constants, because that module is what actually reads
> them.

**`/conddiff-selftest` answers the questions that decide whether a request will succeed.** It is
already mode-aware, because it simply returns `conddiff_inference.selftest()`:

```bash
curl -s https://<backend-host>/conddiff-selftest | python -m json.tool
```

On a healthy gallery deployment, check these fields:

| Field | Expected |
|---|---|
| `mode` | `"gallery"` |
| `ready` / `reason` | `true` / `null`. `ready: false` means `_check_assets()` failed at import and **every** request will refuse |
| `cells_total` | `57` |
| `cells_covered` | `57`, or 57 minus the number of legitimately empty cells |
| `cells_missing` | `[]`, or exactly the cells the UI disables. Every id here is a dropdown combination the site offers and the backend will refuse |
| `variants_total` | `cells_covered × K` |
| `rendered_steps` | `200` — what the images were *actually* rendered at, not this container's `CONDDIFF_STEPS` |
| `rendered_utc`, `rendered_device` | when and on what the gallery was made |
| `core_sha256` | must equal the SHA-256 of this container's own `inference/conddiff_core.py` — the proof that the shipped code and the code that made the pictures are the same code |

That last row closes the loop opened in §5.4. Nothing else in the system can detect a drifted
core after the fact.

On a healthy **live** deployment — which is now the default — `selftest()` returns a different
block, and these are the rows that decide whether the GPU migration actually happened:

| Field | Expected | What the wrong value means |
|---|---|---|
| `mode` | `"live"` | anything else and the env var overrode the default |
| `ready` / `reason` | `true` / `null` | `false` means `_check_assets()` failed at import and **every** request will refuse |
| `device` | `"cuda"` | **`"cpu"` on a GPU host is the silent failure this whole migration exists to avoid.** Every request still succeeds, at ~2 min a sample instead of ~1 s, with correct images and no error anywhere |
| `cuda_available` | `true` | `false` with a GPU present ⇒ the wheel or the runtime is wrong, not the card |
| `torch_version` | ends in `+cu…` | a `+cpu` suffix can never see the card, however many GPUs `nvidia-smi` lists |
| `cuda_built` | e.g. `"11.8"` | must be a CUDA the host driver supports; a mismatch presents as `cuda_available: false` |
| `gpu_name` | e.g. `"Tesla T4"` | `null` means torch is not talking to the card |
| `checkpoint_found` | `true` | `false` ⇒ none of the four `CKPT_CANDIDATES` exist; the checkpoint never landed |
| `bank_slices` | the exporter's `--per-cell` × covered cells | `0` ⇒ the bundle did not land, or `manifest.json` parsed to nothing |
| `ddim_steps` | whatever `CONDDIFF_STEPS` is set to | `50` is a **placeholder**, not a measured recommendation (see §8.8) |

Check `device` and `cuda_available` **before** timing anything. They are cheap and they answer
the question a stopwatch only answers ambiguously.

Finally, generate one image through the real UI and confirm the recorded parameters carry
`"rendering": "gallery"`, a `gallery_cell` matching the four dropdowns, a `gallery_variant`, the
`seed` that actually produced the image (from the manifest) alongside `selection_seed` (this
request's seed, which only chose *which* variant to serve), and either the full lesion
statistics or an explicit `not_recorded` list naming what the manifest did not carry.

Note that at runtime `os.getcwd() == /app` (the Dockerfile sets `WORKDIR /app` and copies the
*contents* of `backend/` into it), so every path above resolves to `/app/inference/...`.
**Locally the server must be started from inside `backend/`** or every path in every inference
module breaks — that is not specific to this model.

---

### 8.8 The road not taken — what live mode *on Lightsail* would have cost

> **Read this as the argument for the GPU host, not as a plan.** Live mode is now deployed
> (§8.10), so items 1 and 2 below are not work anyone has to do — they are the *price of the
> alternative*, kept because "just bump the Lightsail power and flip the flag" is the proposal
> that keeps coming back, and this is the reply. Item 3 has since been done concretely (§8.6.0)
> and item 4 is §8.10.3. Nothing here is needed for the gallery deployment either.

The environment variable was never the whole change: on Lightsail it was one of *four* things
that all had to be true.

**1. A bigger instance.** `micro` (0.25 vCPU / 1 GB) cannot hold the model. Estimated peak
resident memory for a single 256 × 256 sample is **~1.3–1.6 GB**: ~350 MB for the torch import,
341 MB of fp32 weights, ~165 MB of skip connections held live through the forward pass, a
transient attention score tensor of ~268 MB (up to ~536 MB with the softmax buffer, because
torch 2.0.1 has no CPU flash-attention kernel), and ~150 MB of numpy / Supabase client /
conditioning data. The failure presents as an unexplained container restart with **no Python
traceback**. `medium` (4 GB) is the first plausibly safe power; `large` (8 GB, 2 vCPU) also
halves wall-clock. That takes the bill from ~$10/month to ~$40–80/month. No existing deploy
script changes power — `deploy-to-lightsail.sh` only creates a deployment — so it takes:

```bash
aws lightsail update-container-service \
  --service-name socr-image-gen-backend --power large --region us-east-1
```

**2. An async job pattern this application does not have.** Even on `large`, 200 steps is
~72 minutes and 50 steps ~18 minutes (§9.1) — both far beyond the ~65 s health-check kill
window, which cannot be raised away while `/generate` blocks the event loop. Live mode requires
reworking `/generate` into submit-and-poll: `202 Accepted` plus a job id, the work handed off so
the loop stays free and `/health` keeps answering (`run_in_executor`, a worker process, a task
queue — any of them), a `GET /jobs/{id}` status endpoint, and a frontend that polls it. Three
further timeouts stack behind that and only one is ours: the Vercel route handler's function
duration (10 s on Hobby, 300 s when raised, 800 s hard maximum on Pro), the Lightsail load
balancer's non-configurable idle timeout (widely reported ~60 s — measure it, do not assume),
and the browser `fetch`, which has no `AbortController` and simply spins. This is a scoped
project of its own, not a flag.

**3. `requirements.txt` additions — this one has since been done, concretely.** The pins and
the reasoning now live in §8.6.0 and in `requirements-common.txt`'s header:
`diffusers==0.38.0` (the version that wrote the checkpoint, measured) and `safetensors==0.8.0`
(forced by diffusers' `Requires-Dist`, and an unsatisfiable requirement set below it). The
torch pin is the one genuinely open question and has its own decision tree at §8.10.5.

Three cautions from the original analysis that are still live, and one that is now settled:

- **Never bump numpy** past 1.24.x — the existing pin is what keeps it compatible with
  torch 2.0.1 and with the five GAN checkpoints.
- **Pin the transitives if a rebuild ever drifts.** `diffusers` only requires
  `huggingface-hub>=0.23.2`, so an unpinned rebuild months from now can resolve a hub 1.x
  release and break the image with zero code changes. Today's build is reproducible only
  because it was built today; pin them the first time it is not.
- **Side effect to know about:** installing `diffusers` pulls in `requests`, which was not
  previously installed. The Dockerfile's `HEALTHCHECK` runs
  `python -c "import requests; requests.get(...)"` and therefore used to throw `ImportError`
  every 30 s — harmlessly, because Lightsail uses its own `publicEndpoint` health check
  instead. Adding `diffusers` silently **activates** that Docker-level healthcheck, which on a
  plain `docker run` host (§8.10.2) is the healthcheck that actually reports. Decide
  deliberately whether to keep or delete it; do not let it change behaviour by accident.
- **Settled:** "the diffusers version that trained the checkpoint is not recorded anywhere" was
  true when written and is no longer. It was **measured on Great Lakes, 2026-09-08: diffusers
  0.38.0, torch 2.11.0+cu128.** `strict=True` (§2.2) is what turns a version skew — diffusers
  has historically renamed attention parameters, `query/key/value/proj_attn` →
  `to_q/to_k/to_v/to_out.0` — into a loud failure instead of a randomly-initialised network.
- **`scipy` is *not* needed**, in either mode. This supersedes an earlier draft of this document
  that recommended adding it: the two `scipy.ndimage` calls now live as numpy reimplementations
  in the shared core (§5.4). Verify them against scipy on the cluster (§8.2) rather than
  trusting them.

**4. The live-mode assets in the layout the backend actually reads** — the whole bundle
`export_for_website.py` emits, copied unrearranged to `inference/conddiff_bundle/`
(`manifest.json` + `cond_slices/*.npy` + its copy of `diffusion_ema.pt`); the checkpoint may
instead sit at `inference/model/diffusion_ema.pt`, which `CKPT_CANDIDATES` prefers. The older
`conddiff_bank/manifest.csv` + `conddiff_bank/slices/*.npy` layout described here previously
was never emitted by anything and is no longer what `conddiff_inference.py` opens (§8.5).

**And on Lightsail it would still have been a step down in image quality**, unless the instance
were large enough to afford 200 steps inside whatever time budget the async job got — which is
precisely the trade the GPU host removes: at ~1 s a sample, the paper's 200 steps are
affordable *and* synchronous. That is the sentence to lead with if anyone proposes going back
to a bigger CPU instance as an improvement: it buys a job queue, a slower image, and a bill,
and the capability it is reaching for (live sampling on arbitrary conditioning) is what
§8.10 already delivers.

---

### 8.9 Registration points — all six, or the failure is silent

| # | File | What it needs |
|---|---|---|
| 1 | `backend/image_generation.py` | the import and the `elif model_name == "braingen_CondDiffuser_BraTS_v1 (2D)":` branch — the **only** functional dispatch; it also clamps `n_images` to `cdiff.MAX_IMAGES` |
| 2 | `backend/inference/conddiff_inference.py` | the mode dispatch, the inference entry point, and its own tag-keyed `save_images` |
| 3 | `backend/inference/conddiff_core.py` | the shared core — byte-identical to `Summer2026/src/conddiff_core.py` (§5.4) |
| 4 | `components/projects/ParameterControlPanel.tsx` | the `models2D` entry and a `renderParameterFields()` branch (Tumour, Lobe, Slice Location, Tumour Size) |
| 5 | `components/projects/GenerationPageClient.tsx` | state for the four parameters and a `getModelParams()` branch emitting the keys `tumour` / `lobe` / `slice_location` / `tumour_size` |
| 6 | `backend/api.py` | `/models`, `/check-models`, `/conddiff-selftest` |

The model id string must be **byte-identical** in the frontend `models2D` entry, the
`getModelParams()` comparison, and the Python `elif` — including the space before `(2D)`.
Registering in only some of these produces failures that look like success (§9.3).

---

### 8.10 The GPU host runbook — the one to actually follow

This is the deployment. §8.1–8.4 still run first (they are what produces the bundle); §8.5–8.7
still describe where files go and how to verify. This section is the ordered sequence for
standing up the **live** deployment on an EC2 GPU instance, and every step says what it proves
so that a failure is attributable to a step rather than to "it doesn't work".

**Topology after this runbook.** The whole backend container moves — not just this model:

```
Vercel (Next.js)  ──►  app/api/generate/route.ts (server-side proxy)  ──►  EC2 g4dn.xlarge
                                                                            docker: api.py
                                                                            six models, one
                                                                            container, CUDA
Lightsail container service ──► stays deployable as the CPU fallback (CONDDIFF_MODE=gallery)
```

Two consequences to be clear-eyed about, both of which follow from "one container, six models":

- **The five GAN models move to the GPU box too.** That is safe — they hardcode CPU on the line
  after their `cuda` check (§8.0.1.1) — but it is not nothing: when the GPU host is stopped,
  *every* model on the site is unavailable, not just this one (§8.10.8).
- **The frontend never talks to the backend directly.** Every call is proxied server-side
  (`app/api/generate/route.ts:30`, `app/api/conddiff-cells/route.ts:29`), so there is no
  browser-to-backend CORS surface and no https-page-to-http-backend mixed-content block. TLS on
  the GPU host is still the right thing (the traffic crosses the public internet), but it is a
  security decision, not a functional blocker — which is why it is not step 1.

---

#### 8.10.1 Step 0 — the preconditions, checked before anything is launched

| Precondition | How you know |
|---|---|
| The bundle exists on Great Lakes | §8.1 ran; `web_bundle/manifest.json` + `cond_slices/*.npy` + `diffusion_ema.pt` are in `/scratch/...` |
| The two copies of the shared core are byte-identical | §8.2, or a manual `sha256sum` of both files. **Do this before deploying, not after** — in live mode nothing downstream will catch a drift, because there is no manifest hash to compare against (§5.4 mechanism 3 only closes the loop in gallery mode) |
| The five GAN `.pth` files are in `backend/inference/model/` | `ls backend/inference/model/` — they are **not in git** and `COPY . .` bakes whatever is there. An empty directory produces five silently broken models |
| An AWS account with EC2 GPU capacity in the target region | `g4dn.xlarge` requires a vCPU quota for "Running On-Demand G and VT instances" > 0. A fresh account's quota is often **0**, and the request takes hours-to-days. Check this *first*: it is the only step with a lead time you cannot compress |

---

#### 8.10.2 Step 1 — build the GPU image

The GPU image differs from the CPU image in exactly one installed package (§8.6.0), so it is
the same `Dockerfile` recipe pointed at `requirements-gpu.txt`. Two decisions worth making
deliberately:

**Build on the instance, not on the laptop.** The cu118 torch wheel is ~2 GB; building locally
means pushing a multi-gigabyte image over a home connection to a registry and pulling it down
again. Building on the instance from a git checkout plus the out-of-band weight files skips
both transfers. (The exception: if the lab wants an immutable, registry-tagged artefact for
reproducibility, build locally or in CI and push to ECR. That is a better practice and a slower
first deploy; do it once the thing works.)

**Mount the large assets, do not bake them.** Lightsail's container service has no volumes, so
the CPU deployment has to `COPY` everything into the image. A plain Docker host does not, and
that difference is worth using: the 325 MiB checkpoint and the `.npy` bank change on their own
schedule, and baking them makes every unrelated code change a multi-hundred-megabyte rebuild.
Mount them read-only instead. `conddiff_inference.py` resolves its paths from `os.getcwd()`,
which the Dockerfile sets to `/app`, so the mount targets are exactly
`/app/inference/conddiff_bundle` and `/app/inference/model`.

```bash
# on the EC2 instance, as the deploy user
git clone -b feat/conddiff-brats-v1 <repo-url> braingen
cd braingen/backend

# The GAN .pth weights are not in git (§8.5). Put them in place BEFORE the build, because
# `COPY . .` bakes the directory as it finds it and an empty model/ fails silently.
ls inference/model/*.pth        # expect the five GAN checkpoints

docker build -f Dockerfile.gpu -t braingen-backend:gpu .
```

> `Dockerfile.gpu` is the CPU `Dockerfile` with two changes: it installs
> `requirements-gpu.txt` instead of `requirements.txt`, and it must `COPY` **both**
> `requirements-gpu.txt` and `requirements-common.txt` before the `pip install`, because the
> `-r` include is resolved relative to the file that contains it (§8.6.0). A base image with
> the CUDA runtime already present (`nvidia/cuda:*-runtime-*`) is *not* required — the pip
> `torch==2.0.1+cu118` wheel bundles the CUDA runtime libraries it needs; what must come from
> the host is the **driver**, via the NVIDIA Container Toolkit (§8.10.3).

A build that succeeds proves that the pin set resolves. **It proves nothing about whether
diffusers can import on torch 2.0.1** — see §8.10.5, which is the single most important
paragraph in this section.

---

#### 8.10.3 Step 2 — launch the instance and give Docker the GPU

**Instance:** `g4dn.xlarge` — 1 × NVIDIA T4 (16 GB VRAM), 4 vCPU, 16 GB RAM. Sized by what the
job needs, not by what is cheapest: 16 GB VRAM is far more than an 85 M-parameter U-Net at
256 × 256 needs (the model plus a batch of 1 is comfortably under 2 GB), so the T4 is chosen
for being the smallest current-generation NVIDIA card AWS rents rather than for headroom. The
16 GB of *system* RAM is what actually matters for the rest of the container: torch's import,
the fp32 checkpoint, and five GAN models that all run on CPU.

**Root volume:** allow for the image (torch cu118 is ~2 GB installed, so budget 8–10 GB for the
image alone), the bundle, and Docker's build cache. 60–100 GB gp3 is unremarkable; the default
8 GB is not enough and fails partway through `docker build` with a confusing error.

**AMI:** use an AMI that already has the NVIDIA driver and the container toolkit — AWS's Deep
Learning AMI (Ubuntu) is the standard choice and removes the single most annoying failure mode
of this step. **Do not copy an AMI id out of this document**: they are region-specific and are
replaced constantly. Look up the current one:

```bash
aws ec2 describe-images --owners amazon --region us-east-1 \
  --filters "Name=name,Values=Deep Learning*Ubuntu*" "Name=state,Values=available" \
  --query 'reverse(sort_by(Images,&CreationDate))[:5].[ImageId,Name,CreationDate]' \
  --output table
```

```bash
aws ec2 run-instances \
  --region us-east-1 \
  --image-id <ami-from-the-query-above> \
  --instance-type g4dn.xlarge \
  --key-name <your-keypair> \
  --security-group-ids <sg-...> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=braingen-gpu}]'
```

**Security group:** inbound `22` from **your IP only**, and `443` (plus `80` only if you are
terminating TLS with an ACME challenge) from anywhere. Do **not** expose `8000` to the world —
put the container behind nginx/Caddy, or bind it to localhost and reverse-proxy. Uvicorn is
listening with `--workers 1` and no rate limiting; `/generate` performs a GPU computation and
writes to Supabase on every call, which makes an open port a way to spend the lab's money.

Then prove the driver is visible from *inside* a container — not just on the host, which is the
mistake that costs an afternoon:

```bash
nvidia-smi                                    # host: driver + the max CUDA version it supports
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If the second command fails but the first succeeds, the driver is fine and the **NVIDIA
Container Toolkit** is missing or not configured (`nvidia-ctk runtime configure --runtime=docker`
followed by a Docker restart, on an AMI that did not ship it). Everything downstream will
silently fall back to CPU without it — `torch.cuda.is_available()` returns `False` inside the
container and every request still succeeds, slowly. That is the failure §8.0.4 warns about, and
this is where it is cheapest to catch.

---

#### 8.10.4 Step 3 — get the bundle onto the instance

The bundle is produced on Great Lakes (§8.1). Two routes; prefer the first.

**Direct, cluster → instance.** `sftp` *from* the EC2 box, pulling. Use `sftp`, not `scp`: the
U-M login banner is printed into the same channel `scp` uses for its own protocol and corrupts
the transfer, whereas `sftp` negotiates a subsystem after login and the banner is harmless
(§8.4).

```bash
# on the EC2 instance
sudo mkdir -p /opt/braingen && sudo chown $USER /opt/braingen
cd /opt/braingen

sftp <uniqname>@greatlakes.arc-ts.umich.edu
sftp> lcd /opt/braingen
sftp> cd /scratch/dinov_root/dinov1/<uniqname>
sftp> get -r web_bundle
sftp> bye

# THE RENAME IS LOAD-BEARING: conddiff_inference.py:343 looks for inference/conddiff_bundle,
# and the exporter writes a directory called web_bundle. Rename the folder; do NOT rearrange
# what is inside it (§8.5).
mv web_bundle conddiff_bundle
```

**Via the laptop** if the cluster cannot reach out or the instance cannot reach in: `sftp` down
to the laptop (§8.4), then `scp -i <key> -r conddiff_bundle ubuntu@<instance>:/opt/braingen/`.
This moves ~325 MB + the bank twice over a home connection; it works, it is just slow.

Verify what landed, on the instance, before starting anything:

```bash
cd /opt/braingen/conddiff_bundle
ls diffusion_ema.pt manifest.json cond_slices | head
python3 -c "import json; m=json.load(open('manifest.json')); print(len(m['slices']), 'slices,', len(m.get('supported_cells',[])), 'supported cells,', len(m.get('empty_cells',[])), 'empty')"
ls cond_slices/*.npy | wc -l          # must equal the slice count above
sha256sum diffusion_ema.pt            # compare with the hash export_for_website.py printed
stat -c %s diffusion_ema.pt           # expect 341171155 bytes
```

The manifest read on its own cannot notice a `.npy` that the transfer dropped — that is what
the file count and the checkpoint hash are for. A truncated checkpoint does not fail politely:
`torch.load` raises somewhere unhelpful, or worse, `strict=True` fails on a key list that looks
like a version skew and sends you debugging the wrong thing.

Then run the container:

```bash
cd /opt/braingen/braingen/backend            # wherever the checkout is

cat > /opt/braingen/backend.env <<'EOF'
SUPABASE_URL=https://xxxxx.supabase.co
SUPABASE_KEY=<service_role key>
FRONTEND_URL=https://<your-app>.vercel.app
CONDDIFF_MODE=live
CONDDIFF_STEPS=50
EOF
chmod 600 /opt/braingen/backend.env

docker run -d --name braingen --restart unless-stopped \
  --gpus all \
  -p 127.0.0.1:8000:8000 \
  --env-file /opt/braingen/backend.env \
  -v /opt/braingen/conddiff_bundle:/app/inference/conddiff_bundle:ro \
  braingen-backend:gpu
```

- `--env-file`, not `-e SUPABASE_KEY=...`: the service_role key is a full-access credential and
  a command line is visible to every process on the box via `ps`.
- `-p 127.0.0.1:8000:8000` binds to loopback so the reverse proxy is the only way in.
- `CONDDIFF_MODE=live` is set **explicitly** even though it is the code default (§8.0): an
  operator reading the run command should be able to tell which mode production is in without
  reading Python.
- `CONDDIFF_STEPS=50` is the placeholder the code defaults to. Raise it to **200** — the
  paper's value — once §8.10.6 has measured what a request actually costs.

---

#### 8.10.5 Step 4 — the smoke test, and the torch pin decision tree

**This is the step that answers the one genuinely open question in this deployment**, and it
must be run before anyone is told the site works.

##### The question

`requirements-gpu.txt` pins **`torch==2.0.1+cu118`**. That is deliberate and it is the
*preferred* answer: it is the **same torch version already in production**, only the CUDA
build, so the five GAN models' numerics are unchanged by construction — they load the same
weights through the same kernels on the same CPU code path (§8.0.1.1). Nothing about them has
to be re-validated for a version that did not change.

The checkpoint, however, was written under **torch 2.11.0+cu128 / diffusers 0.38.0** (measured
on Great Lakes, 2026-09-08). A bare `state_dict` of plain tensors should load across that gap.
What is **genuinely open** is whether **diffusers 0.38.0 imports at all on torch 2.0.1**.

##### Why pip cannot answer it, and why a green build is not evidence

`diffusers` declares torch only under an **optional extra**, so its install metadata requires no
torch at all. The resolver is therefore happy with `torch==2.0.1`, happy with a newer torch, and
happy with **no torch** — it has no opinion to express. An incompatibility can only appear as an
`ImportError` (or an `AttributeError` on some `torch.*` symbol that did not exist in 2.0.1) at
the moment `diffusers` is first imported, which in this codebase is *inside a request handler*,
lazily, in `_require_diffusers()`.

> **A successful `docker build` is not evidence that live mode works.** It is evidence that a
> set of version strings is mutually satisfiable. Treat the import as a question the smoke test
> answers — never as something to assert, in this document or in a status update.

##### The test

```bash
# The narrow version, ten seconds, answers the import question alone:
docker run --rm --gpus all --entrypoint python braingen-backend:gpu -c \
  "import torch, diffusers; from diffusers import UNet2DModel, DDIMScheduler; \
   print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), diffusers.__version__)"

# The real one: every model in the container, weights and all.
docker run --rm --gpus all \
  -v /opt/braingen/conddiff_bundle:/app/inference/conddiff_bundle:ro \
  --entrypoint python braingen-backend:gpu smoke_test_models.py
```

`backend/smoke_test_models.py` exists to answer, in one run and in one place, the questions that
are otherwise answered by six separate acts of faith:

1. do `torch` and `diffusers` import, and on what CUDA build;
2. is the GPU actually visible from inside the container (`torch.cuda.is_available()`,
   `torch.cuda.get_device_name(0)`);
3. does **each of the five GAN checkpoints** load — and **how many keys actually matched**;
4. does the conddiff checkpoint load into `build_model()` with `strict=True`;
5. does one end-to-end live sample run, and how long does it take.

##### Read the keys-matched column, not the exit status

Point 3 is the one that is easy to skim past and is the reason the script prints a column
instead of a tick. Every GAN loader in this repo calls
`load_state_dict(torch.load(...), strict=False)` (`cgan_inference.py:174`,
`cgan_inference_v2.py:134`, `wgan_inference.py:168-169`, `gan_3d_inference_v1.py:88`).
`strict=False` means a checkpoint whose key names no longer match the module **loads nothing,
raises nothing, and returns a randomly-initialised generator** — which still emits
brain-shaped output. There is no visual check that catches this. The only signal is
*"n of m keys matched"*, and the only acceptable value is **m of m**.

##### The decision tree

```
Run the smoke test on torch==2.0.1+cu118
│
├─ Everything imports, all five GANs report m-of-m keys, conddiff loads strict=True,
│  one sample runs on device=cuda
│      -> DONE. This is the preferred outcome: production torch version unchanged,
│         GAN numerics unchanged by construction. Go to §8.10.6.
│
├─ diffusers fails to import (ImportError / AttributeError on a torch symbol)
│      -> Bump torch, IN BOTH VARIANT FILES TOGETHER, KEEPING THE SUFFIXES PAIRED
│         (a +cpu and a +cu118 of the same version -- otherwise the two targets are
│         no longer the same deployment and the whole point of §8.6.0 is lost).
│         Then RE-RUN THE SMOKE TEST and read the keys-matched column for all five
│         GANs, because you have now changed the thing that can move their numerics.
│         Bump the minimum distance that works: try what diffusers 0.38.0 actually
│         requires, not the newest release.
│      -> HARD CEILING: torch >= 2.6 flips torch.load's default to weights_only=True,
│         which breaks every strict=False load in backend/inference/. Crossing it is
│         not a version bump, it is a code change (explicit weights_only=False, plus
│         re-testing all five models against known-good images).
│
├─ diffusers imports but the conddiff checkpoint fails strict=True
│      -> That is a DIFFUSERS skew, not a torch one: the attention parameter names
│         moved (query/key/value/proj_attn -> to_q/to_k/to_v/to_out.0). The pin must
│         track the version that WROTE the checkpoint (0.38.0). Do not "fix" it by
│         switching to strict=False -- that is the failure mode strict=True exists to
│         expose (§2.2).
│
└─ Everything imports but torch.cuda.is_available() is False
       -> Not a pin problem. Either the wheel is the +cpu build (check the version
          suffix -- "2.0.1+cpu" on a GPU host is the smoking gun) or the container
          cannot see the driver (§8.10.3's `docker run --gpus all ... nvidia-smi`).
```

Record whichever branch you landed on, with the versions, in the deployment notes. The next
person to rebuild this image will otherwise repeat the whole investigation.

---

#### 8.10.6 Step 5 — verify from outside, in this order

```bash
BE=http://127.0.0.1:8000        # on the box; or https://<your-backend-host> once proxied

curl -s $BE/health
curl -s $BE/check-models     | python3 -m json.tool
curl -s $BE/conddiff-selftest| python3 -m json.tool
curl -s $BE/conddiff-cells   | python3 -m json.tool
```

**1. `/check-models`** answers only *"does the file exist"* — the five GAN `.pth` files, the
four conddiff checkpoint/bundle candidates, and the gallery manifest. In a live deployment the
gallery entry is *correctly* reported missing; do not chase it. Read the block that matches the
mode you are in (§8.7).

**2. `/conddiff-selftest`** answers the questions that decide whether a request will succeed.
The live-mode rows and what a wrong value means are tabulated in §8.7. The five that decide
whether this migration actually happened:

| Field | Must be | If it is not |
|---|---|---|
| `mode` | `"live"` | the env var overrode the default — check the `--env-file` |
| `device` | `"cuda"` | **the silent failure this whole migration exists to avoid.** Every request still succeeds, at ~2 min instead of ~1 s, with correct images and no error anywhere |
| `torch_version` | ends in `+cu…` | a `+cpu` wheel can never see the card, however many GPUs `nvidia-smi` lists |
| `checkpoint_found` | `true` | none of the four `CKPT_CANDIDATES` exist — the mount path is wrong, or the rename in §8.10.4 was skipped |
| `bank_slices` | the exporter's `--per-cell` × covered cells | `0` means the bundle did not land or `manifest.json` parsed to nothing |

**3. `/conddiff-cells`** is what the UI consumes to grey out the impossible combinations. Expect
**11 of the 18** (lobe × slice-location) pairs (§9.2). If it returns `ready: false` with empty
lists, the frontend will disable *every* option — which is correct behaviour for a backend that
cannot generate, and is also exactly what a stopped instance looks like (§8.10.8).

**4. Then measure, because everything in §8.0 rests on it.** Generate one image through the
real UI and time it end to end. Confirm the recorded parameters carry `"rendering": "live"`, the
seed, and the *realised* lesion statistics measured from the mask that was actually drawn.
Then decide `CONDDIFF_STEPS`: at ~1 s for 50 steps, 200 steps — the paper's value, the one
`eval_metrics.py` used — costs ~4 s and is still far inside the ~65 s health-check window and
inside Vercel's function duration. **Raise it to 200 and re-measure**; leaving it at 50 ships a
step count no one has ever quality-checked (§8.0.3).

One ceiling to keep in view while measuring: `app/api/generate/route.ts` awaits the backend
inside a Vercel route handler, so total request time is bounded by Vercel's function duration
(10 s on Hobby, raisable to 300 s, 800 s hard maximum on Pro). At 1–4 s there is a comfortable
margin. There is no margin if anyone raises `MAX_IMAGES` above 1 (§8.0.4).

Note that at runtime `os.getcwd() == /app` (the Dockerfile sets `WORKDIR /app` and copies the
*contents* of `backend/` into it), so every path resolves to `/app/inference/...`. **Locally the
server must be started from inside `backend/`** or every path in every inference module breaks
— that is not specific to this model.

---

#### 8.10.7 Step 6 — repoint the frontend

Only after §8.10.6 is green. In Vercel → Project → Settings → Environment Variables:

```
NEXT_PUBLIC_BACKEND_URL = https://<gpu-backend-host>
```

Then **redeploy** — `NEXT_PUBLIC_*` values are inlined at build time, so changing the variable
without a rebuild changes nothing. Change it for Production and Preview deliberately, not
necessarily together: pointing Preview at the GPU box means every preview branch spends GPU
time and writes to the same Supabase project.

Set `FRONTEND_URL` in the backend's env file to the Vercel origin as well. In this architecture
that is belt-and-braces rather than load-bearing — every backend call is proxied server-side
through `app/api/`, so CORS is not on the path a browser takes — but leaving it wrong means a
future direct client-side call fails in a way that is confusing to debug.

**The old Lightsail service is the rollback.** Leave it deployed and healthy in gallery mode.
Rolling back is then one environment variable and one redeploy, with no rebuild and no data
migration — which is the only reason it is safe to do this migration at all before a demo.

---

#### 8.10.8 Cost and operations — honestly

**The shape of the bill, not the number.** Do not plan against any price written in this
document; look up current AWS pricing for the region you deploy in, because these change and a
stale number in a document is worse than no number.

- **The GPU instance bills per second it is *running*, whether or not anyone visits the site.**
  There is no per-request pricing and no scale-to-zero. A demo that gets forty visitors in one
  afternoon costs exactly the same as an idle month, if the instance is up for the same time.
  This is the fundamental difference from the ~$10/month Lightsail container, which was sized
  for a service that is almost always idle.
- **A GPU instance is roughly an order of magnitude more per hour than the CPU container's
  monthly cost implies.** That statement is a *shape*, not a quote — check the g4dn.xlarge
  On-Demand rate for your region, multiply by 730 hours, and compare that to the Lightsail line
  item before deciding to leave it running.
- **Stopping the instance stops the instance charge but not the storage charge.** The EBS root
  volume bills while the instance is stopped. A 100 GB gp3 volume is not free.
- **A stopped instance loses its public IPv4 address** unless an Elastic IP is attached — and an
  Elastic IP that is *not* attached to a running instance bills by the hour, which is exactly
  the state a stopped-between-demos instance is in. Either accept re-pointing DNS on each start,
  or pay for the Elastic IP; know which one you chose.
- **Data transfer out** is billed, and public IPv4 addresses now carry their own hourly charge.
  Small here, but they are line items people are surprised by.
- **Cheaper shapes exist and each costs something else.** Spot instances are much cheaper and can
  be reclaimed with two minutes' notice, mid-request. A scheduled start/stop (EventBridge, or a
  cron on a tiny always-on box) automates the discipline below but adds a moving part.

**Stopping it between demos is the expected mode of operation, not a failure.** Say so plainly
to whoever pays for it:

```bash
aws ec2 stop-instances  --instance-ids <i-...> --region us-east-1
aws ec2 start-instances --instance-ids <i-...> --region us-east-1
```

`--restart unless-stopped` on the container means it comes back on boot, so a start is
"start, wait for `/health`, check `/conddiff-selftest` says `device: cuda`" and nothing more.

**What a visitor sees when the backend is down.** This must be stated, because it is not a
graceful degradation and nobody should discover it during a demo:

- **Every model on the site is unavailable, not just this one.** All six live in the same
  container on the same host.
- The parameter panel calls `/api/conddiff-cells`, which catches the failure and returns
  `ready: false` with empty lists (`app/api/conddiff-cells/route.ts`), so **every lobe and slice
  location is greyed out**. That is the designed behaviour for "cannot generate" and it is not
  distinguishable, from the UI, from "this deployment has an empty conditioning bank".
- Clicking Generate produces an **error toast** — `app/api/generate/route.ts` returns 500 on an
  unreachable backend and `GenerationPageClient.tsx` surfaces `"Failed to generate image"`.
  There is no "the service is asleep, try again in two minutes" state, because nothing in the
  application models one.
- **There is no cold start.** The instance takes a minute or two to boot and pull the container
  up; nothing in the frontend waits for that or retries.

If the site is expected to be continuously available to the public, the honest options are: keep
the GPU instance running continuously and pay for it; or point `NEXT_PUBLIC_BACKEND_URL` back at
the Lightsail gallery deployment between demos and accept that the model is then serving
pre-rendered images (with the disclosure that requires — §10, decision 3). **Silently serving a
gallery image from a page that says the model sampled it is the one option that is not
available.**

---

## 9. Honest limitations

### 9.1 CPU latency — the hard one, and the reason gallery mode exists

> This section is the *evidence* for §8.0. The conclusion has since been acted on: production
> runs `CONDDIFF_MODE=gallery`, so nothing below is a live risk on the deployed site. It is
> kept in full because it is the argument, and because every number in it becomes live again
> the moment anyone proposes Phase 2 (§8.8).

A single forward pass of this U-Net at 256 × 256 is roughly **950 GFLOP** (≈ 475 GMAC,
from a layer-by-layer enumeration that reproduces the ≈ 85.3 M parameter count, which is what
validates the count). DDIM at 200 steps is ~190 TFLOP per image. For comparison, the site's
existing `VanillaGenerator` is ~3.9 GFLOP for one forward pass — the diffusion model at even
50 steps is **~10⁴ times** the compute of a GAN generation.

Estimated wall-clock per image on CPU, assuming ~22 GFLOP/s sustained per full vCPU
(calibrated against a known reference point: Stable Diffusion 1.5's UNet is ~680 GFLOP/forward
and runs ~4 s/step on 8 desktop cores):

| Lightsail power | vCPU / RAM | ≈ sec/step | 50 steps | 200 steps |
|---|---|---|---|---|
| micro (deployed today) | 0.25 / 1 GB | ~173 s | 2.4 h | 9.6 h — *and it OOMs first* |
| small | 0.5 / 2 GB | ~86 s | 1.2 h | 4.8 h |
| medium | 1.0 / 4 GB | ~43 s | 36 min | 2.4 h |
| large | 2.0 / 8 GB | ~22 s | 18 min | 72 min |
| xlarge | 4.0 / 16 GB | ~11 s | 9 min | 37 min |

These are estimates and should be replaced by a measurement. The cheapest decisive
experiment, before any further design work:

```bash
docker run --rm --memory=1g --cpus=0.25 <image> python -c "
import torch, time
from <module> import build_unet
m = build_unet().eval()
x = torch.randn(1, 9, 256, 256)
t0 = time.time(); m(x, torch.tensor(999)); print('one step', time.time() - t0)"
```

One command validates the torch/diffusers pairing, measures the real per-step time on a
metered vCPU, and demonstrates the OOM — all three unknowns at once.

**Four independent timeouts stack, and only one is under our control:** the Vercel route
handler's function duration (default 10 s on Hobby; 300 s when raised; 800 s hard maximum on
Pro), the Lightsail load balancer's non-configurable idle timeout (widely reported ~60 s —
measure it, do not assume), the Lightsail health check (`timeoutSeconds: 5`,
`intervalSeconds: 30`, `unhealthyThreshold: 2` → the container is declared unhealthy after
~65 s), and the browser fetch, which has no `AbortController` and so just spins indefinitely.

The health check is the one that cannot be raised away. `api.py`'s `/generate` is an
`async def` that calls a **synchronous** function directly, so a generation blocks the
uvicorn event loop, `/health` stops answering, and Lightsail replaces the container roughly
65 seconds in — **mid-generation, every time.** A multi-minute synchronous run can therefore
never complete, at any instance power. Live server-side sampling requires reworking
`/generate` into a submit-and-poll flow (202 + job id + a status endpoint), which is a
scoped project in its own right and is not part of this integration.

Also note the UI allows up to 5 images per request and the backend loops sequentially, so
every number above would multiply by up to 5 — which is why `image_generation.py` clamps
`n_images` to `conddiff_inference.MAX_IMAGES` (= 1) for this model and this model only.

**Consequence, and what was decided:** the model cannot be sampled live behind a synchronous
HTTP request on the current deployment. The two realistic options were (a) bump the instance
*and* rework the endpoint to be asynchronous, or (b) precompute on Great Lakes at the paper's
full 200 steps and serve the results. **(b) was chosen and built** — that is `CONDDIFF_MODE=gallery`
(§8.0). It is faster, cheaper, adds no new Python dependencies, and shows *higher*-quality
images than any CPU-affordable step count. (a) remains available as Phase 2 (§8.8).

The condition attached to (b) has not been discharged and is still owed: **the UI must say
so.** A "Generate" button that returns a file rendered three weeks ago, disclosed only in a
database column that no page renders and that is not written at all in playground mode, is not
disclosure. The backend now records `"rendering": "gallery"` plus `rendered_utc` on every
image, which makes the disclosure *possible*; putting it in front of the user is a frontend
change and a lab decision (§10, decision 3), not an implementation detail.

### 9.2 The fixed conditioning bank — and, in gallery mode, a fixed finite set of images

**The site cannot accept a user's own scan.** The model requires a T1 co-registered to SRI24
and six atlas channels warped into the same space; producing those from an arbitrary uploaded
NIfTI would require running the registration pipeline per request (ANTs SyN — minutes of CPU
even before diffusion) plus skull-stripping and the percentile normalisation, and would raise
an entirely separate set of privacy and data-handling questions. Nothing in the current
architecture supports it and nothing in this integration adds it.

The practical consequence is **repetition**: with a bank of *k* patients per (lobe, level)
cell, the entire visible anatomy space of a "generator" is 18 × *k* brains. At *k* = 4 that
is 72. Click Generate twice on the same settings and you may well see the same skull, which
reads immediately as a lookup table. Budget for more variants than feels necessary — the
files are ~1 MB each and the storage is cheap relative to the credibility cost.

Note also that the six atlas channels depend only on the z index, not on the patient, so they
are heavily redundant across a bank. Deduplicating them is worth doing only above roughly
130 slices; below that, per-slice arrays are both smaller and much simpler.

**Gallery mode makes this sharper, and the honest statement is this:** the deployed site does
not have a fixed *anatomy* space, it has a fixed *image* space. There are exactly 57 cells × *K*
variants of them — at *K* = 4, **228 FLAIR images, and that is every image the site can ever
return** until someone re-renders on the cluster. Two users who choose identical parameters are
drawing from the same *K* files. And because the variant is chosen uniformly at random from the
request's seed with no memory across requests, one user clicking Generate on a single setting
has a 1 − (3/4)(2/4)(1/4) ≈ **91 % chance of a repeat within four clicks**, and by the fifth
click a repeat is certain by pigeonhole. That reads as a lookup table because, at the level of
what leaves the server, it *is* one.

Four things follow, and they are cheap:

- **Budget more variants than feels necessary.** *K* = 8 doubles a ~12 MB payload and halves the
  collision rate; the storage is free next to the credibility cost. The GPU render is the same
  job, just longer.
- **Say it, do not hide it.** The `seed` recorded with every image is the seed that actually
  produced it on the cluster, and `selection_seed` is the request seed that only chose which
  variant to serve. Both are in the record; the distinction is what makes the image
  regenerable, which is the property that makes this defensible as reproducible research.
- **Any change to a shape constant means a re-render**, not a redeploy. `PROB_TH`,
  `AREA_FRAC_BY_SIZE`, `CORE_FRAC`, the noise field — all of them are baked into the shipped
  PNGs. Changing one in `conddiff_core.py` without re-rendering makes the container's core hash
  disagree with `manifest.json["core_sha256"]`, which is exactly what `/conddiff-selftest`
  exists to surface (§8.7).
- **Missing cells are visible, not silent.** `cells_missing` in the selftest lists every
  dropdown combination the UI offers and the gallery cannot serve. Those must be disabled in
  the UI, never substituted with a neighbouring cell — substituting would put a label on an
  image that does not match how it was made, which is the one failure this whole design is
  built to prevent.

### 9.3 Failure modes that look like success

This backend converts almost every error into a green "Success" toast. `image_generation.py`
catches every exception from an inference call, prints it to container stdout, and substitutes
an empty list; `api.py` then returns **HTTP 200** with an empty result. A missing checkpoint,
a missing bundle file, a missing `params` key, and a `diffusers` import failure are all
indistinguishable to the user: a blank viewer and a success message.

There is precedent in this repo for something worse. The existing
`braingen_diffuser_BraTS_v1 (2D)` entry has never actually run in production — `diffusers` is
not installed, so its import guard fails, and its error path *uploads a black
`np.zeros((128,128,4))` square into the user's library as if it were a result*. It also calls
`.to("cuda")` on a CPU-only deployment, and its weights directory is empty. **Do not use it
as a template.** Take from it only the idea of guarding the import.

The new model instead fails at container start: `_check_assets()` runs at import, validates
**the active mode's** assets only (a gallery deployment must not be marked broken for shipping
without a 341 MB checkpoint it will never open, and vice versa), prints the resolved absolute
paths, sets `READY`/`REASON`, and every subsequent request refuses rather than publishing
anything. `/conddiff-selftest` (§8.7) is how that state is read from outside the container,
since container stdout is otherwise the only evidence.

Two specific traps, both now avoided in `conddiff_inference.py` — described here because they
are easy to reintroduce, not because they are open:

- **A FLAIR-only output will be published as `t1.png`.** The shared `save_images` hardcodes
  `tags = ['t1','t2','flair','seg']` and indexes by array position, so a single-channel image
  lands at index 0 and is labelled T1 in the filename, in the database, and in the viewer.
  The fix, which this module implements, is a module-local `save_images` keyed by tag (an
  `OrderedDict` of tag → PNG bytes), not by index. Do not pad the array to four channels —
  that publishes three junk images into the user's library. Note this applies identically in
  gallery mode: the pre-rendered bytes go through the same `save_images`, so the naming
  contract is enforced in exactly one place for both modes.
- **Filenames are a load-bearing contract.** Playground mode routes images by
  `url.includes('flair.png')` with no fallback; authenticated mode splits the database name on
  `' - '`. A file named `flair_frontal.png` matches nothing and vanishes silently. Encode
  parameters in the folder name and in the recorded params, never in the filename.

### 9.4 Orientation and colour — cosmetic, but visible

Every figure in the research repo is rendered as `imshow(arr.T, origin='lower')`, i.e. the
displayed image is `np.flipud(arr.T)`, while PIL puts row 0 at the top. Encoding naively
produces a brain rotated 90° relative to every figure in the paper. But the site's other four
models do *no* such transform, so matching the paper means this model's brains will be
oriented differently from every other model in the same viewer, side by side in the same
library grid. **Pick one deliberately** — site consistency is probably the right call, with
the divergence from the paper figures noted here — and verify by rendering the same array both
ways before shipping.

Separately, the site's image viewer defaults to the `magma` colormap. The paper renders FLAIR
in grayscale. A pseudo-coloured FLAIR is arguably misleading for a contrast whose whole
diagnostic content is relative brightness; worth checking whether grayscale can be defaulted
for this model without touching the shared viewer component.

Also relevant if a Left/Right control is ever added: BraTS/SRI24 is stored **LPS** (verified
via `nib.aff2axcodes`), the midline is at axis-0 index 128, and **image-left is patient-right**
(radiological convention).

### 9.5 Not for clinical use

**Every image this model produces is synthetic and must not be used for diagnosis, treatment
planning, training of clinical personnel, or any other clinical purpose.**

It is a research artefact. Concretely: the lesions are drawn by an algorithm, not observed in
a patient; the FLAIR contrast is a generative model's rendering, not a measurement; the model
has never been validated against any clinical endpoint; PSNR ~17 dB means the output is *not*
a faithful reconstruction of any real scan; and the model has only ever been trained on adult
glioma from a single challenge dataset, so it has no notion of any other pathology, of
paediatric anatomy, of post-surgical change, or of any scanner or protocol outside BraTS 2023.

### 9.6 Other limitations, stated for completeness

- **2-D slices only.** No through-plane continuity; stacking generated slices does not produce
  a coherent volume. A separate 3-D effort exists in the research repo and is not this model.
- **Full-reference metrics under-credit the goal** (§4.1), and the distributional and
  downstream-utility evaluations that would credit it properly have not been run.
- **Tumour fidelity is not yet quantified** — there is no metric answering "does the generated
  FLAIR actually show abnormal signal where the mask says it should?"
- **The realised lesion size can miss the requested band.** A small lobe genuinely cannot host
  a large lesion, and the largest-connected-component step can shrink a lesion below its target
  area. Record the realised area rather than trusting the requested label.
- **Both randomness sources in the reference script are wrong for a web service.** The mask RNG
  is hard-seeded to 0, so it produces an identical tumour every call; the diffusion noise is
  unseeded, so images are irreproducible. Take one integer seed per request, derive both from
  it, and return it — that is also what makes any image on the site exactly regenerable, which
  is the property that makes the feature defensible as reproducible research.

---

## 10. Decisions still open

Nothing below is an implementation detail; each needs a person, not a commit.

| # | Question | Who decides |
|---|---|---|
| 1 | Does the BraTS/TCIA DUA permit publishing per-patient derived renderings on a public unauthenticated endpoint? (§7) | Prof. Dinov / lab, in writing, **before** the bank is exported |
| 2 | ~~Live server-side sampling or precomputed outputs?~~ **Decided: precomputed.** `CONDDIFF_MODE=gallery` is built and is the default (§8.0). Live mode remains implemented and is Phase 2 (§8.8), which needs an instance bump *and* an async endpoint (~$40–80/mo) | — (revisit only with a reason; the cost figures are in §8.8) |
| 3 | **Still open, and now the most urgent one.** Given precomputed images: what does the UI say, and where? (§8.0.3, §9.1, §7) | Lab — a "Generate" button must not misrepresent what it did. The backend records `"rendering": "gallery"` and `rendered_utc`; someone has to surface it |
| 4 | Are all 18 (lobe × level) conditioning cells fillable — and therefore all 57 gallery cells — and what happens to the ones that are not? (§6.1, §8.7) | Answered by running the export's selection pass; the UI behaviour (disable, never substitute) is a design call |
| 5 | Keep or drop the "Without Tumor" option, given it is a near-reconstruction of a real held-out scan? (§7) | Lab |
| 6 | Match the paper's orientation, or the site's? (§9.4) | Whoever implements — but decide, do not inherit. In gallery mode this is baked into the shipped PNGs, so getting it wrong costs a re-render |
| 7 | How many seed variants *K* per cell? (§9.2) | Whoever runs the render. 4 is the exporter's default; 8 costs one more GPU hour and ~12 MB and halves the repeat rate |

---

## 11. Where things live

### Research repository (`Summer2026`) — the model and the science

| File | Role |
|---|---|
| `src/model.py` | `build_model()` — the architecture, single source of truth |
| `src/conddiff_core.py` | **the shared core (§5.4)** — byte-identical to `backend/inference/conddiff_core.py`. Constants, `synth_mask*`, `params_to_spec`, `cell_id`, `iter_cells`, `build_cond`, `render_flair`, `to_png_bytes` |
| `src/dataset.py` | `SliceDataset` — `target = stack[0:1]`, `cond = stack[1:9]` |
| `scripts/02_register_atlas.py` | ICBM452 → SRI24 registration, ANTs SyN, run once |
| `scripts/03_preprocess.py` | builds the 9-channel `.npy` slices; the normalisation and the split live here |
| `scripts/train.py` | training, mixed precision + EMA |
| `scripts/sample.py`, `eval_grid.py` | qualitative sampling |
| `scripts/eval_metrics.py` | **the citable PSNR / SSIM / bgSNR numbers** |
| `scripts/augment_tumor.py` | **the reference implementation for the website feature** — atlas-guided synthetic tumour placement |
| `scripts/check_atlas_fit.py` | verifies containment / coverage / circularity with no GPU |
| `scripts/06_captions_v2.py` | source of the size and slice-level vocabulary |
| `scripts/export_for_website.py` | **runbook step 1 (§8.1)** — selects the conditioning bank from the val split, writes `cond_slices/*.npy` + `manifest.json`, copies and hashes the checkpoint |
| `scripts/verify_conddiff_port.py` | **runbook step 2 (§8.2) — NOT YET WRITTEN.** SHA-256 equality of the two core copies + numpy-vs-scipy agreement + 57-cell round-trip |
| `scripts/render_gallery.py`, `render_gallery.sbatch` | **runbook step 3 (§8.3) — NOT YET WRITTEN.** The GPU render: 57 cells × *K* seeds × 200 DDIM steps → PNGs + `manifest.json` |
| `PAPER_MATERIALS.md` | the full methods and results reference |
| `models/diffusion_ema.pt` | the checkpoint — **Great Lakes only, not in git** |

Important index note that has caused real confusion: `cond = stack[1:9]` re-indexes
everything by −1, so **inside `cond`, index 0 is T1, index 1 is the tumour mask, and index
`2+k` is atlas lobe *k***. That is why `sample.py` and `eval_grid.py` read the mask as
`cond[1]` and not `cond[2]`. Also, the shape comment `# (1, 7, 256, 256)` at `sample.py:32`
is stale and wrong — the tensor is `(1, 8, 256, 256)`. Use `augment_tumor.py` as the
reference, which has it right.

### This repository (`Brain-Image-Generator`)

| Path | Role |
|---|---|
| `backend/image_generation.py` | the only functional model dispatch; also clamps `n_images` to `MAX_IMAGES` for this model |
| `backend/inference/` | one module per model; each has its own `save_images` |
| `backend/inference/conddiff_inference.py` | **this model's entry point** — mode dispatch, gallery index, live sampler, tag-keyed `save_images`, `_check_assets`, `selftest` |
| `backend/inference/conddiff_core.py` | **the shared core (§5.4)** — byte-identical to `Summer2026/src/conddiff_core.py`. The only file in this repo that must never diverge from the research repo |
| `backend/inference/conddiff_gallery/` | **gallery mode's entire payload** — `manifest.json` + `<cell_id>/vNN_*.png`. Out of band, not in git |
| `backend/inference/model/` | `.pth` weights — empty in git, filled by hand before `docker build`. Also where the ≈ 341 MB checkpoint would go in live mode |
| `backend/inference/conddiff_bundle/` | live mode only — `manifest.json` + `cond_slices/*.npy` (+ the exporter's copy of `diffusion_ema.pt`). Absent in a gallery deployment, deliberately |
| `backend/supabase_storage.py` | upload + database record; shared, do not modify for one model |
| `backend/api.py` | `/generate`, `/models`, `/check-models`, `/conddiff-selftest` |
| `backend/Dockerfile`, `deploy-to-lightsail.sh`, `lightsail-config.json` | deployment |
| `components/projects/ParameterControlPanel.tsx` | the hardcoded 2D model list and parameter UI |
| `components/projects/GenerationPageClient.tsx` | all parameter state; builds the request |

`backend/config.py` defines `MODEL_BASE_DIR` and `MODEL_PATHS` and is **imported by nothing**
— it is dead code. Model paths are hardcoded inline at the point of use throughout the
backend; match that style rather than reviving config.py.

---

## 12. Citation

If this model or the atlas-guided placement is referenced in a paper or talk, cite the
BraTS 2023 challenge data (Menze et al. 2015; Baid et al. 2021), the ICBM452 lobular atlas,
the SRI24 template (Rohlfing et al. 2010), DDPM (Ho et al. 2020) and DDIM (Song et al. 2021).
The model itself is `braingen_CondDiffuser_BraTS_v1`, SOCR / University of Michigan,
Summer 2026.

---

*Last updated: 2026-09-07 — revised for the two-mode (gallery / live) architecture: §0 and the
header, the new §5.4 (the shared core), the mode notes in §6 and §7, a rewritten §8
(§8.0 why gallery, §8.1–8.7 the runbook, §8.8 Phase 2, §8.9 registration), and §9.1 / §9.2 /
§9.3 / §10 / §11 updated to match.*

*Numbers in §2.2 (parameter count, checkpoint size), §9.1 (FLOPs and latency), the RAM budget
in §8.8 and the ~10–15 MB gallery size in §8.0.2 are analytic estimates and are flagged as such
in the text — replace them with measurements from Great Lakes, from a metered container, and
from the first real render before anyone plans around them.*

*Two scripts named in the §8 runbook — `verify_conddiff_port.py` (§8.2) and
`render_gallery.py`/`.sbatch` (§8.3) — do not exist yet; their contracts are specified there so
that what gets written can be checked against them. Two backend items are also still open:
`/check-models` does not list the gallery manifest (§8.7), and `lightsail-config.json` does not
set `CONDDIFF_MODE` explicitly (§8.6).*
