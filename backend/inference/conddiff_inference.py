"""conddiff_inference.py -- braingen_CondDiffuser_BraTS_v1 (2D), the paper's model.

TWO MODES, CHOSEN BY THE CONDDIFF_MODE ENVIRONMENT VARIABLE
------------------------------------------------------------
    CONDDIFF_MODE=gallery   (the DEFAULT, and what production runs)
        Serve a PRE-RENDERED image. Never imports torch or diffusers, never opens the
        checkpoint, never touches the .npy conditioning bank. A request is a dictionary
        lookup, a random choice among the K seed variants of that cell, and a file read:
        milliseconds, a few MB of RSS.

    CONDDIFF_MODE=live
        Run the reverse diffusion at request time. Exactly the behaviour this file had
        before the gallery existed, unchanged.

WHY GALLERY IS THE DEFAULT -- THE ARITHMETIC THAT FORCED IT
------------------------------------------------------------
backend/lightsail-config.json asks for "power": "micro" = 0.25 vCPU and 1 GB RAM, with a
health check of timeoutSeconds 5 / intervalSeconds 30 / unhealthyThreshold 2 -- so a container
that stops answering /health for ~65 s is REPLACED. Meanwhile api.py:81 declares an
`async def` around synchronous work, so a long generation blocks uvicorn's event loop and
/health stops answering for the duration. And this U-Net is ~950 GFLOP per forward pass at
256x256 with a ~341 MB fp32 checkpoint. Live CPU diffusion on that instance is not "slow"; it
is a deployment that deletes itself mid-request.

The way out is that the user-visible parameter space is FINITE. All conditioning comes from a
fixed shipped bank, so the only things a visitor can vary are the four dropdowns:
Tumour(2) x Lobe(6) x Level(3) x Size(3) -- of which lobe and size are meaningless without a
tumour, so it is 6*3*3 = 54 with-tumour cells + 3 without-tumour cells = 57 real cells (see
conddiff_core.iter_cells). Render K seed variants per cell ONCE, on Great Lakes' GPU, at the
paper's full 200 DDIM steps, ship the PNGs (~12 MB), serve them. The gallery therefore shows
BETTER images than the live CPU path could -- 200 steps instead of a degraded 50 -- while
fitting inside the micro instance with room to spare.

Gallery mode is not a mock and it is not a placeholder. Every pixel it serves came out of this
same model through this same code, and the realised lesion statistics of each individual image
are carried in the gallery manifest and recorded to the database with it (see SECTION 11).

THE CORRECTNESS REQUIREMENT: inference/conddiff_core.py
--------------------------------------------------------
A pre-rendered gallery is only honest if the cluster renderer and the live path compute the
SAME image for the same four choices. If they diverged -- a different PROB_TH, a Gaussian blur
with a different boundary mode, a differently seeded random field -- the site would advertise
a parameter it did not actually honour, silently and forever.

So everything the two sides must agree on lives in conddiff_core.py, which exists
BYTE-IDENTICALLY at two paths (verify with sha256 before trusting a rendered gallery):

    Summer2026/src/conddiff_core.py                          <- the cluster renderer imports this
    Brain-Image-Generator/backend/inference/conddiff_core.py <- this file imports this

That module holds the constants, the scipy-free numpy reimplementations of the two
scipy.ndimage calls, the mask synthesiser, params_to_spec, cell_id, build_cond, render_flair
and to_png_bytes. What is left in THIS file is everything web-specific, which must NOT be
shared: paths, env vars, device selection, the conditioning-bank CSV, the gallery index,
checkpoint loading, Supabase publishing and the database record.

WHAT THIS MODEL IS
------------------
Every other model on this site turns a noise vector into a picture in ONE forward pass and
invents the anatomy from scratch. This one does not. It is a *conditional* denoising
diffusion model: a U-Net that is handed eight channels of context and asked to hallucinate
only the ninth. The nine input channels, in the order the network was trained on, are

    ch0     noisy FLAIR   <- the ONLY thing being generated; starts as pure Gaussian noise
    ch1     patient T1    <- real anatomy from a HELD-OUT (validation-split) BraTS subject
    ch2     tumour mask   <- raw BraTS labels {0 none, 1 necrotic, 2 edema, 3 enhancing core}
    ch3..8  atlas maps    <- 6 ICBM452 lobe probability maps (frontal, parietal, temporal,
                             occipital, cerebellum, insula) warped into SRI24 space

Channels 1-8 are the "conditioning". They are held CONSTANT for the whole reverse diffusion
loop; only ch0 is updated. Because we get to draw ch2 ourselves, we control exactly where the
tumour appears -- and because ch3..8 tell us where each lobe physically is in this particular
slice, we can place that tumour in an anatomically valid spot in a named lobe. That
atlas-guided lobe placement is the paper's novel controllable axis and no other model on this
site has it.

The consequence worth stating plainly, because it changes how the output should be read: the
ANATOMY in the generated FLAIR comes from the real patient's T1, not from the model. What the
model synthesises is the FLAIR contrast and the lesion. "Synthetic FLAIR of a real held-out
brain" is a different claim from "synthetic brain", and the UI copy should say so.

WHERE THIS CODE CAME FROM
-------------------------
The reference implementation is scripts/augment_tumor.py in the research repo
(Research/socr/Summer2026). Everything numerical -- the mask synthesiser, the [-1,1] <-> [0,1]
rescaling, the channel indexing, the DDIM loop itself -- is a faithful port of that script,
and now lives one level down in conddiff_core.py so the cluster renderer runs the identical
code. The DDIM scheduler CONSTRUCTION stayed here, in _get_scheduler(), because it needs the
deployment's step count; its arguments are still transcribed from train.py and every one of
them is load-bearing. Where I deviated from the research script I say so in a comment and give
the reason.

TWO THINGS THIS FILE DELIBERATELY DOES DIFFERENTLY FROM ITS NEIGHBOURS
----------------------------------------------------------------------
1. `load_state_dict(..., strict=True)`.  Every other inference module in this folder uses
   strict=False (cgan_inference.py:174, cgan_inference_v2.py:134, wgan_inference.py:168,
   gan_3d_inference_v1.py:88). strict=False means a key-name mismatch loads NOTHING and the
   network happily emits initialisation noise with no error and no log line. For a diffusion
   U-Net that failure is especially nasty because a partially-loaded model still produces
   blurry brain-shaped output. We use strict=True and let it raise.
2. A module-local `save_images()` keyed by TAG, not by array index.  The shared helper at
   cgan_inference.py:20 hardcodes `tags = ['t1','t2','flair','seg']` and then does
   `for i in range(images.shape[-1]): filename = f"{tags[i]}.png"`. A FLAIR-only output has
   shape (256,256,1), so the loop runs once at i=0 and our FLAIR would be published as
   **t1.png** and stored in the database as "t1 - mri_<uuid>". The frontend keys entirely off
   that filename, so the image would render under the T1 tab. Keying by tag makes that class
   of bug impossible to write.

WHAT MUST BE ON DISK BEFORE THIS RUNS -- AND IT DEPENDS ON THE MODE
--------------------------------------------------------------------
gallery mode needs ONLY:
    inference/conddiff_gallery/manifest.json      (the index: cells, variants, provenance)
    inference/conddiff_gallery/<cell_id>/*.png    (the pre-rendered images)

live mode needs ONLY (this is the layout scripts/export_for_website.py emits, dropped in whole
and NOT rearranged -- SECTION 1's BANK_DIR/MANIFEST_PATH are what actually get opened):
    inference/conddiff_bundle/manifest.json       (the index; JSON, not CSV)
    inference/conddiff_bundle/cond_slices/*.npy   (9-channel float16 slices)
    inference/conddiff_bundle/diffusion_ema.pt    (the exporter's copy of the checkpoint)
  ...or, if you would rather keep the weights with the five GAN .pth files, the checkpoint may
  instead live at inference/model/diffusion_ema.pt (or diffusion_ema_fp16.safetensors), which
  is the first entry in CKPT_CANDIDATES and therefore wins. Only ONE copy is needed.

_check_assets() validates the ACTIVE mode's assets and ignores the other mode's, on purpose: a
gallery deployment must not be marked unhealthy for shipping without a 341 MB checkpoint it
will never open, and a live deployment must not be marked unhealthy for shipping without a
gallery. None of these files are in git -- all are delivered out of band exactly like the five
.pth GAN weights. See _check_assets() below for the exact error messages, and the "HOW TO
PRODUCE THE ASSETS" section at the bottom of this docstring.

DEPENDENCIES -- READ THIS BEFORE EDITING THE IMPORTS
----------------------------------------------------
image_generation.py imports this module at the TOP of the file, alongside the four GAN
modules. That means an exception raised at import time here takes down all five working
models. So:

  * module scope imports ONLY the standard library, numpy and conddiff_core (which itself
    imports only the standard library and numpy). torch is NO LONGER imported at module scope
    -- gallery mode must be able to run without it ever being touched, both to keep the
    container small and because a torch import is the single most expensive thing that could
    go wrong at boot;
  * `torch` is imported lazily inside _device(), _load_state_dict() and _get_model(), and
    inside conddiff_core.build_cond/render_flair. Only the live path reaches any of them;
  * `diffusers` is imported LAZILY, inside _require_diffusers(). It IS pinned in
    requirements.txt now (diffusers==0.38.0), but the laziness still earns its keep: gallery
    mode never pays for the import, and because diffusers declares torch only under an optional
    extra, a diffusers/torch incompatibility cannot be caught by pip and surfaces as an
    ImportError. Lazy + guarded turns that into a precise RuntimeError at request time instead
    of a dead backend at boot;
  * `PIL` is imported lazily by conddiff_core.to_png_bytes, which gallery mode never calls
    (it serves bytes that are already PNG);
  * `scipy` is not imported at all. augment_tumor.py uses scipy.ndimage.gaussian_filter and
    scipy.ndimage.label inside synth_mask; both are reimplemented in ~40 lines of pure numpy
    in conddiff_core (_gaussian_blur, _largest_connected_component). That was a deliberate
    choice -- adding scipy means a rebuild of the shared Docker image and a resolver run that
    could break numpy==1.24.3 / Pillow==10.0.0 for the five models that currently work. Two
    small, well-understood functions are cheaper than that risk. They match scipy's defaults
    exactly (truncate=4.0, mode='reflect' == numpy's 'symmetric', 4-connectivity).

HOW TO PRODUCE THE ASSETS (on Great Lakes, where the checkpoint and the val slices live)
----------------------------------------------------------------------------------------
    # GALLERY (what production needs). The renderer walks conddiff_core.iter_cells(), runs the
    # full pipeline per cell with K distinct seeds at 200 DDIM steps, and writes the PNGs plus
    # manifest.json. It imports conddiff_core from Summer2026/src, which must be byte-identical
    # to this folder's copy -- the renderer records that file's sha256 in the manifest so the
    # backend can prove the two agree.
    scp -r /scratch/.../conddiff_gallery <laptop>:.../backend/inference/

    # LIVE (only if this is ever deployed on a machine that can afford it)
    scp models/diffusion_ema.pt <laptop>:.../backend/inference/model/
    # bank: any val-split slice that passes the same three gates augment_tumor.py:130 uses
    #   (s[1] > 0.05).sum() > 15000        enough brain -> not a tiny inferior sliver
    #   (s[2] > 0.5).sum() < 50            tumour-FREE -> we add exactly the lesion we draw
    #   ((s[3+k] > 0.4) & brain).sum() > 500   the target lobe is genuinely present
    # copy those .npy files verbatim (do NOT recompute the atlas channels -- 03_preprocess.py
    # normalises all six lobe maps by ONE global scalar, and PROB_TH=0.4 is calibrated
    # against exactly that scaling) and write a manifest.json describing them.
    # scripts/export_for_website.py in the research repo does all of this and emits the
    # web_bundle/ folder described above -- copy that folder whole to
    # backend/inference/conddiff_bundle/ rather than assembling the pieces by hand.
"""

# --- standard library + numpy. NOTE WHAT IS ABSENT: torch and PIL used to be imported here
# --- and no longer are. Gallery mode must be able to import this module without either of
# --- them being touched; both are now imported lazily on the live path only. -----------------
import os
import json
import uuid
import secrets
import numpy as np
from collections import OrderedDict

# --- the SHARED core: byte-identical to Summer2026/src/conddiff_core.py ----------------------
# THE DUAL IMPORT IS NOT DEFENSIVE PROGRAMMING FOR ITS OWN SAKE. In production this module is
# imported as `inference.conddiff_inference` from a process whose cwd is /app, so the package
# form is what resolves (it is also the house pattern -- cgan_inference.py:3 does
# `from inference.Generators import *`). The fallback covers running a script from inside
# backend/inference/ itself, e.g. a quick `python -c "import conddiff_inference"` while
# debugging on the laptop, where "inference" is not an importable package name. Both branches
# load the SAME FILE; there is no configuration in which they could disagree.
try:
    from inference import conddiff_core as core
except ImportError:                                        # running with backend/inference/ on sys.path
    import conddiff_core as core

# =============================================================================================
# SECTION 1 -- CONSTANTS
#
# The constants that DEFINE THE PICTURE now live in conddiff_core so the cluster renderer and
# this file cannot drift apart. They are re-bound into this module's namespace below rather
# than referenced as core.X everywhere, for two reasons: the rest of this file (and selftest())
# reads exactly as it did before the split, and this block doubles as an explicit, greppable
# inventory of what moved. Anything NOT in that list is web-deployment-specific and is defined
# here.
# =============================================================================================

# --- moved to conddiff_core.py; re-exported here so this module's API is unchanged -----------
LOBES             = core.LOBES              # lobe order == atlas channel order (03_preprocess.py:10)
ATLAS0            = core.ATLAS0             # atlas lobe k lives at stack channel 3 + k
PROB_TH           = core.PROB_TH            # "inside this lobe" atlas-probability cut, 0.4
MIN_LOBE          = core.MIN_LOBE           # the lobe must occupy > 500 px of the chosen slice
AREA_FRAC_DEFAULT = core.AREA_FRAC_DEFAULT  # the paper's (0.15, 0.35) lesion-area band
CORE_FRAC         = core.CORE_FRAC          # enhancing core as a fraction of lesion area
MIN_AREA          = core.MIN_AREA           # px floor below which a "tumour" is meaningless
COMPACT           = core.COMPACT            # blob compactness around the seed
NOISE_SIGMA       = core.NOISE_SIGMA        # smoothing of the random field, in px
NOISE_AMP         = core.NOISE_AMP          # how irregular the lesion outline gets
SMALL_MAX         = core.SMALL_MAX          # paper's small/moderate boundary, px
LARGE_MIN         = core.LARGE_MIN          # paper's moderate/large boundary, px
INFERIOR_MAX      = core.INFERIOR_MAX       # paper's inferior/middle boundary, slice index
SUPERIOR_MIN      = core.SUPERIOR_MIN       # paper's middle/superior boundary, slice index
UI_TUMOUR         = core.UI_TUMOUR          # dropdown string -> algorithm value, x4
UI_LOBE           = core.UI_LOBE
UI_LEVEL          = core.UI_LEVEL
UI_SIZE           = core.UI_SIZE
AREA_FRAC_BY_SIZE = core.AREA_FRAC_BY_SIZE  # Tumour Size -> lesion-area band

# --- functions likewise moved; same re-export, same reasoning --------------------------------
size_word        = core.size_word           # lesion px      -> small|moderate|large
level_word       = core.level_word          # slice index z  -> inferior|middle|superior
params_to_spec   = core.params_to_spec      # 4 dropdowns    -> validated spec dict
cell_id          = core.cell_id             # spec           -> stable gallery key
synth_mask       = core.synth_mask          # the atlas-guided mask synthesiser
synth_mask_sized = core.synth_mask_sized    # ...with the Tumour Size control wired in
build_cond       = core.build_cond          # (9,256,256) stack + mask -> (1,8,256,256) tensor
render_flair     = core.render_flair        # the DDIM reverse loop
to_png_bytes     = core.to_png_bytes        # (256,256) float32 -> PNG bytes, paper orientation

# The model id. This string is compared with `==` in image_generation.py and is also the
# `id` field of the entry in ParameterControlPanel.tsx's models2D array, which the frontend
# sends back as `model_name`. All three must match BYTE FOR BYTE, including the space before
# "(2D)".
MODEL_NAME = "braingen_CondDiffuser_BraTS_v1 (2D)"

# MIN_BRAIN IS THE ONE THRESHOLD THAT IS *NOT* IN conddiff_core, AND THAT IS DELIBERATE.
# Everything in core is something both sides must agree on exactly. MIN_BRAIN is not: it is a
# gate on which slices are ADMISSIBLE into a bank, it is cosmetic rather than correctness, it
# is allowed to be looser on the cluster than here, and it is overridable per deployment by an
# env var. Putting a deployment-tunable value in a file whose whole purpose is to be
# byte-identical across two repos would defeat the file.
#
# MIN_BRAIN is the one gate the bank builder is expected to be able to relax. augment_tumor.py
# comments it as "require a full mid-axial slice, not a tiny inferior sliver" -- it is a
# COSMETIC gate, not a correctness one, and some (lobe, level) cells cannot be filled without
# lowering it (the cerebellum only exists in the inferior slices that 15000 was written to
# reject). _pick_slice re-verifies every candidate against the real array with this same
# threshold, so a bank deliberately built at a lower threshold would have EVERY one of its
# slices rejected at request time, with the misleading error "the manifest is stale". The env
# var lets the deployment match whatever the bank was exported with. Default is unchanged.
try:
    MIN_BRAIN = int(os.environ.get("CONDDIFF_MIN_BRAIN", "15000"))
except (TypeError, ValueError):
    MIN_BRAIN = 15000
    print("[conddiff] CONDDIFF_MIN_BRAIN is not an integer -- falling back to 15000")

# (The rest of the constants that used to live here -- PROB_TH and its calibration note,
# AREA_FRAC_*, CORE_FRAC, MIN_AREA, COMPACT, NOISE_*, the caption vocabulary and the four UI
# maps -- moved to conddiff_core.py and are re-exported above. Read them there; every one of
# them still carries its original comment explaining why it has the value it has.)

# =============================================================================================
# SECTION 2 -- RUNTIME CONFIGURATION AND ASSET PATHS
# =============================================================================================

# --- WHICH MODE IS THIS CONTAINER IN? --------------------------------------------------------
# "live" (DEFAULT) runs the diffusion at request time; "gallery" serves PNGs pre-rendered on the
# cluster.
#
# WHY LIVE IS THE DEFAULT, DESPITE THE COST. This is a research demonstrator for a generative
# model, and a generator that returns one of four pre-baked images per parameter cell is not
# generating anything at the point of use -- two visitors choosing the same options would get
# the same picture, and the site would be a lookup table wearing a model's name. The scientific
# claim the site exists to demonstrate is that this model SAMPLES a FLAIR conditioned on real
# anatomy, so the site has to actually sample.
#
# WHAT THAT COSTS, STATED PLAINLY SO NOBODY IS SURPRISED IN PRODUCTION. The U-Net is ~950 GFLOP
# per forward pass at 256x256 and DDIM needs one per step, so a request is 50-200 of those. On
# the currently-declared Lightsail "micro" (0.25 vCPU, 1 GB) that does not merely run slowly:
# the 341 MB fp32 checkpoint plus activations does not fit in 1 GB, and api.py's `async def`
# wraps this synchronous work, so /health stops answering and the healthCheck (5 s timeout,
# 30 s interval, threshold 2) replaces the container after ~65 s. LIVE MODE REQUIRES A LARGER
# INSTANCE AND AN ASYNC JOB PATTERN. See CONDDIFF_MODEL.md.
#
# GALLERY MODE IS RETAINED, opt-in via CONDDIFF_MODE=gallery, for a constrained deployment or a
# demo on the existing instance. It is not what the site serves by default.
#
# PARSED DEFENSIVELY AND CASE-INSENSITIVELY, and it cannot raise: this is module scope, and
# image_generation.py imports this module at the top of the file next to the four GAN modules.
_MODE_RAW = os.environ.get("CONDDIFF_MODE", "live")
MODE = str(_MODE_RAW).strip().lower() if _MODE_RAW is not None else "live"
if MODE not in ("gallery", "live"):
    print(f"[conddiff] CONDDIFF_MODE={_MODE_RAW!r} is not 'gallery' or 'live' -- falling back "
          f"to 'live' (the configured default)")
    MODE = "live"

# Paths are built with os.path.join(os.getcwd(), "inference/...") and hardcoded inline at the
# point of use. That matches the house style (cgan_inference_v2.py:122, wgan_inference.py:159,
# gan_3d_inference_v1.py:77) rather than backend/config.py, which defines MODEL_PATHS and is
# imported by nothing -- it is dead code.
#
# WHY os.getcwd() WORKS: Dockerfile sets `WORKDIR /app` and `COPY . .` copies the CONTENTS of
# backend/ into /app, then runs uvicorn from /app. So os.getcwd() == "/app" at runtime and
# these resolve to /app/inference/... . Locally the server MUST be started from inside
# backend/ or every path in every inference module breaks, not just this one.
_CWD           = os.getcwd()
MODEL_DIR      = os.path.join(_CWD, "inference/model")

# --- LIVE-mode assets: the checkpoint and the conditioning bank ------------------------------
# THESE MATCH scripts/export_for_website.py's OUTPUT LAYOUT EXACTLY, deliberately. That script
# writes a folder shaped like this:
#
#     web_bundle/                     (renamed to conddiff_bundle/ on arrival)
#       manifest.json                 the index -- JSON, not CSV
#       cond_slices/<patient>_z###.npy    the conditioning arrays
#       diffusion_ema.pt              the checkpoint
#
# An earlier revision of this file expected `conddiff_bank/manifest.csv` with the slices at the
# bank root. Nothing produced that layout -- three separate mismatches (folder name, file
# format, slice subdirectory) -- so live mode would have died at startup on a missing file. The
# bug was invisible while gallery mode was the default, because gallery mode never reads the
# bank at all. The fix is to read what the tooling actually emits rather than to ask a human to
# rearrange 42 files by hand after an sftp, which is precisely where transfers go wrong.
BANK_DIR       = os.path.join(_CWD, "inference/conddiff_bundle")
MANIFEST_PATH  = os.path.join(BANK_DIR, "manifest.json")
SLICE_SUBDIR   = "cond_slices"        # export_for_website.py:148, SLICE_SUBDIR

# --- GALLERY-mode assets: the pre-rendered PNGs and their index ------------------------------
# A SEPARATE FOLDER FROM THE BANK, ON PURPOSE. The bank is ~1.1 MiB per .npy slice of raw
# float16 conditioning and is useless without torch; the gallery is finished PNGs and is
# useless with it. Keeping them apart means a gallery deployment can simply not ship
# conddiff_bundle/ at all (and vice versa), which is the entire size argument for gallery mode,
# and it means _check_assets can test exactly one of the two trees.
GALLERY_DIR      = os.path.join(_CWD, "inference/conddiff_gallery")
GALLERY_MANIFEST = os.path.join(GALLERY_DIR, "manifest.json")

# The checkpoint. train.py:57-60 in the research repo does `ema.copy_to(model.parameters())`
# then `torch.save(model.state_dict(), ...)`, so this file is a BARE state_dict of the EMA
# weights -- there is no ["model"] / ["state_dict"] / ["epoch"] wrapper to unwrap.
# We accept either the original pickle or an fp16 safetensors export (smaller, and safetensors
# has no arbitrary-code-execution surface the way torch.load's pickle does). Note that
# .gitignore already ignores backend/inference/model/*.safetensors, so the safetensors name
# also buys accidental-commit protection for free.
# The BUNDLE ROOT is listed too, and last, because export_for_website.py copies the checkpoint
# in beside manifest.json. Dropping the transferred bundle in place therefore works with no
# further file shuffling, while a checkpoint deliberately placed in inference/model/ (next to
# the five GAN .pth files, which is where a reader would look for it) still wins.
CKPT_CANDIDATES = [
    os.path.join(MODEL_DIR, "diffusion_ema_fp16.safetensors"),
    os.path.join(MODEL_DIR, "diffusion_ema.safetensors"),
    os.path.join(MODEL_DIR, "diffusion_ema.pt"),
    os.path.join(BANK_DIR, "diffusion_ema.pt"),
]

# Number of DDIM denoising steps. augment_tumor.py:27 uses 200 (the paper's value, via the
# AUG_STEPS env var). 50 is the DEPLOYMENT default and it is a PLACEHOLDER, not a measured
# recommendation: nothing in either repo has ever compared quality below 200 steps. Run the
# step sweep on the cluster and set this from that number.
#
# Cost context, so the number is not mistaken for a free parameter: this U-Net is ~950 GFLOP
# per forward pass at 256x256, so 50 steps is ~47.5 TFLOP -- roughly four orders of magnitude
# more compute than the single ~3.9 GFLOP forward pass of the GAN generators on this site.
#
# PARSED DEFENSIVELY, ON PURPOSE. This is module scope, and image_generation.py imports this
# module at the top of the file alongside the four GAN modules -- so a bare
# `int(os.environ["CONDDIFF_STEPS"])` on a typo'd value (it is set by hand in the Lightsail
# deployment JSON) would raise ValueError during import and take the whole backend, including
# all five working GAN models, offline before uvicorn ever starts. Nothing at module scope in
# this file is allowed to raise; that is the same reason _check_assets() is wrapped below.
try:
    STEPS = int(os.environ.get("CONDDIFF_STEPS", "50"))
    if STEPS < 1:
        raise ValueError("must be >= 1")
except (TypeError, ValueError) as _steps_err:
    STEPS = 50
    print(f"[conddiff] CONDDIFF_STEPS={os.environ.get('CONDDIFF_STEPS')!r} is not a positive "
          f"integer ({_steps_err}) -- falling back to 50 DDIM steps")

# DEVICE IS RESOLVED LAZILY, AND THAT IS THE WHOLE REASON THIS IS A FUNCTION.
# It used to be a module-scope `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`.
# Answering that question requires importing torch, and importing torch is exactly what
# gallery mode must never do -- torch is ~200 MB of RSS on a 1 GB instance, for an answer no
# gallery request will ever use. So the constant became a memoised accessor, and every caller
# on the live path goes through it.
#
# WHICH ANSWER YOU GET DEPENDS ON WHICH IMAGE THIS IS, and there are now two:
#   Dockerfile      -> requirements.txt     -> torch==2.0.1+cpu    -> always "cpu"
#   Dockerfile.gpu  -> requirements-gpu.txt -> torch==2.0.1+cu118  -> "cuda", IF the host has a
#                      driver and the container was started with `--gpus all`
# Same torch VERSION in both, deliberately: only the wheel's CUDA build differs, so the five GAN
# models keep bit-identical numerics across the two images.
#
# THE FAILURE MODE TO WATCH FOR is a GPU deployment that silently resolves to "cpu" anyway --
# from a CUDA base image whose version does not match the wheel's (11.8 here), a host driver
# older than R520, or a forgotten `--gpus all`. Nothing raises; you just get ~2 min/sample
# instead of ~1 s and correct-looking images. backend/smoke_test_models.py prints
# torch.cuda.is_available() first and loudest for exactly this reason.
#
# (cgan_inference.py:120 calls .to("cuda") unconditionally, which is why the old unconditional
# diffuser entry has never once executed on the CPU container. Do not copy that.)
DEVICE = None      # None means "not resolved yet"; stays None forever in gallery mode


def _device():
    """Return "cuda"/"cpu", importing torch on first use. LIVE PATH ONLY."""
    global DEVICE
    if DEVICE is None:
        import torch                                       # lazy -- see the comment above
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return DEVICE

# One generation should not be multiplied by the UI's n_images slider (max 5). image_generation
# .py loops `for i in range(n_images)` synchronously in one HTTP request; five sequential DDIM
# runs is the worst case this model must survive. Exposed here so the dispatcher can clamp.
#
# IT STAYS 1 IN GALLERY MODE TOO, EVEN THOUGH FIVE FILE READS WOULD BE FREE. Two reasons.
# First, most cells ship only a handful of seed variants, so n_images=5 would very often
# publish the SAME pre-rendered image to the user's library more than once -- which looks like
# a bug and quietly overstates how much material the gallery holds. Second, MAX_IMAGES is read
# by image_generation.py at dispatch, before this module knows anything about the request; a
# value that changed with the mode would make the dispatcher's behaviour depend on an env var
# in a different file. If the gallery ever ships enough variants to make this worth revisiting,
# clamp to the size of the smallest cell, not to 5.
MAX_IMAGES = 1

# Module-level singletons. The GAN modules rebuild and reload their generator on EVERY request
# (cgan_inference_v2.py:132-134). At ~85.3M fp32 parameters (~341 MB) that pattern would be
# fatal here, so we load once and cache. `None` means "not loaded yet".
_MODEL = None      # the UNet2DModel, on DEVICE, in eval mode
_SCHED = None      # the DDIMScheduler, already set_timesteps(STEPS)
_SCHED_STEPS = None  # which step count _SCHED was built for, so an env change is picked up

# Populated by _check_assets() at import time. READY gates every request.
READY  = False
REASON = "startup check has not run"
BANK   = []        # list of dicts, one per usable bank slice (see _load_manifest)


# =============================================================================================
# SECTIONS 3, 4 AND 5 MOVED TO inference/conddiff_core.py
#
# What used to be here:
#   SECTION 3  _gaussian_blur, _largest_connected_component -- the scipy-free numpy
#              reimplementations of the only two scipy.ndimage calls on the inference path
#   SECTION 4  size_word, level_word, params_to_spec -- the paper's vocabulary and the
#              UI -> algorithm mapping (plus cell_id, which is new and lives there too)
#   SECTION 5  synth_mask, synth_mask_sized -- the atlas-guided mask synthesiser
#
# They moved because the CLUSTER RENDERER that pre-renders the gallery has to run exactly this
# code. If the renderer had gone on calling scipy.ndimage while this file called the numpy
# versions, the same seed would have produced two different random fields, and therefore two
# different lesions, on the two sides -- with no error and two perfectly plausible pictures.
#
# The names are re-exported in SECTION 1, so `conddiff_inference.synth_mask_sized` and friends
# still resolve for anything that imports them. The section numbering below is deliberately
# NOT renumbered: this file has been reviewed and referenced by section number, and a gap that
# is explained is cheaper to read than a renumbering that silently invalidates every reference.
# =============================================================================================

# =============================================================================================
# SECTION 6 -- THE CONDITIONING BANK  *** LIVE MODE ONLY ***
# The website has no concept of a patient T1 or a lobe atlas, so those eight channels have to
# be SHIPPED. The bank is a folder of 9-channel .npy slices -- byte-for-byte the output of
# scripts/03_preprocess.py -- plus a manifest.json that lets us pick one without opening them
# all. They are drawn from the VAL split only (patient-level split, seed 42,
# 03_preprocess.py:15), so the site never displays anatomy the model was trained on.
#
# NOTHING IN THIS SECTION RUNS IN GALLERY MODE. A gallery deployment ships no bank at all: the
# conditioning was consumed on the cluster when the images were rendered, and what survives of
# it is the provenance recorded per image in the gallery manifest (which held-out subject, which
# slice index z, how much brain). _check_assets() therefore does not even look for the bank
# unless MODE == "live". Keep it working anyway -- live mode is the reference implementation
# that the gallery is generated FROM, and it is what a GPU deployment would switch back on.
# =============================================================================================

def _slice_ok(stack, lobe_idx, min_brain=MIN_BRAIN):
    """The three conjunctive admissibility tests, verbatim from augment_tumor.py:130.

    stack    : (9,256,256) float32
    lobe_idx : int index into LOBES, or None to skip the lobe test (the no-tumour case)

    Returns (ok: bool, stats: dict). The tests are:
      1. enough brain      -- (stack[1] > 0.05).sum() > 15000. Channel 1 is the T1. 0.05 on a
                              percentile-normalised volume is the brain/background cut used
                              everywhere in the research repo. This rejects the tiny
                              inferior/superior slivers where a lesion would look absurd.
      2. tumour-FREE       -- (stack[2] > 0.5).sum() < 50. We want to add exactly the one
                              lesion we drew to an otherwise-healthy slice, with no pre-existing
                              pathology confusing the picture. `> 0.5` catches any label >= 1.
      3. the lobe is there -- ((atlas > 0.4) & brain).sum() > 500 for the requested lobe.
                              Without this, asking for "occipital" on a slice where the
                              occipital lobe barely appears yields a 60-px pinprick.
    """
    brain = stack[1] > 0.05                                # (256,256) bool -- channel 1 = T1
    brain_px  = int(brain.sum())
    tumour_px = int((stack[2] > 0.5).sum())
    lobe_px = 0
    if lobe_idx is not None:
        al = stack[ATLAS0 + lobe_idx]                      # (256,256) this lobe's atlas map
        lobe_px = int(((al > PROB_TH) & brain).sum())

    ok = (brain_px > min_brain) and (tumour_px < 50)
    if lobe_idx is not None:
        ok = ok and (lobe_px > MIN_LOBE)
    return ok, {"brain_px": brain_px, "tumour_px": tumour_px, "lobe_px": lobe_px}


def _load_manifest():
    """Read conddiff_bundle/manifest.json into a list of dicts, one per shipped slice.

    THE FORMAT IS JSON BECAUSE THAT IS WHAT PRODUCES THE BANK. scripts/export_for_website.py
    (line 150, MANIFEST = "manifest.json") writes a single JSON object whose "slices" key holds
    one record per shipped array:

        {"file": "BraTS-GLI-00183-000_z066.npy",   relative to conddiff_bundle/cond_slices/
         "patient": "BraTS-GLI-00183-000", "z": 66, "level": "middle",
         "brain_px": 21044, "tumour_px": 3,
         "lobes": {"frontal": 1840, "parietal": 318}}    ONLY the lobes that passed MIN_LOBE

    Note `lobes` is a {name: pixel_count} OBJECT, not a delimited string: the export script
    already measured each lobe's in-slice area while scanning, so the count comes along free and
    _pick_slice can prefer the most prominent lobe rather than just any viable one.

    An earlier revision of this function parsed a CSV with a `lobes_ok` column at
    conddiff_bank/manifest.csv. Nothing has ever written that file. Left as it was, live mode
    would have raised FileNotFoundError at startup -- which gallery mode hid, because gallery
    mode never calls this.

    WHY A MANIFEST AT ALL: each slice is 8*256*256 float16 = 1 MiB. Opening all 42 to answer
    "which slices have a big temporal lobe at a middle z?" would read the whole bank on every
    request. The manifest answers it from ~18 KB. Anything the manifest claims is re-verified
    against the actual array after loading (see _pick_slice), so a stale manifest degrades to
    "tries the next candidate" rather than to a wrong picture.
    """
    rows = []
    # encoding="utf-8-sig", not "utf-8". json.load rejects a leading byte-order mark with
    # "Expecting value: line 1 column 1", which reads like a corrupt file rather than an
    # encoding detail. The bundle is written on Linux (no BOM) but passes through a Windows
    # laptop on the way here, and some transfer paths add one. "utf-8-sig" strips a BOM if
    # present and is plain UTF-8 if not, so it is free insurance against a confusing error.
    with open(MANIFEST_PATH, encoding="utf-8-sig") as fh:
        man = json.load(fh)

    if not isinstance(man, dict):
        raise ValueError(f"{MANIFEST_PATH} is not a JSON object")
    slices = man.get("slices")
    if not isinstance(slices, list) or not slices:
        raise ValueError(
            f"{MANIFEST_PATH} has no non-empty 'slices' array. Re-run "
            "scripts/export_for_website.py on Great Lakes and copy the whole bundle folder."
        )

    for raw in slices:
        if not isinstance(raw, dict):
            continue
        rel = str(raw.get("file") or "").strip()
        if not rel:
            continue
        # The arrays live in the cond_slices/ SUBDIRECTORY of the bundle, and the manifest
        # records only the bare filename -- so the subdirectory is joined here rather than
        # being baked into each record.
        path = os.path.join(BANK_DIR, SLICE_SUBDIR, rel.replace("\\", "/"))
        try:
            z = int(float(raw.get("z")))
        except (TypeError, ValueError):
            z = None
        level = str(raw.get("level") or "").strip().lower()
        if level not in ("inferior", "middle", "superior"):
            level = level_word(z) if z is not None else "middle"
        # Accept the {name: px} object the exporter writes; tolerate a bare list, and a
        # semicolon-delimited string from the old CSV, so an older bundle still loads.
        raw_lobes = raw.get("lobes")
        if isinstance(raw_lobes, dict):
            lobes_ok = [str(k).strip().lower() for k in raw_lobes]
        elif isinstance(raw_lobes, list):
            lobes_ok = [str(s).strip().lower() for s in raw_lobes if str(s).strip()]
        else:
            lobes_ok = [s.strip().lower()
                        for s in str(raw.get("lobes_ok") or "").split(";") if s.strip()]

        # At the LOOP's indentation, not inside the else. Nested one level deeper this appends
        # only for the legacy string branch, so a real exporter-written bundle parses to zero
        # slices and _check_assets reports an empty bank -- with the manifest sitting right
        # there, correctly formatted.
        rows.append({
            "path":     path,
            "file":     rel,
            "patient":  str(raw.get("patient") or "").strip(),
            "z":        z,
            "level":    level,
            # An empty lobes_ok means "unknown, check it yourself" rather than "none".
            "lobes_ok": lobes_ok or list(LOBES),
            # Kept for _pick_slice: prefer the slice where the requested lobe is largest.
            "lobe_px":  raw_lobes if isinstance(raw_lobes, dict) else {},
            "brain_px": raw.get("brain_px"),
        })
    return rows


def _load_stack(path):
    """Load one bank slice as a (9,256,256) float32 array.

    THE .astype(np.float32) IS MANDATORY AND MUST BE THE FIRST OPERATION. 03_preprocess.py:88
    stores the stack as float16 to halve the bank size. If you index or assign into the half
    array before casting -- e.g. `stack[2] = synthetic_mask` -- numpy silently DOWNCASTS the
    float32 mask into float16, and later `torch.from_numpy` hands the fp32 U-Net a half tensor.
    Both are silent. Cast first, assert, then touch anything.
    """
    stack = np.load(path).astype(np.float32)               # (9,256,256) float32
    if stack.shape != (9, 256, 256):
        raise ValueError(f"bank slice {path} has shape {stack.shape}, expected (9, 256, 256)")
    assert stack.dtype == np.float32
    return stack


def _pick_slice(spec, rng):
    """Choose one bank slice satisfying the requested (lobe, slice level).

    Selection strategy, in order:
      1. candidates whose manifest level matches AND whose lobes_ok contains the target lobe;
      2. shuffle them (so repeated clicks on the same settings show different anatomy);
      3. load each in turn and RE-VERIFY with _slice_ok against the real array -- the manifest
         is a hint, the array is the truth;
      4. return the first that verifies.

    If nothing at the requested level verifies we raise rather than silently substituting a
    different level. Returning a middle slice while the UI still says "Superior" would be the
    site fabricating a labelled result for a combination it cannot satisfy, and that is worse
    than an honest error. The right fix for an unsupported combination is to disable it in the
    dropdown, using the coverage table the bank export prints.

    Returns (stack (9,256,256) float32, row dict, stats dict).
    """
    lobe = spec["lobe"]                                    # None for the no-tumour case
    level = spec["level"]

    candidates = [r for r in BANK if r["level"] == level and (lobe is None or lobe in r["lobes_ok"])]
    if not candidates:
        raise RuntimeError(
            f"the conditioning bank contains no {level} slice"
            + (f" with a usable {lobe} lobe. " if lobe else ". ")
            + "This (lobe, slice location) combination is not supported by the shipped bank -- "
              "either rebuild the bank to cover it, or disable the combination in the UI."
        )

    order = rng.permutation(len(candidates))               # deterministic given the seed
    last_stats = None
    for i in order.tolist():
        row = candidates[i]
        try:
            stack = _load_stack(row["path"])
        except Exception as e:                             # a truncated / missing file
            print(f"[conddiff] bank slice unreadable, skipping: {row['path']}: {e}")
            continue
        ok, stats = _slice_ok(stack, spec["lobe_idx"])
        last_stats = stats
        if ok:
            return stack, row, stats
        print(f"[conddiff] manifest/array disagree for {row['file']}: {stats} -- trying next")

    raise RuntimeError(
        f"every {level} candidate for lobe={lobe} failed re-verification against the real "
        f"array (last stats: {last_stats}). The manifest is stale -- rebuild the bank."
    )


# =============================================================================================
# SECTION 6B -- THE PRE-RENDERED GALLERY  *** GALLERY MODE ONLY ***
#
# Gallery mode's entire data path is this section. It replaces SECTION 6 (pick a conditioning
# slice), SECTION 7 (build and run an 85.3M-parameter U-Net) and the DDIM loop with: read a
# JSON index once at startup, look up a cell, choose a variant, read a file.
#
# THE ON-DISK LAYOUT the cluster renderer produces and this code consumes:
#
#     inference/conddiff_gallery/
#         manifest.json
#         with_frontal_middle_moderate/
#             v00_flair.png   v00_seg.png
#             v01_flair.png   v01_seg.png
#             ...                              (K seed variants per cell)
#         without_middle/
#             v00_flair.png                    (no seg: an all-zero mask is not an image)
#         ...
#
# THE MANIFEST SCHEMA. Written by the renderer, read by _load_gallery(). Only the fields marked
# REQUIRED are load-bearing here; everything else is provenance that is copied verbatim into
# the database record, so adding a field on the cluster side needs no change in this file.
#
#     {
#       "schema": "conddiff_gallery/1",              REQUIRED -- refuse a version we do not know
#       "model_version": "braingen_CondDiffuser_BraTS_v1",
#       "rendered_utc": "2026-09-06T18:22:04Z",
#       "core_sha256": "<sha256 of conddiff_core.py as the RENDERER ran it>",
#       "renderer": {"ddim_steps": 200, "device": "cuda", "torch": "...", "diffusers": "..."},
#       "cells": {                                   REQUIRED
#         "with_frontal_middle_moderate": {
#           "tumour": "With Tumor", "lobe": "Frontal",
#           "slice_location": "Middle", "tumour_size": "Moderate",
#           "variants": [                            REQUIRED, non-empty
#             {"flair": "with_frontal_middle_moderate/v00_flair.png",   REQUIRED
#              "seg":   "with_frontal_middle_moderate/v00_seg.png",
#              "seed": 1234567, "ddim_steps": 200,
#              "bank_file": "...", "bank_patient": "BraTS20_...", "bank_z": 88,
#              "bank_brain_px": 21044,
#              "lesion_px": 1502, "core_px": 450, "lobe_px": 4820, "coverage": 0.3116,
#              "centre_axis0": 118, "centre_axis1": 142, "tries": 2,
#              "size_requested": "moderate", "size_realized": "moderate",
#              "size_in_band": true}
#           ]
#         }
#       },
#       "empty_cells": ["with_cerebellum_superior_large", ...]
#     }
#
# WHY THE PER-VARIANT STATS ARE IN THE MANIFEST AT ALL, given that nobody looks at them: this
# is the honesty requirement. The site tells the user "Large lesion, temporal lobe" and stores
# that claim in their library forever. In live mode we measure what we actually drew and record
# it. Gallery mode must be able to make exactly the same statement about an image that was made
# months earlier on another machine -- so the renderer writes the measurements down at render
# time and inference() copies them into the database record. If a field is missing from the
# manifest it is simply absent from the record; NOTHING IS EVER SYNTHESISED HERE TO FILL A GAP,
# because a plausible-looking invented lesion area is exactly the kind of quiet lie this whole
# split-the-core exercise exists to prevent.
# =============================================================================================

# The parsed manifest. Populated once by _check_assets() at import; {} means "not loaded".
GALLERY = {}                # the whole manifest dict
GALLERY_CELLS = {}          # convenience alias for GALLERY["cells"]

# The only manifest schema this code understands. The renderer stamps it. Bumping it on the
# cluster without teaching this file about the new version is meant to fail LOUDLY at startup
# rather than half-read a layout that has changed underneath us.
GALLERY_SCHEMA = "conddiff_gallery/1"


def _load_gallery():
    """Read and validate conddiff_gallery/manifest.json. Returns the manifest dict.

    Raises on anything it cannot trust. The caller (_check_assets) turns that into READY=False
    plus a printed reason, so a broken gallery is a refused request with a diagnosable log
    line, never a wrong picture.

    VALIDATION IS DONE HERE, ONCE, AT STARTUP -- NOT PER REQUEST. Checking that every PNG
    listed actually exists costs ~57*K stat() calls, which is nothing at boot and is wasteful
    on every request. The per-request path (_pick_gallery_variant) therefore assumes the
    manifest is internally consistent and only handles the file disappearing underneath it.
    """
    with open(GALLERY_MANIFEST, encoding="utf-8") as fh:
        man = json.load(fh)

    if not isinstance(man, dict):
        raise ValueError(f"{GALLERY_MANIFEST} is not a JSON object")

    schema = man.get("schema")
    if schema != GALLERY_SCHEMA:
        raise ValueError(
            f"gallery manifest schema is {schema!r}, this backend understands "
            f"{GALLERY_SCHEMA!r}. The renderer in the research repo and this file are out of "
            "step -- re-export the gallery, or teach _load_gallery about the new schema."
        )

    cells = man.get("cells")
    if not isinstance(cells, dict) or not cells:
        raise ValueError(f"{GALLERY_MANIFEST} has no 'cells' object (or it is empty)")

    # Every cell must have at least one variant, and every variant must name a flair PNG that
    # exists. A cell that lists zero variants would pass a naive `cell in cells` test at request
    # time and then blow up inside rng.integers(0) with an unhelpful traceback.
    bad = []
    for cid, cell in cells.items():
        variants = (cell or {}).get("variants")
        if not isinstance(variants, list) or not variants:
            bad.append(f"{cid}: no variants")
            continue
        for i, v in enumerate(variants):
            rel = (v or {}).get("flair")
            if not rel:
                bad.append(f"{cid}[{i}]: no 'flair' key")
            elif not os.path.isfile(os.path.join(GALLERY_DIR, str(rel).replace("\\", "/"))):
                bad.append(f"{cid}[{i}]: missing file {rel}")
        if len(bad) > 8:                              # do not print hundreds of lines
            break
    if bad:
        raise ValueError(
            "gallery manifest and gallery folder disagree; first problems: "
            + "; ".join(bad[:8])
            + ". The PNGs and manifest.json must be copied out of the cluster together."
        )
    return man


def _gallery_cell_ids():
    """The cell ids the shipped gallery actually covers, as a set. Cheap; used by selftest."""
    return set(GALLERY_CELLS.keys())


def _pick_gallery_variant(spec, rng):
    """Choose one pre-rendered variant for this request. Returns (cell_id, cell, variant, idx).

    SELECTION. `rng` is seeded from the request's seed (see inference()), so the choice is
    reproducible: the recorded params carry that seed, and replaying it picks the same variant.
    That is why this uses the passed-in Generator rather than `random.choice` -- a request must
    be a pure function of its inputs in both modes, not just in live mode.

    THE MISSING-CELL CASE IS AN ERROR, NOT A SUBSTITUTION. If the gallery has no
    "with_cerebellum_superior_large" we raise. Serving a middle slice while the UI still says
    "Superior", or a frontal lesion while it says "Cerebellum", would be the site fabricating a
    labelled result for a combination it cannot satisfy. _pick_slice() refuses for exactly the
    same reason in live mode; the correct fix for a genuinely impossible cell (the cerebellum
    does not exist in superior slices) is to disable the combination in the dropdown, using the
    coverage report selftest() prints.
    """
    cid = cell_id(spec)
    cell = GALLERY_CELLS.get(cid)
    if not cell:
        raise RuntimeError(
            f"the shipped gallery has no cell {cid!r}. Either that (tumour, lobe, slice "
            f"location, size) combination was empty when the gallery was rendered -- the "
            f"manifest's 'empty_cells' list says which -- or the gallery is stale. Re-export "
            f"it, or disable the combination in ParameterControlPanel.tsx. "
            f"Gallery covers {len(GALLERY_CELLS)} of 57 cells."
        )
    variants = cell["variants"]
    idx = int(rng.integers(len(variants)))             # uniform over the K seed variants
    return cid, cell, variants[idx], idx


def _read_gallery_png(rel):
    """Read one gallery PNG as bytes. `rel` is manifest-relative, e.g. "cell/v00_flair.png".

    The `.replace("\\\\", "/")` is not cosmetic: the manifest may be written on Windows during a
    local test render, and os.path.join on Linux would treat a backslash as part of the
    FILENAME rather than as a separator, producing a "file not found" for a file that is right
    there. Normalising to "/" works on both platforms.
    """
    path = os.path.join(GALLERY_DIR, str(rel).replace("\\", "/"))
    with open(path, "rb") as fh:
        return fh.read()                               # already PNG-encoded by the renderer


# =============================================================================================
# SECTION 7 -- THE MODEL: ARCHITECTURE, WEIGHTS AND SCHEDULER  *** LIVE MODE ONLY ***
# (The DDIM loop that uses them, render_flair, moved to conddiff_core -- see the note below
#  _get_scheduler. Nothing in this section is reachable when MODE == "gallery": that is what
#  keeps torch and diffusers out of the container's memory entirely.)
# =============================================================================================

def _require_diffusers():
    """Import diffusers, or raise a message that says exactly what to do about it.

    diffusers IS now pinned in backend/requirements.txt (diffusers==0.38.0), but the import
    stays LAZY and stays guarded. Two reasons, and neither is hypothetical: this module is
    imported at the top of image_generation.py alongside the four GAN modules, so a hard
    ImportError here would take the whole backend down and break five working models; and
    diffusers declares torch only under an optional extra, so pip will happily install
    diffusers 0.38.0 next to torch 2.0.1 and any incompatibility between them surfaces HERE,
    at first live request, rather than at build time.

    This is also deliberately NOT the pattern at cgan_inference.py:11-18, which catches the
    import failure, sets DIFFUSERS_AVAILABLE=False, and then UPLOADS A BLACK SQUARE
    (np.zeros((128,128,4))) to the user's library as if it were a result. Papering over a
    missing dependency with a fake image is worse than failing.
    """
    try:
        from diffusers import UNet2DModel, DDIMScheduler   # noqa: F401  (imported for the check)
        return UNet2DModel, DDIMScheduler
    except Exception as e:
        raise RuntimeError(
            "braingen_CondDiffuser_BraTS_v1 needs the `diffusers` package, which is not "
            "installed in this container.\n"
            "  You are in CONDDIFF_MODE=live. Gallery mode never reaches this code path and "
            "needs neither diffusers nor torch -- if you did not mean to run live, unset "
            "CONDDIFF_MODE.\n"
            "  The version is NOT free: models/diffusion_ema.pt was written on Great Lakes by "
            "diffusers 0.38.0 / torch 2.11.0+cu128 (measured 2026-09-08). The attention block "
            "was renamed AttentionBlock -> Attention across diffusers versions, moving the "
            "parameter names query/key/value/proj_attn to to_q/to_k/to_v/to_out.0, so a skewed "
            "version fails the strict=True load below rather than loading silently.\n"
            "  requirements.txt ALREADY pins `diffusers==0.38.0`, so if you are reading this the "
            "image was built before that pin landed (rebuild and redeploy), or diffusers is "
            "present but failed to import -- read the underlying error at the bottom before "
            "changing any pin.\n"
            "  BUT NOTE THE REAL BLOCKER: requirements.txt pins torch==2.0.1+cpu, which is far "
            "older than the torch that wrote this checkpoint. Expect to have to raise torch as "
            "well, and re-test the five GAN models afterwards -- they load with strict=False, "
            "so a torch upgrade that broke them would NOT raise, it would quietly return "
            "randomly-initialised generators. That risk is exactly why gallery mode is the "
            "default and live mode is Phase 2.\n"
            f"  Underlying import error: {e}"
        )


def _build_unet():
    """Rebuild the paper's denoiser EXACTLY as src/model.py:build_model() defines it.

    Every single argument below is load-bearing. diffusers derives the state_dict key names
    and tensor shapes from this configuration, so any deviation -- a different
    block_out_channels tuple, an Attn block in the wrong position, layers_per_block=3 --
    produces a network whose parameter names do not match diffusion_ema.pt, and the strict
    load below fails. src/model.py's own docstring says "DO NOT change it, or the deployed
    checkpoint models/diffusion_ema.pt will no longer load."

    Architecture, spelled out:
      sample_size=256          256x256 slices (BraTS's 240x240 zero-padded by 8 px per side)
      in_channels=9            noisy FLAIR + 8 conditioning channels, concatenated
      out_channels=1           the model predicts the NOISE that was added to the FLAIR
      layers_per_block=2       two ResNet blocks per resolution level
      block_out_channels       (128, 256, 384, 512) -> four levels at 256, 128, 64, 32 px
      down/up_block_types      self-attention only at the DEEPEST level (32x32), where
                               all-pairs attention over 1024 tokens is affordable; running it
                               at 256x256 would be 65,536 tokens attending to each other.

    Returns an un-loaded UNet2DModel with ~85.3M parameters.
    """
    UNet2DModel, _ = _require_diffusers()
    return UNet2DModel(
        sample_size=256,
        in_channels=9,
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(128, 256, 384, 512),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )


def _find_checkpoint():
    """Return the first existing checkpoint path, or raise a message naming every path tried."""
    for p in CKPT_CANDIDATES:
        if os.path.isfile(p):
            return p
    raise RuntimeError(
        "braingen_CondDiffuser_BraTS_v1 checkpoint NOT FOUND. Looked for, in order:\n  "
        + "\n  ".join(CKPT_CANDIDATES)
        + "\nThe weights are delivered out of band, exactly like the five GAN .pth files -- "
          "they are not in git. Produce them by copying models/diffusion_ema.pt from Great "
          "Lakes into backend/inference/model/ before `docker build`.\n"
          f"(cwd at import time was {_CWD}; if that is not the backend/ directory, the server "
          "was started from the wrong place and EVERY model's weight path is broken.)"
    )


def _load_state_dict(path):
    """Read a checkpoint into a plain {name: fp32 tensor} dict, whatever container it is in.

    train.py:57-60 in the research repo does `ema.copy_to(model.parameters())` then
    `torch.save(model.state_dict(), ...)`. So the .pt file is a BARE state_dict of the EMA
    weights -- there is no ["model"] / ["state_dict"] / ["epoch"] wrapper. We check for a
    wrapper anyway and say so clearly if we find one, because a wrapped checkpoint fed to
    load_state_dict produces a confusing "unexpected key" wall rather than an explanation.

    Everything is upcast to float32. A fp16 export halves the shipped file, but torch 2.0.1's
    CPU half-precision kernels are poor, so we pay the memory and run in fp32.
    """
    import torch                                           # lazy: gallery mode never gets here

    if path.endswith(".safetensors"):
        try:
            from safetensors.torch import load_file
        except Exception as e:
            raise RuntimeError(
                f"checkpoint {os.path.basename(path)} is a safetensors file but the "
                f"`safetensors` package is not installed ({e}). Add `safetensors==0.8.0` to "
                "requirements.txt, or ship the .pt checkpoint instead. (0.8.0, not 0.4.5: "
                "diffusers 0.38.0 declares safetensors>=0.8.0-rc.0, so a lower pin makes the "
                "whole requirements file unresolvable and breaks `docker build`.)"
            )
        sd = load_file(path, device="cpu")
    else:
        sd = torch.load(path, map_location="cpu")

    # Unwrap the containers people commonly save, with a loud note if we had to.
    if isinstance(sd, dict) and not all(isinstance(v, torch.Tensor) for v in sd.values()):
        for key in ("state_dict", "model", "ema", "weights"):
            if key in sd and isinstance(sd[key], dict):
                print(f"[conddiff] checkpoint was wrapped in a '{key}' key -- unwrapping it. "
                      "The research repo saves a bare state_dict, so this file did not come "
                      "from scripts/train.py unmodified.")
                sd = sd[key]
                break
    if not isinstance(sd, dict) or not sd or not all(isinstance(v, torch.Tensor) for v in sd.values()):
        raise RuntimeError(
            f"{path} did not contain a state_dict of tensors (got {type(sd).__name__}). "
            "Expected the bare EMA state_dict written by scripts/train.py."
        )
    return {k: v.float() for k, v in sd.items()}          # fp16 export -> fp32 for CPU kernels


def _get_model():
    """Build + load the U-Net once, then hand back the cached instance.

    WHY A SINGLETON: the GAN modules reconstruct their generator and re-read the .pth on every
    single request (cgan_inference_v2.py:132-134). Those generators are a few MB. This one is
    ~85.3M fp32 parameters, i.e. ~341 MB, and reloading it per request would dominate the
    request AND spike memory on a container that has very little of it.
    """
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    ckpt = _find_checkpoint()
    model = _build_unet()
    sd = _load_state_dict(ckpt)

    # strict=True, ON PURPOSE. See the module docstring: every other loader in this folder uses
    # strict=False, which on a key-name mismatch loads NOTHING, returns quietly, and leaves you
    # with a randomly-initialised network that still emits brain-shaped blur. The most likely
    # cause of a mismatch here is a diffusers version skew (the AttentionBlock -> Attention
    # rename changed attention parameter names from query/key/value/proj_attn to
    # to_q/to_k/to_v/to_out.0), and that is exactly the failure we need to be loud.
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        import diffusers
        raise RuntimeError(
            "FAILED to load diffusion_ema into the U-Net with strict=True. The architecture "
            "here does not match the one that saved the checkpoint.\n"
            f"  checkpoint : {ckpt}\n"
            f"  diffusers  : {getattr(diffusers, '__version__', 'unknown')} (serving)\n"
            "  Most likely cause: a diffusers version skew renaming the attention block's "
            "parameters. Check `pip show diffusers` in the brainmri env on Great Lakes and pin "
            "that version here. Do NOT 'fix' this by switching to strict=False -- that would "
            "load a randomly-initialised model that still produces plausible-looking output.\n"
            f"  torch says: {e}"
        )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[conddiff] loaded {os.path.basename(ckpt)}: {n_params:,} parameters "
          f"({n_params / 1e6:.1f}M, expected ~85.3M) on device={_device()}")
    if not (0.99 * 85.3e6 <= n_params <= 1.01 * 85.3e6):
        raise RuntimeError(
            f"parameter count {n_params:,} is not within 1% of the paper's 85.3M. The "
            "architecture in _build_unet() has drifted from src/model.py:build_model()."
        )

    model.to(_device()).eval()                             # eval(): no dropout, no batchnorm updates
    _MODEL = model
    return _MODEL


def _get_scheduler(steps):
    """Build the DDIM scheduler, cached per step count.

    EVERY UNSPECIFIED ARGUMENT IS A LOAD-BEARING DEFAULT. scripts/train.py:20-21 defines the
    forward noising process as

        betas      = torch.linspace(1e-4, 0.02, 1000)
        alpha_bars = torch.cumprod(1.0 - betas, dim=0)

    which is byte-for-byte diffusers' defaults beta_start=0.0001, beta_end=0.02,
    beta_schedule="linear", num_train_timesteps=1000. And train.py:45 computes
    `F.mse_loss(pred, noise)`, i.e. the network predicts EPSILON, which fixes
    prediction_type="epsilon" (also the default). eta defaults to 0 -> deterministic DDIM.

    Do NOT pass beta_schedule="scaled_linear" (that is Stable Diffusion's schedule, not ours)
    and do NOT pass prediction_type="v_prediction" or "sample". Any of those produces noise
    instead of brains, with no error.

    With set_timesteps(50) and the default timestep_spacing="leading", sched.timesteps is a
    descending int64 tensor [980, 960, ..., 20, 0] (step ratio 1000 // 50 = 20).
    """
    global _SCHED, _SCHED_STEPS
    if _SCHED is not None and _SCHED_STEPS == steps:
        return _SCHED
    _, DDIMScheduler = _require_diffusers()
    sched = DDIMScheduler(num_train_timesteps=1000, clip_sample=True)
    sched.set_timesteps(steps)
    _SCHED, _SCHED_STEPS = sched, steps
    return _SCHED


# =============================================================================================
# THE REST OF SECTION 7, AND SECTION 8, MOVED TO inference/conddiff_core.py
#
# What used to be here:
#   build_cond    -- (9,256,256) stack + mask -> the (1,8,256,256) conditioning tensor, with
#                    the cond[1]-is-the-mask off-by-one that bites everyone
#   render_flair  -- the DDIM reverse-diffusion loop itself
#   SECTION 8     to_png_bytes -- PNG encoding in the paper's orientation and intensity mapping
#
# Same reason as the block above, and here it is at its most literal: the gallery's PNGs were
# encoded ONCE, on the cluster, months before the request that serves them. If this file
# encoded with a different orientation or a different intensity mapping, the identical request
# would produce a visibly different picture depending on which mode the container was in.
#
# Note what did NOT move: _get_scheduler() above stays here, because it needs the deployment's
# STEPS, and STEPS is a per-deployment decision (200 on the cluster, fewer if this were ever
# served live). The step count is recorded with every image so that difference is never silent.
# =============================================================================================

# =============================================================================================
# SECTION 9 -- PUBLISHING
# =============================================================================================

def _json_safe(obj):
    """Coerce numpy scalars/arrays into plain Python types so json.dumps() cannot fail.

    WHY THIS EXISTS: supabase_storage.add_to_database() does `json.dumps(parameters_used)`
    inside a try whose except merely PRINTS and returns None (supabase_storage.py:36-63). A
    single np.int64 in the params dict raises "Object of type int64 is not JSON serializable",
    which becomes a None row id, which becomes `[None]` from save_images, which the frontend
    posts as a null id and renders as... nothing. Green toast, empty viewer, no error anywhere
    except container stdout. Every stat we record comes out of numpy, so every stat goes
    through here.
    """
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):        # np.int64, np.float32, np.bool_ ...
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def save_images(images, save_path, user_id=None, project_id=None,
                model_name=None, params=None, is_playground=False):
    """Publish the generated images. `images` is an OrderedDict of tag -> PNG bytes.

    THE DIVERGENCE FROM THE SHARED HELPER, AND WHY IT MATTERS
    ---------------------------------------------------------
    cgan_inference.py:20 hardcodes `tags = ['t1','t2','flair','seg']` and then does

        for i in range(images.shape[-1]):
            image = images[:, :, i]
            filename = f"{tags[i]}.png"

    i.e. the FILENAME is chosen by the array's channel INDEX. Our output is FLAIR-only, so a
    (256,256,1) array runs that loop exactly once at i=0 and gets published as **t1.png** and
    recorded in the database as "t1 - mri_<uuid>". The frontend keys entirely off that name
    (playground: `url.includes('flair.png')`; authenticated: split the DB name on ' - '), so
    the FLAIR would render under the T1 tab, forever, with no error.
    Keying by TAG instead makes that unrepresentable. The alternative fix -- padding the array
    out to four channels with dummies -- would publish three junk images into the user's
    library, so it is not a fix.
    Writing a per-module save_images is the house pattern anyway: cgan_inference_v2.py:35 and
    wgan_inference.py:79 each have their own with different signatures and different tags.

    FILENAMES ARE A HARD CONTRACT. They must be exactly "flair.png" and "seg.png". Playground
    mode routes images by substring-matching the URL with no else-clause, so a filename like
    "flair_frontal.png" matches nothing and the image silently vanishes behind a green Success
    toast. Encode the lobe/size in the FOLDER name (already unique) and in `params`, never in
    the filename. Also note 't1.png' is tested first, so no filename may contain that substring.

    Returns a LIST (of DB record ids, or public URLs in playground mode). api.py accepts either
    a list or a bare string, but the 2D convention in this codebase is a list.
    """
    unique_id = uuid.uuid4().hex
    folder_name = f"mri_{unique_id}"

    # Fail here rather than inside Supabase's swallow-everything try block: if the params dict
    # is not serialisable, add_to_database silently returns None and the image never appears.
    params = _json_safe(params or {})
    json.dumps(params)                                     # raises now, loudly, if not safe

    if user_id and project_id:
        # Imported here rather than at module scope to avoid a circular import, matching
        # cgan_inference_v2.py:46. (In production this branch is ALWAYS taken: both ids come
        # from the request and image_generation.py:23-24 sanitises but never blanks them.)
        from supabase_storage import upload_image_to_supabase

        results = []
        for tag, png in images.items():                    # iterate by TAG, never by index
            filename = f"{tag}.png"
            value = upload_image_to_supabase(
                image_data=png,                            # BYTES -> bypasses the min-max stretch
                filename=filename,
                user_id=user_id,
                project_id=project_id,
                folder_name=folder_name,
                model_name=model_name,                     # <- see below; must not be None
                parameters=params,
                is_playground=is_playground,
            )
            # WHY model_name MUST BE PASSED: supabase_storage.py:112 returns the DATABASE
            # RECORD ID when model_name is set and is_playground is False, but the public URL
            # otherwise. The authenticated frontend posts what it gets back to
            # /api/images/batch as `ids`, so a URL there means the image is never fetched and
            # never appears. cgan_inference.py:138 gets this wrong today (passes None).

            if value is None:
                raise RuntimeError(
                    f"publishing {filename} returned None. That means the Supabase insert "
                    "failed (add_to_database swallows its own exception and returns None), so "
                    "the image exists in storage but has no database row and will never be "
                    "displayed. Check the container log for 'Error inserting record'."
                )
            if isinstance(value, str) and os.path.isabs(value) and not value.startswith("http"):
                # upload_image_to_supabase's except-branch returns a LOCAL PATH -- and because
                # we passed bytes rather than an ndarray, its `if isinstance(image_data,
                # np.ndarray)` guard is False and it writes NO FILE. The returned path points
                # at nothing. Better a clear error than a broken-image icon under a green
                # "Success" toast.
                raise RuntimeError(
                    f"Supabase upload failed and fell back to the local path {value}, which "
                    "was never written (the fallback only handles numpy arrays, and this "
                    "model uploads pre-encoded PNG bytes). Check SUPABASE_URL / SUPABASE_KEY "
                    "and the container log for 'Error uploading to Supabase'."
                )
            results.append(value)
        return results

    # --- local-disk branch: effectively dead in production, but kept working ---------------
    folder_path = os.path.join(save_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    image_paths = []
    for tag, png in images.items():
        filepath = os.path.join(folder_path, f"{tag}.png")
        with open(filepath, "wb") as fh:                   # already PNG-encoded; just write it
            fh.write(png)
        image_paths.append(filepath)
    return image_paths


# =============================================================================================
# SECTION 10 -- STARTUP ASSET CHECK
# =============================================================================================

def _check_assets():
    """Verify the ACTIVE MODE's assets are present, and print a diagnosable banner.

    Runs ONCE at import. It is wrapped by the caller so that it can never raise -- an
    exception here would propagate through image_generation.py's top-level import and take the
    whole backend, including the five working GAN models, offline.

    IT CHECKS ONE MODE'S ASSETS, NOT BOTH, AND THAT IS THE POINT.
    A gallery deployment ships no checkpoint and no .npy bank -- that is the entire reason it
    fits on the micro instance. Checking for them anyway would set READY=False and the model
    would refuse every request while being, in fact, perfectly deployable. The mirror image is
    equally wrong: a live GPU deployment has no gallery folder and must not be failed for it.
    So the checks below are behind `if MODE == ...`, and the banner prints which mode it
    validated so that a log line can never be misread as validating the other one.

    The point of printing the RESOLVED ABSOLUTE PATHS is that the container has no shell you
    are likely to use; `docker logs` has to be enough to diagnose "the assets did not land".
    """
    global READY, REASON, BANK, GALLERY, GALLERY_CELLS

    problems = []
    ckpt = None

    if MODE == "gallery":
        # ---- gallery mode: one JSON index and a tree of PNGs. No torch, no checkpoint. ------
        if not os.path.isfile(GALLERY_MANIFEST):
            problems.append(
                f"gallery manifest NOT FOUND at {GALLERY_MANIFEST}.\n"
                "  This deployment is in gallery mode (CONDDIFF_MODE=gallery -- note that "
                "UNSET means 'live', not 'gallery'), which "
                "serves images that were pre-rendered on the GPU cluster at the paper's full "
                "200 DDIM steps. It cannot render them itself -- that is the whole point, "
                "because live CPU diffusion on a 0.25 vCPU instance would exceed the "
                "Lightsail health check and get the container replaced mid-request.\n"
                "  Produce it by running the gallery renderer in the research repo and copying "
                "the resulting conddiff_gallery/ folder (manifest.json + the per-cell PNG "
                "directories, ~12 MB) into backend/inference/ before `docker build`.\n"
                "  Alternatively set CONDDIFF_MODE=live and ship the checkpoint + .npy bank "
                "instead -- but only on hardware that can actually afford it."
            )
        else:
            try:
                GALLERY = _load_gallery()
                GALLERY_CELLS = GALLERY.get("cells", {})
            except Exception as e:
                GALLERY, GALLERY_CELLS = {}, {}
                problems.append(f"could not use {GALLERY_MANIFEST}: {e}")
    else:
        # ---- live mode: exactly the checks this function has always done --------------------
        try:
            ckpt = _find_checkpoint()
        except RuntimeError as e:
            problems.append(str(e))

        if not os.path.isfile(MANIFEST_PATH):
            problems.append(
                f"conditioning bank manifest NOT FOUND at {MANIFEST_PATH}.\n"
                "  This model cannot invent its own conditioning: it needs a real patient T1 and "
                "six SRI24-warped atlas lobe maps, and the atlas is the output of a one-time ANTs "
                "registration that is normalised by a single global scalar -- recomputing it "
                "anywhere else would silently change every lesion size.\n"
                # THE INSTRUCTION MUST NAME THE LAYOUT THIS FILE ACTUALLY READS. It used to say
                # "copy slices into conddiff_bank/slices/ and write a manifest.csv" -- the layout
                # of an earlier revision that nothing ever produced (see _load_manifest's
                # docstring and the BANK_DIR comment in SECTION 1). An operator who followed it
                # would build conddiff_bank/manifest.csv by hand, redeploy, and get this exact
                # same error again, because MANIFEST_PATH points at conddiff_bundle/manifest.json.
                "  Produce it on Great Lakes with scripts/export_for_website.py, which writes a "
                "web_bundle/ folder containing manifest.json, cond_slices/<patient>_z###.npy and "
                "diffusion_ema.pt. Copy that folder WHOLE to "
                "backend/inference/conddiff_bundle/ before `docker build` -- do not rearrange it, "
                "and do not hand-write a manifest: this reader wants the exporter's JSON "
                "('slices' array of {file, patient, z, level, lobes}), not a CSV."
            )
        else:
            try:
                BANK = _load_manifest()
                if not BANK:
                    problems.append(f"{MANIFEST_PATH} parsed to zero usable rows.")
                missing = [r["file"] for r in BANK if not os.path.isfile(r["path"])][:5]
                if missing:
                    problems.append(
                        f"{len(missing)}+ slices listed in {MANIFEST_PATH} are missing from disk, "
                        f"e.g. {missing}. The manifest and the {SLICE_SUBDIR}/ folder are out of "
                        "sync -- copy the whole exported bundle rather than parts of it."
                    )
            except Exception as e:
                problems.append(f"could not parse {MANIFEST_PATH}: {e}")

    print("=" * 78)
    print(f"[conddiff] {MODEL_NAME}")
    # The default in this string must track the default in the os.environ.get() call at the top
    # of SECTION 1. It said 'gallery' while the code already defaulted to 'live', which is the
    # single most misleading thing a boot banner can do: an operator reading `docker logs` on a
    # container with no CONDDIFF_MODE set would conclude it was serving pre-rendered PNGs when
    # it was in fact configured to run the DDIM loop.
    print(f"[conddiff]   mode       : {MODE}   (env CONDDIFF_MODE; default 'live')")
    print(f"[conddiff]   cwd        : {_CWD}")
    if MODE == "gallery":
        n_var = sum(len(c.get("variants", [])) for c in GALLERY_CELLS.values())
        print(f"[conddiff]   gallery    : {GALLERY_DIR}")
        print(f"[conddiff]                {len(GALLERY_CELLS)} of 57 cells, "
              f"{n_var} pre-rendered images")
        # Provenance of the pictures we are about to serve as this model's output. Printed at
        # boot because it is the only place an operator can see, without opening the manifest,
        # WHICH code produced them -- core_sha256 is the hash of conddiff_core.py as the
        # RENDERER ran it, and it should equal the hash of this container's own copy.
        if GALLERY:
            r = GALLERY.get("renderer", {}) or {}
            print(f"[conddiff]   rendered   : {GALLERY.get('rendered_utc', 'unknown')} "
                  f"at {r.get('ddim_steps', '?')} DDIM steps on {r.get('device', '?')}")
            print(f"[conddiff]   core sha256: {GALLERY.get('core_sha256', 'not recorded')}")
            empty = GALLERY.get("empty_cells") or []
            if empty:
                print(f"[conddiff]   EMPTY cells: {len(empty)} -- disable these in the UI: "
                      f"{', '.join(empty[:6])}{' ...' if len(empty) > 6 else ''}")
    else:
        print(f"[conddiff]   checkpoint : {ckpt or 'MISSING'}")
        print(f"[conddiff]   bank       : {BANK_DIR}  ({len(BANK)} slices in manifest)")
        # _device() is called here rather than earlier because calling it IMPORTS TORCH. In
        # gallery mode we never reach this line, so the boot banner costs no torch import.
        # Wrapped because this is still module-scope work: if torch were somehow missing from a
        # live deployment, an unguarded call would abort the banner BEFORE it printed the
        # problems list, and the log would say nothing useful about why the model is refusing.
        try:
            dev = _device()
        except Exception as e:                             # pragma: no cover -- torch is pinned
            dev = f"UNKNOWN ({e})"
            problems.append(f"live mode needs torch and it could not be imported: {e}")
        print(f"[conddiff]   device     : {dev}   ddim steps: {STEPS} "
              f"(env CONDDIFF_STEPS)")
        if BANK:
            by_level = {}
            for r in BANK:
                by_level[r["level"]] = by_level.get(r["level"], 0) + 1
            print(f"[conddiff]   levels     : {by_level}")
    if problems:
        READY = False
        REASON = "\n".join(problems)
        print("[conddiff]   STATUS     : NOT READY -- this model will refuse requests:")
        for line in REASON.splitlines():
            print(f"[conddiff]     {line}")
    else:
        READY = True
        REASON = "ok"
        print("[conddiff]   STATUS     : ready")
    print("=" * 78)


try:
    _check_assets()
except Exception as _e:                                    # belt and braces -- see the docstring
    READY, REASON = False, f"startup check itself failed: {_e}"
    print(f"[conddiff] startup check raised (backend is unaffected): {_e}")


def supported_combinations():
    """Which (Lobe, Slice Location) pairs this deployment can actually generate.

    WHY THIS EXISTS. The conditioning bank cannot fill every cell, and this is anatomy, not a
    bug: scripts/export_for_website.py's real run admitted only 11 of the 18 (lobe, level)
    combinations. The cerebellum lives in inferior slices, so `cerebellum_superior` has no
    candidate slice; the insula is small enough that 03_preprocess.py:55's SINGLE global atlas
    maximum can leave its peak probability under PROB_TH = 0.4, so `insula_inferior` never
    clears MIN_LOBE = 500 px. Those cells will never be renderable from this bank.

    Without this endpoint the frontend offers all 18, the user picks one of the 7 that do not
    exist, _pick_gallery_variant raises KeyError, image_generation.py:65 catches it and
    substitutes an empty list, and FastAPI returns HTTP 200 -- a green "Success" toast over a
    blank viewer, with the only evidence in container stdout. So the UI must GREY THE CELL OUT.

    WHY NOT JUST SUBSTITUTE A NEARBY SLICE LEVEL: because the site would then label the image
    "Cerebellum, Superior" while showing anatomy generated at a different level. export_for_
    website.py's own transfer checklist says this explicitly -- "do not silently substitute a
    different slice level, or the site will label an image with a location it did not
    generate." Refusing to offer the cell is honest; quietly serving a different one is not.

    RETURN SHAPE. Values are the EXACT dropdown strings the frontend uses ("Frontal",
    "Middle"), not the lowercase internal vocabulary, because the manifest already records them
    that way (see THE MANIFEST SCHEMA above) and a case-mapping layer in TypeScript is one more
    place for the two sides to disagree.

        {"mode": "gallery", "ready": True,
         "pairs":  [{"lobe": "Frontal", "slice_location": "Middle"}, ...],   # generatable
         "lobes":  ["Frontal", ...],                                        # any level works
         "levels": ["Inferior", "Middle", "Superior"],
         "sizes":  ["Small", "Moderate", "Large"]}

    A deployment that is not READY returns ready:False with empty lists, which the frontend
    renders as "every option disabled" -- the correct behaviour when nothing can be generated.
    """
    # `core` is already bound at module scope (see the try/except import above); re-importing it
    # here as a bare `conddiff_core` would work when the server is started from backend/ and
    # fail when it is imported as the `inference` package, which is exactly the split the
    # try/except exists to paper over. Use the binding that is already correct.
    UI_LOBE, UI_LEVEL, UI_SIZE = core.UI_LOBE, core.UI_LEVEL, core.UI_SIZE

    out = {"mode": MODE, "ready": bool(READY), "reason": REASON,
           "pairs": [], "lobes": [], "levels": [], "sizes": list(UI_SIZE.keys())}
    if not READY:
        return out

    # Collect the (lobe, level) pairs that have at least one RENDERABLE cell. A pair counts only
    # if some cell using it carries a non-empty `variants` list -- a cell key with zero variants
    # is a cell the renderer skipped, and offering it would fail exactly like a missing key.
    pairs = set()
    if MODE == "gallery":
        for cell in GALLERY_CELLS.values():
            if not cell.get("variants"):
                continue
            lobe, level = cell.get("lobe"), cell.get("slice_location")
            if lobe and level:                        # no-tumour cells carry no lobe; skip them
                pairs.add((lobe, level))
    else:
        # LIVE mode reads the bank instead of the gallery. BANK rows carry the lowercase internal
        # vocabulary, so invert UI_LOBE / UI_LEVEL to get back to dropdown strings.
        inv_lobe  = {v: k for k, v in UI_LOBE.items()}
        inv_level = {v: k for k, v in UI_LEVEL.items()}
        for row in BANK:
            level = inv_level.get(row.get("level"))
            if not level:
                continue
            # _load_manifest calls this key "lobes_ok" and stores a LIST of lowercase lobe names
            # (an empty one in the CSV is normalised to "all six", meaning "unknown, verify at
            # request time"). It is not a {lobe: px} dict -- reading it as one would iterate the
            # characters of nothing and silently yield zero pairs, disabling the entire control.
            for lobe in (row.get("lobes_ok") or []):
                ui = inv_lobe.get(lobe)
                if ui:
                    pairs.add((ui, level))

    out["pairs"]  = [{"lobe": l, "slice_location": z} for l, z in sorted(pairs)]
    # Keep the declaration order of the dropdowns rather than sorting alphabetically, so the UI
    # lists Frontal..Insula and Inferior..Superior the way it always has.
    out["lobes"]  = [l for l in UI_LOBE  if any(p[0] == l for p in pairs)]
    out["levels"] = [z for z in UI_LEVEL if any(p[1] == z for p in pairs)]
    return out


def selftest():
    """Cheap diagnostic payload for an HTTP endpoint. Imports nothing heavy, runs no model.

    Served by api.py's GET /conddiff-selftest. It exists because image_generation.py catches
    every inference exception, prints it, and substitutes an empty list -- so a broken
    deployment returns HTTP 200 with a green "Success" toast and a blank viewer, and container
    stdout is otherwise the only evidence. This is that evidence over HTTP.

    The payload is MODE-DEPENDENT: reporting bank coverage on a gallery deployment (or gallery
    coverage on a live one) would be reporting on assets that are not there and are not
    supposed to be. The shared keys come first, then exactly one mode's block.
    """
    out = {
        "model_name":     MODEL_NAME,
        "mode":           MODE,
        "ready":          READY,
        "reason":         REASON,
        "lobes":          list(LOBES),
        "sizes":          {k: list(v) for k, v in AREA_FRAC_BY_SIZE.items()},
        # The full parameter space, so an operator can compare it against what is covered.
        "cells_total":    sum(1 for _ in core.iter_cells()),      # 57
    }

    if MODE == "gallery":
        covered = _gallery_cell_ids()
        # WHAT IS MISSING IS MORE USEFUL THAN WHAT IS PRESENT: every id in this list is a
        # dropdown combination the UI currently offers and the backend will refuse. Either
        # re-render those cells or disable the combination in ParameterControlPanel.tsx.
        missing = [cid for cid, _ in core.iter_cells() if cid not in covered]
        renderer = (GALLERY.get("renderer") or {}) if GALLERY else {}
        out.update({
            "gallery_dir":       GALLERY_DIR,
            "gallery_schema":    GALLERY.get("schema") if GALLERY else None,
            "cells_covered":     len(covered),
            "cells_missing":     missing,
            "variants_total":    sum(len(c.get("variants", [])) for c in GALLERY_CELLS.values()),
            # Provenance of the pictures being served. ddim_steps here is what the images were
            # ACTUALLY rendered at (200 on the cluster), NOT this container's STEPS env var --
            # which is meaningless in gallery mode and is deliberately not reported.
            "rendered_utc":      GALLERY.get("rendered_utc") if GALLERY else None,
            "rendered_steps":    renderer.get("ddim_steps"),
            "rendered_device":   renderer.get("device"),
            # sha256 of conddiff_core.py AS THE RENDERER RAN IT. Compare against this
            # container's own copy to prove the shared core did not drift between the machine
            # that made the images and the machine serving them.
            "core_sha256":       GALLERY.get("core_sha256") if GALLERY else None,
        })
    else:
        # WHY THIS RESOLVES THE DEVICE EAGERLY, WHEN _device() IS DELIBERATELY LAZY EVERYWHERE
        # ELSE. The lazy design exists so GALLERY mode never imports torch (~200 MB of RSS for
        # an answer no gallery request uses) -- and this branch only runs in LIVE mode, where
        # torch is imported by the very first /generate anyway, so there is nothing left to
        # save by deferring.
        #
        # What deferring COST was the single most dangerous failure this deployment has, and it
        # is a SILENT one: the host is a GPU box, but the installed torch wheel is a CPU-only
        # build (or a CUDA build whose runtime does not match the driver), so
        # torch.cuda.is_available() is False, _device() returns "cpu", and every request still
        # SUCCEEDS -- at roughly two minutes a sample instead of one second, with correct
        # images and no error anywhere. The whole point of moving to a GPU host is then
        # silently unachieved. Reporting `"device": "not resolved yet"` until someone happens to
        # generate an image means the one endpoint built to answer "will this deployment work"
        # cannot answer the one question the migration turns on.
        #
        # Wrapped, because selftest's contract is that it CANNOT fail: api.py serves it as an
        # unguarded `return cdiff.selftest()`, and an ImportError here (torch absent, or a
        # broken CUDA build raising on import) would turn the diagnostic endpoint into an
        # HTTP 500 exactly when it is needed most. A torch that cannot even be imported is
        # itself the answer, so it is reported as a string rather than raised.
        try:
            _dev = _device()                                # memoises DEVICE; imports torch
            import torch                                    # already imported by the line above
            _cuda = {
                "cuda_available": bool(torch.cuda.is_available()),
                "torch_version":  getattr(torch, "__version__", "unknown"),
                # "2.0.1+cpu" here on a GPU host is the smoking gun for the failure above:
                # a CPU-only wheel can never see the card, however many GPUs nvidia-smi lists.
                "cuda_built":     getattr(getattr(torch, "version", None), "cuda", None),
                "gpu_name":       (torch.cuda.get_device_name(0)
                                   if torch.cuda.is_available() else None),
            }
        except Exception as _dev_err:                       # noqa: BLE001 -- diagnostics only
            _dev  = f"unavailable: {type(_dev_err).__name__}: {_dev_err}"
            _cuda = {"cuda_available": None, "torch_version": None,
                     "cuda_built": None, "gpu_name": None}

        out.update({
            "device":           _dev,
            **_cuda,
            "ddim_steps":       STEPS,
            "checkpoint_found": any(os.path.isfile(p) for p in CKPT_CANDIDATES),
            "bank_slices":      len(BANK),
            "bank_levels":      sorted({r["level"] for r in BANK}),
            "bank_lobes":       sorted({l for r in BANK for l in r["lobes_ok"]}),
            # bank_levels and bank_lobes are UNIONS over the whole bank, so on their own they
            # can advertise coverage that does not exist: a bank with a cerebellum slice at
            # "inferior" and a frontal slice at "superior" reports both lobes and both levels
            # while the (cerebellum, superior) cell -- which _pick_slice would refuse -- is
            # empty. This is the actual 6-lobes x 3-levels grid the UI can satisfy, so an
            # operator can see which dropdown combinations to disable. Counts are manifest
            # CLAIMS; _pick_slice still re-verifies the chosen slice against the real array.
            "bank_coverage":    {lv: {lb: sum(1 for r in BANK
                                              if r["level"] == lv and lb in r["lobes_ok"])
                                      for lb in LOBES}
                                 for lv in ("inferior", "middle", "superior")},
        })
    return out


# =============================================================================================
# SECTION 11 -- THE ENTRY POINT image_generation.py CALLS
#
# Structure, because it changed when gallery mode arrived:
#
#     inference()            the public entry point. Does everything that is TRUE IN BOTH
#                            MODES -- the readiness gate, params_to_spec, the seed -- then
#                            dispatches, then assembles the database record and publishes.
#       _generate_gallery()  look the cell up, pick a variant, read two PNGs.  (gallery)
#       _generate_live()     pick a slice, draw a mask, run 50-200 U-Net steps. (live)
#
# Both helpers return the SAME PAIR: (images, provenance) -- an OrderedDict of tag -> PNG
# bytes, and a dict of "what actually happened". Neither of them publishes anything and
# neither of them writes the four user-facing controls into the record. That is deliberate:
# the honesty rules (below) are then enforced in exactly ONE place, at the bottom of
# inference(), instead of being duplicated in two branches that could drift apart.
# =============================================================================================

# Provenance keys copied verbatim from a gallery variant into the database record.
#
# THIS LIST IS THE HONESTY CONTRACT FOR GALLERY MODE. In live mode we measure what we just did
# and record the measurement. In gallery mode the image was made months ago on another machine,
# so the measurements have to be carried in the manifest and copied here. Every key below was
# WRITTEN AT RENDER TIME by the renderer, from the values conddiff_core actually produced --
# they are measurements, not intentions. If a key is absent from a variant it is absent from
# the record and its name is listed in "not_recorded", so the gap is visible rather than
# papered over. NOTHING IS EVER COMPUTED OR GUESSED HERE TO FILL A HOLE.
#
# THE LIST IS SPLIT IN TWO because the lesion measurements only exist when there is a lesion. A
# "Without Tumor" cell legitimately records no lesion_px, and reporting those ten keys as
# "not_recorded" on every no-tumour request would be crying wolf -- it would train whoever
# reads the logs to ignore the one warning that actually matters. This mirrors params_to_spec
# returning None for lobe and size in the same situation: absent because meaningless, not
# absent because lost.
_GALLERY_PROVENANCE_KEYS = (
    # what the sampler did
    "ddim_steps",                      # 200 on the cluster -- the paper's value
    "seed",                            # the seed that produced THIS image
    # provenance of the ANATOMY (a real held-out validation subject, not model output)
    "bank_file", "bank_patient", "bank_z", "bank_brain_px",
)
_GALLERY_LESION_KEYS = (
    # The realised lesion -- what was actually drawn, not what was asked for. This is exactly
    # the `stats` dict conddiff_core.synth_mask_sized returns, which is what the renderer wrote
    # into the manifest at render time.
    "lesion_px", "core_px", "lobe_px", "coverage",
    "centre_axis0", "centre_axis1", "tries",
    "size_requested", "size_realized", "size_in_band",
)


def _generate_gallery(spec, rng, seed):
    """GALLERY MODE: serve a pre-rendered image. Returns (images, provenance).

    No torch, no diffusers, no checkpoint, no .npy bank -- a dict lookup, a uniform choice
    among the cell's K seed variants, and one or two file reads.

    WHY THE VARIANTS EXIST AT ALL: without them, two visitors who both ask for
    "With Tumor / Temporal / Middle / Large" would get the identical PNG, and the site would
    look like a lookup table rather than a generative model. K distinct seeds per cell keeps
    repeated clicks interesting while every image remains a genuine sample from the model. The
    choice is driven by the REQUEST's rng, so it is reproducible from the recorded
    selection_seed rather than being unrepeatable global randomness.

    spec : the dict from params_to_spec
    rng  : numpy Generator seeded from the request seed
    seed : the request seed, recorded as selection_seed (NOT as the image's seed -- see below)
    """
    cid, cell, variant, idx = _pick_gallery_variant(spec, rng)

    images = OrderedDict()
    images["flair"] = _read_gallery_png(variant["flair"])
    if spec["with_tumour"]:
        seg_rel = variant.get("seg")
        if seg_rel:
            # Same reasoning as live mode: the mask is ground truth (the renderer drew it), it
            # files under the 'seg' key the viewer already labels "Tumor Mask", and having two
            # images unlocks the viewer's Grid View toggle.
            images["seg"] = _read_gallery_png(seg_rel)
        else:
            # Publish the FLAIR anyway rather than failing the whole request -- but say so, and
            # do not pretend a mask was shown.
            print(f"[conddiff] gallery variant {cid}[{idx}] has no seg PNG; publishing FLAIR only")
    # For "Without Tumor" we publish only the FLAIR: an all-zero mask carries no information
    # and would be a constant image, which divides by zero in any consumer that min-max
    # normalises. The renderer does not write a seg for those cells either.

    provenance = {
        "sampler":        "DDIM",
        "conditioning":   "held-out BraTS validation subject (T1 + ICBM452/SRI24 lobe atlas)",
        # WHICH pre-rendered image this is, so it can be found on disk and re-examined.
        "gallery_cell":    cid,
        "gallery_variant": idx,
        "gallery_flair":   variant["flair"],
        # TWO SEEDS, AND THE DISTINCTION MATTERS.
        #   "seed"           (copied from the variant below) is the seed that MADE the image,
        #                    on the cluster. Replaying it in live mode reproduces the picture.
        #   "selection_seed" is this request's seed, which only chose WHICH variant to show.
        # Recording the request seed as "seed" would be a lie: no seed on this container ever
        # produced this image, and someone re-running it would get something else entirely.
        "selection_seed":  seed,
        # Provenance of the RENDER RUN as a whole (as opposed to this one image).
        "rendered_utc":    GALLERY.get("rendered_utc"),
        "core_sha256":     GALLERY.get("core_sha256"),
        "render_device":   (GALLERY.get("renderer") or {}).get("device"),
    }

    # Copy only what was actually recorded, and name what was not. The lesion keys are only
    # EXPECTED when this cell has a lesion at all -- see the comment on the key lists above.
    expected = _GALLERY_PROVENANCE_KEYS + (_GALLERY_LESION_KEYS if spec["with_tumour"] else ())
    missing = []
    for k in expected:
        if k in variant:
            provenance[k] = variant[k]
        else:
            missing.append(k)
    if missing:
        provenance["not_recorded"] = missing
        print(f"[conddiff] gallery variant {cid}[{idx}] does not record {missing} -- those "
              f"fields are omitted from the database record rather than invented")

    print(f"[conddiff] gallery hit: cell={cid} variant={idx}/{len(cell['variants'])} "
          f"rendered_seed={variant.get('seed')} steps={variant.get('ddim_steps')} "
          f"lesion={variant.get('lesion_px')} px")
    return images, provenance


def _generate_live(spec, rng, seed):
    """LIVE MODE: run the model now. Returns (images, provenance).

    This is the original code path, unchanged in behaviour. It is also the REFERENCE the
    gallery is generated from: the cluster renderer performs these same steps, through the same
    conddiff_core functions, at 200 DDIM steps.
    """
    import torch                                            # lazy -- live path only

    device = _device()

    # ---- pick the conditioning slice --------------------------------------------------------
    stack, row, slice_stats = _pick_slice(spec, rng)         # (9,256,256) float32
    brain = stack[1] > 0.05                                  # (256,256) bool -- channel 1 = T1

    # ---- build the tumour mask (channel 2) --------------------------------------------------
    if spec["with_tumour"]:
        atlas_lobe = stack[ATLAS0 + spec["lobe_idx"]]        # (256,256) the chosen lobe's map
        mask, centre, mask_stats = synth_mask_sized(atlas_lobe, brain, rng, spec["size"])
        print(f"[conddiff]   lesion {mask_stats['lesion_px']} px "
              f"({100 * mask_stats['coverage']:.0f}% of a {mask_stats['lobe_px']} px lobe), "
              f"core {mask_stats['core_px']} px, realised size '{mask_stats['size_realized']}'"
              f"{'' if mask_stats['size_in_band'] else ' (OUTSIDE the requested band)'}")
    else:
        # "Without Tumor" ZEROES the mask channel -- it does not synthesise a tiny lesion and
        # it does not reuse the bank slice's own mask. The bank slices are already selected to
        # be tumour-free ((stack[2] > 0.5).sum() < 50), but zeroing is explicit and cannot be
        # defeated by a stale manifest.
        mask = np.zeros((256, 256), dtype=np.float32)
        centre, mask_stats = None, {}

    # ---- assemble conditioning and run the reverse diffusion --------------------------------
    cond = build_cond(stack, mask, device)                   # (1,8,256,256) on `device`
    model_obj = _get_model()                                 # cached after the first request
    sched = _get_scheduler(STEPS)

    # Use every core the container is given. The default thread count is often wrong inside a
    # CPU-limited container, and this loop is ~950 GFLOP per step -- it is the whole request.
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    flair = render_flair(model_obj, sched, cond, seed)       # (256,256) float32 in [0,1]

    # ---- encode ------------------------------------------------------------------------------
    # An OrderedDict so the FLAIR is uploaded first and therefore becomes the first tab.
    images = OrderedDict()
    images["flair"] = to_png_bytes(flair, kind="flair")
    if spec["with_tumour"]:
        # The mask is ground truth -- we drew it -- so publishing it is free extra value. It
        # files under the existing 'seg' key, which the viewer already labels "Tumor Mask", and
        # having two images unlocks the viewer's Grid View toggle.
        # For "Without Tumor" we publish only the FLAIR: an all-zero mask carries no
        # information and would be a constant image, which divides by zero in any consumer
        # that min-max normalises.
        images["seg"] = to_png_bytes(mask, kind="seg")

    provenance = {
        "sampler":       "DDIM",
        "ddim_steps":    STEPS,
        "seed":          seed,                               # this seed really did make this image
        "device":        device,
        # provenance of the ANATOMY -- this is not model output, it is a real held-out subject
        "conditioning":  "held-out BraTS validation subject (T1 + ICBM452/SRI24 lobe atlas)",
        "bank_file":     row["file"],
        "bank_patient":  row["patient"],
        "bank_z":        row["z"],
        "bank_brain_px": slice_stats["brain_px"],
    }
    provenance.update(mask_stats)          # lesion_px, coverage, size_realized, size_in_band, ...
    return images, provenance


def inference(model, tumour, lobe, slice_location, tumour_size, save_path,
              user_id=None, project_id=None, is_playground=False, seed=None):
    """Generate one atlas-guided synthetic-tumour FLAIR and publish it.

    In gallery mode "generate" means "serve one of the K images that were rendered for exactly
    these four choices, on the GPU cluster, at the paper's full 200 DDIM steps". In live mode it
    means run the reverse diffusion here and now. Which one happened is recorded per image in
    params["rendering"], so the two are never confusable after the fact.

    SIGNATURE. The argument order deliberately mirrors cgan_inference_v2.py:98 --
    (model, <the params>, save_path, user_id=None, project_id=None, is_playground=False) --
    which is the convention the 2D dispatcher already uses, so image_generation.py can call
    this positionally like every other model. `seed` is appended last (keyword-only in
    practice) so that adding it cannot disturb the positional call.

    model          : the full model_name string, e.g. "braingen_CondDiffuser_BraTS_v1 (2D)".
                     Passed straight through to save_images because supabase_storage keys the
                     "return an id, not a URL" behaviour off it.
    tumour         : "With Tumor" | "Without Tumor"
    lobe           : "Frontal" | "Parietal" | "Temporal" | "Occipital" | "Cerebellum" | "Insula"
    slice_location : "Inferior" | "Middle" | "Superior"
    tumour_size    : "Small" | "Moderate" | "Large"
    seed           : optional int; if omitted a fresh one is drawn and RETURNED in the params.
                     In live mode it makes the image exactly reproducible; in gallery mode it
                     selects the variant and is recorded as selection_seed (the image's own
                     seed comes from the manifest).

    Returns a list of published ids/URLs (one per image: flair, and seg when there is a tumour).
    """
    # ---- 0. refuse clearly if the shipped assets are not there ----------------------------
    # image_generation.py:65 catches every exception, prints it, and substitutes an empty list,
    # so the API returns HTTP 200 and the user gets a green "Success" toast and a blank viewer.
    # We cannot change that (it is shared with five working models), so instead we make the
    # container log unambiguous and never publish a placeholder image.
    # READY reflects the ACTIVE MODE's assets only -- see _check_assets.
    if not READY:
        print("=" * 78)
        print(f"[conddiff] REFUSING TO GENERATE (mode={MODE}) -- required assets are missing:"
              f"\n{REASON}")
        print("=" * 78)
        raise RuntimeError(f"{MODEL_NAME} is not deployable in {MODE} mode: {REASON}")

    # ---- 1. normalise the four dropdown strings into a validated spec ----------------------
    # Both modes go through the SAME funnel, which is what makes cell_id() a valid key: the
    # renderer built the gallery by walking core.iter_cells() through this same function.
    spec = params_to_spec(tumour, lobe, slice_location, tumour_size)
    if spec["substituted"]:
        print(f"[conddiff] substituted defaults for {spec['substituted']} -- the frontend sent "
              f"tumour={tumour!r} lobe={lobe!r} slice_location={slice_location!r} "
              f"tumour_size={tumour_size!r}")

    # ---- 2. one seed in, every randomness source out ---------------------------------------
    # secrets.randbelow rather than random.randint: it needs no global RNG state, so two
    # concurrent requests cannot collide on a shared seeded generator.
    if seed is None:
        seed = secrets.randbelow(2 ** 31)
    seed = int(seed)
    # live mode : drives slice choice AND the mask shape (and, inside render_flair, the torch
    #             Generator for the diffusion noise -- so the whole image is a pure function of
    #             this one number)
    # gallery   : drives the choice of variant, and nothing else
    rng = np.random.default_rng(seed)

    print(f"[conddiff] generating [{MODE}]: tumour={spec['with_tumour']} lobe={spec['lobe']} "
          f"level={spec['level']} size={spec['size']} cell={cell_id(spec)} seed={seed}")

    # ---- 3. do the work, whichever way this container is configured to -----------------------
    if MODE == "gallery":
        images, provenance = _generate_gallery(spec, rng, seed)
    else:
        images, provenance = _generate_live(spec, rng, seed)

    # ---- 4. the record that goes into the database -------------------------------------------
    # THE HONESTY RULES, ENFORCED HERE AND ONLY HERE:
    #   * the four user-facing controls are echoed back exactly as they arrived, so the record
    #     shows what was ASKED for;
    #   * "rendering" says which path produced the bytes -- "gallery" or "live" -- and is
    #     always present, because a viewer of the library months later cannot otherwise tell
    #     whether the image was made on demand or picked from a pre-rendered set;
    #   * everything else comes from the helper and describes what ACTUALLY HAPPENED: the DDIM
    #     step count, the seed, which held-out subject supplied the anatomy, which slice index,
    #     and the realised lesion measurements. In gallery mode those were recorded at render
    #     time and copied across; in live mode they were measured a few seconds ago;
    #   * nothing is claimed that was not recorded. A field the gallery manifest did not carry
    #     is absent and is named in "not_recorded", never reconstructed from the request.
    params = {
        # what was asked for
        "tumour":         tumour,
        "lobe":           lobe,
        "slice_location": slice_location,
        "tumour_size":    tumour_size,
        # which path produced these bytes
        "rendering":      MODE,
        "model_version":  "braingen_CondDiffuser_BraTS_v1",
    }
    if spec["substituted"]:
        params["substituted_defaults"] = spec["substituted"]
    params.update(provenance)                  # what actually happened -- see the helpers

    # ---- 5. publish ---------------------------------------------------------------------------
    return save_images(images, save_path, user_id, project_id, model, params, is_playground)
