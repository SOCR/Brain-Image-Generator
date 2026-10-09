# Running the Brain Image Generator on the Hugging Face free plan

**Design approved:** 2026-09-25 by Alex Liu, one decision at a time. Each decision in §3 lists the alternatives considered and the reason for the choice.
**Scope:** an MVP. All six models run on Hugging Face, and the website generates through it. The "later" items are in §7.
**Code:** `backend/hf_space/` (`app.py`, `requirements.txt`, `README.md`, `push_space.py`), `lib/hfSpace.ts`, and small edits to `app/api/generate/route.ts`, `app/api/conddiff-cells/route.ts` and `components/projects/GenerationPageClient.tsx`.
**Unchanged:** every inference module, `image_generation.py`, `supabase_storage.py`, `api.py`, both Dockerfiles and the deploy scripts. There is no production FastAPI host anymore: AWS access, including Lightsail, is gone.

---

## 1. What runs where

```
 visitor's browser
   │  page, parameter panel, library
   ▼
 Vercel (Next.js, free plan)  /api/generate, /api/conddiff-cells
   │  HF_SPACE_URL set → lib/hfSpace.ts → HTTPS, with the lab's HF_TOKEN on every request
   ▼
 HF Space  ALIUD1/braingen-backend   [Gradio 6.28, ZeroGPU, PUBLIC]
   app.py  (the Gradio replacement for api.py)
   ├─ endpoint "generate"        = api.py POST /generate        (same JSON in and out)
   ├─ endpoint "conddiff_cells"  = api.py GET  /conddiff-cells  (same JSON out)
   ├─ main process, CPU, no quota:     image_generation.py → 4 GAN modules (5 models)
   └─ @spaces.GPU worker, uses quota:  conddiff _generate_live → 200 DDIM steps
   ▲ at boot: snapshot_download (Space secret HF_TOKEN)        │ publish
   │                                                           ▼
 HF model repo  ALIUD1/braingen-assets  [PRIVATE]           Supabase (storage + DB)  [unchanged]
   model/          7 GAN weight files (~450 MB)
   conddiff_bundle/  = web_bundle/ (checkpoint + 42 BraTS slices + manifest, 367 MB)
```

---

## 2. The free-plan facts that forced this shape

Checked against huggingface.co/docs and the installed `spaces` 0.51.3 source on 2026-09-23, with an independent verifier for each claim.

| Fact | Consequence |
|---|---|
| Since 2026-07-21 a free account **cannot create Docker or CPU Gradio Spaces**. The only free compute is **up to 2 Gradio Spaces on ZeroGPU**, for a personal account with a verified e-mail that is older than 30 days. | Everything goes in one ZeroGPU Gradio Space. `ALIUD1` was created 2026-04-22, so it qualifies. |
| ZeroGPU = Gradio SDK only; Python **3.10.13 or 3.12.12**; torch **2.8.0–2.13.0**. The builder installs `gradio[oauth,mcp]==<sdk_version>`. | `app.py` wraps the backend in Gradio. torch moves 2.0.1 → 2.11.0 and supabase moves 2.0.3 → 2.31.0 (see 2.7). |
| The GPU exists only inside `@spaces.GPU` functions, which run in a **forked worker**. Arguments and results are copied between processes. | GANs run on the CPU in the main process. Only diffusion's generation loop is sent to the GPU. |
| ZeroGPU bills the **caller**, identified by the request's HF token. Quotas: anonymous 2 min/day, free account 5 min/day, PRO 40 min/day. A free account can't buy extra. | See §4. |
| Before running, `spaces` multiplies the requested duration by **1.5 on the RTX Pro 6000 Blackwell**, and refuses the call if the caller has less quota left than that. | `duration=30` is checked as **45 s** (see 2.9). |
| `preload_from_hub` can't read private repos; a Space's disk is wiped on every restart; a Space repo is capped at ~1 GB. | Weights and data live in a private model repo and are downloaded at boot. |

---

## 3. Decisions (as approved)

| # | Decision | Chosen | Over | Why |
|---|---|---|---|---|
| 1 | Host | **Hugging Face, free plan** | Globus Compute, PRO | The goal is to get the site back online for free. |
| 2.1 | Kind of app | **Gradio app with API endpoints (`gr.api`)** mirroring `api.py`'s JSON | Gradio Server mode; mounting `api.py` in Gradio; a Docker Space | ZeroGPU only runs Gradio apps. A GPU call must happen inside a Gradio event, because that's how ZeroGPU knows whose quota to charge (`spaces/zero/wrappers.py`). Docker needs PRO and can't use ZeroGPU. |
| 2.2 | Number of Spaces | **One for all six models** | One Space per model family | Fewer moving parts. The second free slot stays open for staging. |
| 2.3 | CPU vs GPU | **GANs on CPU, diffusion on GPU** | Everything on GPU; everything on CPU | The GANs take 0.01–0.2 s on CPU and cost no quota. Diffusion on CPU takes ~9 min per image. |
| 2.4 | Getting diffusion onto the GPU | **Rebind `cdiff._generate_live` from `app.py`** | A hook in the model file; `@spaces.GPU` inside the model file; wrapping the whole request | Python looks names up at call time, so one rebinding in memory routes the unmodified pipeline through the GPU. The model file on disk stays byte-identical. The original function is kept as `_generate_live_cpu` for the life of the process. |
| 2.5 | Errors | **Fix in `app.py`:** re-raise as `gr.Error` inside the worker, stash the message, re-raise it from `generate()` | Editing `image_generation.py`; leaving it | Otherwise the worker boundary drops messages and `image_generation.py` swallows them. The real reason ("quota exceeded, try again in …") now reaches the user. |
| 2.6 | Python / torch | **3.10 / 2.11.0** | 3.12; torch 2.13 / 2.8 | 2.11.0 trained the checkpoint. 3.10 keeps numpy 1.24.3 / matplotlib 3.7.2, which have no 3.12 builds, so the GAN numbers stay identical. |
| 2.7 | Other libraries | **diffusers 0.38.0, supabase 2.31.0, gradio 6.28.0, the rest unchanged** | supabase 2.0.3; older Gradio; rewriting the upload code | diffusers fixes the checkpoint's weight names. supabase 2.0.3 makes the build unsatisfiable (httpx conflict with `gradio[mcp]`). 2.31.0 was tested against a fake Supabase: identical requests. |
| 2.8 | Denoising steps | **200** (Space variable `CONDDIFF_STEPS`) | 50; measure first; user-selectable | The paper's value. 50 was a CPU placeholder whose quality was never measured. Every image records its step count. |
| 2.9 | GPU time per call | **30 s default; set to 15 on the Space on 2026-09-25** after measuring (cold 11.0 s end to end, `gpu_seconds` ≈ 6.3; requested window 15 × 1.5 = 22.5 s) (`CONDDIFF_GPU_DURATION`) | A formula based on steps; 60 s; 10 s | Safe for cold starts on day one. Tighten to ~1.5 × the measured cold-call time, and keep it ≤ 80. A typo falls back to 30. |
| 2.10 | Model loading | **Once at boot, `.to("cuda")`, device forced to cuda** | Lazy load; per-call move; fp16 | ZeroGPU's documented pattern. The checkpoint is read once per boot, and a broken checkpoint fails at boot, not on a user's click. |
| 2.11 | Simultaneous requests | **One at a time** (Gradio's default) | 2–4 at once; separate queues | `supabase_storage.py` writes a fixed `temp_nifti.nii.gz`, so running requests in parallel could mix up users' 3D volumes. The old backend also ran one at a time. |
| 2.12 | Space page | **A small diffusion test form that publishes nothing** | API only; a full demo | Used to verify deploys, measure `gpu_seconds` and run quota tests. Visitors who use it spend their own quota. |
| 3.1 | Where data lives | **One private HF model repo** | Model + dataset repos; a Supabase bucket; a Storage Bucket; the Space repo | Free, token-gated and versioned. It mirrors `backend/inference/`, so files land where the code looks. |
| 3.2 | Space visibility | **Public for the MVP** | Private first (`push_space.py --private`); Protected (PRO) | Known to work on the free plan. The data stays private (3.1). The accepted risks are in §6. |
| 3.3 | Getting data in | **`snapshot_download` at boot, then check `inference/model/` arrived; not pinned** | `preload_from_hub`; mounting the repo; downloading on the first request | With a bad token, `snapshot_download` silently returns if the folder already has files (reproduced), so the check makes the boot fail loudly. Unpinned: each restart serves the latest upload. |
| 3.4 | Tokens | **Three tokens:** Space = fine-grained read on `braingen-assets`; Vercel = fine-grained read on the Space, used for quota identity; yours = write (it creates the repos), never stored in the Space or Vercel | One read token everywhere; one write token everywhere | A leak stays small, and each token can be revoked independently. |
| 3.5 | Approval | **Cleared 2026-09-25:** everything may be uploaded | Waiting; splitting the question | Alex has full permission. |
| 3.6 | Uploads | **From where each set lives, with `hf upload`; only the 7 GAN files that are loaded** | Everything through the laptop or through Great Lakes; all 8 `.pth` files | No copies between machines, and the repo holds exactly what's served. |
| 4.1 | Website → Space | **The Vercel server calls the Space** (`lib/hfSpace.ts`) | The browser calling the Space directly; both behind a switch | Works like the old backend: the token stays on the server and the login check still applies. |
| 4.2 | Token on which requests | **Every request, playground included** | Signed-in only; none; no diffusion in the playground | Playground visitors get the account's quota and priority. Accepted risk 6.2. |
| 4.3 | Website wait limit | **`maxDuration = 60`** | 300 s (needs Fluid compute); Vercel's default (10 s) | Allowed on every Vercel plan. It covers the normal case. |
| 4.4 | When `HF_SPACE_URL` is unset | **Fall back to the old FastAPI `fetch`** | Always call the Space; a mode setting | Local development with `uvicorn api:app` keeps working, and it's the smallest change to the shared repo. |
| 4.5 | Gradio protocol | **A hand-written 20-line `fetch` helper** | `@gradio/client`; a Python proxy | No dependency. Tested against a real Gradio 6.28 server; the test caught that Gradio 6 sends errors as an object. |
| 4.6 | Error text | **Show the backend's message as-is** | Friendly rewording; a generic message | Users can tell quota (wait) from bug (report it) from waking up (retry). |
| 4.7 | Available combinations | **Ask the Space's `conddiff_cells` on every page load** | Hard-coding them; caching | Always matches the deployed bank, and opening the page wakes a sleeping Space. |
| 5.1 | Open `generate` endpoint | **Accepted risk for the MVP** | A shared-secret header; a private Space | See §6. |
| 5.2 | Quota draining via the playground | **Accepted for now** | Signed-in-only token; rate limiting; sign-in required | See §6. The escape hatch is one line. |
| 5.3 | Images per request | **Capped at 1–5 in `app.py`** | No cap; also capping in the route | Matches the UI. One huge request can't freeze the queue or flood storage. |
| 6.1 | Deploying code | **`push_space.py`: one command, one atomic commit of 13 files** | Manual `hf upload`s; git; GitHub auto-deploy | One rebuild per deploy, and weights can never be uploaded by accident. |
| 7.1 | 8- vs 9-channel slices *(found on the first real GPU run)* | **Fix `_load_stack` in `conddiff_inference.py`:** accept 8-channel slices and prepend a zero channel 0 | Another name swap in `app.py`; re-export with `--keep-flair`; padding the files | The exporter ships 8 channels by default (the patient's real FLAIR is deliberately dropped), but the loader only accepted 9, so live mode rejected every real slice. Channel 0 is never read at inference (`build_cond` takes `stack[1:9]`), so the zeros change no computation. This is the one edit to the model file, overriding 2.4's "untouched". `conddiff_core.py` is still unchanged (sha256 `87785659…`). |

---

## 4. Whose GPU quota pays

The website's server sends the lab account's token with **every** request, so **all diffusion generated through the site shares that account's 5 GPU-minutes/day** (40 on PRO), at medium priority. The GANs use none.

**The arithmetic.** Worst case, HF deducts the full requested 45 s per image (the docs say actual runtime, one HF source says requested): 300 / 45 = **6 images/day**. If actual runtime is charged and an image takes ~5 s, it's **~60/day**.

**Two tests to run once the Space works** (about 10 minutes):
1. **Which time is charged:** note the remaining quota before and after one run (refusal messages print "Ns requested vs. Ms left"). A drop of ~`gpu_seconds` means actual time is charged; ~45 s means the requested time is.
2. **Run cap:** HF staff have mentioned an undocumented ~3 runs/day limit for free accounts. Run the Space page's test form 4 times while logged in. A refusal on the 4th, with quota left, means the cap is real.

If the quota isn't enough: tighten `CONDDIFF_GPU_DURATION`, lower `CONDDIFF_STEPS` (after the step sweep), or move to **PRO ($9/mo)**, which needs no code change.

---

## 5. Runbook

### 0. Before you start
- [x] Permission to upload the weights and BraTS slices to a private HF repo (cleared 2026-09-25).
- [x] HF account `ALIUD1` is older than 30 days. Also check that the e-mail is verified (Settings → Account).
- [ ] **Review and push `feat/conddiff-brats-v1` to SOCR/Brain-Image-Generator.** The Space is public, so it will show this code.
- [ ] Create the upload token: huggingface.co → Settings → Access Tokens → **Write** token named `braingen-upload`. It has to be able to *create* the two repos, and a fine-grained token can only be scoped to repos that already exist. It is for you only and never goes into the Space or Vercel. The two read-only tokens are created later, in steps 3 and 5, once their repos exist.
- [ ] `pip install -U huggingface_hub` (≥ 1.32 provides the `hf` command), then `hf auth login` with `braingen-upload`.

### 1. Upload the weights and the bank
Laptop (the 7 GAN files the code loads):
```bash
hf repos create ALIUD1/braingen-assets --private
hf upload ALIUD1/braingen-assets C:/Users/alexl/code/Research/socr/_socr_checkpoints/app/inference/model model --include generator_epoch_200_seg.pth --include generator_epoch_145.pth --include generator_epoch_95_seg.pth --include generator_coarse_epoch_16_2500.pth --include generator_fine_epoch_16_2500.pth --include generator_epoch_6100.pth --include generator_epoch_3d64.pth
```
Great Lakes **login node**, which has internet (compute nodes don't):
```bash
module load python && source ~/envs/brainmri/bin/activate
hf version            # if `hf` is missing: pip install -U huggingface_hub  (or use `huggingface-cli upload`)
hf auth login
hf upload ALIUD1/braingen-assets ~/Summer2026/web_bundle conddiff_bundle
hf auth logout
```
Use `web_bundle/` whole and unrearranged. Adjust the path if it was exported with `--out`. If `selection.min_brain` in its `manifest.json` isn't 15000, set the Space variable `CONDDIFF_MIN_BRAIN` to it in step 3.

### 2. Create and push the Space
From the repo root on the laptop:
```bash
python backend/hf_space/push_space.py ALIUD1/braingen-backend --assets ALIUD1/braingen-assets
```
This creates the Space (Gradio, ZeroGPU, public), sets `ASSETS_REPO`, uploads 13 files in one commit, and prints the `conddiff_core.py` hash. The build takes ~5–15 min. **The first boot is expected to fail** with "ASSETS_REPO … download failed", because the Space has no `HF_TOKEN` yet. Step 3 adds it and restarts.

### 3. Add the Space's secrets
First create a **fine-grained** token `braingen-space-read` with **read** access to `ALIUD1/braingen-assets` only. Then Space → Settings → *Variables and secrets* → **New secret**: `HF_TOKEN` = `braingen-space-read`; `SUPABASE_URL` = `https://jfabbyhrypmamraqluyv.supabase.co` (the lab's production project) and `SUPABASE_KEY` = the **anon** key, i.e. the value of `NEXT_PUBLIC_SUPABASE_ANON_KEY` in the repo-root `.env.local`. **Not** `backend/.env`: that file points at a local test Supabase (`http://localhost:54321`). The anon key is enough because the lab's schema lets it upload to the `images`/`volumes` buckets ("Public Access" policy) and insert into `generated_images` (no row-level security). It's also already public, since browsers receive it. Only if a publish fails with a row-level-security or 401/403 error, use the `service_role` key (Supabase → Project Settings → API) instead. Optional variables: `CONDDIFF_STEPS` (200), `CONDDIFF_GPU_DURATION` (30, whole seconds), `CONDDIFF_MIN_BRAIN`. Then **Restart**.

### 4. Verify on HF (the MVP's success checks)
- [ ] The logs show `STATUS : ready`, `device : cuda`, `loaded diffusion_ema.pt: 85,261,185 parameters`. A bad token fails the boot with "ASSETS_REPO … download failed".
- [ ] The Space page's test form produces a FLAIR, and the JSON shows `"device": "cuda"` and `gpu_seconds`. Time the **first** run after a restart (cold), then set `CONDDIFF_GPU_DURATION` ≈ 1.5 × that (≤ 80).

### 5. Point the website at it
Create a **fine-grained** token `braingen-vercel` with **read** access to `ALIUD1/braingen-backend` only. Then Vercel → Settings → Environment Variables, then redeploy:
- `HF_SPACE_URL` = `https://aliud1-braingen-backend.hf.space` (check it on the Space page: ⋮ → *Embed this Space* → Direct URL)
- `HF_TOKEN` = `braingen-vercel`

Then check:
- [ ] All 6 models generate from the website.
- [ ] A website diffusion image counts against **ALIUD1's** ZeroGPU quota; your account's usage goes up. If it doesn't, ZeroGPU isn't attributing the minimal token: give `braingen-vercel` broader read access.
- [ ] The first diffusion image and its database row appear correctly in the lab's **real** Supabase. supabase 2.31.0 was only tested against a fake server.
- [ ] The parameter panel greys out the unsupported lobe/level combinations.

To take the Space out of the path, delete `HF_SPACE_URL` and redeploy. The routes then call `NEXT_PUBLIC_BACKEND_URL`, which only helps if some FastAPI host exists.

---

## 6. Known limits and accepted risks

| # | Limit / risk | What happens | Fix, when it matters |
|---|---|---|---|
| 6.1 | **The public `generate` endpoint** (accepted, 5.1) | Anyone who finds the Space can publish images into the lab's Supabase under any `user_id`/`project_id`, or keep the one-at-a-time queue busy with GAN calls (≤ 5 images per call). | **A shared-secret header** (~6 lines + one secret on each side). Also: tie `user_id` to `session.user.id` in `/api/generate`; or a private Space. |
| 6.2 | **Quota draining** (accepted, 4.2 / 5.2) | The playground skips login, and every request carries the token, so a script can use up the day's 5 GPU-minutes. Signed-in users then get "quota exceeded" until the reset. It can't cost money or touch data. | **One line** in `route.ts`: `const token = is_playground ? undefined : process.env.HF_TOKEN;` |
| 6.3 | Daily GPU quota | Toast: "You have exceeded your GPU quota … try again in H:MM:SS". | Tighten the duration, reduce steps after the sweep, or PRO. |
| 6.4 | Vercel's 60 s cap | Under load (up to 60 s waiting for a GPU, plus queueing behind another user), Vercel returns 504, but the Space may still finish, publish the image and spend the quota. | If the Vercel project has Fluid compute on, raise `maxDuration` to 300. |
| 6.5 | The Space sleeps when idle | First request: "The model server is not answering… starting up". While the Space can't be reached, `/api/conddiff-cells` answers "not ready", and the panel then leaves **every** lobe/level option **enabled** (`ParameterControlPanel.tsx:84`: it only filters when `ready` is true). A missing combination is then refused with its reason (4.6). | Retry. Later: a keep-alive ping. |
| 6.6 | Legacy `braingen_diffuser_BraTS_v1` | Not served (it uses `.to("cuda")` outside a GPU function). | It was never offered in the UI. |

---

## 7. Later (logged, not in the MVP)

- **Security:** the shared-secret header (6.1); tying `user_id` to the session; the playground-token escape hatch (6.2); a private Space.
- **Shared-code fixes (separate lab-repo changes):** a unique temp file per request in `supabase_storage.py`, then raise the request limit; the root fix for `image_generation.py` hiding errors.
- **Research:** a DDIM step sweep (25/50/100/200 vs 200, PSNR/SSIM) to decide whether fewer steps lose quality.
- **Operations:** pin the weights' revision; mount the repo instead of downloading it; GitHub auto-deploy; a keep-alive ping; the quota tests in §4.
- **Polish:** friendly error text; a 300 s limit if Fluid compute is on.
- **Upgrades:** Python 3.12 (3.10 loses security fixes in October 2026), with numpy/matplotlib upgrades and a GAN re-test.

---

## 8. What was tested (laptop, CPU, Python 3.10.21)

The environment was the exact dependency set the HF builder installs: `requirements.txt` + `gradio[oauth,mcp]==6.28.0` + `spaces`, resolved with uv (torch 2.11.0, supabase 2.31.0, httpx 0.28.1).

- **End to end:** the real `lib/hfSpace.ts` against the staged Space tree running `app.py`, with the real GAN weights and a fake conddiff bundle (random weights, exact architecture, synthetic slices). It covered:
  - conddiff with and without tumour
  - diffusion clamped to 1 image
  - the 1–5 `n_images` cap
  - an unsupported cell refused with its reason
  - all 5 GAN models
  - a malformed request
  - an unreachable Space
  - a malformed `CONDDIFF_GPU_DURATION`

  The fake checkpoint loads `strict=True` at 85,261,185 parameters. The fake slices are **8-channel** (1,048,704 bytes each), exactly like the real exporter output. An earlier 9-channel fake bundle is why the 7.1 bug stayed hidden until the first real run.
- **The real bundle and checkpoint (on HF):** 42 slices from 22 patients, 11 lobe × level pairs. The real `diffusion_ema.pt` loads `strict=True` at 85,261,185 parameters.
- **A failed asset download** (401) now stops the boot with a clear error. It no longer boots silently without weights.
- **`smoke_test_models.py`, torch 2.0.1 vs 2.11.0:** every GAN checkpoint matches every key (30/30, 30/30, 30/30, 87/87, 30/30, 41/41), and the outputs are equal to within ≤ 1e-7.
- **`supabase_storage.py` (unmodified), supabase 2.0.3 vs 2.31.0,** against a fake Supabase server: the same upload paths, the same multipart PNG body, the same database row and ID.
- **Not testable before the Space exists:** real ZeroGPU behaviour and real GPU timings (§5 step 4).
