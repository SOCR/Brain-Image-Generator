"""app.py -- the whole SOCR Brain-Image-Generator backend as ONE Hugging Face ZeroGPU Space.

WHY THIS FILE EXISTS INSTEAD OF api.py ON A DOCKER SPACE
--------------------------------------------------------
Since July 2026 a FREE Hugging Face account can no longer create Docker or CPU Gradio Spaces.
The only free compute left is up to 2 Gradio Spaces on ZeroGPU (verified e-mail, account older
than 30 days). ZeroGPU is Gradio-only, so FastAPI's api.py cannot be the entry point here. This
file is the Gradio replacement for it and exposes the same two calls the website makes:

    Gradio endpoint "/generate"        <-> api.py POST /generate        (same JSON in, same JSON out)
    Gradio endpoint "/conddiff_cells"  <-> api.py GET  /conddiff-cells  (same JSON out)

Everything below those two calls is the UNMODIFIED backend: image_generation.py dispatches, the
four GAN modules generate and publish to Supabase exactly as on Lightsail, and
conddiff_inference.py runs its own live path. The only thing this file changes is WHERE the
diffusion model's reverse loop runs: on a ZeroGPU GPU, via one wrapper (see _render_on_gpu).

HOW ZEROGPU SPLITS THE WORK (and why the five GANs cost no GPU quota)
---------------------------------------------------------------------
A ZeroGPU Space has a long-lived MAIN process (this file, Gradio, the queue) with NO real GPU.
A GPU exists only inside a function decorated with @spaces.GPU. Each call to one is run in a
short-lived worker process that is forked with a GPU attached, and its arguments and return
values are pickled across that boundary. So:

    * the four GAN modules force device='cpu' (cgan_inference_v2.py:109, wgan_inference.py:152)
      and run in the main process on the Space's CPU. They never touch the GPU and cost 0 s of
      quota, just as they cost 0 GPU on Lightsail;
    * only conddiff_inference._generate_live -- pick a slice, draw a mask, 200 DDIM steps -- is
      sent to the GPU. It takes (spec dict, numpy Generator, int) and returns (OrderedDict of
      PNG bytes, dict), all plain picklable CPU objects, which is exactly what the boundary needs.

WHOSE QUOTA IS SPENT
--------------------
ZeroGPU bills the CALLER, not the Space owner. The website's Vercel server is the caller, and it
sends the lab account's HF token with every request (lib/hfSpace.ts, playground included), so every
diffusion image generated through the site draws on that one account: 5 min/day free, 40 min/day on
PRO. The five GANs draw nothing. Someone who calls this public Space directly, or uses the test form
below, spends their own quota, not the lab's. When a quota runs out, ZeroGPU refuses the call and
its message ("...try again in H:MM:SS") is passed through to the user's toast -- see _render_live
and generate(). DEPLOY.md §4 has the arithmetic.
"""

# `spaces` MUST be imported before anything imports torch. On ZeroGPU it patches torch's CUDA
# layer so that `.to("cuda")` in the main process (which has no GPU) is recorded instead of
# failing, and replayed on the real GPU inside @spaces.GPU. Off ZeroGPU (a laptop) the package
# is inert and @spaces.GPU is a pass-through, so this same file runs locally on the CPU.
import spaces

import io
import os
import secrets
import threading
import time

import numpy as np
from huggingface_hub import snapshot_download

# Every inference module builds its paths as os.path.join(os.getcwd(), "inference/..."), so the
# working directory IS the configuration (see the note above _CWD in conddiff_inference.py). On
# the Space the cwd is already this folder; pinning it here makes a local run from anywhere else
# behave identically instead of failing with a missing-weights error for all six models.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)

# --- 1. THE WEIGHTS AND THE CONDITIONING BANK --------------------------------------------------
# None of them are in the Space repo. They live in a PRIVATE model repo (Space variable
# ASSETS_REPO, e.g. "ALIUD1/braingen-assets") that mirrors backend/inference/:
#     model/generator_*.pth            the 7 GAN weight files the code loads (~450 MB)
#     conddiff_bundle/manifest.json    export_for_website.py's output,   (~367 MB)
#     conddiff_bundle/cond_slices/*.npy    copied whole and unrearranged
#     conddiff_bundle/diffusion_ema.pt
# so downloading it INTO inference/ drops every file exactly where the unmodified modules look.
#
# WHY A SEPARATE PRIVATE REPO AND NOT THE SPACE REPO: the conditioning slices are real BraTS
# patients' T1 scans. A private model repo keeps them unreadable to anyone but the token holder
# whatever the Space's own visibility is. WHY AT STARTUP AND NOT AT BUILD TIME: the Space README's
# preload_from_hub option does not support private repos, and a Space's disk is wiped on every
# restart anyway, so a download at boot is the one mechanism that works. The HF_TOKEN Space secret
# is picked up by huggingface_hub from the environment automatically; it needs read access to
# ASSETS_REPO only. Downloading inside the HF network takes seconds.
#
# Unset ASSETS_REPO = "the files are already on disk", which is how a local test runs.
ASSETS_REPO = os.environ.get("ASSETS_REPO", "").strip()
if ASSETS_REPO:
    snapshot_download(repo_id=ASSETS_REPO, local_dir=os.path.join(HERE, "inference"))
    # snapshot_download does NOT raise when the repo is unreachable (bad/expired HF_TOKEN, typo in
    # ASSETS_REPO, Hub outage) if local_dir already has files in it -- and inference/ always does,
    # because the Space repo ships the .py modules there. It logs a warning and returns, and the
    # Space would boot "Running" with no weights: every GAN request a blank viewer. So check that
    # something actually arrived, and crash the boot visibly if not. Only model/ (the GANs) is
    # required: a repo holding just the GAN weights -- e.g. before the BraTS slices are approved
    # for upload -- is a valid deployment, and conddiff reports its own missing bundle clearly.
    if not os.path.isdir(os.path.join(HERE, "inference", "model")):
        raise RuntimeError(f"ASSETS_REPO={ASSETS_REPO!r} download failed: inference/model/ is missing. "
                           "Check the HF_TOKEN secret (read access to that repo) and the repo name.")

# --- 2. CONFIGURE THE DIFFUSION MODEL BEFORE IT IS IMPORTED ------------------------------------
# conddiff_inference reads both at IMPORT time, so they must be set first. setdefault, so a Space
# variable of the same name still wins.
#
# 200 DDIM steps, the paper's value, instead of the module's CPU placeholder of 50. The 50 was a
# compromise for a 0.25 vCPU box ("nothing in either repo has ever compared quality below 200
# steps"). On a GPU, 200 steps take a few seconds, so the site now serves the same step count as
# the paper and the cluster gallery. If the daily quota turns out to be the bottleneck, lower it
# with a Space variable -- every image records its own ddim_steps, so the change is never silent.
os.environ.setdefault("CONDDIFF_MODE", "live")
os.environ.setdefault("CONDDIFF_STEPS", "200")

import gradio as gr                                  # noqa: E402  (must follow `import spaces`)
from PIL import Image                                # noqa: E402

import image_generation                              # noqa: E402  the unmodified dispatcher + 4 GAN modules
import inference.conddiff_inference as cdiff         # noqa: E402  (already imported by the line above)

# --- 3. PUT THE DIFFUSION MODEL "ON CUDA" ONCE, IN THE MAIN PROCESS -----------------------------
# ZeroGPU's documented pattern: build the model at module level and call .to("cuda") there. In the
# main process that move is intercepted (the weights are parked, not uploaded); every GPU worker
# then starts with the model already resident. Loading inside the GPU function instead would
# re-read the 341 MB checkpoint on every request, on the user's quota.
#
# cdiff.DEVICE is forced rather than trusted because _check_assets() already resolved it at import
# via torch.cuda.is_available(), and whether ZeroGPU's emulation answers True in the main process
# is an implementation detail we should not bet the device on. SPACES_ZERO_GPU is the variable
# the `spaces` package itself uses to detect ZeroGPU hardware.
if os.environ.get("SPACES_ZERO_GPU"):
    cdiff.DEVICE = "cuda"
if cdiff.READY:
    cdiff._get_model()                     # strict=True load + parameter-count check, then .to(DEVICE)
    cdiff._get_scheduler(cdiff.STEPS)      # built once here so forked workers inherit it

# --- 4. SEND THE REVERSE-DIFFUSION LOOP TO THE GPU ---------------------------------------------
# GPU_DURATION is the GPU time REQUESTED per call, in seconds. ZeroGPU checks it against the
# caller's remaining daily quota BEFORE running and refuses the call if less is left, so it should
# be the smallest SAFE upper bound. Two things inflate it (spaces 0.51.3, zero/client.py):
#   * on the RTX Pro 6000 Blackwell it is multiplied by duration_factor = 1.5 before the check,
#     so 30 is requested -- and shown in quota messages -- as 45 s;
#   * a cold GPU worker (first call after boot, or after any failed call) also spends the window on
#     CUDA init and on moving the 341 MB of weights to the GPU, which gpu_seconds does NOT include.
# So size it from a COLD call's end-to-end time, not from a warm gpu_seconds. Keep it <= 80:
# 80 x 1.5 = 120 s, the practical per-call cap for free callers.
#
# Parsed defensively for the same reason conddiff_inference.py parses CONDDIFF_STEPS that way: a
# typo'd Space variable ("12.4", "12s") must not crash the import and take all six models down.
try:
    GPU_DURATION = max(1, round(float(os.environ.get("CONDDIFF_GPU_DURATION", "30"))))
except (ValueError, OverflowError):                 # "12s" -> ValueError; "inf" -> OverflowError
    GPU_DURATION = 30
    print(f"[hf_space] CONDDIFF_GPU_DURATION={os.environ.get('CONDDIFF_GPU_DURATION')!r} is not a "
          "number -- using 30")

_generate_live_cpu = cdiff._generate_live            # the original function, captured before rebinding


@spaces.GPU(duration=GPU_DURATION)
def _render_on_gpu(spec, rng, seed):
    """Run conddiff's live path inside a ZeroGPU worker. Same arguments, same return value.

    spec : dict from params_to_spec   rng : numpy Generator seeded with `seed`   seed : int
    returns (OrderedDict{"flair": png bytes[, "seg": png bytes]}, provenance dict)

    gpu_seconds is added to the provenance, so it lands in the database record with the image.
    It is a measurement, like every other provenance field, and it is the number to tune
    GPU_DURATION and CONDDIFF_STEPS from.

    WHY EVERY EXCEPTION IS RE-RAISED AS gr.Error: this body runs in the forked GPU worker, and
    `spaces` carries only gr.Error across the process boundary intact. Anything else reaches the
    main process as "ZeroGPU worker error: RuntimeError" -- the class name, with the message
    (e.g. "the conditioning bank contains no superior slice with a usable cerebellum lobe")
    left behind in the Space log (spaces/zero/wrappers.py, exception_result).
    """
    t0 = time.perf_counter()
    try:
        images, provenance = _generate_live_cpu(spec, rng, seed)
    except Exception as e:
        raise gr.Error(f"{type(e).__name__}: {e}") from e
    provenance["gpu_seconds"] = round(time.perf_counter() - t0, 3)
    return images, provenance


# image_generation.py catches EVERY inference exception, prints it, and returns an empty list
# (image_generation.py:118), so a refused GPU call -- most importantly "quota exceeded, try again
# in 3:41:09" -- would reach the user as a blank "backend returned no images". The message is
# therefore stashed here on its way past that except, and generate() re-raises it. threading.local
# because Gradio runs each request on a worker thread, and the whole chain generate ->
# generate_image -> cdiff.inference -> _render_live runs on that one thread.
_last_error = threading.local()


def _render_live(spec, rng, seed):
    try:
        return _render_on_gpu(spec, rng, seed)
    except Exception as e:
        # gr.Error keeps its text in .message; its str() is a quoted repr.
        _last_error.message = getattr(e, "message", None) or str(e)
        raise


# THE ONE REBINDING. cdiff.inference() looks _generate_live up in its module globals at call
# time, so rebinding the name routes the unmodified inference() -> readiness check, seed,
# honesty record, Supabase publish -- through the GPU without editing conddiff_inference.py.
# The GPU worker calls _generate_live_cpu (captured above), never the rebound name.
cdiff._generate_live = _render_live


# --- 5. THE TWO CALLS THE WEBSITE MAKES ----------------------------------------------------------
def generate(payload: dict) -> dict:
    """api.py's POST /generate, as a Gradio endpoint. Same request body, same response body.

    payload: {user_id, project_id, model_name, n_images=1, params={}, is_playground=False}
    returns: {"image_paths": [[id_or_url, ...], ...]}  one inner list per generated image set

    n_images is clamped to 1..5, the UI's own range. api.py never bounded it, but this endpoint is
    public and the queue runs one call at a time, so one n_images=10000 GAN call would block every
    user and fill the Supabase bucket.
    """
    _last_error.message = None
    try:
        paths = image_generation.generate_image(
            payload["user_id"],
            payload["project_id"],
            payload["model_name"],
            max(1, min(int(payload.get("n_images", 1)), 5)),
            payload.get("params") or {},
            bool(payload.get("is_playground", False)),
        )
    except Exception as e:
        # gr.Error, not a bare exception: Gradio sends a bare exception's message to the caller
        # only in debug mode, and the caller is the website's error toast.
        raise gr.Error(f"Error generating images: {e}")

    if not any(paths) and _last_error.message:        # the diffuser failed; say why
        raise gr.Error(_last_error.message)
    # Same normalisation as api.py:88-97: every entry becomes a list.
    return {"image_paths": [p if isinstance(p, list) else [p] for p in paths]}


def conddiff_cells() -> dict:
    """api.py's GET /conddiff-cells: the (lobe, level) pairs the shipped bank can generate."""
    return cdiff.supported_combinations()


def preview(tumour, lobe, slice_location, tumour_size, seed):
    """The Space page's own test form: one image through the SAME GPU path, published nowhere.

    Exists so the deployment can be checked, and gpu_seconds measured, from the Space page
    before the website is pointed at it.
    """
    if not cdiff.READY:
        raise gr.Error(cdiff.REASON)
    seed = secrets.randbelow(2 ** 31) if seed is None else int(seed)
    spec = cdiff.params_to_spec(tumour, lobe, slice_location, tumour_size)
    images, provenance = cdiff._generate_live(spec, np.random.default_rng(seed), seed)
    return ([Image.open(io.BytesIO(png)) for png in images.values()],
            cdiff._json_safe({"seed": seed, **provenance}))


with gr.Blocks(title="SOCR Brain Image Generator backend") as demo:
    gr.Markdown(
        "## SOCR Brain Image Generator: backend\n"
        "API backend for the SOCR Brain-Image-Generator website (5 GANs on CPU + the atlas-guided "
        "conditional diffusion model on ZeroGPU). The form below runs one diffusion sample "
        "without publishing it."
    )
    with gr.Row():
        tumour = gr.Dropdown(list(cdiff.UI_TUMOUR), value="With Tumor", label="Tumour")
        lobe = gr.Dropdown(list(cdiff.UI_LOBE), value="Frontal", label="Lobe")
        level = gr.Dropdown(list(cdiff.UI_LEVEL), value="Middle", label="Slice location")
        size = gr.Dropdown(list(cdiff.UI_SIZE), value="Moderate", label="Tumour size")
        seed = gr.Number(value=None, precision=0, label="Seed (blank = random)")
    run = gr.Button("Generate one FLAIR", variant="primary")
    out_images = gr.Gallery(label="FLAIR, tumour mask", columns=2)
    out_record = gr.JSON(label="What happened (the provenance the website records)")
    run.click(preview, [tumour, lobe, level, size, seed], [out_images, out_record])

    # The website's two calls: API-only endpoints, no UI components.
    gr.api(generate, api_name="generate")
    gr.api(conddiff_cells, api_name="conddiff_cells")

# Gradio's queue runs one call per endpoint at a time by default (default_concurrency_limit=1).
# That is deliberate here: supabase_storage.py writes a fixed temp_nifti.nii.gz path, and the
# GANs' torch CPU threads would fight each other. Requests wait in line instead of racing.
if __name__ == "__main__":
    demo.launch(show_error=True)
