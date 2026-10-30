"""wavedit_inference.py -- braingen_WaveDiT_v1 (3D): the WaveDiT 128^3 wavelet flow-matching model.

WHAT THIS MODEL IS. WaveDiT generates a whole 3D brain volume in ONE pass per ODE step, in the
3D Haar wavelet domain, conditioned only on patient AGE. It is a flow-matching model (not a GAN
and not a diffusion sampler): ``steps`` is the number of ODE integration steps, so 1 step is a
single network evaluation and 50 is the finest the UI offers.

WHY THERE IS NO ``torch``/``wavedit`` IMPORT AT MODULE SCOPE. ``image_generation.py`` imports this
file at module scope alongside the four GAN modules, and an ImportError raised here would happen
before FastAPI starts and would take down every other model with it. The vendored ``wavedit``
package (backend/wavedit/) pulls in torch, einops, PyYAML and tqdm, so it is imported lazily
inside :func:`get_wavedit_model`, exactly as conddiff_inference.py treats diffusers. A deployment
that forgot to ship the package fails only this model, at request time, with a readable error.

THE WEIGHTS. ``inference/model/WaveDiT-128.pth`` (~1.7 GB, fp32), delivered out of band like every
other .pth in this repo -- it is not in git. The checkpoint is SELF-CONTAINED: alongside
``model_state_dict`` it carries the resolved ``config`` and the dataset-derived condition metadata
(``condition_config``/``condition_ranges``/``categorical_maps``/``null_conditions``), so the
architecture is rebuilt from the file itself rather than from a constants table duplicated here.

DEVICE. ``WAVEDIT_DEVICE`` overrides (e.g. ``cpu``/``cuda:1``); otherwise CUDA when available.
On the Hugging Face ZeroGPU Space this runs inside the worker that hf_space/app.py attaches to the
model (see the note there); on the Lightsail CPU container it runs on the CPU.

ORIENTATION. The saved volume uses the identity affine, like the legacy 2D/diffusion paths. WaveDiT
is trained on the project's own preprocessed OpenBHB volumes, so it emits a volume in that
canonical layout already; orient3d.py's atlas fit is a fix for the 3D GANs, whose AP/SI direction
is random per sample, and applying it here would risk flipping a correctly oriented volume.
"""

import os
import time
import uuid

import numpy as np
import nibabel as nib

# The 128^3 checkpoint this module serves by default. The name is also the RESOLUTIONS key below,
# which is what actually selects it.
CKPT_FILENAME = "WaveDiT-128.pth"

# Output resolutions this deployment can serve, keyed by the string the UI sends. Each entry is
# (checkpoint, native volume size). Only the native 128^3 checkpoint ships today, so there is one
# entry -- but the resolution the UI sends IS honoured here, so adding a second checkpoint later is
# a one-line change here plus a dropdown item in ParameterControlPanel.tsx, not a new code path.
RESOLUTIONS = {
    "128": (CKPT_FILENAME, (128, 128, 128)),
}
DEFAULT_RESOLUTION = "128"

# The UI's sliders. MIN_STEPS=1 is a legitimate setting (a single Euler step), not a mistake.
MIN_STEPS, MAX_STEPS = 1, 50
DEFAULT_STEPS = 10

# One request generates ONE volume. Unlike the 2D GANs (a single forward pass, milliseconds) every
# sample here is a full ODE loop producing 128^3 voxels from a 1.7 GB model, and
# image_generation.py runs the n_images loop SEQUENTIALLY inside one synchronous request. The
# dispatcher clamps to this, and ParameterControlPanel caps its Number of Images input to match.
MAX_IMAGES = 1

# Fixed for this app. Euler is one network call per step, and cfg_scale == 1.0 lets the model skip
# the second, unconditional forward pass, so generation stays at one NFE per step. Neither is
# exposed in the UI: change them here if a future checkpoint needs otherwise.
FIXED_SAMPLER = "euler"
FIXED_CFG_SCALE = 1.0
CFG_RESCALE = 0.7

# Age conditioning range the model was trained on (the UI's input is clamped to it).
MIN_AGE, MAX_AGE = 6.0, 95.0

# TURN torch.compile OFF FOR THE VENDORED PACKAGE, BEFORE IT IS IMPORTED.
# wavedit/models/hdit/*.py decorates its hot functions with @flags.compile_wrap, which reads
# K_DIFFUSION_USE_COMPILE (default "1") when the decorator is first called. Two reasons that
# default is wrong for a request-serving process:
#   * the cost is paid by the FIRST request, not at boot: compile_wrap resolves lazily, so a 60 s
#     ZeroGPU window -- or Lightsail's 0.25 vCPU -- would be spent on compilation, and measured
#     here it turned a 31 s one-step sample into a 40 s one while emitting a torch._dynamo
#     traceback per function;
#   * with no C++ toolchain (this dev box, and any slim CPU image) every one of those attempts
#     FAILS and falls back to eager anyway, so the logs warn for nothing.
# setdefault, so an operator can still opt in with K_DIFFUSION_USE_COMPILE=1. Must run before
# wavedit is imported, which it does: that import is lazy, inside load_wavedit_model().
os.environ.setdefault("K_DIFFUSION_USE_COMPILE", "0")

# Loaded once per process. A checkpoint this size is not re-read per request.
# Keyed by RESOLUTIONS key ("128"), so adding a second checkpoint later caches independently
# instead of silently serving the first model loaded.
_cached = {}


def _checkpoint_path(resolution=DEFAULT_RESOLUTION) -> str:
    """Path to the checkpoint that serves ``resolution`` (falling back to the default)."""
    ckpt_filename = RESOLUTIONS.get(str(resolution), RESOLUTIONS[DEFAULT_RESOLUTION])[0]
    return os.path.join(os.getcwd(), "inference", "model", ckpt_filename)


def _resolve_device():
    """WAVEDIT_DEVICE override, else CUDA if available, else CPU."""
    import torch

    requested = os.getenv("WAVEDIT_DEVICE", "").strip().lower()
    if requested:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print(f"[WaveDiT] WAVEDIT_DEVICE={requested} requested but CUDA is unavailable; using CPU.")
            return torch.device("cpu")
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_wavedit_model(device=None, resolution=DEFAULT_RESOLUTION):
    """Build the WaveDiT model that serves ``resolution``, load its weights, move it to ``device``
    and cache it.

    Returns ``(model, cfg)``. ``device`` defaults to :func:`_resolve_device`.

    This is separate from :func:`get_wavedit_model` so a host that manages placement itself can
    build the model once and choose the device -- the Hugging Face Space does exactly that, since
    ZeroGPU requires the model to be constructed and moved in the MAIN process (see
    hf_space/app.py).

    Raises FileNotFoundError when the checkpoint is absent, so a deployment can tell "weights not
    uploaded" from "the package is missing" from any other failure.
    """
    # Lazy, for the reason in the module docstring: importing the package cannot be allowed to
    # break image_generation.py's import of every other model.
    from wavedit import Config
    from wavedit.models import build_model
    from wavedit.training.checkpoint import load_model_weights

    import torch

    key = str(resolution) if str(resolution) in RESOLUTIONS else DEFAULT_RESOLUTION
    ckpt_path = _checkpoint_path(key)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"WaveDiT checkpoint not found at: {ckpt_path}")

    device = torch.device(device) if device is not None else _resolve_device()
    if device.type == "cuda":
        label = f"{device} ({torch.cuda.get_device_name(device)})"
    else:
        label = str(device)
    print(f"[WaveDiT] Loading {ckpt_path} on {label}...")

    # weights_only=False is required: the checkpoint is a dict of plain Python metadata plus a
    # state_dict, not a bare tensor collection.
    #
    # map_location is "cpu" even when the target is a GPU, matching conddiff_inference._get_model
    # and for a reason that matters on the HF Space: its MAIN process has no GPU (ZeroGPU only
    # attaches one inside @spaces.GPU), so a direct map_location="cuda" there would fail at boot.
    # Loading to CPU and then calling .to(device) is the pattern ZeroGPU's .to("cuda") patch is
    # built to intercept, and it costs nothing measurable on a real GPU box.
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = Config.from_dict(ck["config"])
    model = build_model(
        cfg,
        ck["condition_config"],
        ck["condition_ranges"],
        ck["categorical_maps"],
        ck["null_conditions"],
    )
    load_model_weights(model, ck)
    model.to(device)
    model.eval()

    _cached[key] = {"model": model, "config": cfg, "device": device}
    print(f"[WaveDiT] Model loaded and cached ({os.path.basename(ckpt_path)}, key='{key}').")
    return model, cfg


def get_wavedit_model(resolution=DEFAULT_RESOLUTION):
    """Return the cached model for ``resolution``, loading it on first use."""
    key = str(resolution) if str(resolution) in RESOLUTIONS else DEFAULT_RESOLUTION
    if key in _cached:
        return _cached[key]["model"], _cached[key]["config"]
    return load_wavedit_model(resolution=key)


def save_wavedit_volume(
    generated_volume,
    save_path,
    user_id=None,
    project_id=None,
    model_name=None,
    params=None,
    is_playground=False,
):
    """Publish the generated volume: Supabase when we have a user/project, else local disk.

    Mirrors gan_3d_inference_v1.save_images so the site receives the same thing it already knows
    how to display: a database row id for signed-in users, a public URL in playground mode, and a
    local folder path when there is no Supabase at all.
    """
    folder_name = f"wavedit_{uuid.uuid4().hex}"

    if user_id and project_id:
        from supabase_storage import upload_nifti_to_supabase

        return upload_nifti_to_supabase(
            nifti_data=generated_volume,
            filename="mri3D.nii.gz",
            affine=np.eye(4),
            user_id=user_id,
            project_id=project_id,
            folder_name=folder_name,
            model_name=model_name,
            parameters=params,
            is_playground=is_playground,
        )

    folder_path = os.path.join(save_path, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    file_path = os.path.join(folder_path, "mri3D.nii.gz")
    nib.save(nib.Nifti1Image(generated_volume, affine=np.eye(4)), file_path)
    return folder_path


def generate_volume(age=50.0, steps=DEFAULT_STEPS, seed=42, resolution=DEFAULT_RESOLUTION):
    """Sample one volume. Returns ``(volume float32, params dict)`` in [0, 1].

    Split out from :func:`inference` so a host can run ONLY this part on a GPU and do the upload
    elsewhere: the volume and the params are plain picklable Python objects, which is exactly what
    the HF Space needs to cross its ZeroGPU process boundary (see hf_space/app.py).

    ``resolution`` selects the checkpoint AND the output size (see RESOLUTIONS). An unrecognised
    value falls back to the default rather than raising, because it arrives from the UI.
    """
    import torch

    key = str(resolution) if str(resolution) in RESOLUTIONS else DEFAULT_RESOLUTION
    out_shape = RESOLUTIONS[key][1]

    try:
        steps = int(steps)
    except (TypeError, ValueError):
        steps = DEFAULT_STEPS
    steps = max(MIN_STEPS, min(MAX_STEPS, steps))

    try:
        age = float(age)
    except (TypeError, ValueError):
        age = 50.0
    age = max(MIN_AGE, min(MAX_AGE, age))

    seed = _coerce_seed(seed)
    if seed is not None:
        torch.manual_seed(seed)

    model, _cfg = get_wavedit_model(key)
    model.num_flow_steps_sampling = steps

    print(f"[WaveDiT] age={age} steps={steps} seed={seed} resolution={key} out={out_shape} "
          f"sampler={FIXED_SAMPLER} cfg_scale={FIXED_CFG_SCALE}")
    t0 = time.time()
    with torch.no_grad():
        vol = model.sample(
            num_samples=1,
            raw_conditions={"age": age},
            cfg_scale=FIXED_CFG_SCALE,
            sampler=FIXED_SAMPLER,
            morpheus_scale=None,
            cfg_rescale=CFG_RESCALE,
            autocast_dtype=None,
        )

    # The model's range is [-1, 1]; every viewer in this app expects [0, 1].
    volume = torch.clamp((vol.float() + 1.0) / 2.0, 0.0, 1.0)[0, 0].cpu().numpy().astype(np.float32)
    print(f"[WaveDiT] Generated {volume.shape} in {time.time() - t0:.2f}s")

    params = {
        "age": age,
        "steps": steps,
        "seed": seed,
        "sampler": FIXED_SAMPLER,
        "cfg_scale": FIXED_CFG_SCALE,
        "resolution": key,
    }
    return volume, params


def _coerce_seed(seed):
    """A seed the UI sends as a number; None (or junk) means "no seeding"."""
    if seed is None:
        return None
    try:
        return int(seed)
    except (TypeError, ValueError):
        return 42


def inference(
    model_name,
    age=50.0,
    steps=DEFAULT_STEPS,
    seed=42,
    resolution=DEFAULT_RESOLUTION,
    save_path=None,
    user_id=None,
    project_id=None,
    is_playground=False,
    volume=None,
    params=None,
    **kwargs,
):
    """Generate one volume conditioned on ``age`` and publish it.

    ``resolution`` picks the checkpoint and output size (see RESOLUTIONS); an unrecognised value
    falls back to the default. ``kwargs`` swallows anything else the API may forward (e.g.
    ``cfg_scale``/``sampler``) rather than raising: those axes are pinned in this module, and a
    stray key must not turn a generation into a KeyError caught by image_generation.py's except.

    ``volume``/``params`` are for a caller that already ran :func:`generate_volume` on another
    device or process (the HF Space, which samples on ZeroGPU). Passing them skips sampling and
    only publishes.
    """
    if volume is None:
        volume, generated_params = generate_volume(
            age=age, steps=steps, seed=seed, resolution=resolution
        )
        if params is None:
            params = generated_params

    return save_wavedit_volume(
        generated_volume=volume,
        save_path=save_path,
        user_id=user_id,
        project_id=project_id,
        model_name=model_name,
        params=params,
        is_playground=is_playground,
    )
