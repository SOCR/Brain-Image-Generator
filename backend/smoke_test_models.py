#!/usr/bin/env python3
"""smoke_test_models.py -- the pre-deploy gate for a torch / diffusers version change.

WHY THIS FILE EXISTS
====================
We are moving this backend from a CPU-only Lightsail container (torch==2.0.1+cpu) onto an
EC2 GPU host so that braingen_CondDiffuser_BraTS_v1 can run LIVE -- a real DDIM reverse
diffusion per request rather than a lookup into a folder of pre-rendered PNGs. That move
means changing the torch wheel, and possibly the torch VERSION.

Changing torch is dangerous here for one specific, non-obvious reason:

    ALL FIVE GAN MODELS LOAD THEIR WEIGHTS WITH strict=False.

        cgan_inference.py:174     generator.load_state_dict(torch.load(...), strict=False)
        cgan_inference_v2.py:134  generator.load_state_dict(torch.load(...), strict=False)
        wgan_inference.py:168-169 (twice, coarse + fine)
        gan_3d_inference_v1.py:88

    torch's `strict=False` means "silently ignore every key in the checkpoint that the model
    does not have, and every key the model has that the checkpoint does not supply". If a
    torch upgrade changed a key name, changed a pickle format, or otherwise made the
    checkpoint unreadable-as-expected, load_state_dict would NOT raise. It would return
    quietly, leave the generator at its RANDOM Kaiming/normal initialisation, and the model
    would go on producing 128x128 tensors of smooth grey blobs. Those blobs look enough like
    a low-quality brain MRI that eyeballing the website would not catch it. The site would be
    serving noise under the name of a trained model, indefinitely.

    This script exists to catch exactly that. It replicates each module's load EXACTLY and
    then inspects what strict=False threw away: the `_IncompatibleKeys(missing_keys,
    unexpected_keys)` object that the production code discards.

WHAT "FAIL" MEANS HERE
======================
  * checkpoint present but ZERO state_dict keys matched -> FAIL. This is the silent failure
    above, caught red-handed.
  * generated output is CONSTANT (max == min) or contains NaN -> FAIL. A constant image is
    the fingerprint of a generator that never learned anything, and it also divides by zero
    in every downstream consumer (each save_images() in this repo min-max normalises).
  * checkpoint absent -> SKIP, never a crash. The five .pth files and the diffusion
    checkpoint are delivered out of band and are not in git, so on a fresh clone NOTHING is
    present. A gate that crashes when the weights are missing is a gate nobody runs.

HOW TO USE IT AS A GATE
=======================
    # on the CURRENT production image, before touching requirements.txt
    python smoke_test_models.py --json > before.json

    # after changing the torch pin and rebuilding
    python smoke_test_models.py --json > after.json

    # then diff them. "keys_matched" must be IDENTICAL. The output min/max/mean should be
    # identical too: every generation below is run from a FIXED seed, so on the same device
    # the same weights must produce the same numbers. A changed mean is a changed model.
    diff before.json after.json

Exit code is 0 when nothing FAILed and non-zero otherwise, so it can sit in front of a
`docker push` in deploy.sh.

USAGE
    python smoke_test_models.py                 # every model
    python smoke_test_models.py conddiff        # just the conditional diffuser
    python smoke_test_models.py gans            # the five GAN entries only
    python smoke_test_models.py --list          # show the subset keys and exit
    python smoke_test_models.py --strict-keys   # promote partial key matches to FAIL
    python smoke_test_models.py --json          # append a machine-diffable JSON blob

ENVIRONMENT
    SMOKE_EXPECT_CUDA=1   turn "torch.cuda.is_available() is False" into a hard FAIL. Set
                          this in the EC2 GPU deploy script: a GPU box that quietly fell back
                          to CPU is the single most likely silent failure of this migration,
                          and at ~2 minutes per CPU DDIM sample it would take the site down
                          by timeout rather than by error.
    CONDDIFF_MODE         read by conddiff_inference itself ("live" default / "gallery").
    CONDDIFF_STEPS        DDIM step count, default 50.

This script never writes to Supabase and never writes an image to disk. It deliberately does
NOT call each module's public inference() function, because those functions END in
save_images(), which either uploads to the user's real library or writes PNGs into
backend/data/. It instead replicates the load-and-forward part of each inference() -- the
part a torch change can break -- and stops before publication. Every replication below is
annotated with the file and line it mirrors so the two can be kept in step.
"""

# =================================================================================================
# SECTION 0 -- BOOTSTRAP
#
# This has to happen before ANY project import, because of two pieces of global state that the
# inference modules capture at import time and never re-read.
# =================================================================================================

import os
import sys

# (a) matplotlib backend. cgan_inference.py:9, cgan_inference_v2.py:9 and wgan_inference.py:12
#     all do `import matplotlib.pyplot as plt` AT MODULE SCOPE. On a fresh headless EC2 box with
#     no DISPLAY, pyplot's default interactive backend selection can raise, which would make the
#     import of image_generation.py fail and take down all six models -- an error that has
#     nothing to do with torch but would look exactly like a torch problem in the deploy log.
#     "Agg" is the non-interactive raster backend; it needs no display. Set as an env var rather
#     than via matplotlib.use() so it is in effect no matter who imports pyplot first.
os.environ.setdefault("MPLBACKEND", "Agg")

# (b) the working directory. EVERY weight path in this repo is built as
#         os.path.join(os.getcwd(), "inference/model/....pth")
#     (cgan_inference.py:166, cgan_inference_v2.py:122, wgan_inference.py:159,
#      gan_3d_inference_v1.py:77, conddiff_inference.py:325). In production the Dockerfile sets
#     WORKDIR /app and copies the CONTENTS of backend/ there, so os.getcwd() == "/app" and the
#     paths resolve. Run from anywhere else and every path in every module is wrong.
#     conddiff_inference.py additionally FREEZES the value into a module constant (`_CWD`) at
#     import time, so chdir'ing later would not help -- it must happen here, first.
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../Brain-Image-Generator/backend
os.chdir(_HERE)

# (c) import path. The modules import each other as `inference.Generators`, i.e. `inference` must
#     be an importable package directory on sys.path. Running this file directly already puts its
#     own directory at sys.path[0], but say it explicitly so `python -m`, a debugger, or a
#     pytest collection all behave the same.
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import argparse
import io
import json
import platform
import time
import traceback

import numpy as np


# =================================================================================================
# SECTION 1 -- CONSTANTS
# =================================================================================================

# ONE FIXED SEED FOR EVERY GENERATION IN THIS SCRIPT. This is the point of the whole design.
# With a fixed seed and unchanged weights, `min`, `max` and `mean` of every model's output are
# reproducible numbers. Run the script before and after a torch change and compare: identical
# numbers prove the weights loaded identically; a completely different mean proves they did not.
# (Tiny last-digit differences are expected across kernel implementations -- especially CPU vs
# GPU, where reduction order differs. A different ORDER OF MAGNITUDE is not.)
SEED = 20260909

# Lightsail's container healthCheck, from lightsail-config.json / the deploy scripts. These three
# numbers decide whether SYNCHRONOUS serving of a multi-second model is viable on that platform,
# and the conddiff verdict below is computed from them rather than from a remembered rule of
# thumb. api.py:81 declares `async def generate_images(...)` but the work inside it is fully
# synchronous, so a long request occupies uvicorn's single event loop thread and /health
# (api.py:71) cannot answer while it runs.
HEALTH_TIMEOUT_S = 5      # a check that does not answer within this is a failed check
HEALTH_INTERVAL_S = 30    # gap between checks
HEALTH_THRESHOLD = 2      # this many consecutive failures replaces the container
HEALTH_REPLACE_S = HEALTH_TIMEOUT_S + HEALTH_INTERVAL_S * HEALTH_THRESHOLD   # ~65 s

# Status strings. Ordered by severity for the summary line.
PASS, FAIL, SKIP, NOTE = "PASS", "FAIL", "SKIP", "NOTE"

RULE = "=" * 92
THIN = "-" * 92


# =================================================================================================
# SECTION 2 -- REPORTING SCAFFOLDING
# =================================================================================================

class Result:
    """One row of the final table, plus everything printed in that model's detail block.

    Kept as a plain object rather than a dict so the field names are checked by the reader
    rather than by nobody. `to_json()` at the bottom is what --json emits, and it is the thing
    you actually diff across a torch upgrade.
    """

    def __init__(self, key, label):
        self.key = key                 # the argv subset key, e.g. "wavelet"
        self.label = label             # the model_name string the frontend POSTs
        self.status = SKIP             # PASS | FAIL | SKIP | NOTE
        self.verdict = "not run"       # short phrase for the table's last column
        self.loaded = None             # True / False / None (never attempted)
        self.checkpoints = []          # [{"path":..., "exists":..., "bytes":...}]
        self.n_params = None           # parameter count of the CONSTRUCTED module
        self.keys_total = None         # keys the module's state_dict expects
        self.keys_matched = None       # ...of which this many were supplied by the checkpoint
        self.missing = []              # model wants it, checkpoint lacks it  -> random init
        self.unexpected = []           # checkpoint has it, model does not     -> ignored
        self.gen_seconds = None        # wall clock of ONE generation
        self.load_seconds = None       # wall clock of the one-off weight load
        self.stats = None              # shape/dtype/min/max/mean of the output
        self.extra = {}                # per-model facts (mode, device, ddim steps, ...)
        self.lines = []                # free-text detail printed under the heading

    def say(self, line=""):
        """Record and immediately print one line of this model's detail block."""
        self.lines.append(line)
        print(f"  {line}" if line else "")

    def fail(self, verdict, detail=None):
        self.status = FAIL
        self.verdict = verdict
        if detail:
            self.say("")
            for ln in str(detail).splitlines():
                self.say(f"FAIL: {ln}")

    def skip(self, verdict, detail=None):
        self.status = SKIP
        self.verdict = verdict
        if detail:
            self.say(f"SKIP: {detail}")

    def to_json(self):
        return {
            "key": self.key,
            "model": self.label,
            "status": self.status,
            "verdict": self.verdict,
            "loaded": self.loaded,
            "checkpoints": self.checkpoints,
            "n_params": self.n_params,
            "keys_total": self.keys_total,
            "keys_matched": self.keys_matched,
            # Truncated: a fully-mismatched checkpoint would otherwise dump 400 key names into
            # the diff and drown the numbers that matter. The counts above are the signal.
            "missing_sample": self.missing[:8],
            "n_missing": len(self.missing),
            "unexpected_sample": self.unexpected[:8],
            "n_unexpected": len(self.unexpected),
            "load_seconds": self.load_seconds,
            "gen_seconds": self.gen_seconds,
            "stats": self.stats,
            "extra": self.extra,
        }


def heading(text):
    print()
    print(RULE)
    print(text)
    print(RULE)


def fmt_bytes(n):
    """Human-size a file, so '325 MB' in the log matches what the operator sftp'd."""
    if n is None:
        return "-"
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0


def array_stats(arr):
    """shape / dtype / min / max / mean of a generated tensor, as plain Python types.

    Returns a dict, and NEVER raises: it is called on model output, which is exactly the thing
    that might be all-NaN. np.nanmin on an all-NaN array warns and returns nan; np.min on an
    empty array raises. Both are guarded, because a crash here would lose the very diagnosis we
    came for.
    """
    a = np.asarray(arr)
    out = {"shape": tuple(int(d) for d in a.shape), "dtype": str(a.dtype), "size": int(a.size)}
    if a.size == 0:
        out.update({"min": None, "max": None, "mean": None, "n_nan": 0, "constant": True})
        return out
    is_float = a.dtype.kind == "f"
    n_nan = int(np.isnan(a).sum()) if is_float else 0
    finite = a[np.isfinite(a)] if is_float else a
    if finite.size == 0:
        out.update({"min": None, "max": None, "mean": None, "n_nan": n_nan, "constant": True})
        return out
    mn, mx = float(finite.min()), float(finite.max())
    out.update({"min": mn, "max": mx, "mean": float(finite.mean()),
                "n_nan": n_nan,
                # THE CONSTANT-OUTPUT TEST. Exact equality is the right comparison, not a
                # tolerance: a real generator's output spans a wide range, and a network whose
                # weights failed to load produces either literal zeros or one repeated value.
                "constant": mn == mx})
    return out


def report_stats(res, arr, seconds, what="output"):
    """Print the generation stats and apply the two hard output rules (constant, NaN)."""
    st = array_stats(arr)
    res.stats = st
    res.gen_seconds = seconds
    res.say(f"{what:<12}: shape={st['shape']} dtype={st['dtype']}")
    if st["min"] is None:
        res.say(f"{'values':<12}: ALL non-finite ({st['n_nan']} NaN of {st['size']})")
    else:
        res.say(f"{'values':<12}: min={st['min']:+.6f}  max={st['max']:+.6f}  "
                f"mean={st['mean']:+.6f}" + (f"  NaN={st['n_nan']}" if st["n_nan"] else ""))
    res.say(f"{'wall clock':<12}: {seconds:.3f} s for one generation")

    if st["min"] is None or st["constant"]:
        res.fail(
            "CONSTANT output",
            "the generated tensor has max == min (or is entirely non-finite).\n"
            "A trained generator never does this. This is the exact fingerprint of a network\n"
            "whose checkpoint did not load: strict=False left it at its random initialisation,\n"
            "or the forward pass collapsed. Check the 'keys matched' line above -- if it says\n"
            "0, the checkpoint and the module have no key names in common.\n"
            "It is also a downstream crash: every save_images() in this repo computes\n"
            "(image - min) / (max - min), which divides by zero here.")
        return False
    if st["n_nan"]:
        res.fail("NaN in output",
                 f"{st['n_nan']} of {st['size']} values are NaN. Numerically broken weights, or a\n"
                 "kernel that behaves differently on this torch build. Do not deploy.")
        return False
    return True


# =================================================================================================
# SECTION 3 -- THE ENVIRONMENT REPORT
#
# Printed FIRST, before a single weight is touched, because on this migration the environment
# IS the risk. Everything below is a fact about the box, not about our code.
# =================================================================================================

def report_environment():
    """Print the interpreter / torch / CUDA / library facts. Returns (env_dict, hard_failures).

    hard_failures is a list of strings; a non-empty list makes the whole run exit non-zero even
    if every model passes, because a model that passes on the wrong device has not been tested
    on the device it will serve from.
    """
    heading("ENVIRONMENT  (read this block before anything else)")
    env, hard = {}, []

    env["python"] = sys.version.split()[0]
    env["platform"] = platform.platform()
    env["machine"] = platform.machine()
    env["cwd"] = os.getcwd()
    env["cpu_count"] = os.cpu_count()
    print(f"  python        : {env['python']}  ({env['machine']})")
    print(f"  platform      : {env['platform']}")
    print(f"  cwd           : {env['cwd']}")
    print(f"    ^ every weight path is os.path.join(os.getcwd(), 'inference/...'), so this MUST")
    print(f"      be the backend/ directory (it is /app inside the container).")
    print(f"  cpu count     : {env['cpu_count']}")
    print(f"  numpy         : {np.__version__}")
    env["numpy"] = np.__version__

    # ---- torch -----------------------------------------------------------------------------
    # torch.__version__ carries the BUILD SUFFIX, and that suffix is the whole question of this
    # migration: "2.0.1+cpu" is the CPU-only wheel currently pinned in requirements.txt and it
    # physically cannot see a GPU no matter what hardware it is on; "2.0.1+cu118" is the same
    # source at the same version compiled against CUDA 11.8, which is the preferred change
    # because the GAN numerics stay identical while the diffuser gains a device.
    try:
        import torch
    except Exception as e:
        env["torch"] = None
        print()
        print("  !!! torch COULD NOT BE IMPORTED:", e)
        print("      Nothing below can run. Every model on this site is a torch model.")
        hard.append(f"torch import failed: {e}")
        return env, hard

    env["torch"] = torch.__version__
    env["torch_cuda_build"] = torch.version.cuda            # None on a +cpu wheel
    env["cuda_available"] = bool(torch.cuda.is_available())
    print()
    print(f"  torch         : {torch.__version__}")
    print(f"    build cuda  : {torch.version.cuda or 'NONE  (this is a CPU-only wheel)'}")
    try:
        env["cudnn"] = torch.backends.cudnn.version()
    except Exception:
        env["cudnn"] = None
    print(f"    cudnn       : {env['cudnn'] or '-'}")
    print(f"  cuda avail.   : {env['cuda_available']}")

    if env["cuda_available"]:
        try:
            n = torch.cuda.device_count()
            env["cuda_devices"] = []
            for i in range(n):
                props = torch.cuda.get_device_properties(i)
                dev = {"index": i, "name": props.name,
                       "capability": f"{props.major}.{props.minor}",
                       "total_mem_gb": round(props.total_memory / 1024 ** 3, 2)}
                env["cuda_devices"].append(dev)
                print(f"    device {i}    : {dev['name']}  "
                      f"sm_{props.major}{props.minor}  {dev['total_mem_gb']} GB")
        except Exception as e:                              # never let diagnostics crash the gate
            print(f"    device probe failed: {e}")
    else:
        # THE LOUD BLOCK. This is requirement #1: a GPU deploy where cuda.is_available() is
        # False must be impossible to miss. It is the most likely silent failure of this
        # migration because NOTHING errors -- the site keeps working, the GAN models are
        # unaffected (see below), and only the diffuser quietly takes ~2 minutes instead of
        # ~1 second, which surfaces as a gateway timeout rather than as a stack trace.
        print()
        print("  " + "*" * 88)
        print("  ***  torch.cuda.is_available() is FALSE -- THIS PROCESS WILL RUN ON THE CPU  ***")
        print("  " + "*" * 88)
        if torch.version.cuda is None:
            print("  *  Cause: this is a CPU-ONLY WHEEL (no '+cuXXX' build). requirements.txt still")
            print("  *  pins `torch==2.0.1+cpu` behind `--extra-index-url .../whl/cpu`. On the EC2")
            print("  *  GPU host that pin must become the cu118 build of the SAME version:")
            print("  *      --extra-index-url https://download.pytorch.org/whl/cu118")
            print("  *      torch==2.0.1+cu118")
            print("  *  Same source, same numerics for the five GAN models, different backend.")
        else:
            print("  *  Cause: this IS a CUDA build, but no usable device was found. Check, in order:")
            print("  *    - the instance actually has a GPU        (nvidia-smi)")
            print("  *    - the host NVIDIA driver is installed and new enough for this CUDA")
            print("  *    - the container was started with GPU access")
            print("  *      (docker run --gpus all ...  /  nvidia-container-toolkit installed)")
            print("  *    - CUDA_VISIBLE_DEVICES is not set to '' or '-1'")
        print("  *")
        print("  *  CONSEQUENCE, stated precisely:")
        print("  *    The five GAN models DO NOT CARE. Every one of them hardcodes")
        print("  *    `device = torch.device('cpu')` on the line immediately after the")
        print("  *    `torch.cuda.is_available()` line, so the cuda line is dead code")
        print("  *    (cgan_inference.py:150-151, cgan_inference_v2.py:108-109,")
        print("  *     wgan_inference.py:151-152, gan_3d_inference_v1.py:61-62), and every")
        print("  *    torch.load passes map_location='cpu'. They are unaffected by the device.")
        print("  *    braingen_CondDiffuser_BraTS_v1 IS THE ONE THAT CARES. Its DDIM loop is")
        print("  *    ~950 GFLOP per step for 50-200 steps: ~1 s/sample on a GPU, ~2 min on a")
        print("  *    full multi-core laptop CPU. Serving that synchronously is not viable.")
        print("  " + "*" * 88)
        if os.environ.get("SMOKE_EXPECT_CUDA", "").strip().lower() in ("1", "true", "yes"):
            hard.append("SMOKE_EXPECT_CUDA is set but torch.cuda.is_available() is False")
            print("  ***  SMOKE_EXPECT_CUDA is set -> this is a HARD FAILURE of the gate.  ***")

    # A CUDA-capable build on a box with a GPU still means nothing until a tensor actually
    # lands there, so prove it with the cheapest possible allocation.
    if env["cuda_available"]:
        try:
            t0 = time.perf_counter()
            _probe = torch.zeros(8, device="cuda")
            _probe = (_probe + 1).sum().item()
            env["cuda_probe_seconds"] = round(time.perf_counter() - t0, 3)
            print(f"    probe       : allocated + summed a cuda tensor in "
                  f"{env['cuda_probe_seconds']:.3f} s (includes context creation)")
        except Exception as e:
            print(f"    probe       : FAILED to allocate on cuda: {e}")
            hard.append(f"cuda reported available but allocation failed: {e}")

    # ---- the rest of the stack ---------------------------------------------------------------
    # diffusers is the OTHER open question. It is needed only by the conditional diffuser's live
    # path. diffusers declares torch only under an optional extra, so pip will cheerfully install
    # diffusers 0.38.0 alongside torch 2.0.1 and any incompatibility appears ONLY as an
    # ImportError at first use. That is precisely what this line tests. Do not assert an answer
    # from the pin file; read it from the interpreter.
    print()
    for mod, attr in (("diffusers", "__version__"), ("safetensors", "__version__"),
                      ("PIL", "__version__"), ("pandas", "__version__"),
                      ("pywt", "__version__"), ("nibabel", "__version__")):
        try:
            m = __import__(mod)
            v = getattr(m, attr, "?")
            env[mod] = v
            print(f"  {mod:<13}: {v}")
        except Exception as e:
            env[mod] = None
            print(f"  {mod:<13}: NOT IMPORTABLE -- {type(e).__name__}: {e}")
            if mod == "diffusers":
                print("      ^ THE OPEN QUESTION OF THIS MIGRATION, ANSWERED. diffusers is what")
                print("        the conditional diffuser's live path needs, and pip cannot")
                print("        predict this failure (torch is an optional extra of diffusers).")
                print("        If the error above mentions a torch symbol, the torch pin is too")
                print("        old for diffusers==0.38.0 and must be raised -- after which every")
                print("        GAN below must be re-run and its 'keys matched' compared.")

    print()
    print("  Reminder: gallery mode imports NEITHER torch NOR diffusers. Only CONDDIFF_MODE=live")
    print("  reaches them, which is why the Lightsail deploy scripts force CONDDIFF_MODE=gallery.")
    print(f"  CONDDIFF_MODE = {os.environ.get('CONDDIFF_MODE') or '(unset -> live)'}   "
          f"CONDDIFF_STEPS = {os.environ.get('CONDDIFF_STEPS') or '(unset -> 50)'}")
    return env, hard


# =================================================================================================
# SECTION 4 -- CHECKPOINT AND KEY-DIFF HELPERS
#
# The heart of the script. Everything here is about making visible what strict=False hides.
# =================================================================================================

def note_checkpoints(res, paths):
    """Record + print each checkpoint path, whether it exists, and its size.

    Returns True only if EVERY path exists. A missing file is a SKIP, never a crash: the
    weights are delivered out of band (they are not in git), so a fresh clone legitimately has
    none of them, and a gate that explodes on a fresh clone is a gate nobody runs.
    """
    all_there = True
    for p in paths:
        exists = os.path.isfile(p)
        size = os.path.getsize(p) if exists else None
        res.checkpoints.append({"path": p, "exists": exists, "bytes": size})
        res.say(f"{'checkpoint':<12}: {p}")
        res.say(f"{'':<12}  exists={exists}" + (f"  size={fmt_bytes(size)}" if exists else ""))
        all_there = all_there and exists
    return all_there


def torch_load_checkpoint(path):
    """Read a checkpoint EXACTLY the way the production modules read it.

    Deliberately mirrors, character for character, what the GAN modules do:

        torch.load(model_path, map_location=torch.device('cpu'))

    with NO `weights_only=` argument. That omission is load-bearing for this test. torch 2.6
    flipped the default of `weights_only` from False to True, which makes torch.load refuse to
    unpickle anything that is not a plain tensor container and raise
    `_pickle.UnpicklingError: Weights only load failed`. If a torch upgrade crosses that
    boundary and one of these five .pth files contains anything unusual, PRODUCTION would raise
    here -- so the smoke test must raise here too. Passing weights_only=False to "be safe" would
    hide the very regression we are gating on.

    Returns the loaded object (normally a plain {name: tensor} dict).
    """
    import torch
    return torch.load(path, map_location=torch.device("cpu"))


def unwrap_state_dict(obj):
    """Return a {name: tensor} mapping, unwrapping the usual training-checkpoint containers.

    A bare state_dict is what all six checkpoints in this repo are, but a checkpoint saved as
    {"epoch":..., "state_dict":...} fed straight to load_state_dict produces a wall of
    'unexpected key' noise instead of an explanation, so name the case if we meet it.
    """
    import torch
    if isinstance(obj, dict) and obj and all(isinstance(v, torch.Tensor) for v in obj.values()):
        return obj, None
    if isinstance(obj, dict):
        for k in ("state_dict", "model", "ema", "weights", "generator"):
            if k in obj and isinstance(obj[k], dict):
                return obj[k], k
    return obj, None


def load_and_diff(res, module, ckpt_path, strict=False):
    """THE CORE CHECK. Load `ckpt_path` into `module` and report what actually matched.

    Two independent measurements, on purpose:

      1. A MANUAL DIFF, computed BEFORE the load, of
             set(module.state_dict().keys())     -- what the architecture expects
         against
             set(checkpoint.keys())              -- what the file supplies
         plus a shape comparison over the intersection. This is the ground truth and it does
         not depend on torch's return value at all.

      2. THE RETURNED `_IncompatibleKeys(missing_keys, unexpected_keys)` namedtuple from
         torch's own load_state_dict. This is the object production code throws away by
         writing `generator.load_state_dict(...)` and ignoring the result. Printing it is
         the entire reason this function exists.

    A subtlety worth knowing: `strict=False` does NOT forgive a SHAPE mismatch. If a key is
    present in both but the tensors are different shapes, load_state_dict raises RuntimeError
    regardless of strict. So we check shapes ourselves first and report the mismatch as a
    diagnosis rather than as an uncaught traceback.

    Returns True if the load happened at all (even partially), False if it raised.
    """
    model_sd = module.state_dict()
    n_params = sum(p.numel() for p in module.parameters())
    res.n_params = int(n_params)
    res.say(f"{'parameters':<12}: {n_params:,} ({n_params / 1e6:.2f} M) in "
            f"{len(model_sd)} state_dict entries")

    # ---- read the file -----------------------------------------------------------------------
    try:
        raw = torch_load_checkpoint(ckpt_path)
    except Exception as e:
        res.fail("torch.load raised",
                 f"torch.load('{os.path.basename(ckpt_path)}', map_location='cpu') raised\n"
                 f"  {type(e).__name__}: {e}\n"
                 "If this mentions 'Weights only load failed' / 'WeightsUnpickler', the torch\n"
                 "upgrade crossed the 2.6 boundary where weights_only defaults to True. That is\n"
                 "a REAL production break: the four GAN modules call torch.load with no\n"
                 "weights_only argument, so they would raise here too.")
        res.loaded = False
        return False

    sd, wrapper = unwrap_state_dict(raw)
    if wrapper:
        res.say(f"{'note':<12}: checkpoint was wrapped in a '{wrapper}' key -- unwrapped it")
    if not isinstance(sd, dict) or not sd:
        res.fail("not a state_dict",
                 f"the checkpoint deserialised to {type(raw).__name__}, not a dict of tensors.")
        res.loaded = False
        return False

    # ---- measurement 1: the manual diff -------------------------------------------------------
    ck_keys, md_keys = set(sd.keys()), set(model_sd.keys())
    shared = ck_keys & md_keys
    shape_mismatch = []
    for k in sorted(shared):
        a, b = sd[k], model_sd[k]
        if tuple(getattr(a, "shape", ())) != tuple(getattr(b, "shape", ())):
            shape_mismatch.append((k, tuple(a.shape), tuple(b.shape)))

    res.keys_total = len(md_keys)
    res.keys_matched = len(shared) - len(shape_mismatch)
    res.missing = sorted(md_keys - ck_keys)
    res.unexpected = sorted(ck_keys - md_keys)

    pct = (100.0 * res.keys_matched / res.keys_total) if res.keys_total else 0.0
    res.say(f"{'keys':<12}: matched {res.keys_matched}/{res.keys_total} ({pct:.1f}%)   "
            f"missing {len(res.missing)}   unexpected {len(res.unexpected)}   "
            f"shape-mismatch {len(shape_mismatch)}")
    if res.missing:
        res.say(f"{'':<12}  missing (model wants, file lacks -> STAYS RANDOM): "
                f"{res.missing[:5]}{' ...' if len(res.missing) > 5 else ''}")
    if res.unexpected:
        res.say(f"{'':<12}  unexpected (file has, model lacks -> IGNORED): "
                f"{res.unexpected[:5]}{' ...' if len(res.unexpected) > 5 else ''}")
    for k, a, b in shape_mismatch[:5]:
        res.say(f"{'':<12}  SHAPE {k}: file{a} vs model{b}")

    if shape_mismatch:
        # Report before attempting, because the load below will raise on this even with
        # strict=False and the traceback is less informative than these three lines.
        res.fail("shape mismatch",
                 f"{len(shape_mismatch)} tensor(s) exist in both but with different shapes. The\n"
                 "architecture in inference/Generators.py has drifted from the one that trained\n"
                 "this checkpoint. This is NOT a torch problem.")
        res.loaded = False
        return False

    # ---- measurement 2: torch's own verdict, the one production discards ----------------------
    try:
        incompatible = module.load_state_dict(sd, strict=strict)
    except Exception as e:
        res.fail("load_state_dict raised", f"{type(e).__name__}: {e}")
        res.loaded = False
        return False

    if incompatible is not None:
        # _IncompatibleKeys is a namedtuple; getattr keeps this working if a future torch
        # returns something else, since this is diagnostics and must not itself break the run.
        mk = list(getattr(incompatible, "missing_keys", []) or [])
        uk = list(getattr(incompatible, "unexpected_keys", []) or [])
        res.say(f"{'strict=False':<12}: torch returned _IncompatibleKeys("
                f"missing={len(mk)}, unexpected={len(uk)})")
        res.say(f"{'':<12}  ^ PRODUCTION THROWS THIS AWAY. cgan_inference.py:174 and its three")
        res.say(f"{'':<12}    siblings ignore the return value entirely.")

    res.loaded = True

    # ---- the verdict on the keys --------------------------------------------------------------
    if res.keys_matched == 0:
        res.fail(
            "ZERO keys matched",
            "the checkpoint and the module have NO state_dict key names in common.\n"
            "load_state_dict(strict=False) returned WITHOUT RAISING and the generator is still\n"
            "at its random initialisation. In production this model would keep serving\n"
            "brain-shaped noise with no error anywhere. THIS IS THE FAILURE THIS SCRIPT EXISTS\n"
            "FOR. Do not deploy.")
        return True                                     # loaded (badly); caller may still run it

    # Distinguish benign from serious partial matches. `num_batches_tracked` is a BatchNorm
    # bookkeeping counter, not a learned parameter; it is absent from checkpoints saved by older
    # torch and its absence changes nothing in eval() mode. Anything else missing IS a learned
    # tensor left at its random initialisation.
    critical = [k for k in res.missing if not k.endswith("num_batches_tracked")]
    res.extra["critical_missing"] = len(critical)
    if critical:
        res.say("")
        res.say(f"WARNING: {len(critical)} LEARNED tensor(s) were not supplied by the checkpoint")
        res.say(f"         and remain RANDOMLY INITIALISED: "
                f"{critical[:5]}{' ...' if len(critical) > 5 else ''}")
        res.say("         Compare this count against a run from BEFORE the torch change. If it")
        res.say("         grew, the upgrade broke this checkpoint. If it was always this, the")
        res.say("         model has always been partly random and that is a separate bug.")
        res.say("         (--strict-keys turns this warning into a FAIL.)")
    elif res.missing:
        res.say(f"{'note':<12}: the only missing keys are BatchNorm num_batches_tracked counters,")
        res.say(f"{'':<12}  which are bookkeeping, not learned weights. Harmless in eval().")
    return True


# =================================================================================================
# SECTION 5 -- THE PER-MODEL CHECKS
#
# Each function replicates ONE production inference path up to (but not including) publication,
# and cites the file:line it mirrors. Signatures are (res) -> None; they record into `res`.
# =================================================================================================

def _vanilla_gan_check(res, ckpt_rel, out_channels, embedding_dim, latent_dim, n_classes,
                       label_value, mirrors):
    """Shared body for the three VanillaGenerator models (they differ only in constants).

    VanillaGenerator (inference/Generators.py:6) is a conditional DCGAN generator:
        label  (int)         -> nn.Embedding(n_classes, embedding_dim) -> Linear(->16) -> (1,4,4)
        noise  (latent_dim)  -> Linear(-> 4*4*512) -> LeakyReLU        -> (512,4,4)
        concat on the channel axis                                     -> (513,4,4)
        five ConvTranspose2d(stride 2) blocks, each doubling            -> (out_channels,128,128)
        Tanh                                                            -> values in [-1, 1]
    The 513 is not a typo: 512 latent channels plus the single label channel. That is why the
    first ConvTranspose2d is declared as `nn.ConvTranspose2d(513, ...)`.
    """
    import torch
    from inference.Generators import VanillaGenerator

    res.say(f"{'mirrors':<12}: {mirrors}")
    ckpt = os.path.join(os.getcwd(), ckpt_rel)
    if not note_checkpoints(res, [ckpt]):
        res.skip("weights absent",
                 "checkpoint not on disk. The five .pth files are delivered out of band (they "
                 "are not in git); copy them into backend/inference/model/ before `docker build`.")
        return

    # Construction mirrors cgan_inference_v2.py:132 -- note the ARGUMENT ORDER, which is
    # (ouput_channels, embedding_dim, latent_dim, n_classes) and is NOT the order of the
    # commented-out line above it in that file. Getting it wrong builds a differently-shaped
    # network whose keys still all match by name, so the shape diff above is the only guard.
    device = torch.device("cpu")            # what production hardcodes; see the module header
    gen = VanillaGenerator(out_channels, embedding_dim, latent_dim, n_classes).to(device)

    t0 = time.perf_counter()
    ok = load_and_diff(res, gen, ckpt, strict=False)
    res.load_seconds = round(time.perf_counter() - t0, 3)
    res.say(f"{'load time':<12}: {res.load_seconds:.3f} s")
    if not ok:
        return

    gen.eval()                              # eval(): BatchNorm uses running stats, no dropout

    # Forward pass, mirroring cgan_inference_v2.py:143-149 exactly.
    torch.manual_seed(SEED)                 # fixed -> the stats below are comparable across runs
    with torch.no_grad():
        labels = (torch.ones(1) * label_value).to(device).unsqueeze(1).long()   # (1,1) int64
        noise = torch.randn(1, latent_dim, device=device)                        # (1,latent_dim)
        t0 = time.perf_counter()
        imgs = gen((noise, labels))                                              # (1,C,128,128)
        seconds = time.perf_counter() - t0
        # permute(1,2,0): (C,128,128) -> (128,128,C), which is what save_images() indexes as
        # images[:, :, i] to peel off t1 / t2 / flair / seg.
        arr = imgs[0].permute(1, 2, 0).cpu().numpy()
        arr = (arr + 1) / 2.0 * 255.0       # Tanh [-1,1] -> [0,255], as production does

    res.say(f"{'label':<12}: class {label_value} of {n_classes}")
    if report_stats(res, arr, seconds, what="output"):
        res.status, res.verdict = PASS, "loaded + generated"


def check_gan_seg_tcga(res):
    """braingen_GAN_seg_TCGA_v1 (2D) -- cgan_inference.inference()

    Two-class conditional GAN (tumour / no tumour), 4 output channels (t1, t2, flair, seg).
    NOTE the constants differ from the v2 module: embedding_dim is 100 here, 20 there.
    """
    _vanilla_gan_check(
        res,
        ckpt_rel="inference/model/generator_epoch_200_seg.pth",
        out_channels=4, embedding_dim=100, latent_dim=100, n_classes=2,
        label_value=1,                       # "With Tumor" -> 1 (cgan_inference.py:148)
        mirrors="inference/cgan_inference.py:143-189 (inference)")


def check_cgan_multi(res):
    """braingen_cGAN_Multicontrast_BraTS_v1 (2D) -- cgan_inference_v2.inference(), 3 channels.

    18 classes = 3 orientations x 3 locations x 2 tumour states, encoded as
        orientation{Axial 0, Coronal 6, Sagittal 12} + location{0,2,4} + tumour{0,1}
    (cgan_inference_v2.py:11-33). We ask for Axial / Middle / With Tumor = 0 + 2 + 1 = 3.
    """
    _vanilla_gan_check(
        res,
        ckpt_rel="inference/model/generator_epoch_145.pth",
        out_channels=3, embedding_dim=20, latent_dim=100, n_classes=18,
        label_value=3,
        mirrors="inference/cgan_inference_v2.py:98-152 (inference)")


def check_cgan_multi_seg(res):
    """braingen_cGAN_Multicontrast_seg_BraTS_v1 (2D) -- same module, 4 channels (adds seg)."""
    _vanilla_gan_check(
        res,
        ckpt_rel="inference/model/generator_epoch_95_seg.pth",
        out_channels=4, embedding_dim=20, latent_dim=100, n_classes=18,
        label_value=3,
        mirrors="inference/cgan_inference_v2.py:98-152 (inference)")


def check_wavelet(res):
    """braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D) -- wgan_inference.inference()

    THE ONLY MODEL ON THE SITE THAT IS TWO NETWORKS, so it is the only one where a partial
    checkpoint failure could affect half the pipeline:

      generator_coarse : VanillaGenerator(5, 20, 128, 18)
                         (noise 128, label) -> (1, 5, 128, 128)   the 5 wavelet APPROXIMATION
                                                                  bands, one per contrast
      generator_fine   : SpatialAttentionUNetGenerator(5, 15, 128)
                         (1,5,128,128)      -> (1,15,128,128)     the 5x3 DETAIL bands
                                                                  (horizontal, vertical, diagonal)

    Then wgan_inference.transform_to_real_images() de-normalises both with the per-band global
    min/max in inference/min_max_values.csv and runs pywt.idwt2 (Haar, 'db1') to reconstruct
    five 256x256 images. We run that too, because it exercises pandas + PyWavelets and the CSV,
    all of which sit on this model's request path and none of which is torch.
    """
    import torch
    from inference.Generators import VanillaGenerator, SpatialAttentionUNetGenerator
    import inference.wgan_inference as wg

    res.say(f"{'mirrors':<12}: inference/wgan_inference.py:142-194 (inference)")
    coarse_p = os.path.join(os.getcwd(), "inference/model/generator_coarse_epoch_16_2500.pth")
    fine_p = os.path.join(os.getcwd(), "inference/model/generator_fine_epoch_16_2500.pth")
    csv_p = os.path.join(os.getcwd(), "inference/min_max_values.csv")
    if not note_checkpoints(res, [coarse_p, fine_p]):
        res.skip("weights absent", "one or both wavelet checkpoints are not on disk.")
        return
    res.say(f"{'bands csv':<12}: {csv_p}  exists={os.path.isfile(csv_p)}")
    if not os.path.isfile(csv_p):
        res.fail("min_max_values.csv missing",
                 "transform_to_real_images() reads the per-band global min/max from this file.\n"
                 "Unlike the .pth weights it IS tracked in git, so its absence means the image\n"
                 "was built from an incomplete copy of inference/.")
        return

    device = torch.device("cpu")
    latent_dim, n_classes, embedding_dim, n_channels = 128, 18, 20, 5

    # Two nets, two independent key diffs. We accumulate the counts so the summary table can
    # show one honest "matched/total" for the model as a whole, but each is reported separately
    # in the detail block -- a half-loaded pipeline is a real and confusing failure mode.
    totals = {"total": 0, "matched": 0}
    t0 = time.perf_counter()
    for name, module, path in (
        ("coarse", VanillaGenerator(n_channels, embedding_dim, latent_dim, n_classes).to(device),
         coarse_p),
        ("fine", SpatialAttentionUNetGenerator(input_channels=5, output_channels=15,
                                               feature_dim=128).to(device), fine_p),
    ):
        res.say("")
        res.say(f"--- {name} generator ---")
        if not load_and_diff(res, module, path, strict=False):
            return
        totals["total"] += res.keys_total or 0
        totals["matched"] += res.keys_matched or 0
        module.eval()
        if name == "coarse":
            gen_coarse = module
        else:
            gen_fine = module
        if res.status == FAIL:              # ZERO-key match on either half is fatal for both
            res.keys_total, res.keys_matched = totals["total"], totals["matched"]
            return
    res.load_seconds = round(time.perf_counter() - t0, 3)
    res.keys_total, res.keys_matched = totals["total"], totals["matched"]
    res.say("")
    res.say(f"{'combined':<12}: {totals['matched']}/{totals['total']} keys matched across both nets")
    res.say(f"{'load time':<12}: {res.load_seconds:.3f} s for both")

    # Forward pass + reconstruction, mirroring wgan_inference.py:175-191.
    torch.manual_seed(SEED)
    label = wg.class_embedding("With Tumor", "Axial", "Middle")     # -> 0 + 2 + 1 = 3
    t0 = time.perf_counter()
    with torch.no_grad():
        labels = (torch.ones(1) * label).to(device).unsqueeze(1).long()
        noise = torch.randn(1, latent_dim, device=device)
        c = gen_coarse((noise, labels))                             # (1, 5,128,128)
        f = gen_fine(c)                                             # (1,15,128,128)
        image_c = c[0].cpu().detach().permute(1, 2, 0).numpy()      # (128,128, 5)
        image_f = f[0].cpu().detach().permute(1, 2, 0).numpy()      # (128,128,15)
    try:
        # Returns a list of 5 arrays of (256,256): idwt2 doubles 128 -> 256.
        recon = wg.transform_to_real_images(image_c, image_f, csv_p)
    except Exception as e:
        res.fail("wavelet reconstruction raised",
                 f"{type(e).__name__}: {e}\n"
                 "The two generators loaded and ran; the failure is in pywt.idwt2 / pandas /\n"
                 "min_max_values.csv, i.e. downstream of torch.")
        return
    seconds = time.perf_counter() - t0

    arr = np.stack(recon, axis=-1)                                   # (256,256,5)
    res.say(f"{'coarse out':<12}: shape={tuple(image_c.shape)}  "
            f"min={image_c.min():+.4f} max={image_c.max():+.4f}")
    res.say(f"{'fine out':<12}: shape={tuple(image_f.shape)}  "
            f"min={image_f.min():+.4f} max={image_f.max():+.4f}")
    if report_stats(res, arr, seconds, what="reconstructed"):
        res.status, res.verdict = PASS, "loaded + generated"


def _gan3d_check(res, resolution, ckpt_rel, cls, noise_size):
    """Shared body for the two 3D resolutions of braingen_gan3d_BraTS_64_v1 (3D).

    THE OFF-BY-ONE IS REAL AND INTENTIONAL. gan_3d_inference_v1.py:78 constructs
    `Generator3D_32(noise_size=(noise_size + 1), ...)` with noise_size=200, i.e. the module is
    built for 201 input channels -- but line 95 draws `z` with only 200. The extra channel is
    the CONDITION: Generators.py:157 does
        condition_tensor = condition * torch.ones([x.shape[0], 1])
        x = torch.cat([x, condition_tensor], dim=1)
    so the 200-vector plus one condition scalar becomes the 201 channels the first
    ConvTranspose3d expects. Passing 201 noise values here would be a shape error at layer 1.

    The generator is a stack of stride-2 ConvTranspose3d layers from a 1x1x1 seed:
      _32: 5 layers,  1 -> 2 -> 4 -> 8 -> 16 -> 32   -> sigmoid -> (B,1,32,32,32)
      _64: 6 layers,  1 -> 2 -> 4 -> 8 -> 16 -> 32 -> 64 -> (B,1,64,64,64), with two 1x1
           skip connections trilinearly resampled onto the deeper stages
    `.squeeze()` at the end drops the singleton channel, giving (B,32,32,32) / (B,64,64,64).
    """
    import torch

    res.say(f"{'mirrors':<12}: inference/gan_3d_inference_v1.py:58-108 (inference), "
            f"resolution='{resolution}'")
    ckpt = os.path.join(os.getcwd(), ckpt_rel)
    if not note_checkpoints(res, [ckpt]):
        res.skip("weights absent", "3D checkpoint not on disk.")
        return

    device = torch.device("cpu")
    gen = cls(noise_size=(noise_size + 1), cube_resolution=32)      # +1: the condition channel

    t0 = time.perf_counter()
    ok = load_and_diff(res, gen, ckpt, strict=False)
    res.load_seconds = round(time.perf_counter() - t0, 3)
    res.say(f"{'load time':<12}: {res.load_seconds:.3f} s")
    if not ok:
        return

    gen.eval()
    # eval() matters MORE here than anywhere else on the site. These generators are dense with
    # BatchNorm3d, and in eval() BN uses the checkpoint's running_mean/running_var. If those
    # buffers were among the keys strict=False silently dropped, BN falls back to the
    # constructor defaults (mean 0, var 1) and the output is still a plausible-looking blob.
    # That is why the "missing" list above is worth reading even when the picture looks fine.

    torch.manual_seed(SEED)
    # num_images=2, matching gan_3d_inference_v1.py:90 -- production generates two volumes and
    # publishes only [0]. Kept identical because batch size can change BatchNorm behaviour.
    z = torch.randn([2, noise_size], device=device)
    t0 = time.perf_counter()
    with torch.no_grad():
        vols = gen(z, 0)                    # condition=0, as production hardcodes (line 100)
    seconds = time.perf_counter() - t0
    arr = vols.detach().cpu().numpy()[0]    # (res, res, res) -- what save_images() writes as NIfTI

    res.say(f"{'batch out':<12}: {tuple(vols.shape)} -> publishing index 0")
    if report_stats(res, arr, seconds, what="volume"):
        res.status, res.verdict = PASS, "loaded + generated"


def check_gan3d_32(res):
    """braingen_gan3d_BraTS_64_v1 (3D) with params['resolution'] == '32'."""
    from inference.Generators import Generator3D_32
    _gan3d_check(res, "32", "inference/model/generator_epoch_6100.pth", Generator3D_32, 200)


def check_gan3d_64(res):
    """braingen_gan3d_BraTS_64_v1 (3D) with params['resolution'] == '64'."""
    from inference.Generators import Generator3D_64
    _gan3d_check(res, "64", "inference/model/generator_epoch_3d64.pth", Generator3D_64, 400)


def check_legacy_diffuser(res):
    """braingen_diffuser_BraTS_v1 (2D) -- cgan_inference.inference_diffuser(). NOTE ONLY.

    THIS MODEL IS REPORTED, NOT TESTED, AND THAT IS DELIBERATE.

    cgan_inference.py:120-121 does

        model.to("cuda")
        noisy_sample = noisy_sample.to("cuda")

    UNCONDITIONALLY -- there is no `if torch.cuda.is_available()` and no map_location. On the
    CPU-only container that has always raised
        AssertionError: Torch not compiled with CUDA enabled
    which image_generation.py:117 catches, prints, and turns into an empty list, so the API
    returns HTTP 200 with a blank viewer. In other words this entry has NEVER produced an image
    in production, and its failure has never been visible outside container stdout.

    It is called out here because the GPU migration CHANGES ITS BEHAVIOUR: on a CUDA box those
    two lines would suddenly succeed and the model would start running -- 50 DDIM steps of a
    128x128 UNet2DModel loaded with `from_pretrained("inference/model/ddim-brain-128/unet")`.
    Whether that is desirable is a product decision, not a smoke-test decision, so this function
    reports the facts and takes no action. It never contributes a FAIL.
    """
    repo = os.path.join(os.getcwd(), "inference/model/ddim-brain-128/unet")
    exists = os.path.isdir(repo)
    res.checkpoints.append({"path": repo, "exists": exists, "bytes": None})
    res.say(f"{'mirrors':<12}: inference/cgan_inference.py:74-140 (inference_diffuser)")
    res.say(f"{'weights dir':<12}: {repo}")
    res.say(f"{'':<12}  exists={exists}")
    res.say("")
    res.say("NOT TESTED, ON PURPOSE. cgan_inference.py:120-121 calls .to('cuda')")
    res.say("UNCONDITIONALLY, so on the current CPU-only container this model has never once")
    res.say("executed -- it raises 'Torch not compiled with CUDA enabled', which")
    res.say("image_generation.py:117 swallows into an empty list and an HTTP 200.")
    res.say("")
    res.say("WHAT THE GPU MIGRATION CHANGES: on a CUDA host those two lines would start")
    res.say("working, and this entry would begin serving images for the first time. That is a")
    res.say("behaviour change nobody asked for. Decide deliberately whether to keep it exposed")
    res.say("in the UI; do not let it switch on by accident.")
    res.status, res.verdict = NOTE, "not tested (see notes)"


def check_conddiff(res):
    """braingen_CondDiffuser_BraTS_v1 (2D) -- the reason for the migration.

    Handles BOTH modes, because which one runs is an env-var decision and the smoke test must
    validate whatever this container is actually configured to do:

      CONDDIFF_MODE=gallery : no torch, no diffusers, no checkpoint. Serves a PNG that was
                              pre-rendered on the cluster at 200 DDIM steps. We decode one with
                              PIL and stat it.
      CONDDIFF_MODE=live    : builds the 85.3M-parameter UNet2DModel, loads diffusion_ema.pt
                              with strict=True, and runs the DDIM reverse loop for CONDDIFF_STEPS
                              iterations. We replicate _generate_live()'s body up to and
                              including render_flair(), stopping before PNG encoding and before
                              save_images().

    We do NOT call cdiff.inference(): its last statement is save_images(), which uploads to the
    user's real Supabase library when user_id/project_id are given and otherwise writes PNGs
    into backend/data/. A smoke test must not leave artefacts.
    """
    import inference.conddiff_inference as cdiff

    res.extra["mode"] = cdiff.MODE
    res.extra["ddim_steps"] = cdiff.STEPS
    res.extra["ready"] = bool(cdiff.READY)
    res.say(f"{'mode':<12}: {cdiff.MODE}   (env CONDDIFF_MODE; default 'live')")
    res.say(f"{'ddim steps':<12}: {cdiff.STEPS}  (env CONDDIFF_STEPS; the paper uses 200)")
    res.say(f"{'max images':<12}: {cdiff.MAX_IMAGES}  (image_generation.py:56 clamps n_images "
            f"to this)")
    res.say(f"{'READY':<12}: {cdiff.READY}")

    if not cdiff.READY:
        # READY is set by _check_assets() at import and reflects the ACTIVE MODE's assets only.
        # Missing assets are a SKIP, not a FAIL, for the same reason a missing .pth is: the
        # bundle and the gallery are both delivered out of band.
        res.skip("assets absent", "conddiff_inference reports NOT READY:\n"
                 + "\n".join(f"      {ln}" for ln in str(cdiff.REASON).splitlines()))
        return

    # ---- choose a parameter cell this deployment can actually serve --------------------------
    # Hardcoding "Frontal / Middle" would make the smoke test fail on a perfectly good bank that
    # simply does not contain that pair. supported_combinations() is the same function the
    # frontend uses to grey out impossible cells, so asking it is exactly right.
    combos = cdiff.supported_combinations()
    pairs = combos.get("pairs") or []
    if not pairs:
        res.fail("no supported cells",
                 "supported_combinations() returned READY but zero (lobe, slice_location) pairs.\n"
                 "Every dropdown combination the UI offers would fail. The bank or gallery\n"
                 "manifest parsed but contains nothing usable.")
        return
    lobe = pairs[0]["lobe"]
    level = pairs[0]["slice_location"]
    res.say(f"{'cells':<12}: {len(pairs)} supported (lobe, level) pairs; testing "
            f"{lobe}/{level}")

    spec = cdiff.params_to_spec("With Tumor", lobe, level, "Moderate")
    rng = np.random.default_rng(SEED)
    res.extra["cell"] = cdiff.cell_id(spec)
    res.say(f"{'cell id':<12}: {res.extra['cell']}")

    if cdiff.MODE == "gallery":
        _conddiff_gallery(res, cdiff, spec, rng)
    else:
        _conddiff_live(res, cdiff, spec, rng, lobe)


def _conddiff_gallery(res, cdiff, spec, rng):
    """Gallery mode: a dict lookup and a file read. Decode the PNG so we can stat real pixels."""
    from PIL import Image

    t0 = time.perf_counter()
    images, provenance = cdiff._generate_gallery(spec, rng, SEED)
    seconds = time.perf_counter() - t0

    res.loaded = True
    res.extra["device"] = "n/a (no torch on the gallery path)"
    res.extra["provenance"] = {k: provenance.get(k) for k in
                               ("ddim_steps", "seed", "device", "rendering") if k in provenance}
    res.say(f"{'returned':<12}: {list(images.keys())} "
            f"({sum(len(v) for v in images.values())} bytes of PNG)")
    res.say(f"{'rendered at':<12}: {provenance.get('ddim_steps', '?')} DDIM steps on "
            f"{provenance.get('device', '?')} (on the cluster, months ago)")

    # Decode the FLAIR so the constant-output rule applies to actual pixels rather than to the
    # length of a byte string. A PNG of a uniform grey would pass a byte-length check and fail
    # this one, which is the point.
    arr = np.asarray(Image.open(io.BytesIO(images["flair"])))
    if report_stats(res, arr, seconds, what="decoded PNG"):
        res.status = PASS
        res.verdict = f"gallery hit ({seconds * 1000:.0f} ms)"
    res.say("")
    res.say("HEALTH-CHECK VERDICT: not applicable. Gallery mode is a file read; it cannot")
    res.say("block the event loop long enough to matter. This is the mode the Lightsail deploy")
    res.say("scripts force via CONDDIFF_MODE=gallery precisely because that box (0.25 vCPU,")
    res.say("1 GB) cannot run the live path at all.")


def _conddiff_live(res, cdiff, spec, rng, lobe):
    """Live mode: replicate _generate_live() (conddiff_inference.py:1704) up to render_flair.

    Steps, in the order the production function performs them:
      1. _pick_slice(spec, rng)      -> (9,256,256) float32 conditioning stack from the bank
                                        (real held-out BraTS anatomy + 6 SRI24 atlas lobe maps)
      2. synth_mask_sized(...)       -> (256,256) float32 synthetic lesion mask in the chosen lobe
      3. build_cond(stack, mask, dev)-> (1,8,256,256) the conditioning tensor, on `device`
      4. _get_model()                -> the UNet2DModel, loaded with strict=True, cached
      5. _get_scheduler(STEPS)       -> DDIMScheduler with set_timesteps(STEPS)
      6. render_flair(...)           -> (256,256) float32 in [0,1]   <-- the actual image
    We stop there; production then PNG-encodes and publishes.
    """
    import torch

    device = cdiff._device()
    res.extra["device"] = device
    res.say(f"{'device':<12}: {device}   "
            f"(cuda.is_available()={torch.cuda.is_available()})")
    if device == "cpu":
        res.say(f"{'':<12}  ^ LIVE MODE ON CPU. ~950 GFLOP per DDIM step x {cdiff.STEPS} steps.")
        res.say(f"{'':<12}    Measured elsewhere at ~2 min per sample on a full laptop CPU.")

    # ---- 1 + 2: conditioning ------------------------------------------------------------------
    try:
        t0 = time.perf_counter()
        stack, row, slice_stats = cdiff._pick_slice(spec, rng)      # (9,256,256) float32
        cond_seconds = time.perf_counter() - t0
    except Exception as e:
        res.fail("conditioning bank unusable",
                 f"_pick_slice raised {type(e).__name__}: {e}\n"
                 "The bank manifest parsed at import but no slice for this cell verified against\n"
                 "the real array. Rebuild the bank, or disable this cell in the UI.")
        return
    res.say(f"{'bank slice':<12}: {row['file']}  patient={row['patient']} z={row['z']} "
            f"({cond_seconds:.2f} s to load)")
    res.say(f"{'stack':<12}: shape={tuple(stack.shape)} dtype={stack.dtype}  "
            f"[0]=FLAIR-target [1]=T1 [2]=seg [3..8]=6 atlas lobes")

    brain = stack[1] > 0.05                                          # (256,256) bool, T1 > 0.05
    atlas_lobe = stack[cdiff.ATLAS0 + spec["lobe_idx"]]              # (256,256) chosen lobe map
    mask, centre, mask_stats = cdiff.synth_mask_sized(atlas_lobe, brain, rng, spec["size"])
    res.say(f"{'lesion mask':<12}: {mask_stats.get('lesion_px')} px, realised size "
            f"'{mask_stats.get('size_realized')}' (requested '{spec['size']}')")

    # ---- 3: the (1,8,256,256) conditioning tensor ---------------------------------------------
    cond = cdiff.build_cond(stack, mask, device)
    res.say(f"{'cond tensor':<12}: {tuple(cond.shape)} {cond.dtype} on {cond.device}")

    # ---- 4: the model. THIS is the load a diffusers/torch skew breaks. -------------------------
    # Key accounting is done BEFORE _get_model() and the checkpoint dict is released immediately,
    # so we never hold two 341 MB copies of the weights at once. (Order matters on a box that is
    # tight on RAM; getting it wrong is an OOM, not an error message.)
    try:
        ckpt = cdiff._find_checkpoint()
    except Exception as e:
        res.skip("checkpoint absent", str(e).splitlines()[0])
        return
    note_checkpoints(res, [ckpt])

    try:
        sd = cdiff._load_state_dict(ckpt)                  # {name: fp32 tensor}, wrappers unwrapped
        ckpt_keys = set(sd.keys())
        del sd                                             # free ~341 MB before building the model
    except Exception as e:
        res.fail("checkpoint unreadable", f"{type(e).__name__}: {e}")
        return

    t0 = time.perf_counter()
    try:
        model = cdiff._get_model()
    except Exception as e:
        # _get_model raises with a long, deliberately diagnostic message covering the two known
        # causes: diffusers missing (torch/diffusers skew) and strict=True key mismatch (the
        # AttentionBlock -> Attention rename that moved query/key/value/proj_attn to
        # to_q/to_k/to_v/to_out.0). Print it whole; it is the most useful text in this file.
        res.loaded = False
        res.fail("model load failed", f"{type(e).__name__}: {e}")
        return
    res.load_seconds = round(time.perf_counter() - t0, 3)
    res.loaded = True

    model_keys = set(model.state_dict().keys())
    res.keys_total = len(model_keys)
    res.keys_matched = len(ckpt_keys & model_keys)
    res.missing = sorted(model_keys - ckpt_keys)
    res.unexpected = sorted(ckpt_keys - model_keys)
    res.n_params = int(sum(p.numel() for p in model.parameters()))
    res.say(f"{'parameters':<12}: {res.n_params:,} ({res.n_params / 1e6:.1f} M, expected ~85.3 M)")
    res.say(f"{'keys':<12}: matched {res.keys_matched}/{res.keys_total}   "
            f"missing {len(res.missing)}   unexpected {len(res.unexpected)}")
    res.say(f"{'':<12}  ^ UNLIKE THE FIVE GANs, THIS ONE LOADS WITH strict=True")
    res.say(f"{'':<12}    (conddiff_inference.py:_get_model). A mismatch RAISES rather than")
    res.say(f"{'':<12}    silently leaving a randomly-initialised network, so reaching this")
    res.say(f"{'':<12}    line already proves the keys are exact. The counts confirm it.")
    res.say(f"{'load time':<12}: {res.load_seconds:.3f} s -- ONE-OFF. _MODEL is a module")
    res.say(f"{'':<12}  singleton, so only the FIRST request after a container start pays it.")

    # ---- 5 + 6: the DDIM loop -----------------------------------------------------------------
    sched = cdiff._get_scheduler(cdiff.STEPS)
    res.say(f"{'scheduler':<12}: DDIM, {len(sched.timesteps)} timesteps, "
            f"first={int(sched.timesteps[0])} last={int(sched.timesteps[-1])}")
    if device == "cpu":
        # Mirrors _generate_live: the default thread count inside a CPU-limited container is
        # usually wrong, and this loop IS the request.
        torch.set_num_threads(max(1, os.cpu_count() or 1))

    t0 = time.perf_counter()
    try:
        flair = cdiff.render_flair(model, sched, cond, SEED)         # (256,256) float32 in [0,1]
    except Exception as e:
        res.fail("DDIM sampling raised", f"{type(e).__name__}: {e}\n"
                 + traceback.format_exc(limit=3))
        return
    seconds = time.perf_counter() - t0
    per_step = seconds / max(1, cdiff.STEPS)
    res.extra["seconds_per_step"] = round(per_step, 4)

    ok = report_stats(res, flair, seconds, what="FLAIR")
    res.say(f"{'per step':<12}: {per_step * 1000:.0f} ms x {cdiff.STEPS} steps")
    res.say(f"{'at 200 steps':<12}: {per_step * 200:.1f} s (the paper's step count, for scale)")
    if ok:
        res.status = PASS
        res.verdict = f"live, {seconds:.1f} s/image"

    _print_healthcheck_verdict(res, seconds, res.load_seconds or 0.0, cdiff.STEPS, device)


def _print_healthcheck_verdict(res, gen_s, load_s, steps, device):
    """Requirement #4: does the measured time fit inside the Lightsail health-check window?

    THE ARITHMETIC, stated so it can be checked rather than believed:

      api.py:81 is `async def generate_images(...)` but everything it awaits is synchronous
      Python. An `async def` that never awaits does not yield the event loop -- so for the whole
      duration of a generation, uvicorn's single loop thread is busy and GET /health (api.py:71)
      cannot be answered.

      Lightsail's container healthCheck: timeout {t}s, interval {i}s, unhealthyThreshold {n}.
      A check that goes unanswered for {t}s is a failure; {n} CONSECUTIVE failures replace the
      container. Consecutive checks are {i}s apart, so a blockage must last roughly
      {t} + {i}*{n} = {r}s to be fatal -- and a blockage between {t}s and that is survivable
      (one failed check, threshold not reached) but still shows as a health blip.

      NOTE THE FIRST-REQUEST CASE SEPARATELY: the U-Net is a module singleton, so request #1
      after a container start pays load + generate, and every later request pays generate only.
      A deploy that only ever measures the steady state can be surprised exactly once, at the
      worst possible moment.
    """
    first = load_s + gen_s
    res.extra["healthcheck"] = {"generate_s": round(gen_s, 2),
                                "first_request_s": round(first, 2),
                                "replace_after_s": HEALTH_REPLACE_S}
    res.say("")
    res.say("HEALTH-CHECK VERDICT  (Lightsail container healthCheck arithmetic)")
    res.say(f"  timeout {HEALTH_TIMEOUT_S}s x threshold {HEALTH_THRESHOLD} at {HEALTH_INTERVAL_S}s"
            f" intervals -> container replaced after ~{HEALTH_REPLACE_S}s of blockage")
    res.say(f"  measured : {gen_s:.1f}s per image ({steps} DDIM steps on {device})")
    res.say(f"  first req: {first:.1f}s  (adds the {load_s:.1f}s one-off checkpoint load)")

    def band(t):
        if t >= HEALTH_REPLACE_S:
            return ("RED", "EXCEEDS the replacement window -- the container is killed MID-REQUEST "
                           "and the user gets a 502, forever, on every attempt")
        if t >= HEALTH_TIMEOUT_S:
            return ("AMBER", f"survivable but not clean: /health goes unanswered for longer than "
                             f"the {HEALTH_TIMEOUT_S}s timeout, so checks fail intermittently and "
                             f"a second concurrent request would push past the threshold")
        return ("GREEN", f"fits inside the {HEALTH_TIMEOUT_S}s timeout -- synchronous serving is "
                         f"viable with no async job queue")

    for tag, t in (("steady state", gen_s), ("first request", first)):
        colour, why = band(t)
        res.say(f"  {tag:<14} {colour:<6} {why}")

    worst, _ = band(first)
    res.extra["healthcheck"]["verdict"] = worst
    res.say("")
    if worst == "GREEN":
        res.say("  => SYNCHRONOUS SERVING IS VIABLE ON THIS HOST. No job queue, no frontend")
        res.say("     change, no async pattern needed. This is the number the whole design")
        res.say("     rests on, and it has now been measured rather than assumed.")
    else:
        res.say("  => DO NOT SERVE THIS SYNCHRONOUSLY ON A LIGHTSAIL CONTAINER. Either move to")
        res.say("     hardware where the measurement lands GREEN, cut CONDDIFF_STEPS, or set")
        res.say("     CONDDIFF_MODE=gallery on this box (which is exactly what the four deploy")
        res.say("     scripts already do as a safety interlock).")
    res.say("")
    res.say("  CAVEAT, so this number is not over-read: the health check above is LIGHTSAIL's.")
    res.say("  A plain EC2 GPU host running the container directly has no such supervisor, so")
    res.say("  there is nothing to replace the process -- the only cost of a slow request is")
    res.say("  latency and a blocked worker. The arithmetic still tells you when to worry.")


# =================================================================================================
# SECTION 6 -- REGISTRY, ARGV, MAIN
# =================================================================================================

# key -> (model_name as the frontend POSTs it, check function, is_gan)
# The order here is the order of the final table.
REGISTRY = [
    ("gan_seg_tcga", "braingen_GAN_seg_TCGA_v1 (2D)", check_gan_seg_tcga, True),
    ("cgan_multi", "braingen_cGAN_Multicontrast_BraTS_v1 (2D)", check_cgan_multi, True),
    ("cgan_multi_seg", "braingen_cGAN_Multicontrast_seg_BraTS_v1 (2D)", check_cgan_multi_seg, True),
    ("wavelet", "braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)", check_wavelet, True),
    ("gan3d_32", "braingen_gan3d_BraTS_64_v1 (3D) res=32", check_gan3d_32, True),
    ("gan3d_64", "braingen_gan3d_BraTS_64_v1 (3D) res=64", check_gan3d_64, True),
    ("conddiff", "braingen_CondDiffuser_BraTS_v1 (2D)", check_conddiff, False),
    ("legacy_diffuser", "braingen_diffuser_BraTS_v1 (2D)", check_legacy_diffuser, False),
]

GROUPS = {
    "all": [k for k, _, _, _ in REGISTRY],
    # "gans" excludes the two diffusers: it is the set whose risk is a TORCH change, as opposed
    # to the conditional diffuser whose risk is a DIFFUSERS change.
    "gans": [k for k, _, _, is_gan in REGISTRY if is_gan],
}


def build_parser():
    p = argparse.ArgumentParser(
        prog="smoke_test_models.py",
        description="Pre-deploy gate: prove every model's checkpoint really loaded and really "
                    "generates, after a torch/diffusers change.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="subset keys:\n  " + "\n  ".join(f"{k:<16} {label}" for k, label, _, _ in REGISTRY)
               + "\n  " + f"{'gans':<16} the five GAN entries (everything torch can break)"
               + "\n  " + f"{'all':<16} default")
    p.add_argument("subset", nargs="*", default=[],
                   help="model keys / groups to test (default: all)")
    p.add_argument("--list", action="store_true", help="print the subset keys and exit")
    p.add_argument("--strict-keys", action="store_true",
                   help="promote 'some learned tensors are missing' from WARNING to FAIL")
    p.add_argument("--json", action="store_true",
                   help="append a machine-diffable JSON blob (run before and after a pin change)")
    return p


def resolve_subset(tokens):
    """Expand argv tokens into an ordered, de-duplicated list of registry keys.

    Unknown tokens are a usage error rather than a silent no-op: a typo'd key would otherwise
    make the gate 'pass' by testing nothing at all, which is the worst possible failure mode
    for a gate.
    """
    known = {k for k, _, _, _ in REGISTRY}
    if not tokens:
        return GROUPS["all"], []
    chosen, unknown = [], []
    for tok in tokens:
        t = tok.strip().lower()
        if t in GROUPS:
            for k in GROUPS[t]:
                if k not in chosen:
                    chosen.append(k)
        elif t in known:
            if t not in chosen:
                chosen.append(t)
        else:
            unknown.append(tok)
    return chosen, unknown


def print_table(results):
    """The deliverable: model | loaded | keys matched | gen time | verdict."""
    heading("SUMMARY")
    cols = ("model", "loaded", "keys matched", "gen time", "verdict")
    widths = (46, 7, 14, 10, 11)
    print("  " + "  ".join(c.ljust(w) for c, w in zip(cols, widths)))
    print("  " + "  ".join("-" * w for w in widths))
    for r in results:
        loaded = {True: "yes", False: "NO", None: "-"}[r.loaded]
        if r.keys_total:
            keys = f"{r.keys_matched}/{r.keys_total}"
            if r.keys_matched == 0:
                keys += " !!"
        else:
            keys = "-"
        gen = f"{r.gen_seconds:.3f} s" if r.gen_seconds is not None else "-"
        name = r.label if len(r.label) <= widths[0] else r.label[:widths[0] - 1] + "…"
        print("  " + "  ".join(c.ljust(w) for c, w in zip(
            (name, loaded, keys, gen, r.status), widths)))
        # The prose verdict goes on its own indented line so the columns stay aligned no matter
        # how long the explanation is.
        print("  " + " " * (widths[0] + 2) + f"└─ {r.verdict}")
    print()


def main(argv):
    args = build_parser().parse_args(argv)

    if args.list:
        for k, label, _, _ in REGISTRY:
            print(f"{k:<16} {label}")
        for g, keys in GROUPS.items():
            print(f"{g:<16} -> {', '.join(keys)}")
        return 0

    keys, unknown = resolve_subset(args.subset)
    if unknown:
        print(f"unknown model key(s): {unknown}. Run with --list to see the valid ones.",
              file=sys.stderr)
        return 2

    heading("smoke_test_models.py -- checkpoint + generation gate")
    print(f"  seed          : {SEED}  (fixed, so min/max/mean are comparable across runs)")
    print(f"  testing       : {', '.join(keys)}")
    print(f"  strict keys   : {args.strict_keys}")

    env, hard_failures = report_environment()

    # The inference package is imported AFTER the environment report so that, when the import
    # itself is what breaks, the operator has already seen the torch/diffusers versions that
    # caused it. image_generation.py is the single module api.py depends on and it imports all
    # six model modules at its top, so importing it here proves the entire production import
    # chain in one line -- including conddiff_inference's boot banner, which prints below.
    heading("IMPORTING THE PRODUCTION MODULE CHAIN  (image_generation.py imports all six models)")
    try:
        import image_generation                                   # noqa: F401
        print("  OK -- image_generation imported, so api.py would start.")
    except Exception as e:
        print(f"  !!! image_generation FAILED TO IMPORT: {type(e).__name__}: {e}")
        traceback.print_exc()
        print("  Every model on the site is behind this import. api.py would not start.")
        hard_failures.append(f"image_generation import failed: {e}")
        # Keep going: the per-model checks import Generators directly and may still produce a
        # useful diagnosis of WHICH module is at fault.

    results = []
    for key in keys:
        label, fn, _is_gan = next((l, f, g) for k, l, f, g in REGISTRY if k == key)
        res = Result(key, label)
        heading(f"{label}    [{key}]")
        try:
            fn(res)
        except Exception as e:
            # The requirement is "never crash on an absent checkpoint" -- but the stronger rule
            # is that no single model's failure may prevent the other models from being tested.
            # An unexpected exception is a FAIL with its traceback, not the end of the run.
            res.fail(f"crashed: {type(e).__name__}",
                     f"{type(e).__name__}: {e}\n" + traceback.format_exc(limit=6))
        if args.strict_keys and res.status == PASS and res.extra.get("critical_missing"):
            res.status = FAIL
            res.verdict = f"{res.extra['critical_missing']} learned tensors missing"
        results.append(res)

    print_table(results)

    failed = [r for r in results if r.status == FAIL]
    skipped = [r for r in results if r.status == SKIP]
    passed = [r for r in results if r.status == PASS]

    heading("RESULT")
    print(f"  {len(passed)} passed   {len(failed)} FAILED   {len(skipped)} skipped   "
          f"{len([r for r in results if r.status == NOTE])} note-only")
    if skipped:
        print()
        print("  SKIPPED (weights or assets not on this machine -- NOT a pass):")
        for r in skipped:
            print(f"    {r.label}: {r.verdict}")
        print("  A skipped model has been proved NOTHING about. Re-run this inside the built")
        print("  image, where the checkpoints actually live, before treating it as a gate.")
    if hard_failures:
        print()
        print("  ENVIRONMENT FAILURES (independent of any model):")
        for h in hard_failures:
            print(f"    {h}")
    if failed:
        print()
        print("  FAILED:")
        for r in failed:
            print(f"    {r.label}: {r.verdict}")

    if args.json:
        heading("JSON  (redirect stdout and diff this against a run from before the pin change)")
        print(json.dumps({"seed": SEED, "environment": env,
                          "hard_failures": hard_failures,
                          "models": [r.to_json() for r in results]},
                         indent=2, sort_keys=True, default=str))

    # Non-zero on any FAIL so this can gate a deploy: `python smoke_test_models.py && docker push`
    return 1 if (failed or hard_failures) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
