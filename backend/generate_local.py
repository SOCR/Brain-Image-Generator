#!/usr/bin/env python3
"""generate_local.py -- run any SOCR Brain Gen model on your own computer (no web app, no Supabase).

    python backend/generate_local.py conddiff -n 4 --lobe Temporal --size Large
    python backend/generate_local.py cgan --orientation Sagittal --location Left --no-tumour
    python backend/generate_local.py gan3d --res 32 -n 10 --seed 0

Download the weights into backend/inference/ first (they go in model/ and conddiff_bundle/).
Every sample is written to its own folder under --out/<model>/ and the folder is printed.
This calls the same inference functions as the website; with no user/project id they save to disk.
"""
import argparse
import os
import sys
import time

MODELS = {  # CLI name -> model name used by the website and the back end
    "tcga":     "braingen_GAN_seg_TCGA_v1 (2D)",
    "cgan":     "braingen_cGAN_Multicontrast_BraTS_v1 (2D)",
    "cgan-seg": "braingen_cGAN_Multicontrast_seg_BraTS_v1 (2D)",
    "wavelet":  "braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)",
    "gan3d":    "braingen_gan3d_BraTS_64_v1 (3D)",
    "ddim":     "braingen_diffuser_BraTS_v1 (2D)",
    "conddiff": "braingen_CondDiffuser_BraTS_v1 (2D)",
}
LOCATIONS = {"Axial":    ["Inferior", "Middle", "Superior"],
             "Coronal":  ["Anterior", "Middle", "Posterior"],
             "Sagittal": ["Left", "Middle", "Right"]}

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("model", choices=MODELS)
p.add_argument("-n", type=int, default=1, help="number of samples (default 1)")
p.add_argument("--out", default="outputs", help="output folder (default ./outputs)")
p.add_argument("--seed", type=int, help="random seed, for reproducible samples")
p.add_argument("--no-tumour", action="store_true",
               help="tcga, cgan, cgan-seg, wavelet, conddiff: generate a brain without a tumour")
p.add_argument("--orientation", default="Axial", choices=LOCATIONS, help="cgan, cgan-seg, wavelet")
p.add_argument("--location", default="Middle", choices=sorted({x for v in LOCATIONS.values() for x in v}),
               help="slice location: cgan, cgan-seg, wavelet (must match --orientation); "
                    "conddiff (Inferior, Middle or Superior)")
p.add_argument("--res", default="64", choices=["32", "64"], help="gan3d: volume size (default 64)")
p.add_argument("--lobe", default="Frontal",
               choices=["Frontal", "Parietal", "Temporal", "Occipital", "Cerebellum", "Insula"],
               help="conddiff: lobe the tumour is placed in")
p.add_argument("--size", default="Moderate", choices=["Small", "Moderate", "Large"], help="conddiff: tumour size")
p.add_argument("--steps", type=int, default=200, help="conddiff: DDIM steps (default 200, as in the paper)")
a = p.parse_args()

plane = "Axial" if a.model == "conddiff" else a.orientation   # conddiff is axial-only
if a.model in ("cgan", "cgan-seg", "wavelet", "conddiff") and a.location not in LOCATIONS[plane]:
    p.error(f"--location {a.location} is not valid for {plane}; use one of {LOCATIONS[plane]}")

out = os.path.abspath(os.path.join(a.out, a.model))  # resolve before changing directory
os.makedirs(out, exist_ok=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # the modules find weights relative to backend/

import torch  # noqa: E402

if a.seed is not None:
    torch.manual_seed(a.seed)
name = MODELS[a.model]
tumour = "Without Tumor" if a.no_tumour else "With Tumor"
LOCAL = dict(user_id=None, project_id=None)  # no Supabase ids -> the modules save to disk

if a.model == "tcga":
    from inference import cgan_inference as m
    run = lambda i: m.inference(tumour, name, out, **LOCAL)
elif a.model in ("cgan", "cgan-seg"):
    from inference import cgan_inference_v2 as m
    run = lambda i: m.inference(name, tumour, a.orientation, a.location, out, **LOCAL)
elif a.model == "wavelet":
    from inference import wgan_inference as m
    run = lambda i: m.inference(name, tumour, a.orientation, a.location, out, **LOCAL)
elif a.model == "gan3d":
    from inference import gan_3d_inference_v1 as m
    run = lambda i: m.inference(name, a.res, out, **LOCAL)
elif a.model == "ddim":
    from inference import cgan_inference as m
    run = lambda i: m.inference_diffuser(out, **LOCAL)
else:
    os.environ["CONDDIFF_MODE"] = "live"            # both are read when the module is imported
    os.environ["CONDDIFF_STEPS"] = str(a.steps)
    from inference import conddiff_inference as m
    if not m.READY:
        sys.exit(m.REASON)
    run = lambda i: m.inference(name, tumour, a.lobe, a.location, a.size, out, **LOCAL,
                                seed=None if a.seed is None else a.seed + i)

for i in range(a.n):
    t = time.perf_counter()
    result = run(i)  # 2D modules return their file paths, 3D and wavelet return the folder
    folder = os.path.dirname(result[0]) if isinstance(result, list) else result
    print(f"[{i + 1}/{a.n}] {time.perf_counter() - t:.2f} s -> {folder}", flush=True)
print(f"Done: {a.n} sample(s) of {name} in {out}")
