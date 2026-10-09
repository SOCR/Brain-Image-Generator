"""push_space.py -- publish the backend to its Hugging Face ZeroGPU Space, in ONE commit.

    python backend/hf_space/push_space.py USER/braingen-backend --assets USER/braingen-assets
    python backend/hf_space/push_space.py --local C:/tmp/space      # same tree, copied locally

WHY A SCRIPT AND NOT `hf upload backend/`: the Space repo is not a copy of backend/. It needs
hf_space/app.py, requirements.txt and README.md at its ROOT (a Space runs app.py and reads the
README's YAML from there), while backend/ already has a different README.md, requirements.txt and
Dockerfile for Lightsail. It also must NOT receive inference/model/ or any bank files: the weights
come from the private ASSETS_REPO at startup. So the file list is built here, and uploaded as one
atomic commit, because a Space rebuilds on every commit and a half-uploaded tree is a failed build.

WHAT IT DOES
  1. creates the Space if it does not exist: Gradio SDK, ZeroGPU hardware, PUBLIC unless --private
     (exist_ok, so re-running it on an existing Space changes nothing about the Space itself).
     Public for the MVP (DEPLOY.md §3, decision 3.2): HF does not document whether a free account's ZeroGPU
     Space may be private, and a public one is known to work. It exposes this code and its
     endpoints, never the data -- the weights and BraTS slices stay in the private ASSETS_REPO;
  2. sets the Space VARIABLE ASSETS_REPO (not a secret: it is only a repo name);
  3. uploads the files below.
Secrets (HF_TOKEN, SUPABASE_URL, SUPABASE_KEY) are NOT set here -- add them once in the Space's
Settings page, so no credential ever passes through a script or a shell history.

Needs `pip install huggingface_hub` and `hf auth login` (a token with WRITE access) first.
"""
import argparse
import glob
import hashlib
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))       # backend/hf_space
BACKEND = os.path.dirname(HERE)                         # backend

# path in the Space repo -> local file
FILES = {
    "app.py": os.path.join(HERE, "app.py"),
    "requirements.txt": os.path.join(HERE, "requirements.txt"),
    "README.md": os.path.join(HERE, "README.md"),
    "image_generation.py": os.path.join(BACKEND, "image_generation.py"),
    "supabase_storage.py": os.path.join(BACKEND, "supabase_storage.py"),
    "inference/min_max_values.csv": os.path.join(BACKEND, "inference", "min_max_values.csv"),
    **{f"inference/{os.path.basename(p)}": p
       for p in glob.glob(os.path.join(BACKEND, "inference", "*.py"))},
}

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("space", nargs="?", help="Space id, e.g. ALIUD1/braingen-backend")
ap.add_argument("--assets", help="private model repo with the weights + bank, e.g. ALIUD1/braingen-assets")
ap.add_argument("--private", action="store_true", help="create the Space private (server-token calls only)")
ap.add_argument("--local", metavar="DIR", help="copy the Space tree into DIR instead of uploading")
args = ap.parse_args()

# The shared core must be byte-identical to Summer2026/src/conddiff_core.py (PROJECT_LOG section 9).
# Printed on every push so the deployed hash is on record next to the commit.
with open(FILES["inference/conddiff_core.py"], "rb") as fh:
    print(f"conddiff_core.py sha256 {hashlib.sha256(fh.read()).hexdigest()}")

if args.local:
    for dst, src in FILES.items():
        os.makedirs(os.path.dirname(os.path.join(args.local, dst)), exist_ok=True)
        shutil.copy2(src, os.path.join(args.local, dst))
    print(f"copied {len(FILES)} files to {args.local}")
else:
    if not (args.space and args.assets):
        ap.error("give the Space id and --assets (or --local DIR)")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    api.create_repo(args.space, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g",
                    private=args.private, exist_ok=True)        # "zero-a10g" = ZeroGPU's legacy id
    api.add_space_variable(args.space, "ASSETS_REPO", args.assets)
    api.create_commit(
        repo_id=args.space, repo_type="space",
        operations=[CommitOperationAdd(path_in_repo=dst, path_or_fileobj=src) for dst, src in FILES.items()],
        commit_message="Deploy backend from SOCR/Brain-Image-Generator backend/",
    )
    print(f"pushed {len(FILES)} files -> https://huggingface.co/spaces/{args.space}")
