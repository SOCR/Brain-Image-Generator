import os

# Base directory for models
MODEL_BASE_DIR = os.path.join(os.getcwd(), "inference/model")

# Model paths dictionary
MODEL_PATHS = {
    "GAN_SEG_TCGA": os.path.join(MODEL_BASE_DIR, "generator_epoch_200_seg.pth"),
    "WAVELET_GAN_COARSE": os.path.join(MODEL_BASE_DIR, "generator_coarse_epoch_16_2500.pth"),
    "WAVELET_GAN_FINE": os.path.join(MODEL_BASE_DIR, "generator_fine_epoch_16_2500.pth"),
    "GAN_3D_32": os.path.join(MODEL_BASE_DIR, "generator_epoch_6100.pth"),
    "GAN_3D_64": os.path.join(MODEL_BASE_DIR, "generator_epoch_3d64.pth"),
    "MIN_MAX_VALUES": os.path.join(MODEL_BASE_DIR, "../min_max_values.csv"),

    # --- braingen_CondDiffuser_BraTS_v1 (atlas-guided conditional diffuser) -------------------
    # This model needs two kinds of asset. The CHECKPOINT lives in inference/model/ alongside
    # the GAN weights, and may arrive as either an fp16 safetensors export or the original .pt
    # pickle - inference/conddiff_inference.py accepts either and only ONE of the two needs to
    # exist. The CONDITIONING BUNDLE (patient T1 + tumour mask + 6 atlas lobe probability maps
    # per slice, which the website has no way to synthesise) lives in
    # inference/conddiff_bundle/, indexed by a manifest.JSON, with the .npy slices under a
    # cond_slices/ subdirectory and a second copy of the checkpoint at the bundle root. The
    # "../" prefix reaches back out of MODEL_BASE_DIR to inference/, matching the style already
    # used by MIN_MAX_VALUES above.
    #
    # THESE MUST MATCH inference/conddiff_inference.py's CKPT_CANDIDATES / MANIFEST_PATH, which
    # are what actually get read, and the copies re-hardcoded in api.py's /check-models.
    #
    # AND THEY DID NOT. These entries used to read "conddiff_bank/manifest.csv" - a folder name,
    # a file format and a slice layout that NOTHING has ever produced.
    # scripts/export_for_website.py:148-150 defines SLICE_SUBDIR="cond_slices",
    # CKPT_NAME="diffusion_ema.pt", MANIFEST="manifest.json", and conddiff_inference.py:334-336
    # reads exactly that. Corrected here to the layout the tooling actually emits, so that this
    # file cannot mislead someone who reads it as the source of truth for where to sftp the
    # bundle. (The CSV expectation was the bug; do not "restore" it.)
    #
    # IMPORTANT: nothing in the backend actually imports this file - every inference module
    # hardcodes its own os.path.join(os.getcwd(), "inference/...") path inline at the point of
    # use, and api.py re-hardcodes these same paths in /check-models. These entries are recorded
    # for tidiness and documentation; changing a path here will NOT change where anything looks.
    "CONDDIFF_CHECKPOINT_SAFETENSORS": os.path.join(MODEL_BASE_DIR, "diffusion_ema_fp16.safetensors"),
    "CONDDIFF_CHECKPOINT_PT": os.path.join(MODEL_BASE_DIR, "diffusion_ema.pt"),
    "CONDDIFF_CHECKPOINT_IN_BUNDLE": os.path.join(MODEL_BASE_DIR, "../conddiff_bundle/diffusion_ema.pt"),
    "CONDDIFF_BUNDLE_DIR": os.path.join(MODEL_BASE_DIR, "../conddiff_bundle"),
    "CONDDIFF_BUNDLE_MANIFEST": os.path.join(MODEL_BASE_DIR, "../conddiff_bundle/manifest.json"),
    "CONDDIFF_BUNDLE_SLICES": os.path.join(MODEL_BASE_DIR, "../conddiff_bundle/cond_slices"),
    # Gallery mode's payload, a completely separate tree (see conddiff_inference.py:344-345).
    "CONDDIFF_GALLERY_DIR": os.path.join(MODEL_BASE_DIR, "../conddiff_gallery"),
    "CONDDIFF_GALLERY_MANIFEST": os.path.join(MODEL_BASE_DIR, "../conddiff_gallery/manifest.json")
}