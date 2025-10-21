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
    "MIN_MAX_VALUES": os.path.join(MODEL_BASE_DIR, "../min_max_values.csv")
} 