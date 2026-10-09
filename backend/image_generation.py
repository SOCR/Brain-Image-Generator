# image_generation.py
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import inference.wgan_inference as wg
import inference.cgan_inference as cgv1
import inference.cgan_inference_v2 as cgv2
import inference.gan_3d_inference_v1 as gan3d
# The atlas-guided conditional diffusion model from the paper (braingen_CondDiffuser_BraTS_v1).
# Imported at module scope alongside the four GAN modules above so the dispatch chain below
# reads exactly the same way for all six models. An ImportError raised here would happen before
# FastAPI ever starts and would take down all five of the existing, working models with it, so
# inference/conddiff_inference.py is written to import only what is ALREADY installed and
# already imported by the four modules above (torch, numpy, PIL). diffusers is imported lazily
# inside the sampling path, and scipy is not used at all (its two ndimage calls are
# reimplemented in numpy), so a missing optional dependency cannot break this import. Its
# startup asset check is likewise wrapped so that missing weights set a flag instead of raising.
import inference.conddiff_inference as cdiff


def sanitize_filename(filename):
    # Replace invalid characters for Windows
    invalid_chars = ['<', '>', ':', '"', '/', '\\', '|', '?', '*']
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def generate_image(user_id, project_id, model_name, n_images, params, is_playground=False):
    try:
        # Create user directory if it doesn't exist
        user_id = sanitize_filename(user_id)
        project_id = sanitize_filename(project_id)

        # NOTE: this list is not read by anything - it is defined here and never referenced
        # again (the real dispatch is the if/elif chain on model_name further down). The names
        # in it also carry no " (2D)"/" (3D)" suffix, unlike the literals the dispatch compares
        # against. The new model is added for consistency only; adding it here does NOT make it
        # callable, and removing it would not break anything.
        model_list = ["braingen_GAN_seg_TCGA_v1","braingen_cGAN_Multicontrast_BraTS_v1",
                      "braingen_cGAN_Multicontrast_seg_BraTS_v1", "braingen_WaveletGAN_Multicontrast_BraTS_v1",
                      "braingen_diffuser_BraTS_v1","braingen_gan3d_BraTS_64_v1",
                      "braingen_CondDiffuser_BraTS_v1"]
        image_paths = []

        print("in python", params)

        # The conditional diffuser is not a single forward pass like the GANs: it runs a DDIM
        # reverse loop of CONDDIFF_STEPS iterations, each ~950 GFLOP at 256x256, on the CPU.
        # The loop below runs one full inference per requested image, SYNCHRONOUSLY, inside one
        # HTTP request, so honouring the UI's n_images slider (1..5) would multiply an already
        # multi-minute request by up to five and block the single uvicorn worker for all users.
        # conddiff_inference exposes MAX_IMAGES precisely so the dispatcher can clamp here; the
        # four GAN models are untouched by this and keep their full n_images behaviour.
        if model_name == "braingen_CondDiffuser_BraTS_v1 (2D)" and n_images > cdiff.MAX_IMAGES:
            print(f"Clamping n_images {n_images} -> {cdiff.MAX_IMAGES} for {model_name}")
            n_images = cdiff.MAX_IMAGES

        for i in range(n_images):
            save_path = os.path.join(os.getcwd(),f"data\\{user_id}\\{project_id}")
            os.makedirs(save_path) if not os.path.exists(save_path) else None

            try:
                if model_name == "braingen_GAN_seg_TCGA_v1 (2D)":
                    tumour = params["tumour"]
                    image_path = cgv1.inference(tumour, model_name, save_path, user_id, project_id, is_playground)

                elif model_name == "braingen_diffuser_BraTS_v1 (2D)":
                    image_path = cgv1.inference_diffuser(save_path, user_id, project_id, is_playground)

                elif model_name == "braingen_cGAN_Multicontrast_BraTS_v1 (2D)" or model_name == "braingen_cGAN_Multicontrast_seg_BraTS_v1 (2D)":
                    tumour = params["tumour"]
                    slice_orientation = params["slice_orientation"]
                    slice_location = params["slice_location"]

                    image_path = cgv2.inference(model_name, tumour, slice_orientation, slice_location, save_path, user_id, project_id, is_playground)

                elif model_name == "braingen_CondDiffuser_BraTS_v1 (2D)":
                    # The atlas-guided conditional diffuser. The model_name literal must match the
                    # `id` in ParameterControlPanel.tsx's models2D array byte-for-byte, including
                    # the space before "(2D)" - that string is POSTed verbatim as model_name and is
                    # compared with == here. It deliberately does not contain the substring
                    # "braingen_cGAN_Multicontrast", so it cannot be captured by the .includes()
                    # tests the frontend uses for the cGAN models.
                    #
                    # These four keys are what GenerationPageClient.getModelParams() sends for this
                    # model. Unlike the branches above, which index params["..."] directly, these use
                    # .get() with the frontend's own defaults. That is deliberate: a bare KeyError
                    # would be caught by the `except` at the bottom of this loop, turned into an
                    # empty list, and returned as HTTP 200 - so a half-wired frontend would show a
                    # green "Success" toast over an empty viewer with no error anywhere. Defaulting
                    # instead degrades to a sensible image and keeps the failure visible in the logs.
                    tumour = params.get("tumour", "With Tumor")              # "With Tumor" / "Without Tumor" (British key, American value - matches the other models)
                    lobe = params.get("lobe", "Frontal")                     # which ICBM452 atlas lobe the synthetic lesion is placed in
                    slice_location = params.get("slice_location", "Middle")  # "Inferior" / "Middle" / "Superior" - the axial level of the conditioning slice
                    tumour_size = params.get("tumour_size", "Moderate")      # "Small" / "Moderate" / "Large" - lesion area relative to the lobe

                    # Argument order mirrors cgv2.inference above: model first, then the model's own
                    # parameters, then the shared (save_path, user_id, project_id, is_playground)
                    # tail. Returns a list of image identifiers, which api.py appends unchanged.
                    image_path = cdiff.inference(model_name, tumour, lobe, slice_location, tumour_size, save_path, user_id, project_id, is_playground)

                elif model_name == "braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)":
                    tumour = params["tumour"]
                    slice_orientation = params["slice_orientation"]
                    slice_location = params["slice_location"]

                    image_path = wg.inference(model_name, tumour, slice_orientation, slice_location, save_path, user_id, project_id, is_playground)

                elif model_name == "braingen_gan3d_BraTS_64_v1 (3D)":
                    resolution = params["resolution"]
                    image_path = gan3d.inference(model_name, resolution, save_path, user_id, project_id, is_playground)
                else:
                    image_path = ""
                    print("Wrong model called for inference")
            except Exception as e:
                print(f"Error in generate_image with model {model_name}: {str(e)}")
                image_path = []

            image_paths.append(image_path)
        
        return image_paths
    except Exception as e:
        print(f"Error in generate_image: {str(e)}")
        raise
  
