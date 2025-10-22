# image_generation.py
import numpy as np
import matplotlib.pyplot as plt
import os
import uuid
import inference.wgan_inference as wg
import inference.cgan_inference as cgv1
import inference.cgan_inference_v2 as cgv2
import inference.gan_3d_inference_v1 as gan3d


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

        model_list = ["braingen_GAN_seg_TCGA_v1","braingen_cGAN_Multicontrast_BraTS_v1",
                      "braingen_cGAN_Multicontrast_seg_BraTS_v1", "braingen_WaveletGAN_Multicontrast_BraTS_v1",
                      "braingen_diffuser_BraTS_v1","braingen_gan3d_BraTS_64_v1"]
        image_paths = []
        
        print("in python", params)

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
  
