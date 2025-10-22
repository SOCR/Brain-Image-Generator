import torch
import numpy as np
from inference.Generators import *
from PIL import Image
import base64
import io
import uuid
import os

import pywt
import pandas as pd
import matplotlib.pyplot as plt

def transform_to_real_images(coarse_coeffs, fine_coeffs, glob_min_max_csv):
    # Read the global min and max values from the CSV file
    df = pd.read_csv(glob_min_max_csv)
    glob_min = df['Global Min'].values
    glob_max = df['Global Max'].values

    # Initialize a list to store the output images
    real_images = []

    # Iterate over the 5 coarse coefficients
    for i in range(5):
        # Get the corresponding coarse and fine coefficients for each image
        coarse_coeff = coarse_coeffs[:, :, i]*0.5 + 0.5
        fine_coeff = fine_coeffs[:, :, i*3:i*3+3]*0.5 + 0.5  # 3 fine channels per coarse channel
        
        # Check and print coarse/fine_coeff before de-normalization for debugging
        print(f"Before De-normalization, Coarse {i}: Max: {np.max(coarse_coeff)}, Min: {np.min(coarse_coeff)}")

        # De-normalize coarse coefficient
        coarse_coeff = coarse_coeff * (glob_max[i] - glob_min[i]) + glob_min[i]

        # De-normalize fine coefficients (skipping every 3 steps correctly)
        fine_index_base = i * 3 + 5
        for j in range(3):
            fine_coeff[:, :, j] = fine_coeff[:, :, j] * (glob_max[fine_index_base + j] - glob_min[fine_index_base + j]) + glob_min[fine_index_base + j]
        
        # Ensure values stay within reasonable limits (optional debugging)
        print(f"After De-normalization, Coarse {i}: Max: {np.max(coarse_coeff)}, Min: {np.min(coarse_coeff)}")
        print(f"Fine {i}: Max: {np.max(fine_coeff)}, Min: {np.min(fine_coeff)}")

        # Combine coarse and fine coefficients
        coeff = [coarse_coeff, (fine_coeff[:, :, 0], fine_coeff[:, :, 1], fine_coeff[:, :, 2])]

        # Perform the inverse discrete wavelet transform
        wavelet_shape = 'db1'
        recon_img = pywt.idwt2(coeff, wavelet_shape)

        # Store the result
        real_images.append(recon_img)

    return real_images

def class_embedding(tumour,slice_orientation,slice_location):
  
   oreintation_dict = {"Axial":0, "Coronal":6, "Sagittal":12}
   location_dict    = {"Inferior" : 0, "Middle": 2, "Superior":4,
                      "Left":0, "Right":4, "Front":0, "Back":4}
   tumour_dict      = {"Without Tumor": 0, "With Tumor":1}
   
   class_label      = oreintation_dict[slice_orientation] + \
                      location_dict[slice_location] + tumour_dict[tumour]
                      
   return class_label
                  
def save_images(images, slice_orientation, save_path, user_id=None, project_id=None, model_name=None, params=None, is_playground=False):
    # Define the tags for the images
    tags = ['t1', 't1ce', 't2', 'flair', 'seg']

    # Generate a unique folder name
    unique_id = uuid.uuid4().hex
    folder_name = f"mri_{unique_id}"

    image_urls = []
    
    # Check if we're using Supabase or local storage
    if user_id and project_id:
        # Import here to avoid circular imports
        from supabase_storage import upload_image_to_supabase
        
        # Iterate over the images and save them to Supabase
        for i, image in enumerate(images):
            # Create the filename using the provided structure
            filename = f"{tags[i]}.png"
            
            # Rotate the image if needed
            if slice_orientation == 'Sagittal' or slice_orientation == 'Coronal':
                image = np.rot90(image)
                
            # Upload to Supabase and get the URL
            url = upload_image_to_supabase(
                image_data=image,
                filename=filename,
                user_id=user_id,
                project_id=project_id,
                folder_name=folder_name,
                model_name=model_name,
                parameters=params,
                is_playground=is_playground
            )
            image_urls.append(url)
            
        return image_urls
    else:
        # Fallback to local storage (original behavior)
        # Create the folder
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        # Iterate over the images and save them
        for i, image in enumerate(images):
            # Create the filename using the provided structure
            filename = f"{tags[i]}.png"
            filepath = os.path.join(folder_path, filename)

            # Rotate the image if needed
            if slice_orientation == 'Sagittal' or slice_orientation == 'Coronal':
                image = np.rot90(image)

            # Normalize the image to the range [0, 1] (important for `imshow` in matplotlib)
            norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))

            # Save the image using matplotlib
            plt.imsave(filepath, norm_image, cmap='gray')

        # Return the folder path
        return folder_path

def inference(model, tumour, slice_orientation, slice_location, save_path, user_id=None, project_id=None, is_playground=False):
    # Create a params dictionary to record what was used
    params = {
        "tumour": tumour,
        "slice_orientation": slice_orientation,
        "slice_location": slice_location
    }
    
    # Set up the device
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    latent_dim = 128
    n_classes = 18
    embedding_dim = 20

    # Load the appropriate model based on the selection
    if model == "braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)":
        coarse_model_path = os.path.join(os.getcwd(),"inference/model/generator_coarse_epoch_16_2500.pth")
        fine_model_path =  os.path.join(os.getcwd(),"inference/model/generator_fine_epoch_16_2500.pth")
        n_channels = 5

    # Initialize the generators
    generator_coarse_coeff = VanillaGenerator(n_channels, embedding_dim, latent_dim, n_classes).to(device)
    generator_fine_coeff   = SpatialAttentionUNetGenerator(input_channels=5, output_channels=15, feature_dim=128).to(device)

    # Load the pre-trained model weights
    generator_coarse_coeff.load_state_dict(torch.load(coarse_model_path, map_location=torch.device('cpu')), strict=False)
    generator_fine_coeff.load_state_dict(torch.load(fine_model_path, map_location=torch.device('cpu')), strict=False)

    generator_coarse_coeff.eval()
    generator_fine_coeff.eval()

    # Generate noise and labels for inference
    label = class_embedding(tumour, slice_orientation, slice_location)
    num_images = 1  # Generate one image

    with torch.no_grad():
        labels = torch.ones(num_images) * label
        labels = labels.to(device).unsqueeze(1).long()
        noise_vector = torch.randn(num_images, latent_dim, device=device)

        # Generate coarse and fine coefficients
        generated_c_image = generator_coarse_coeff((noise_vector, labels))
        generated_f_image = generator_fine_coeff(generated_c_image)

        image_c = generated_c_image[0].cpu().detach().permute(1, 2, 0).numpy()
        image_f = generated_f_image[0].cpu().detach().permute(1, 2, 0).numpy()

    # Transform the generated coefficients into real images
    images = transform_to_real_images(image_c, image_f, os.path.join(os.getcwd(),"inference/min_max_values.csv"))

    # Save the generated images and return the filenames
    image_filenames = save_images(images, slice_orientation, save_path, user_id, project_id, model, params, is_playground)

    return image_filenames
