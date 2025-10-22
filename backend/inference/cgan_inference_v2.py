import torch
import numpy as np
from inference.Generators import *
from PIL import Image
import base64
import io
import uuid
import os
import matplotlib.pyplot as plt

def class_embedding(tumour, slice_orientation, slice_location):
    # Map the tumour class
    if tumour == "With Tumor":
        tumour_class = 1
    else:
        tumour_class = 0
    
    # Map the slice orientation
    if slice_orientation == "Axial":
        orientation_class = 0
    elif slice_orientation == "Coronal":
        orientation_class = 1
    elif slice_orientation == "Sagittal":
        orientation_class = 2
    else:
        orientation_class = 0
    
    # Map the slice location
    if slice_location == "Anterior" or slice_location == "Superior":
        location_class = 0
    elif slice_location == "Middle":
        location_class = 1
    elif slice_location == "Posterior" or slice_location == "Inferior":
        location_class = 2
    else:
        location_class = 1  # Default to middle
    
    # Calculate the final class label
    class_label = tumour_class * 9 + orientation_class * 3 + location_class
    return class_label
                  
def save_images(images, slice_orientation, save_path, user_id=None, project_id=None, model_name=None, params=None, is_playground=False):
    # Define the tags for the images
    tags = ['t1', 't2', 'flair', 'seg']

    # Generate a unique folder name
    unique_id = uuid.uuid4().hex
    folder_name = f"mri_{unique_id}"

    # Check if we're using Supabase or local storage
    if user_id and project_id:
        # Import here to avoid circular imports
        from supabase_storage import upload_image_to_supabase
        
        image_urls = []
        # Iterate over the images and save them to Supabase
        for i in range(images.shape[-1]):  # Loop through the last axis
            image = images[:, :, i]
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

        image_paths = []
        for i in range(images.shape[-1]):
            image = images[:, :, i]
            filename = f"{tags[i]}.png"
            filepath = os.path.join(folder_path, filename)

            # Rotate the image if needed
            if slice_orientation == 'Sagittal' or slice_orientation == 'Coronal':
                image = np.rot90(image)

            # Normalize the image to the range [0, 1] (important for `imshow` in matplotlib)
            norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            # Save the image using matplotlib
            plt.imsave(filepath, norm_image, cmap='gray')
            image_paths.append(filepath)
        
        return image_paths

def inference(model, tumour, slice_orientation, slice_location, save_path, user_id=None, project_id=None, is_playground=False):
    # Create a params dictionary to record what was used
    params = {
        "tumour": tumour,
        "slice_orientation": slice_orientation,
        "slice_location": slice_location
    }
    
    # torch.manual_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    batch_size = 128
    img_save_pth = "images"
    latent_dim = 100
    n_classes = 18
    embedding_dim = 20

    print("Selected model ",model)
    
    label = class_embedding(tumour,slice_orientation,slice_location)
    
    if(model == "braingen_cGAN_Multicontrast_BraTS_v1 (2D)"):
      # Root directory for the dataset
      model_path =  os.path.join(os.getcwd(),"inference/model/generator_epoch_145.pth")
      n_channels = 3
      
    elif(model == "braingen_cGAN_Multicontrast_seg_BraTS_v1 (2D)"):
      model_path =  os.path.join(os.getcwd(),"inference/model/generator_epoch_95_seg.pth")
      n_channels = 4
    image_shape = (n_channels, 128, 128)
    image_dim = int(np.prod(image_shape))
      
    # generator = Generator(n_classes, embedding_dim,latent_dim,n_channels).to(device)
    generator = VanillaGenerator(n_channels, embedding_dim, latent_dim, n_classes).to(device)

    generator.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')), strict=False)
    print("Model loaded....................")
    num_images = 1
    generator.eval()
  

    print("Inference started...............")

    # Generate images using the generator
    with torch.no_grad():
        labels = torch.ones(num_images) * label
        labels = labels.to(device).unsqueeze(1).long()
        noise_vector = torch.randn(num_images, latent_dim, device=device)
        generated_images = generator((noise_vector, labels))
        images = generated_images[0].permute(1, 2, 0).cpu().numpy()  # Change tensor to numpy array
        images = (images + 1) / 2.0 * 255.0  # Rescale pixel values

    # Save the images and return the filenames
    image_filenames = save_images(images, slice_orientation, save_path, user_id, project_id, model, params, is_playground)

    return image_filenames
