import torch
import numpy as np
from inference.Generators import *
from PIL import Image
import base64
import io
import uuid
import os
import matplotlib.pyplot as plt

# Wrap diffusers import in a try-except to handle incompatible PyTorch versions
try:
    from diffusers import DDIMScheduler
    from diffusers import UNet2DModel
    DIFFUSERS_AVAILABLE = True
except Exception as e:
    print(f"Diffusers import failed: {str(e)}")
    DIFFUSERS_AVAILABLE = False

def save_images(images, save_path, user_id=None, project_id=None, model_name=None, params=None, is_playground=False):
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
            # Normalize the image to the range [0, 1] (important for `imshow` in matplotlib)
            norm_image = (image - np.min(image)) / (np.max(image) - np.min(image))
            
            # Save the image using matplotlib
            plt.imsave(filepath, norm_image, cmap='gray')
            image_paths.append(filepath)
        
        return image_paths

def inference_diffuser(save_path, user_id=None, project_id=None, is_playground=False):
    if not DIFFUSERS_AVAILABLE:
        print("Warning: The diffusers library could not be imported due to compatibility issues with PyTorch.")
        print("To use this model, please upgrade PyTorch or downgrade diffusers.")
        
        # Create a message image explaining the error
        error_img = np.zeros((128, 128, 4), dtype=np.uint8)
        
        # Return the error image or an empty path
        if user_id and project_id:
            from supabase_storage import upload_image_to_supabase
            
            # Create a unique folder name
            unique_id = uuid.uuid4().hex
            folder_name = f"error_{unique_id}"
            
            url = upload_image_to_supabase(
                image_data=error_img,
                filename="error.png",
                user_id=user_id,
                project_id=project_id,
                folder_name=folder_name
            )
            return [url]
        else:
            # Create a local error folder
            unique_id = uuid.uuid4().hex
            folder_name = f"error_{unique_id}"
            folder_path = os.path.join(save_path, folder_name)
            os.makedirs(folder_path, exist_ok=True)
            
            return folder_path
    
    # Original implementation continues below
    repo_id = os.path.join(os.getcwd(),"inference/model/ddim-brain-128/unet")
    model = UNet2DModel.from_pretrained(repo_id)

    scheduler = DDIMScheduler()
    scheduler.set_timesteps(num_inference_steps=50)


    # torch.manual_seed(0)

    noisy_sample = torch.randn(
        1, model.config.in_channels, model.config.sample_size, model.config.sample_size
    )
    model.to("cuda")
    noisy_sample = noisy_sample.to("cuda")

    sample = noisy_sample

    for i, t in enumerate(scheduler.timesteps):
        # 1. predict noise residual
        with torch.no_grad():
            residual = model(sample, t).sample

        # 2. compute less noisy image and set x_t -> x_t-1
        sample = scheduler.step(residual, t, sample).prev_sample

    image_processed = sample.cpu().permute(0, 2, 3, 1)
    image_processed = (image_processed + 1.0) * 127.5
    image_processed = image_processed.numpy().astype(np.uint8)

    # Save the images and return the filenames
    image_filenames = save_images(image_processed, save_path, user_id, project_id, None, None, is_playground)

    return image_filenames


def inference(tumour, model, save_path, user_id=None, project_id=None, is_playground=False):
    # Create a params dictionary to record what was used
    params = {"tumour": tumour}
    
    # torch.manual_seed(1)
    label = 1 if tumour == "With Tumor" else 0

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    batch_size = 128
    latent_dim = 100
    n_classes = 2
    embedding_dim = 100

    print("Selected model ",model)
    
    
    if(model == "brainGen_v1"):
      # Root directory for the dataset
      model_path = "model/generator_epoch_950.pth"
      n_channels = 3
      
    elif(model == "braingen_GAN_seg_TCGA_v1 (2D)"):
      model_path = os.path.join(os.getcwd(),"inference/model/generator_epoch_200_seg.pth")
      n_channels = 4
    image_shape = (n_channels, 128, 128)
    image_dim = int(np.prod(image_shape))
      
    # generator = VanillaGenerator(n_classes, embedding_dim,latent_dim,n_channels).to(device)
    generator = VanillaGenerator(n_channels, embedding_dim, latent_dim, n_classes).to(device)
                # ouput_channels, embedding_dim, latent_dim, n_classes
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
    image_filenames = save_images(images, save_path, user_id, project_id, model, params, is_playground)

    return image_filenames

