import torch
import numpy as np
from inference.Generators import *
from PIL import Image
import base64
import io
import uuid
import os
import nibabel as nib


def save_images(generated_image, save_path, resolution, user_id=None, project_id=None, model_name=None, params=None):
    unique_id = uuid.uuid4().hex
    if(resolution == "32"):
      folder_name = f"{'mri32'}_{unique_id}"
    elif(resolution == "64"):
      folder_name = f"{'mri64'}_{unique_id}"
    
    # Check if we're using Supabase or local storage
    if user_id and project_id:
        # Import here to avoid circular imports
        from supabase_storage import upload_nifti_to_supabase
        
        # Create filename
        filename = 'mri3D.nii.gz'
        
        # Upload to Supabase and get the URL
        url = upload_nifti_to_supabase(
            nifti_data=generated_image,
            filename=filename,
            user_id=user_id,
            project_id=project_id,
            folder_name=folder_name,
            model_name=model_name,
            parameters=params
        )
        
        return url
    else:
        # Fallback to local storage (original behavior)
        folder_path = os.path.join(save_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        
        image_path = os.path.join(folder_path, 'mri3D.nii.gz')
        
        # Generate a random UUID and convert it to a hexadecimal string
        affine = np.eye(4)
        new_img = nib.Nifti1Image(generated_image, affine=affine)
        nib.save(new_img, image_path)
        
        # Return the folder name
        return folder_path
  
  


def inference(model, resolution, save_path, user_id=None, project_id=None):
    # torch.manual_seed(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    batch_size = 128

    cube_len = 32
    epoch_count = 20

    condition_count = 1

    print("Selected model ",model)
    
    # label = class_embedding(tumour,slice_orientation,slice_location)
    
    if(resolution == "32"):
      print("model 64 selected")
      noise_size = 200
      model_path = os.path.join(os.getcwd(),"inference/model/generator_epoch_6100.pth")
      generator = Generator3D_32(noise_size=(noise_size + 1), cube_resolution=cube_len) 
    elif(resolution == "64"):
      print("model 64 selected")
      noise_size = 400
      model_path = os.path.join(os.getcwd(),"inference/model/generator_epoch_3d64.pth")
      generator = Generator3D_64(noise_size=(noise_size + 1), cube_resolution=cube_len) 
    

    image_shape = (cube_len,cube_len,cube_len)
   
    generator.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')), strict=False)
    print("Model loaded....................")
    num_images = 2
    generator.eval()
  

    # Generate random noise
    z = torch.randn([num_images,noise_size], device=device)

    print("Inference started...............")

  
    generated_images = generator(z, 0)
    print(generated_images.shape)

    generated_image = generated_images.detach().cpu().numpy()[0]
    
    # Create a params dictionary to record what was used
    params = {"resolution": resolution}
    
    image_path = save_images(generated_image, save_path, resolution, user_id, project_id, model, params)
    
    # return image_filenames
    return image_path
