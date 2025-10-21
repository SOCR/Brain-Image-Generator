import os
import io
import uuid
from supabase import create_client
from dotenv import load_dotenv
import base64
from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import json

# Load environment variables from .env file
load_dotenv()

# Initialize Supabase client
supabase_url = os.getenv('SUPABASE_URL')
supabase_key = os.getenv('SUPABASE_KEY')
supabase = create_client(supabase_url, supabase_key)

def add_to_database(file_url, user_id, project_id, model_name, parameters_used, image_name):
    """
    Add the generated image record to the database
    
    Args:
        file_url: URL of the uploaded file
        user_id: User ID (UUID)
        project_id: Project ID (UUID)
        model_name: Name of the model used
        parameters_used: Parameters used for generation
        image_name: Name of the image
        
    Returns:
        The ID of the inserted record
    """
    try:
        # Convert parameters to JSONB if it's not already a string
        if not isinstance(parameters_used, str):
            parameters_used = json.dumps(parameters_used)
            
        # Insert record into the generated_images table
        data = {
            "name": image_name,
            "project_id": project_id,
            "file_path": file_url,
            "parameters_used": parameters_used,
            "user_id": user_id,
            # Using a placeholder for model_id since we're not concerned with it now
            "model_id": "00000000-0000-0000-0000-000000000000"
        }
        
        result = supabase.table("generated_images").insert(data).execute()
        
        # Check if insertion was successful
        if len(result.data) > 0:
            return result.data[0].get("id")
        else:
            print("No data returned from insert operation")
            return None
            
    except Exception as e:
        print(f"Error inserting record to database: {str(e)}")
        return None

def upload_image_to_supabase(image_data, filename, user_id, project_id, folder_name, bucket_name="images", model_name=None, parameters=None):
    """
    Upload an image to Supabase Storage and record it in the database
    
    Args:
        image_data: PIL Image or numpy array
        filename: Name of the file
        user_id: User ID
        project_id: Project ID
        folder_name: Folder name for the image
        bucket_name: Supabase bucket name
        model_name: Name of the ML model used
        parameters: Parameters used for generation
        
    Returns:
        URL of the uploaded file
    """
    # Create path for the file
    path = f"{user_id}/{project_id}/{folder_name}/{filename}"
    
    # Convert image to bytes
    if isinstance(image_data, np.ndarray):
        # Convert numpy array to PIL Image
        img = Image.fromarray(
            (((image_data - np.min(image_data)) / 
              (np.max(image_data) - np.min(image_data))) * 255).astype(np.uint8)
        )
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        file_content = buffer.getvalue()
    else:
        # Assume it's already in a format supabase can handle
        file_content = image_data
    
    # Upload file with better error handling
    try:
        result = supabase.storage.from_(bucket_name).upload(
            path=path,
            file=file_content,
            file_options={"content-type": "image/png"}
        )
        
        # Get public URL
        file_url = supabase.storage.from_(bucket_name).get_public_url(path)
        
        # Add record to database if we have a model name
        if model_name:
            # Create a meaningful image name
            image_name = f"{os.path.splitext(filename)[0]} - {folder_name}"
            
            # Add to database - pass file_url not path
            id = add_to_database(
                file_url=file_url,
                user_id=user_id,
                project_id=project_id,
                model_name=model_name,
                parameters_used=parameters or {},
                image_name=image_name
            )
            
        return id
    except Exception as e:
        print(f"Error uploading to Supabase: {str(e)}")
        # Create local fallback storage and return local path 
        local_dir = os.path.join(os.getcwd(), "data", user_id, project_id, folder_name)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        
        if isinstance(image_data, np.ndarray):
            # Save locally using matplotlib
            norm_image = (image_data - np.min(image_data)) / (np.max(image_data) - np.min(image_data))
            plt.imsave(local_path, norm_image, cmap='gray')
        
        return local_path

def upload_nifti_to_supabase(nifti_data, filename, user_id, project_id, folder_name, bucket_name="volumes", model_name=None, parameters=None):
    """
    Upload a NIFTI file to Supabase Storage and record it in the database
    """
    # Create path for the file
    path = f"{user_id}/{project_id}/{folder_name}/{filename}"
    
    # Convert nifti to bytes - first save to a temporary file, then read it back
    try:
        # Create a temporary file
        temp_file = os.path.join(os.getcwd(), "temp_nifti.nii.gz")
        
        # Save the NIFTI data to the temporary file
        affine = np.eye(4)
        nii_img = nib.Nifti1Image(nifti_data, affine=affine)
        nib.save(nii_img, temp_file)
        
        # Read the file back as bytes
        with open(temp_file, 'rb') as f:
            file_content = f.read()
        
        # Clean up the temporary file
        os.remove(temp_file)
        
        # Upload file
        result = supabase.storage.from_(bucket_name).upload(
            path=path,
            file=file_content,
            file_options={"content-type": "application/octet-stream"}
        )
        
        # Get public URL
        file_url = supabase.storage.from_(bucket_name).get_public_url(path)
        
        # Add record to database if we have a model name
        if model_name:
            # Create a meaningful image name
            image_name = f"3D Volume - {folder_name}"
            
            # Add to database
            id = add_to_database(
                file_url=file_url,
                user_id=user_id,
                project_id=project_id,
                model_name=model_name,
                parameters_used=parameters or {},
                image_name=image_name
            )
        
        return id
    except Exception as e:
        print(f"Error uploading NIFTI to Supabase: {str(e)}")
        # Create local fallback storage and return local path
        local_dir = os.path.join(os.getcwd(), "data", user_id, project_id, folder_name)
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, filename)
        
        # Save the NIFTI file locally
        affine = np.eye(4)
        nii_img = nib.Nifti1Image(nifti_data, affine=affine)
        nib.save(nii_img, local_path)
        
        return local_path 