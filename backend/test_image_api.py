import requests
import json
import time

# API base URL
BASE_URL = "http://localhost:8000"

def test_api_connection():
    """Test if the API is running"""
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"API Connection: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Connection Error: {str(e)}")
        return False

def get_available_models():
    """Get list of available models"""
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"Available Models: {len(models)}")
            for i, model in enumerate(models):
                print(f"  {i+1}. {model}")
            return models
        else:
            print(f"Error getting models: {response.status_code} - {response.json()}")
            return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def generate_image(user_id, project_id, model_name, params, n_images=1):
    """Generate images with the specified parameters"""
    try:
        request_data = {
            "user_id": user_id,
            "project_id": project_id,
            "model_name": model_name,
            "n_images": n_images,
            "params": params
        }
        
        print(f"\nRequesting image generation with {model_name}...")
        print(f"Parameters: {json.dumps(params, indent=2)}")
        
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/generate", 
            json=request_data
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            image_paths = response.json().get("image_paths", [])
            print(f"Success! Generated {len(image_paths)} images")
            
            for i, path_item in enumerate(image_paths):
                if isinstance(path_item, list):
                    print(f"  Image set {i+1}:")
                    for j, url in enumerate(path_item):
                        print(f"    {j+1}. {url}")
                else:
                    print(f"  {i+1}. {path_item}")
                    
            return image_paths 
        else:
            print(f"Error generating images: {response.status_code}")
            try:
                error_detail = response.json().get("detail", "Unknown error")
                print(f"Error details: {error_detail}")
            except:
                print(f"Response: {response.text}")
            return []
    except Exception as e:
        print(f"Error: {str(e)}")
        return []

def run_tests():
    """Run a series of tests with different models and parameters"""
    if not test_api_connection():
        print("Cannot connect to API. Exiting.")
        return
    
    models = get_available_models()
    if not models:
        print("No models available. Exiting.")
        return
    
    # Test user and project IDs
    user_id = f"beb6753f-396a-443c-9daa-fb5314dc6758"
    project_id = f"4a35aa93-8ab8-4bbb-be23-ae3ed38470d8"
    
    print(f"\nUsing test user ID: {user_id}")
    print(f"Using test project ID: {project_id}")
    
    # Test 1: 2D GAN with tumor
    if "braingen_GAN_seg_TCGA_v1 (2D)" in models:
        generate_image(
            user_id=user_id,
            project_id=project_id,
            model_name="braingen_GAN_seg_TCGA_v1 (2D)",
            params={"tumour": "With Tumor"}
        )
    
    # Test 2: 2D Wavelet GAN with specific orientation
    # if "braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)" in models:
    #     generate_image(
    #         user_id=user_id,
    #         project_id=project_id,
    #         model_name="braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)",
    #         params={
    #             "tumour": "With Tumor",
    #             "slice_orientation": "Axial",
    #             "slice_location": "Middle"
    #         }
    #     )
    
    # # Test 3: 3D GAN
    # if "braingen_gan3d_BraTS_64_v1 (3D)" in models:
    #     generate_image(
    #         user_id=user_id,
    #         project_id=project_id,
    #         model_name="braingen_gan3d_BraTS_64_v1 (3D)",
    #         params={"resolution": "64"}
    #     )
    
    # # Test 4: Try the diffuser model (which might have compatibility issues)
    # if "braingen_diffuser_BraTS_v1 (2D)" in models:
    #     generate_image(
    #         user_id=user_id,
    #         project_id=project_id,
    #         model_name="braingen_diffuser_BraTS_v1 (2D)",
    #         params={}
    #     )
    
    # Test 5: cGAN Multicontrast
    # if "braingen_cGAN_Multicontrast_BraTS_v1 (2D)" in models:
    #     generate_image(
    #         user_id=user_id,
    #         project_id=project_id,
    #         model_name="braingen_cGAN_Multicontrast_BraTS_v1 (2D)",
    #         params={
    #             "tumour": "Without Tumor",
    #             "slice_orientation": "Sagittal",
    #             "slice_location": "Posterior"
    #         }
    #     )

if __name__ == "__main__":
    run_tests()