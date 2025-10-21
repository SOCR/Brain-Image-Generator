from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import uvicorn
import os
import image_generation

app = FastAPI(
    title="SOCR Image Generation API",
    description="API for generating medical images using various ML models",
    version="1.0.0"
)

# Configure CORS
# For initial deployment, FRONTEND_URL can be "*" to allow all origins
# After frontend is deployed, update to your specific Vercel URL for better security
frontend_url = os.getenv("FRONTEND_URL", "*")
allowed_origins = []

if frontend_url == "*":
    # Allow all origins (for testing and initial deployment)
    allowed_origins = ["*"]
else:
    # Specific origins for production
    allowed_origins = [
        "http://localhost:3000",  # Local development
        frontend_url,  # Your Vercel deployment
    ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageGenerationRequest(BaseModel):
    user_id: str
    project_id: str
    model_name: str
    n_images: int = 1
    params: Dict[str, Any]

class ImageGenerationResponse(BaseModel):
    image_paths: List[Union[str, List[str]]]

@app.get("/")
async def root():
    return {"message": "SOCR Image Generation API is running"}

@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {
        "status": "healthy",
        "service": "SOCR Image Generation API",
        "version": "1.0.0"
    }

@app.post("/generate", response_model=ImageGenerationResponse)
async def generate_images(request: ImageGenerationRequest):
    try:
        image_paths = image_generation.generate_image(
            request.user_id,
            request.project_id,
            request.model_name,
            request.n_images,
            request.params
        )
        
        # Flatten the result if needed - ensuring we return a consistent format
        flattened_paths = []
        for path in image_paths:
            if isinstance(path, list):
                # If it's already a list of URLs, add it
                flattened_paths.append(path)
            else:
                # If it's a single path string, wrap it in a list
                flattened_paths.append([path])
                
        return {"image_paths": flattened_paths}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")

@app.get("/models")
async def list_models():
    models = [
        "braingen_GAN_seg_TCGA_v1 (2D)",
        "braingen_cGAN_Multicontrast_BraTS_v1 (2D)",
        "braingen_cGAN_Multicontrast_seg_BraTS_v1 (2D)",
        "braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)",
        "braingen_diffuser_BraTS_v1 (2D)",
        "braingen_gan3d_BraTS_64_v1 (3D)"
    ]
    return {"models": models}

@app.get("/check-models")
async def check_model_files():
    """Check if all required model files exist"""
    model_files = {
        "GAN_SEG_TCGA": os.path.join(os.getcwd(), "inference/model/generator_epoch_200_seg.pth"),
        "WAVELET_GAN_COARSE": os.path.join(os.getcwd(), "inference/model/generator_coarse_epoch_16_2500.pth"),
        "WAVELET_GAN_FINE": os.path.join(os.getcwd(), "inference/model/generator_fine_epoch_16_2500.pth"),
        "GAN_3D_32": os.path.join(os.getcwd(), "inference/model/generator_epoch_6100.pth"),
        "GAN_3D_64": os.path.join(os.getcwd(), "inference/model/generator_epoch_3d64.pth")
    }
    
    results = {}
    for name, path in model_files.items():
        results[name] = {
            "path": path,
            "exists": os.path.isfile(path)
        }
    
    return {"model_files": results}

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 