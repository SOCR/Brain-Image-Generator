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
# frontend_url = os.getenv("FRONTEND_URL", "*")
# allowed_origins = []

# if frontend_url == "*":
    # Allow all origins (for testing and initial deployment)
#         allowed_origins = [
#     "http://localhost:3000",
#     "http://127.0.0.1:3000",
#     "http://localhost:8080",
#     "http://127.0.0.1:8080",
#     "https://brain-image-generator.vercel.app",
#     "https://ai-bot.statisticalcomputing.org",
#         ]
# allowed_origins = ["*"]
# else:
#     allowed_origins = [
#        "http://localhost:3000",  # Local development
#        frontend_url,  # Your Vercel deployment
#     ]

allowed_origins = [
"http://localhost:3000",
"http://127.0.0.1:3000",
"http://localhost:8080",
"http://127.0.0.1:8080",
"https://brain-image-generator.vercel.app",
"https://ai-bot.statisticalcomputing.org",
"https://braingen.statisticalcomputing.org/",
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
    is_playground: bool = False

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
            request.params,
            request.is_playground
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
        "braingen_gan3d_BraTS_64_v1 (3D)",
        # Atlas-guided conditional diffuser. NOTE: this endpoint is cosmetic - the frontend
        # hardcodes its own model list in ParameterControlPanel.tsx and never calls GET /models
        # (which is why this list already advertises braingen_diffuser_BraTS_v1, a model the UI
        # does not offer). Kept in sync anyway so the API self-describes correctly.
        "braingen_CondDiffuser_BraTS_v1 (2D)"
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
        "GAN_3D_64": os.path.join(os.getcwd(), "inference/model/generator_epoch_3d64.pth"),
        # --- braingen_CondDiffuser_BraTS_v1 assets -------------------------------------------
        # Unlike the GANs, this model needs more than a weights file: it also needs conditioning
        # (a patient T1, a tumour mask and 6 atlas lobe maps) that the website has no way to
        # synthesize, so a preprocessed bank of val-split slices ships alongside the checkpoint.
        # Both are delivered out of band, exactly like the .pth files - neither is in git.
        # Listing the paths here is the only way to confirm from outside the container that the
        # out-of-band copy actually landed before docker build.
        #
        # THESE PATHS MUST STAY IN LOCKSTEP WITH inference/conddiff_inference.py's own constants
        # (CKPT_CANDIDATES and MANIFEST_PATH) - that module is what actually reads them, and this
        # endpoint is only useful if it checks the same places. The checkpoint lives in
        # inference/model/ with the GAN weights; the conditioning bundle produced by
        # scripts/export_for_website.py lands whole in inference/conddiff_bundle/.
        # The checkpoint may arrive as either the original .pt pickle or an fp16 safetensors
        # export, so both are listed and only ONE of the two needs to exist -- and the bundle
        # carries its own copy of the .pt, which is the third candidate the module accepts.
        "CONDDIFF_CHECKPOINT_PT": os.path.join(os.getcwd(), "inference/model/diffusion_ema.pt"),
        "CONDDIFF_CHECKPOINT_SAFETENSORS": os.path.join(os.getcwd(), "inference/model/diffusion_ema_fp16.safetensors"),
        "CONDDIFF_CHECKPOINT_IN_BUNDLE": os.path.join(os.getcwd(), "inference/conddiff_bundle/diffusion_ema.pt"),
        "CONDDIFF_BUNDLE_MANIFEST": os.path.join(os.getcwd(), "inference/conddiff_bundle/manifest.json"),
        "CONDDIFF_GALLERY_MANIFEST": os.path.join(os.getcwd(), "inference/conddiff_gallery/manifest.json")
    }
    
    results = {}
    for name, path in model_files.items():
        results[name] = {
            "path": path,
            "exists": os.path.isfile(path)
        }
    
    return {"model_files": results}

@app.get("/conddiff-selftest")
async def conddiff_selftest():
    """Diagnose the conditional diffuser without a shell on the container.

    /check-models above only answers "does the file exist". This answers the questions that
    actually decide whether a request will succeed: did the startup asset check pass, how many
    conditioning slices did the manifest parse to, which lobes and slice levels does the bank
    actually cover, and how many DDIM steps is this deployment configured for.

    This matters because image_generation.py catches every inference exception, prints it, and
    substitutes an empty list - so a broken deployment returns HTTP 200 with a green "Success"
    toast and a blank viewer. Without this endpoint the only evidence is container stdout.

    Imported inside the handler so that a problem in the diffuser module can never break
    api.py's own import; it runs no model and loads no weights.
    """
    import inference.conddiff_inference as cdiff
    return cdiff.selftest()

@app.get("/conddiff-cells")
async def conddiff_cells():
    """Which (Lobe, Slice Location) combinations this deployment can actually generate.

    The frontend calls this on mount and GREYS OUT the rest. It is not a nicety: the real
    conditioning bank fills only 11 of the 18 combinations, because the anatomy does not
    cooperate - the cerebellum does not appear in superior slices, and the insula is small
    enough that it may never clear the lobe-area gate at any level. Offering all 18 means a
    user can pick one of the 7 that do not exist, and image_generation.py's blanket except
    turns that into HTTP 200 with an empty image list: a green "Success" toast over a blank
    viewer, with the only evidence in container stdout.

    Deliberately NOT solved by substituting a nearby slice level on the backend. That would
    label an image with a location it was not generated at, and the whole point of this model
    on this site is that the location is a controlled, recorded claim.

    Values are the exact dropdown strings the UI uses ("Frontal", "Middle"), so the frontend
    compares them directly with no case-mapping layer to get wrong.

    Imported inside the handler for the same reason as /conddiff-selftest: a problem in the
    diffuser module must never break api.py's own import. It runs no model and loads no weights.
    """
    import inference.conddiff_inference as cdiff
    return cdiff.supported_combinations()

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 
