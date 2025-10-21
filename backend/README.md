# SOCR Image Generation API

A FastAPI-based REST API for generating medical images using various ML models, with storage in Supabase.

## Setup

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Create a `.env` file based on the `.env.example` template:
   ```bash
   cp .env.example .env
   ```

3. Edit the `.env` file with your Supabase credentials:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_service_key
   ```

## Usage

1. Start the API server:
   ```bash
   python api.py
   ```

2. The API will be available at `http://localhost:8000`

3. API Documentation will be available at `http://localhost:8000/docs`

## API Endpoints

- `GET /`: Check if the API is running
- `GET /models`: List available models
- `POST /generate`: Generate images based on specified parameters

### Example Request for Image Generation

```json
{
  "user_id": "user123",
  "project_id": "project456",
  "model_name": "braingen_WaveletGAN_Multicontrast_BraTS_v1 (2D)",
  "n_images": 1,
  "params": {
    "tumour": "With Tumor",
    "slice_orientation": "Axial",
    "slice_location": "Middle"
  }
}
```

## Storage

Generated images are stored in Supabase storage buckets:
- 2D images are stored in the "images" bucket
- 3D volumes are stored in the "volumes" bucket

The storage path follows this structure:
`{user_id}/{project_id}/{folder_name}/{filename}` 