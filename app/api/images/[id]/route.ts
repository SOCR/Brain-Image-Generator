// app/api/generate/route.ts
import { createClient } from "@/utils/supabase/server";
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const supabase = await createClient();
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    
    if (userError || !user) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    const body = await request.json();
    const { user_id, project_id, model_name, n_images = 1, params } = body;
    
    if (!user_id || !project_id || !model_name) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }
    
    // Limit n_images to max 5
    const numImages = Math.min(Math.max(1, n_images), 5);
    
    // Make the request to your Python backend API
    const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';
    
    const response = await fetch(`${backendUrl}/generate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        user_id,
        project_id,
        model_name,
        n_images: numImages,
        params
      })
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      return NextResponse.json(
        { error: errorData.detail || "Failed to generate image" },
        { status: response.status }
      );
    }
    
    const data = await response.json();
    return NextResponse.json({ image_ids: data.image_paths });
    
  } catch (error) {
    console.error("Error generating image:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}