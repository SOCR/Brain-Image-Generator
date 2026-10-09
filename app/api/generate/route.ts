import { createClient } from "@/utils/supabase/server";
import { callSpace } from "@/lib/hfSpace";
import { NextRequest, NextResponse } from "next/server";

// A diffusion request waits for a ZeroGPU slot, runs 200 DDIM steps and uploads to Supabase in
// one call. Vercel's default function timeout can be as short as 10 s; 60 s is allowed on every plan.
export const maxDuration = 60;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { user_id, project_id, model_name, n_images = 1, params, is_playground = false } = body;
    
    // Check authentication only if not in playground mode
    if (!is_playground) {
      const supabase = await createClient();
      const { data: { session } } = await supabase.auth.getSession();
      
      if (!session) {
        return NextResponse.json(
          { error: "Unauthorized" },
          { status: 401 }
        );
      }
    }
    
    if (!user_id || !project_id || !model_name) {
      return NextResponse.json(
        { error: "Missing required fields" },
        { status: 400 }
      );
    }
    
    const payload = { user_id, project_id, model_name, n_images, params, is_playground };
    let data;

    if (process.env.HF_SPACE_URL) {
      // Hugging Face Space backend (backend/hf_space/app.py): same payload, same response shape.
      // The token decides whose daily ZeroGPU quota pays for diffusion, and it goes on EVERY
      // request, playground included, so signed-out visitors get the account's quota and priority.
      // ACCEPTED RISK (DEPLOY.md §6): the playground skips the session check above, so a script
      // could use up the day's quota. Escape hatch, one line:
      //   const token = is_playground ? undefined : process.env.HF_TOKEN;
      try {
        data = await callSpace(process.env.HF_SPACE_URL, "generate", [payload], process.env.HF_TOKEN);
      } catch (e) {
        // The Space's reason (e.g. "GPU quota exceeded ... try again in 3:41:09") goes to the toast.
        return NextResponse.json(
          { error: e instanceof Error ? e.message : "Failed to generate image" },
          { status: 502 }
        );
      }
    } else {
      // Make the request to your Python backend API
      const backendUrl = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

      const response = await fetch(`${backendUrl}/generate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        return NextResponse.json(
          { error: errorData.detail || "Failed to generate image" },
          { status: response.status }
        );
      }

      data = await response.json();
    }
    return NextResponse.json({ image_ids: data.image_paths });
    
  } catch (error) {
    console.error("Error generating image:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
} 