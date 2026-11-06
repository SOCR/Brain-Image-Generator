import { createClient } from "@/utils/supabase/server";
import { callSpace } from "@/lib/hfSpace";
import { NextRequest, NextResponse } from "next/server";

// A diffusion request waits for a ZeroGPU slot, runs 200 DDIM steps and uploads to Supabase in
// one call. Vercel's default function timeout can be as short as 10 s; 60 s is allowed on every plan.
export const maxDuration = 60;

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { model_name, n_images = 1, params, is_playground = false } = body;

    // Whose library the images land in comes from the session, never from the request body: the
    // backend publishes with a key that bypasses row-level security, so a user_id taken from the
    // client would let anyone write into any account. The playground always uses the guest folder.
    let user_id = "guest";
    let project_id = "playground";
    if (!is_playground) {
      const supabase = await createClient();
      // getUser(), not getSession(): getSession() trusts the cookie without asking Supabase.
      const { data: { user } } = await supabase.auth.getUser();
      if (!user) {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
      }
      const { data: project } = await supabase
        .from("projects").select("id")
        .eq("id", body.project_id).eq("user_id", user.id).maybeSingle();
      if (!project) {
        return NextResponse.json({ error: "Project not found" }, { status: 404 });
      }
      user_id = user.id;
      project_id = project.id;
    }

    if (!model_name) {
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
      // api_secret is how the Space tells this route from a direct caller (backend/hf_space/app.py).
      try {
        const args = [{ ...payload, api_secret: process.env.SPACE_API_SECRET }];
        data = await callSpace(process.env.HF_SPACE_URL, "generate", args, process.env.HF_TOKEN);
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