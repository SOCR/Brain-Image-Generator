// app/api/images/project/[id]/route.ts
import { createClient } from "@/utils/supabase/server";
import { NextRequest, NextResponse } from "next/server";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const supabase = await createClient();

    // Get authenticated user
    const { data: { user }, error: userError } = await supabase.auth.getUser();

    if (userError || !user) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }

    const projectId = params.id;

    // Fetch all images for this project
    const { data: images, error } = await supabase
      .from('generated_images')
      .select('*')
      .eq('project_id', projectId)
      .eq('user_id', user.id)
      .order('created_at', { ascending: false });

    if (error) {
      console.error("Error fetching project images:", error);
      return NextResponse.json(
        { error: "Failed to fetch images" },
        { status: 500 }
      );
    }

    return NextResponse.json({ images: images || [] });

  } catch (error) {
    console.error("Error in project images API:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
