// app/api/images/batch/route.ts
import { createClient } from "@/utils/supabase/server";
import { NextRequest, NextResponse } from "next/server";

export async function POST(request: NextRequest) {
  try {
    const supabase = await createClient();
    
    // Get authenticated user with the more secure method
    const { data: { user }, error: userError } = await supabase.auth.getUser();
    
    if (userError || !user) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    // Get IDs from request body
    const { ids } = await request.json();
    
    if (!ids || !Array.isArray(ids) || ids.length === 0) {
      return NextResponse.json(
        { error: "Invalid request. Expected an array of image IDs." },
        { status: 400 }
      );
    }
    
    // Fetch multiple images in one query
    const { data: images, error } = await supabase
      .from('generated_images')
      .select('*')
      .in('id', ids)
      .eq('user_id', user.id);
    
    if (error) {
      return NextResponse.json(
        { error: "Failed to fetch images" },
        { status: 500 }
      );
    }
    
    return NextResponse.json({ images });
    
  } catch (error) {
    console.error("Error fetching images:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}