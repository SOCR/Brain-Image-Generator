import { createClient } from "@/utils/supabase/server";
import { NextRequest, NextResponse } from "next/server";

export async function GET(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const supabase = await createClient();
    const { data: { session } } = await supabase.auth.getSession();
    
    if (!session) {
      return NextResponse.json(
        { error: "Unauthorized" },
        { status: 401 }
      );
    }
    
    const { data: images, error } = await supabase
      .from('generated_images')
      .select('*')
      .eq('project_id', params.id)
      .eq('user_id', session.user.id)
      .order('created_at', { ascending: false })
      .limit(10);
    
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