import { notFound, redirect } from 'next/navigation';
import { getProjectById } from '@/lib/projects';
import { createClient } from "@/utils/supabase/server";
import { ProjectLayoutComponent } from '@/components/layouts/project-layout';
import GenerationPageClient from '@/components/projects/GenerationPageClient';
import { Suspense } from 'react';

interface ProjectPageProps {
  params: {
    id: string;
  };
}

export default async function ProjectPage({ params }: ProjectPageProps) {
  try {
    const supabase = await createClient();
    const { data: { session } } = await supabase.auth.getSession();
    const project = await getProjectById(params.id);

    if (!session) {
      redirect('/sign-in');
    }

    return (
      <ProjectLayoutComponent user={session.user} project={project}>
        <Suspense fallback={
          <div className="flex-1 flex items-center justify-center bg-white">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 mx-auto mb-4"></div>
              <p className="text-gray-500">Loading generation panel...</p>
            </div>
          </div>
        }>
          <GenerationPageClient
            projectId={params.id}
            userId={session.user.id}
          />
        </Suspense>
      </ProjectLayoutComponent>
    );
  } catch (error) {
    return notFound();
  }
}
