import { notFound, redirect } from 'next/navigation';
import { getProjectById } from '@/lib/projects';
import { createClient } from "@/utils/supabase/server";
import { ProjectLayoutComponent } from '@/components/layouts/project-layout';
import GenerationPageClient from '@/components/projects/GenerationPageClient';
import { Suspense } from 'react';

interface ProjectPageProps {
  params: Promise<{
    id: string;
  }>;
}

export default async function ProjectPage({ params }: ProjectPageProps) {
  // Await params first (Next.js 15 requirement)
  const { id } = await params;
  
  // Check if this is playground mode BEFORE checking auth
  const isPlayground = id === 'playground';
  
  if (isPlayground) {
    // Playground mode - no auth required
    const playgroundProject = {
      id: 'playground',
      name: 'Playground',
      description: 'Try out brain image generation',
      user_id: 'guest',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
      settings: {},
    };

    return (
      <ProjectLayoutComponent 
        user={null} 
        project={playgroundProject}
        isPlayground={true}
      >
        <Suspense fallback={
          <div className="flex-1 flex items-center justify-center bg-white dark:bg-gray-900">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-gray-100 mx-auto mb-4"></div>
              <p className="text-gray-500 dark:text-gray-400">Loading generation panel...</p>
            </div>
          </div>
        }>
          <GenerationPageClient
            projectId="playground"
            userId="guest"
            isPlayground={true}
          />
        </Suspense>
      </ProjectLayoutComponent>
    );
  }

  // Normal project mode - auth required
  // Only check auth if NOT playground
  const supabase = await createClient();
  const { data: { session } } = await supabase.auth.getSession();
  
  try {
    if (!session) {
      redirect('/sign-in');
    }

    const project = await getProjectById(id);

    return (
      <ProjectLayoutComponent user={session.user} project={project}>
        <Suspense fallback={
          <div className="flex-1 flex items-center justify-center bg-white dark:bg-gray-900">
            <div className="text-center">
              <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900 dark:border-gray-100 mx-auto mb-4"></div>
              <p className="text-gray-500 dark:text-gray-400">Loading generation panel...</p>
            </div>
          </div>
        }>
          <GenerationPageClient
            projectId={id}
            userId={session.user.id}
          />
        </Suspense>
      </ProjectLayoutComponent>
    );
  } catch (error) {
    return notFound();
  }
}
