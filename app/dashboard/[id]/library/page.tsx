import { notFound, redirect } from 'next/navigation';
import { getProjectById } from '@/lib/projects';
import { createClient } from "@/utils/supabase/server";
import { ProjectLayoutComponent } from '@/components/layouts/project-layout';
import LibraryPageClient from '@/components/projects/LibraryPageClient';

interface ProjectLibraryPageProps {
  params: {
    id: string;
  };
}

export default async function ProjectLibraryPage({ params }: ProjectLibraryPageProps) {
  try {
    const supabase = await createClient();
    const { data: { session } } = await supabase.auth.getSession();
    const project = await getProjectById(params.id);

    if (!session) {
      redirect('/sign-in');
    }

    return (
      <ProjectLayoutComponent user={session.user} project={project}>
        <LibraryPageClient
          projectId={params.id}
          userId={session.user.id}
        />
      </ProjectLayoutComponent>
    );
  } catch (error) {
    return notFound();
  }
} 