import { notFound, redirect } from 'next/navigation';
import { getProjectById } from '@/lib/projects';
import { createClient } from "@/utils/supabase/server";
import { ProjectLayoutComponent } from '@/components/layouts/project-layout';
import { ProjectForm } from '@/components/project-form';
import { DeleteProjectSection } from '@/components/delete-project-section';

interface ProjectSettingsPageProps {
  params: {
    id: string;
  };
}

export default async function ProjectSettingsPage({ params }: ProjectSettingsPageProps) {
  try {
    const supabase = await createClient();
    const { data: { session } } = await supabase.auth.getSession();
    const project = await getProjectById(params.id);
    
    if (!session) {
      redirect('/sign-in');
    }

    return (
      <ProjectLayoutComponent user={session.user} project={project}>
        <div className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-950 p-6">
          <div className="max-w-3xl mx-auto space-y-6">
            {/* Header */}
            <div>
              <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100">Project Settings</h1>
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Manage your project configuration and preferences</p>
            </div>

            {/* General Settings */}
            <div className="bg-white dark:bg-gray-900 rounded-lg border border-gray-200 dark:border-gray-800 overflow-hidden">
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-800">
                <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100">General</h2>
                <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">Update your project name and description</p>
              </div>
              <div className="p-6">
                <ProjectForm project={project} />
              </div>
            </div>

            {/* Danger Zone */}
            <div className="bg-white dark:bg-gray-900 rounded-lg border border-red-200 dark:border-red-900 overflow-hidden">
              <div className="px-6 py-4 border-b border-red-200 dark:border-red-900 bg-red-50 dark:bg-red-950">
                <h2 className="text-base font-semibold text-red-900 dark:text-red-400">Danger Zone</h2>
                <p className="text-sm text-red-600 dark:text-red-400 mt-0.5">Irreversible actions that affect your project</p>
              </div>
              <div className="p-6">
                <DeleteProjectSection projectId={project.id} />
              </div>
            </div>
          </div>
        </div>
      </ProjectLayoutComponent>
    );
  } catch (error) {
    return notFound();
  }
} 