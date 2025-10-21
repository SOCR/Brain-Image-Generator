import { getUserProjects } from '@/lib/projects';
import { createClient } from "@/utils/supabase/server";
import { CreateProjectButton } from '@/components/create-project-button';
import { ProjectsList } from '@/components/projects-list';
import { EditProjectModal } from '@/components/edit-project-modal';
import { DashboardRootLayout } from '@/components/layouts/dashboard-root-layout';
import { redirect } from 'next/navigation';

export default async function DashboardPage() {
  const supabase = await createClient();
  const { data: { session } } = await supabase.auth.getSession();
  const projects = await getUserProjects();

  if (!session) {
    redirect('/sign-in');
  }

  return (
    <DashboardRootLayout user={session.user}>
      <div className="min-h-screen bg-gray-50">
        {/* Header Section */}
        <div className="bg-white border-b border-gray-200">
          <div className="container mx-auto px-8 py-6">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold text-gray-900">Projects</h1>
                <p className="text-sm text-gray-500 mt-1">
                  Manage your brain image generation projects
                </p>
              </div>
              <CreateProjectButton />
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-8 py-8">
          <ProjectsList projects={projects} />
          <EditProjectModal projects={projects} />
        </div>
      </div>
    </DashboardRootLayout>
  );
} 