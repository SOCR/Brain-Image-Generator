'use client';

import { User } from '@supabase/supabase-js';
import { Project } from '@/types/database';
import { DashboardHeader } from '@/components/dashboard/dashboard-header';
import { ProjectTabs } from '@/components/dashboard/project-tabs';

interface ProjectLayoutComponentProps {
  user: User;
  project: Project;
  children: React.ReactNode;
}

export function ProjectLayoutComponent({ user, project, children }: ProjectLayoutComponentProps) {
  // Define tabs specific to this project
  const projectTabs = [
    { label: 'Generation', href: `/dashboard/${project.id}`, active: true },
    { label: 'Library', href: `/dashboard/${project.id}/library`, active: false },
    { label: 'Settings', href: `/dashboard/${project.id}/settings`, active: false },
  ];

  return (
    <div className="h-screen flex flex-col bg-gray-50 overflow-hidden">
      {/* Fixed header - does not scroll */}
      <div className="flex-shrink-0 bg-white border-b border-gray-200">
        {/* Global header with project selector */}
        <DashboardHeader
          user={user}
          showProjectSelector={true}
          currentProject={project}
        />

        {/* Project-specific tabs */}
        <ProjectTabs project={project} tabs={projectTabs} />
      </div>

      {/* Main content - fills remaining height */}
      <main className="flex-1 flex overflow-hidden">
        {children}
      </main>
    </div>
  );
}
