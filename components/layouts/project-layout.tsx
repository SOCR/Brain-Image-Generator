'use client';

import { User } from '@supabase/supabase-js';
import { Project } from '@/types/database';
import { DashboardHeader } from '@/components/dashboard/dashboard-header';
import { ProjectTabs } from '@/components/dashboard/project-tabs';
import { PlaygroundBanner } from '@/components/playground/PlaygroundBanner';

interface ProjectLayoutComponentProps {
  user: User | null;
  project: Project;
  children: React.ReactNode;
  isPlayground?: boolean;
}

export function ProjectLayoutComponent({ user, project, children, isPlayground = false }: ProjectLayoutComponentProps) {
  // Define tabs specific to this project
  const projectTabs = [
    { label: 'Generation', href: `/dashboard/${project.id}`, active: true },
    { label: 'Library', href: `/dashboard/${project.id}/library`, active: false, disabled: isPlayground },
    { label: 'Settings', href: `/dashboard/${project.id}/settings`, active: false, disabled: isPlayground },
  ];

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-950 overflow-hidden">
      {/* Playground banner */}
      {isPlayground && <PlaygroundBanner />}
      
      {/* Fixed header - does not scroll */}
      <div className="flex-shrink-0 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
        {/* Global header with project selector */}
        <DashboardHeader
          user={user}
          showProjectSelector={!isPlayground}
          currentProject={project}
        />

        {/* Project-specific tabs */}
        <ProjectTabs project={project} tabs={projectTabs} isPlayground={isPlayground} />
      </div>

      {/* Main content - fills remaining height */}
      <main className="flex-1 flex overflow-hidden">
        {children}
      </main>
    </div>
  );
}
