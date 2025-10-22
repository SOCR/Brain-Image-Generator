'use client';

import { Project } from '@/types/database';
import { useRouter } from 'next/navigation';
import { useState } from 'react';
import { deleteProjectAction } from '@/app/actions/project-actions';
import { DeleteConfirmationDialog } from "@/components/delete-confirmation-dialog";
import { ProjectCard } from "@/components/ui/project-card";

interface ProjectsListProps {
  projects: Project[];
}

export function ProjectsList({ projects }: ProjectsListProps) {
  const router = useRouter();
  const [deletingProjectId, setDeletingProjectId] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [projectToDelete, setProjectToDelete] = useState<Project | null>(null);

  if (projects.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 px-4">
        <div className="w-20 h-20 mb-6 rounded-full bg-gray-100 dark:bg-gray-800 flex items-center justify-center">
          <svg
            className="w-10 h-10 text-gray-300 dark:text-gray-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
            />
          </svg>
        </div>
        <h3 className="text-xl font-semibold text-gray-600 dark:text-gray-400 mb-2">No projects yet</h3>
        <p className="text-sm text-gray-400 dark:text-gray-500 text-center max-w-sm">
          Create your first project to start generating brain images with AI
        </p>
      </div>
    );
  }

  const handleEditProject = (projectId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    // Dispatch custom event to open edit modal with this project
    document.dispatchEvent(new CustomEvent('open-edit-project-modal', {
      detail: { projectId }
    }));
  };

  const handleDeleteClick = (project: Project, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setProjectToDelete(project);
    setDeleteDialogOpen(true);
  };

  const handleConfirmDelete = async () => {
    if (!projectToDelete) return;

    setDeletingProjectId(projectToDelete.id);
    await deleteProjectAction(projectToDelete.id);
    router.refresh();
    setDeletingProjectId(null);
    setProjectToDelete(null);
  };

  const handleNavigateToSettings = (projectId: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    router.push(`/dashboard/${projectId}/settings`);
  };

  return (
    <>
      {/* Grid layout - responsive */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {projects.map((project) => (
          <ProjectCard
            key={project.id}
            project={project}
            onEdit={handleEditProject}
            onDelete={handleDeleteClick}
            onSettings={handleNavigateToSettings}
            isDeleting={deletingProjectId === project.id}
          />
        ))}
      </div>

      {/* Delete confirmation dialog */}
      {projectToDelete && (
        <DeleteConfirmationDialog
          open={deleteDialogOpen}
          onOpenChange={setDeleteDialogOpen}
          title="Delete Project"
          description={`Are you sure you want to delete "${projectToDelete.name}"? This action cannot be undone.`}
          onConfirm={handleConfirmDelete}
        />
      )}
    </>
  );
}
