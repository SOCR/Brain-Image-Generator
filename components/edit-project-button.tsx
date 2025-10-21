'use client';

import { useState } from 'react';
import { Project } from '@/types/database';
import { EditProjectModal } from '@/components/edit-project-modal';

interface EditProjectButtonProps {
  project: Project;
}

export function EditProjectButton({ project }: EditProjectButtonProps) {
  const [showModal, setShowModal] = useState(false);

  return (
    <>
      <button
        onClick={() => setShowModal(true)}
        className="text-indigo-600 hover:text-indigo-800"
      >
        Edit Project
      </button>

      {showModal && (
        <EditProjectModal 
          project={project} 
          onClose={() => setShowModal(false)} 
        />
      )}
    </>
  );
} 