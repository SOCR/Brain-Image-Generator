'use client'

import { useState } from 'react';
import { deleteProjectAction } from '@/app/actions/project-actions';
import { useRouter } from 'next/navigation';

interface DeleteProjectButtonProps {
  projectId: string;
}

export function DeleteProjectButton({ projectId }: DeleteProjectButtonProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [showConfirm, setShowConfirm] = useState(false);
  const router = useRouter();

  const handleDelete = async () => {
    setIsDeleting(true);
    try {
      const result = await deleteProjectAction(projectId);
      if (result?.success) {
        router.push('/dashboard');
      } else {
        setIsDeleting(false);
        setShowConfirm(false);
      }
    } catch (error) {
      console.error('Failed to delete project', error);
      setIsDeleting(false);
      setShowConfirm(false);
    }
  };

  if (showConfirm) {
    return (
      <div className="flex items-center space-x-2">
        <p className="text-sm text-red-600">Are you sure?</p>
        <button
          onClick={handleDelete}
          disabled={isDeleting}
          className="text-white bg-red-600 hover:bg-red-700 px-3 py-1 rounded text-sm disabled:opacity-50"
        >
          {isDeleting ? 'Deleting...' : 'Yes, Delete'}
        </button>
        <button
          onClick={() => setShowConfirm(false)}
          disabled={isDeleting}
          className="text-gray-700 bg-gray-100 hover:bg-gray-200 px-3 py-1 rounded text-sm disabled:opacity-50"
        >
          Cancel
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={() => setShowConfirm(true)}
      className="text-red-600 hover:text-red-800"
    >
      Delete Project
    </button>
  );
} 