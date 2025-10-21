'use client';

import { useState } from 'react';
import { deleteProjectAction } from '@/app/actions/project-actions';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { AlertCircle, Trash2 } from 'lucide-react';

interface DeleteProjectSectionProps {
  projectId: string;
}

export function DeleteProjectSection({ projectId }: DeleteProjectSectionProps) {
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfirm, setShowConfirm] = useState(false);
  const [confirmText, setConfirmText] = useState('');
  const router = useRouter();

  const handleDelete = async () => {
    setIsDeleting(true);
    setError(null);

    try {
      const result = await deleteProjectAction(projectId);
      if (result?.success) {
        router.push('/dashboard');
      } else if (result?.error) {
        setError(result.error);
        setIsDeleting(false);
        setShowConfirm(false);
      }
    } catch (err) {
      setError('Failed to delete project. Please try again.');
      setIsDeleting(false);
      setShowConfirm(false);
    }
  };

  return (
    <div>
      <p className="text-sm text-gray-600 mb-5">
        Deleting this project will permanently remove all associated images and data. This action cannot be undone.
      </p>

      {error && (
        <div className="flex items-start gap-3 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm mb-4">
          <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {showConfirm ? (
        <div className="space-y-4 p-4 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-gray-700">
            To confirm deletion, type <span className="font-semibold text-red-700">delete</span> in the field below:
          </p>

          <Input
            type="text"
            value={confirmText}
            onChange={(e) => setConfirmText(e.target.value)}
            placeholder="Type 'delete' to confirm"
            className="w-full"
            autoFocus
          />

          <div className="flex gap-3">
            <Button
              onClick={handleDelete}
              disabled={isDeleting || confirmText !== 'delete'}
              variant="destructive"
              className="bg-red-600 hover:bg-red-700"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              {isDeleting ? 'Deleting...' : 'Delete Project'}
            </Button>

            <Button
              onClick={() => {
                setShowConfirm(false);
                setConfirmText('');
              }}
              disabled={isDeleting}
              variant="outline"
            >
              Cancel
            </Button>
          </div>
        </div>
      ) : (
        <Button
          onClick={() => setShowConfirm(true)}
          variant="destructive"
          className="bg-red-600 hover:bg-red-700"
        >
          <Trash2 className="h-4 w-4 mr-2" />
          Delete Project
        </Button>
      )}
    </div>
  );
} 