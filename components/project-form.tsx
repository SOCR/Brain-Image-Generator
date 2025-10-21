'use client'

import { useState } from 'react';
import { Project } from '@/types/database';
import { updateProjectAction } from '@/app/actions/project-actions';
import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { CheckCircle2, AlertCircle } from 'lucide-react';

interface ProjectFormProps {
  project: Project;
}

// Fix TypeScript error by properly typing the result from updateProjectAction
type UpdateProjectResult = { error: string } | Project;

export function ProjectForm({ project }: ProjectFormProps) {
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [success, setSuccess] = useState(false);
  const router = useRouter();

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError(null);
    setSuccess(false);

    const formData = new FormData(e.currentTarget);

    try {
      const result = await updateProjectAction(project.id, formData) as UpdateProjectResult;

      if ('error' in result) {
        setError(result.error);
      } else {
        setSuccess(true);
        router.refresh();
        // Auto-hide success message after 3 seconds
        setTimeout(() => setSuccess(false), 3000);
      }
    } catch (err) {
      setError('An unexpected error occurred');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      {error && (
        <div className="flex items-start gap-3 p-3 bg-red-50 border border-red-200 text-red-700 rounded-lg text-sm">
          <AlertCircle className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {success && (
        <div className="flex items-start gap-3 p-3 bg-green-50 border border-green-200 text-green-700 rounded-lg text-sm">
          <CheckCircle2 className="h-5 w-5 flex-shrink-0 mt-0.5" />
          <span>Project updated successfully</span>
        </div>
      )}

      <div className="space-y-2">
        <Label htmlFor="name" className="text-sm font-medium text-gray-700">
          Project Name
        </Label>
        <Input
          type="text"
          id="name"
          name="name"
          defaultValue={project.name}
          required
          className="w-full"
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="description" className="text-sm font-medium text-gray-700">
          Description
        </Label>
        <Textarea
          id="description"
          name="description"
          rows={4}
          defaultValue={project.description || ''}
          placeholder="Add a description for this project..."
          className="w-full resize-none"
        />
      </div>

      <div className="flex justify-end pt-2">
        <Button
          type="submit"
          disabled={isSubmitting}
          className="bg-gradient-to-r from-[var(--color-1)] via-[var(--color-3)] to-[var(--color-5)] text-white hover:opacity-90"
        >
          {isSubmitting ? 'Saving...' : 'Save Changes'}
        </Button>
      </div>
    </form>
  );
} 