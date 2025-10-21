'use client';

import { useState, useEffect } from 'react';
import { updateProjectAction } from '@/app/actions/project-actions';
import { useRouter } from 'next/navigation';
import { Project } from '@/types/database';

// shadcn components
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";

interface EditProjectModalProps {
  projects: Project[];
}

export function EditProjectModal({ projects }: EditProjectModalProps) {
  const [open, setOpen] = useState(false);
  const [projectId, setProjectId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const router = useRouter();

  // Get the current project based on projectId
  const currentProject = projects.find(p => p.id === projectId) || null;

  // Listen for custom event to open modal
  useEffect(() => {
    const handleOpenModal = (e: CustomEvent) => {
      setProjectId(e.detail.projectId);
      setOpen(true);
    };

    document.addEventListener('open-edit-project-modal', handleOpenModal as EventListener);
    return () => {
      document.removeEventListener('open-edit-project-modal', handleOpenModal as EventListener);
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!projectId) return;
    
    setIsSubmitting(true);
    setError(null);

    const formData = new FormData(e.currentTarget);
    
    try {
      const result = await updateProjectAction(projectId, formData);
      
      if ('error' in result) {
        setError(result.error);
        setIsSubmitting(false);
      } else {
        router.refresh();
        setOpen(false);
      }
    } catch (err) {
      setError('An unexpected error occurred');
      setIsSubmitting(false);
    }
  };

  return (
    <Dialog 
      open={open && currentProject !== null} 
      onOpenChange={(isOpen) => {
        setOpen(isOpen);
        if (!isOpen) {
          setProjectId(null);
          setError(null);
        }
      }}
    >
      <DialogContent className="sm:max-w-[425px]">
        <DialogHeader>
          <DialogTitle>Edit Project</DialogTitle>
          <DialogDescription>
            Update your project details.
          </DialogDescription>
        </DialogHeader>
        
        {error && (
          <div className="bg-red-50 text-red-600 p-3 rounded-md text-sm">
            {error}
          </div>
        )}
        
        {currentProject && (
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="name">Project Name</Label>
              <Input
                id="name"
                name="name"
                defaultValue={currentProject.name}
                placeholder="My Brain Imaging Project"
                required
              />
            </div>
            
            <div className="space-y-2">
              <Label htmlFor="description">Description (optional)</Label>
              <Textarea
                id="description"
                name="description"
                defaultValue={currentProject.description || ''}
                placeholder="Project description..."
                rows={3}
              />
            </div>
            
            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setOpen(false)}
                disabled={isSubmitting}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                disabled={isSubmitting}
              >
                {isSubmitting ? 'Saving...' : 'Save Changes'}
              </Button>
            </DialogFooter>
          </form>
        )}
      </DialogContent>
    </Dialog>
  );
} 