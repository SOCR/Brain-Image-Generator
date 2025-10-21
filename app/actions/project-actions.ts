'use server'

import { revalidatePath } from 'next/cache';
import { redirect } from 'next/navigation';
import { createProject, updateProject, deleteProject } from '@/lib/projects';

export async function createProjectAction(formData: FormData) {
  const name = formData.get('name') as string;
  const description = formData.get('description') as string;
  
  if (!name) {
    return {
      error: 'Project name is required'
    };
  }
  
  try {
    const project = await createProject({
      name,
      description,
      settings: {}
    });
    
    revalidatePath('/dashboard');
    // Return the project so the client can navigate if needed
    return project;
  } catch (error) {
    return {
      error: 'Failed to create project'
    };
  }
}

export async function updateProjectAction(id: string, formData: FormData) {
  const name = formData.get('name') as string;
  const description = formData.get('description') as string;
  
  if (!name) {
    return {
      error: 'Project name is required'
    };
  }
  
  try {
    const project = await updateProject(id, {
      name,
      description
    });
    
    revalidatePath(`/dashboard/${id}`);
    revalidatePath('/dashboard');
    return project;
  } catch (error) {
    return {
      error: 'Failed to update project'
    };
  }
}

export async function deleteProjectAction(id: string) {
  try {
    await deleteProject(id);
    
    revalidatePath('/dashboard');
    return { success: true };
  } catch (error) {
    return {
      error: 'Failed to delete project'
    };
  }
} 