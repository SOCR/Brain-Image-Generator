import { createClient } from "@/utils/supabase/server";
import { Project } from '@/types/database';

// Get all projects for the current user
export async function getUserProjects() {

  const supabase = await createClient();
  
  const { data: projects, error } = await supabase
    .from('projects')
    .select('*')
    .order('created_at', { ascending: false });
    
  if (error) {
    console.error('Error fetching projects:', error);
    throw new Error('Failed to fetch projects');
  }
  
  return projects as Project[];
}

// Get a single project by ID
export async function getProjectById(id: string) {

  const supabase = await createClient();
  
  const { data: project, error } = await supabase
    .from('projects')
    .select('*')
    .eq('id', id)
    .single();
    
  if (error) {
    console.error('Error fetching project:', error);
    throw new Error('Failed to fetch project');
  }
  
  return project as Project;
}

// Create a new project
export async function createProject(projectData: Omit<Project, 'id' | 'created_at' | 'updated_at' | 'user_id'>) {
  const supabase = await createClient();
  
  const { data: user } = await supabase.auth.getUser();
  
  if (!user.user) {
    throw new Error('User not authenticated');
  }
  
  const { data: project, error } = await supabase
    .from('projects')
    .insert({
      ...projectData,
      user_id: user.user.id
    })
    .select()
    .single();
    
  if (error) {
    console.error('Error creating project:', error);
    throw new Error('Failed to create project');
  }
  
  return project as Project;
}

// Update a project
export async function updateProject(id: string, projectData: Partial<Omit<Project, 'id' | 'created_at' | 'user_id'>>) {
  const supabase = await createClient();
  
  const { data: project, error } = await supabase
    .from('projects')
    .update({
      ...projectData,
      updated_at: new Date().toISOString()
    })
    .eq('id', id)
    .select()
    .single();
    
  if (error) {
    console.error('Error updating project:', error);
    throw new Error('Failed to update project');
  }
  
  return project as Project;
}

// Delete a project
export async function deleteProject(id: string) {
  const supabase = await createClient();
  
  const { error } = await supabase
    .from('projects')
    .delete()
    .eq('id', id);
    
  if (error) {
    console.error('Error deleting project:', error);
    throw new Error('Failed to delete project');
  }
  
  return true;
} 