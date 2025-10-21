export interface Project {
  id: string;
  name: string;
  description: string | null;
  created_at: string;
  updated_at: string;
  user_id: string;
  settings: Record<string, any>;
}

export interface MLModel {
  id: string;
  name: string;
  description: string | null;
  type: '2D' | '3D';
  parameters: Record<string, any>;
  created_at: string;
}

export interface GeneratedImage {
  id: string;
  name: string;
  project_id: string;
  model_id: string;
  file_path: string;
  parameters_used: Record<string, any>;
  created_at: string;
  user_id: string;
} 