'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Project } from '@/types/database';
import { ChevronDown, PlusCircle, Settings } from 'lucide-react';

// shadcn components
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";

interface ProjectSwitcherProps {
  currentProject: Project;
}

export function ProjectSwitcher({ currentProject }: ProjectSwitcherProps) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const router = useRouter();
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    async function loadProjects() {
      try {
        // This will need to be updated to a client-side fetch
        const projectsData = await fetch('/api/projects').then(res => res.json());
        setProjects(projectsData);
      } catch (error) {
        console.error('Failed to load projects', error);
      } finally {
        setLoading(false);
      }
    }
    
    loadProjects();
  }, []);

  const getInitials = (name: string) => {
    if (!name) return "P";
    return name
      .split(' ')
      .map(part => part[0])
      .join('')
      .toUpperCase()
      .substring(0, 2);
  };

  return (
    <Popover open={isOpen} onOpenChange={setIsOpen}>
      <PopoverTrigger asChild>
        <Button 
          variant="outline" 
          className="flex items-center gap-2 h-9 pl-3 pr-2 rounded-md border-gray-200 bg-white hover:bg-gray-50 hover:text-gray-900"
        >
          <Avatar className="h-5 w-5 mr-1">
            <AvatarFallback className="bg-indigo-100 text-indigo-600 text-xs">
              {getInitials(currentProject.name)}
            </AvatarFallback>
          </Avatar>
          <span className="text-sm font-medium truncate max-w-[150px]">
            {currentProject.name}
          </span>
          <ChevronDown className="h-4 w-4 text-gray-500" />
        </Button>
      </PopoverTrigger>
      <PopoverContent className="w-56 p-0" align="start">
        <div className="p-2">
          <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider px-2 py-1.5">
            Projects
          </h3>
          <div className="mt-1 max-h-60 overflow-y-auto">
            {loading ? (
              <div className="px-2 py-1.5 text-sm text-gray-500">Loading projects...</div>
            ) : (
              projects.map(project => (
                <Button
                  key={project.id}
                  variant="ghost"
                  className={`w-full justify-start text-left rounded-md px-2 py-1.5 ${
                    project.id === currentProject.id ? 'bg-gray-100 font-medium' : ''
                  }`}
                  onClick={() => {
                    router.push(`/dashboard/${project.id}`);
                    setIsOpen(false);
                  }}
                >
                  <Avatar className="h-5 w-5 mr-2">
                    <AvatarFallback className="bg-gray-100 text-gray-600 text-xs">
                      {getInitials(project.name)}
                    </AvatarFallback>
                  </Avatar>
                  <span className="truncate">{project.name}</span>
                </Button>
              ))
            )}
          </div>
        </div>
        <DropdownMenuSeparator />
        <div className="p-2">
          <Button
            variant="ghost"
            className="w-full justify-start text-left px-2 py-1.5 text-sm"
            onClick={() => {
              router.push('/dashboard');
              setIsOpen(false);
            }}
          >
            View All Projects
          </Button>
          <Button
            variant="ghost"
            className="w-full justify-start text-left px-2 py-1.5 text-sm"
            onClick={() => {
              setIsOpen(false);
              // Open create project modal
              document.dispatchEvent(new CustomEvent('open-create-project-modal'));
            }}
          >
            <PlusCircle className="h-4 w-4 mr-2" />
            Create New Project
          </Button>
        </div>
      </PopoverContent>
    </Popover>
  );
} 