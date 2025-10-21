'use client';

import { Project } from '@/types/database';
import Link from 'next/link';
import { formatDistanceToNow } from 'date-fns';
import { useState } from 'react';
import { MoreVertical, Edit, Settings, Trash2, ImageIcon } from 'lucide-react';
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { GlowEffect } from "@/components/ui/glow-effect";
import { cn } from "@/lib/utils";

interface ProjectCardProps {
  project: Project;
  onEdit: (projectId: string, e: React.MouseEvent) => void;
  onDelete: (project: Project, e: React.MouseEvent) => void;
  onSettings: (projectId: string, e: React.MouseEvent) => void;
  isDeleting?: boolean;
}

export function ProjectCard({
  project,
  onEdit,
  onDelete,
  onSettings,
  isDeleting = false
}: ProjectCardProps) {
  const [isHovered, setIsHovered] = useState(false);

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
    <div
      className="group relative"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Glow Effect Container - OUTSIDE the card, only visible on hover */}
      <div className={cn(
        "absolute -inset-1 rounded-xl transition-opacity duration-500",
        isHovered ? "opacity-100" : "opacity-0"
      )}>
        <GlowEffect
          mode="rotate"
          blur="medium"
          colors={[
            'var(--color-1)',
            'var(--color-5)',
            'var(--color-3)',
            'var(--color-4)',
            'var(--color-2)',
          ]}
        />
      </div>

      {/* Card Content - INSIDE with overflow-hidden */}
      <Link
        href={`/dashboard/${project.id}`}
        className="block"
      >
        <div className={cn(
          "relative bg-white rounded-xl border transition-all duration-200 overflow-hidden",
          "group-hover:scale-[1.02] group-hover:shadow-md",
          isHovered ? "border-gray-300 shadow-sm" : "border-gray-200 shadow-sm"
        )}>
          {/* Card content */}
          <div className="relative z-10 p-6">
            {/* Header */}
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                {/* Avatar */}
                <Avatar className="h-10 w-10">
                  <AvatarFallback className="bg-gray-100 text-gray-600 font-semibold">
                    {getInitials(project.name)}
                  </AvatarFallback>
                </Avatar>

                <div>
                  <h3 className="text-lg font-semibold text-gray-900 group-hover:text-gray-700 transition-colors">
                    {project.name}
                  </h3>
                  <p className="text-xs text-gray-500">
                    Created {formatDistanceToNow(new Date(project.created_at))} ago
                  </p>
                </div>
              </div>

              {/* Actions menu */}
              <DropdownMenu>
                <DropdownMenuTrigger asChild onClick={(e) => e.stopPropagation()}>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                  >
                    <MoreVertical className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end" className="w-48">
                  <DropdownMenuItem onClick={(e) => onEdit(project.id, e)}>
                    <Edit className="mr-2 h-4 w-4" />
                    <span>Rename</span>
                  </DropdownMenuItem>
                  <DropdownMenuItem onClick={(e) => onSettings(project.id, e)}>
                    <Settings className="mr-2 h-4 w-4" />
                    <span>Settings</span>
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                  <DropdownMenuItem
                    className="text-red-600 focus:text-red-600"
                    onClick={(e) => onDelete(project, e)}
                    disabled={isDeleting}
                  >
                    <Trash2 className="mr-2 h-4 w-4" />
                    <span>{isDeleting ? 'Deleting...' : 'Delete'}</span>
                  </DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            </div>

            {/* Description */}
            {project.description && (
              <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                {project.description}
              </p>
            )}

            {/* Footer stats */}
            <div className="flex items-center gap-4 pt-4 border-t border-gray-100">
              <div className="flex items-center gap-1.5 text-xs text-gray-500">
                <ImageIcon className="h-3.5 w-3.5" />
                <span>0 images</span>
              </div>
              <div className="flex items-center gap-1.5 text-xs text-gray-400">
                <span>•</span>
                <span>Updated {formatDistanceToNow(new Date(project.updated_at))} ago</span>
              </div>
            </div>
          </div>
        </div>
      </Link>
    </div>
  );
}
