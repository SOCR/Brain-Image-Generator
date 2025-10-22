'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Project } from '@/types/database';
import { cn } from "@/lib/utils";
import { Lock } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';

interface Tab {
  label: string;
  href: string;
  active?: boolean;
  disabled?: boolean;
}

interface ProjectTabsProps {
  project: Project;
  tabs: Tab[];
  isPlayground?: boolean;
}

export function ProjectTabs({ project, tabs, isPlayground = false }: ProjectTabsProps) {
  const pathname = usePathname();

  return (
    <div className="px-4 sm:px-6">
      <div className="md:hidden py-3">
        <h1 className="text-xl font-bold text-gray-900 dark:text-gray-100">
          {project.name}
        </h1>
      </div>
      
      <nav className="-mb-px flex space-x-8">
        <TooltipProvider>
          {tabs.map((tab) => {
            // Check if current path matches this tab
            const isActive = pathname === tab.href || 
              (tab.href !== `/dashboard/${project.id}` && pathname.startsWith(tab.href));
            
            const isDisabled = tab.disabled || false;
            
            if (isDisabled) {
              return (
                <Tooltip key={tab.href}>
                  <TooltipTrigger asChild>
                    <div
                      className={cn(
                        "inline-flex items-center px-1 py-3 text-sm font-medium border-b-2 cursor-not-allowed",
                        "border-transparent text-gray-400 dark:text-gray-600"
                      )}
                    >
                      <Lock className="h-3 w-3 mr-1.5" />
                      {tab.label}
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Sign in to access {tab.label.toLowerCase()}</p>
                  </TooltipContent>
                </Tooltip>
              );
            }
                
            return (
              <Link
                key={tab.href}
                href={tab.href}
                className={cn(
                  "inline-flex items-center px-1 py-3 text-sm font-medium border-b-2",
                  isActive 
                    ? "border-indigo-500 text-indigo-600 dark:text-indigo-400" 
                    : "border-transparent text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 hover:border-gray-300 dark:hover:border-gray-600"
                )}
              >
                {tab.label}
              </Link>
            );
          })}
        </TooltipProvider>
      </nav>
    </div>
  );
} 