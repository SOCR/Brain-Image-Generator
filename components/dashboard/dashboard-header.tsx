'use client';

import Link from 'next/link';
import { User } from '@supabase/supabase-js';
import { Project } from '@/types/database';
import { ProjectSwitcher } from './project-switcher';
import { HelpCircle, ExternalLink, LogOut, Settings, User as UserIcon, Sun, Moon } from 'lucide-react';
import { useTheme } from 'next-themes';
import { useEffect, useState } from 'react';
import { signOutAction } from '@/app/actions';

// shadcn components
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

interface DashboardHeaderProps {
  user: User | null;
  showProjectSelector: boolean;
  currentProject?: Project;
}

export function DashboardHeader({
  user,
  showProjectSelector = false,
  currentProject
}: DashboardHeaderProps) {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  return (
    <header className="h-16 px-6 flex items-center justify-between">
      <div className="flex items-center gap-4">
        {/* Logo with Gradient Orb */}
        <Link href="/dashboard" className="flex items-center gap-3">
          <div className="relative h-8 w-8 rounded-full bg-gradient-to-br from-[var(--color-1)] via-[var(--color-3)] to-[var(--color-5)] flex items-center justify-center shadow-lg">
            {/* Inner glow effect */}
            <div className="absolute inset-0 rounded-full bg-gradient-to-br from-[var(--color-2)] via-[var(--color-4)] to-[var(--color-1)] opacity-60 blur-sm"></div>
            <div className="relative z-10 h-6 w-6 rounded-full bg-gradient-to-br from-white/20 to-transparent"></div>
          </div>
          <span className="font-bold text-xl bg-gradient-to-r from-gray-900 via-gray-700 to-gray-900 dark:from-gray-100 dark:via-gray-300 dark:to-gray-100 bg-clip-text text-transparent">
            SOCR BrainGen
          </span>
        </Link>

        {/* Project switcher (only show when inside a project) */}
        {showProjectSelector && currentProject && (
          <div className="hidden md:block">
            <ProjectSwitcher currentProject={currentProject} />
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        {/* Documentation/Help links */}
        <Button variant="ghost" size="icon" className="hidden md:flex text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200">
          <HelpCircle className="h-5 w-5" />
        </Button>

        {/* Theme Switcher */}
        {mounted && (
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            className="text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            {theme === 'dark' ? (
              <Sun className="h-5 w-5" />
            ) : (
              <Moon className="h-5 w-5" />
            )}
          </Button>
        )}

        {/* User menu or Auth buttons */}
        {user ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" className="rounded-full">
                <Avatar className="h-8 w-8">
                  {user.user_metadata?.avatar_url ? (
                    <AvatarImage src={user.user_metadata.avatar_url} alt={user.email || 'User'} />
                  ) : (
                    <AvatarFallback className="bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-300 font-semibold">
                      {user.email?.[0].toUpperCase() || 'U'}
                    </AvatarFallback>
                  )}
                </Avatar>
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>
                <div className="flex flex-col space-y-1">
                  <p className="text-sm font-medium">{user.user_metadata?.name || 'User'}</p>
                  <p className="text-xs text-muted-foreground truncate">{user.email}</p>
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <UserIcon className="mr-2 h-4 w-4" />
                <span>Profile</span>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="mr-2 h-4 w-4" />
                <span>Settings</span>
              </DropdownMenuItem>
              <DropdownMenuItem>
                <ExternalLink className="mr-2 h-4 w-4" />
                <span>Documentation</span>
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <form action={signOutAction} className="w-full">
                <DropdownMenuItem asChild>
                  <button type="submit" className="w-full flex cursor-pointer items-center">
                    <LogOut className="mr-2 h-4 w-4" />
                    <span>Sign out</span>
                  </button>
                </DropdownMenuItem>
              </form>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <div className="flex items-center gap-2">
            <Link href="/sign-in">
              <Button variant="ghost" size="sm">
                Sign In
              </Button>
            </Link>
            <Link href="/sign-up">
              <Button variant="default" size="sm">
                Sign Up
              </Button>
            </Link>
          </div>
        )}
      </div>
    </header>
  );
}
