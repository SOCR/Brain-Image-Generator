'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { cn } from "@/lib/utils";

interface Tab {
  label: string;
  href: string;
  active?: boolean;
}

interface DashboardTabsProps {
  tabs: Tab[];
}

export function DashboardTabs({ tabs }: DashboardTabsProps) {
  const pathname = usePathname();

  return (
    <div className="px-4 sm:px-6 border-b border-gray-200 dark:border-gray-800">
      <nav className="-mb-px flex space-x-8">
        {tabs.map((tab) => {
          // Check if current path matches this tab
          const isActive = 
            (tab.href === '/dashboard' && pathname === '/dashboard') || 
            (tab.href !== '/dashboard' && pathname.startsWith(tab.href));
              
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
      </nav>
    </div>
  );
} 