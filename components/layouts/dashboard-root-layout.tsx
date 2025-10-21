'use client';

import { User } from '@supabase/supabase-js';
import { DashboardHeader } from '@/components/dashboard/dashboard-header';
import { DashboardTabs } from '@/components/dashboard/dashboard-tabs';

interface DashboardRootLayoutProps {
  user: User;
  children: React.ReactNode;
}

export function DashboardRootLayout({ user, children }: DashboardRootLayoutProps) {
  // Define the tabs for the main dashboard
  const dashboardTabs = [
    { label: 'Projects', href: '/dashboard', active: true },
    { label: 'Library', href: '/dashboard/library', active: false },
  ];

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* Fixed position header - NO extra padding/margin */}
      <div className="sticky top-0 z-10 bg-white border-b border-gray-200">
        {/* Global header with profile and general app navigation */}
        <DashboardHeader 
          user={user}
          showProjectSelector={false}
        />
        
        {/* Dashboard-specific tabs */}
        <DashboardTabs tabs={dashboardTabs} />
      </div>
      
      {/* Main content - removed extra padding */}
      <main className="flex-1">
        {children}
      </main>
    </div>
  );
}
