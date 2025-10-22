import { redirect } from 'next/navigation';
import { createClient } from "@/utils/supabase/server";
import { headers } from 'next/headers';

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  // Check if this is playground mode via custom header set by middleware
  const headersList = await headers();
  const isPlaygroundHeader = headersList.get('x-is-playground');
  const isPlayground = isPlaygroundHeader === 'true';
  
  console.log('=== LAYOUT DEBUG ===');
  console.log('x-is-playground header:', isPlaygroundHeader);
  console.log('Is playground (layout):', isPlayground);

  if (!isPlayground) {
    const supabase = await createClient();
    const { data: { session } } = await supabase.auth.getSession();

    if (!session) {
      console.log('❌ Layout: No session - redirecting to sign-in');
      // Redirect to login page if not authenticated
      redirect('/sign-in');
    }
  } else {
    console.log('✅ Layout: Playground mode - skipping auth check');
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* All content rendered here, including headers from child layouts */}
      {children}
    </div>
  );
} 