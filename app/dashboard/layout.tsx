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

  if (!isPlayground) {
    const supabase = await createClient();
    const { data: { session } } = await supabase.auth.getSession();

    if (!session) {
      // Redirect to playground if not authenticated
      redirect('/dashboard/playground');
    }
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* All content rendered here, including headers from child layouts */}
      {children}
    </div>
  );
} 