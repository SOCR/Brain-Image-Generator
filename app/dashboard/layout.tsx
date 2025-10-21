import { redirect } from 'next/navigation';
import { createClient } from "@/utils/supabase/server";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const supabase = await createClient();
  const { data: { session } } = await supabase.auth.getSession();

  if (!session) {
    // Redirect to login page if not authenticated
    redirect('/sign-in');
  }

  return (
    <div className="min-h-screen flex flex-col bg-gray-50">
      {/* All content rendered here, including headers from child layouts */}
      {children}
    </div>
  );
} 