import { createClient } from "@/utils/supabase/server";
import { DashboardRootLayout } from '@/components/layouts/dashboard-root-layout';

export default async function LibraryPage() {
  const supabase = await createClient();
  const { data: { session } } = await supabase.auth.getSession();

  return (
    <DashboardRootLayout user={session.user}>
      <div className="container mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">Image Library</h1>
        </div>

        <div className="bg-white dark:bg-gray-900 rounded-lg shadow-md p-6 border border-gray-200 dark:border-gray-800">
          <div className="text-center py-12">
            <h3 className="text-lg font-medium text-gray-500 dark:text-gray-400">All generated images will appear here</h3>
            <p className="mt-2 text-gray-400 dark:text-gray-500">
              This library shows images from all your projects
            </p>
          </div>
        </div>
      </div>
    </DashboardRootLayout>
  );
} 