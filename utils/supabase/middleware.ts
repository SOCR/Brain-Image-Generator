import { createServerClient } from "@supabase/ssr";
import { type NextRequest, NextResponse } from "next/server";

export const updateSession = async (request: NextRequest) => {
  // This `try/catch` block is only here for the interactive tutorial.
  // Feel free to remove once you have Supabase connected.
  try {
    // Create an unmodified response
    let response = NextResponse.next({
      request: {
        headers: request.headers,
      },
    });

    const supabase = createServerClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
      {
        cookies: {
          getAll() {
            return request.cookies.getAll();
          },
          setAll(cookiesToSet) {
            cookiesToSet.forEach(({ name, value }) =>
              request.cookies.set(name, value),
            );
            response = NextResponse.next({
              request,
            });
            cookiesToSet.forEach(({ name, value, options }) =>
              response.cookies.set(name, value, options),
            );
          },
        },
      },
    );

    // This will refresh session if expired - required for Server Components
    // https://supabase.com/docs/guides/auth/server-side/nextjs
    const user = await supabase.auth.getUser();

    console.log('=== MIDDLEWARE DEBUG ===');
    console.log('Path:', request.nextUrl.pathname);
    console.log('Has user:', !user.error);
    
    // Check if this is playground mode FIRST (before any auth checks)
    const isPlayground = request.nextUrl.pathname === "/dashboard/playground" ||
                        request.nextUrl.pathname.startsWith("/dashboard/playground/");
    
    console.log('Is playground:', isPlayground);
    
    // Allow playground mode without authentication - skip all auth checks
    if (isPlayground) {
      console.log('✅ Allowing playground access');
      // Set a custom header so the layout knows it's playground mode
      response.headers.set('x-is-playground', 'true');
      return response;
    }

    // TEMPORARILY DISABLE ALL REDIRECTS FOR DEBUGGING
    /*
    // protected routes
    if (request.nextUrl.pathname.startsWith("/protected") && user.error) {
      return NextResponse.redirect(new URL("/sign-in", request.url));
    }

    if (request.nextUrl.pathname === "/" && !user.error) {
      return NextResponse.redirect(new URL("/dashboard", request.url));
    }
    
    // Redirect to playground if accessing /dashboard without auth
    if (request.nextUrl.pathname === "/dashboard" && user.error) {
      return NextResponse.redirect(new URL("/dashboard/playground", request.url));
    }
    
    // Block other dashboard routes if not authenticated
    if (request.nextUrl.pathname.startsWith("/dashboard") && user.error) {
      return NextResponse.redirect(new URL("/sign-in", request.url));
    }
    */
    
    console.log('✅ Allowing access (all redirects disabled)');
    return response;
  } catch (e) {
    // If you are here, a Supabase client could not be created!
    // This is likely because you have not set up environment variables.
    // Check out http://localhost:3000 for Next Steps.
    return NextResponse.next({
      request: {
        headers: request.headers,
      },
    });
  }
};
