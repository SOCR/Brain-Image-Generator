import Link from "next/link";

export default async function Layout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col min-h-screen items-center justify-center p-4 lg:p-8 bg-muted/40">
      <div className="w-full max-w-4xl">
        {children}
      </div>
      <footer className="mt-6 text-center text-xs text-muted-foreground">
        By continuing, you agree to SOCR&apos;s{" "}
        <Link href="/terms" className="underline hover:text-primary">
          Terms of Service
        </Link>{" "}
        and{" "}
        <Link href="/privacy" className="underline hover:text-primary">
          Privacy Policy
        </Link>
        .
      </footer>
    </div>
  );
}
