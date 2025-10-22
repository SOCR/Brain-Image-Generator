import { forgotPasswordAction } from "@/app/actions";
import { FormMessage, Message } from "@/components/form-message";
import { SubmitButton } from "@/components/submit-button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from "next/link";
import Image from "next/image";

export default async function ForgotPassword(props: {
  searchParams: Promise<Message>;
}) {
  const searchParams = await props.searchParams;
  return (
    <>
      <Card className="overflow-hidden w-full max-w-4xl mx-auto">
        <CardContent className="grid p-0 md:grid-cols-2">
          <form className="p-6 md:p-8 space-y-6">
            <div className="flex flex-col items-center text-center space-y-1">
              {/* SOCR Logo with Glow */}
              <div className="relative h-12 w-12 mb-2">
                <div className="absolute inset-0 bg-blue-500/30 dark:bg-blue-400/30 blur-xl rounded-full"></div>
                <Image
                  src="/socr.png"
                  alt="SOCR Logo"
                  width={48}
                  height={48}
                  className="relative z-10 drop-shadow-xl"
                />
              </div>
              <h1 className="text-2xl font-bold">Reset your password</h1>
              <p className="text-balance text-muted-foreground">
                Enter your email to receive a reset link
              </p>
            </div>
            
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" name="email" type="email" placeholder="you@example.com" required />
              </div>
              <SubmitButton formAction={forgotPasswordAction} className="w-full">
                Reset Password
              </SubmitButton>
            </div>

            <FormMessage message={searchParams} />

            <div className="text-center text-sm">
              Remember your password?{" "}
              <Link href="/sign-in" className="underline underline-offset-4 hover:text-primary">
                Sign in
              </Link>
            </div>
          </form>

          <div className="relative hidden bg-muted md:block">
            <Image 
              src="/signin-1.png" 
              alt="Password Reset" 
              fill
              className="object-cover"
              priority
            />
          </div>
        </CardContent>
      </Card>
    </>
  );
}
