import { signUpAction } from "@/app/actions";
import { FormMessage, Message } from "@/components/form-message";
import { SubmitButton } from "@/components/submit-button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Link from "next/link";
import Image from "next/image";

export default async function Signup(props: {
  searchParams: Promise<Message>;
}) {
  const searchParams = await props.searchParams;
  if ("message" in searchParams) {
    return (
      <div className="w-full flex-1 flex items-center h-screen sm:max-w-md justify-center gap-2 p-4">
        <FormMessage message={searchParams} />
      </div>
    );
  }

  return (
    <>
      <Card className="overflow-hidden w-full max-w-4xl mx-auto">
        <CardContent className="grid p-0 md:grid-cols-2">
          <form className="p-6 md:p-8 space-y-6">
            <div className="flex flex-col items-center text-center space-y-1">
              <h1 className="text-2xl font-bold">Create an account</h1>
              <p className="text-balance text-muted-foreground">
                Sign up for SOCR Brain Gen
              </p>
            </div>
            
            <div className="space-y-4">
              <div className="grid gap-2">
                <Label htmlFor="email">Email</Label>
                <Input id="email" name="email" type="email" placeholder="you@example.com" required />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  placeholder="At least 6 characters"
                  minLength={6}
                  required
                />
              </div>
              <SubmitButton formAction={signUpAction} pendingText="Signing up..." className="w-full">
                Sign up
              </SubmitButton>
            </div>

            <FormMessage message={searchParams} />

            <div className="text-center text-sm">
              Already have an account?{" "}
              <Link href="/sign-in" className="underline underline-offset-4 hover:text-primary">
                Sign in
              </Link>
            </div>
          </form>

          <div className="relative hidden bg-muted md:block">
            <Image 
              src="/signup-1.png" 
              alt="Sign Up" 
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
