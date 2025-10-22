import { resetPasswordAction } from "@/app/actions";
import { FormMessage, Message } from "@/components/form-message";
import { SubmitButton } from "@/components/submit-button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import Image from "next/image";

export default async function ResetPassword(props: {
  searchParams: Promise<Message>;
}) {
  const searchParams = await props.searchParams;
  return (
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
              Please enter your new password below
            </p>
          </div>
          
          <div className="space-y-4">
            <div className="grid gap-2">
              <Label htmlFor="password">New password</Label>
              <Input
                id="password"
                type="password"
                name="password"
                placeholder="New password"
                minLength={6}
                required
              />
            </div>
            <div className="grid gap-2">
              <Label htmlFor="confirmPassword">Confirm password</Label>
              <Input
                id="confirmPassword"
                type="password"
                name="confirmPassword"
                placeholder="Confirm password"
                minLength={6}
                required
              />
            </div>
            <SubmitButton formAction={resetPasswordAction} className="w-full">
              Reset password
            </SubmitButton>
          </div>

          <FormMessage message={searchParams} />
        </form>

        <div className="relative hidden bg-muted md:block">
          <Image 
            src="/signin-1.png" 
            alt="Reset Password" 
            fill
            className="object-cover"
            priority
          />
        </div>
      </CardContent>
    </Card>
  );
}
