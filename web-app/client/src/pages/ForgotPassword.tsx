/**
 * Forgot Password Page for ReUnity
 * Built by REOP Solutions
 */

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Link } from "wouter";
import { trpc } from "@/lib/trpc";
import { Loader2, ArrowLeft, Mail, CheckCircle } from "lucide-react";

export default function ForgotPassword() {
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSubmitted, setIsSubmitted] = useState(false);

  const requestResetMutation = trpc.auth.requestPasswordReset.useMutation();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const result = await requestResetMutation.mutateAsync({ email });
      if (result.success) {
        setIsSubmitted(true);
      } else {
        setError(result.error || "Failed to send reset email");
      }
    } catch (err) {
      // Don't reveal if email exists or not for security
      setIsSubmitted(true);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-4"
      style={{
        backgroundImage: "url('/fractured-spiral-bg-wide.png')",
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundAttachment: "fixed",
      }}
    >
      <div className="absolute inset-0 bg-black/60" />
      
      <div className="relative z-10 w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <Link href="/">
            <img 
              src="/reop-logo.png" 
              alt="REOP Solutions" 
              className="h-16 mx-auto mb-4 cursor-pointer hover:opacity-80 transition-opacity"
            />
          </Link>
          <h1 className="text-3xl font-bold text-white mb-2">Reset Password</h1>
          <p className="text-zinc-400">
            {isSubmitted 
              ? "Check your email for reset instructions" 
              : "Enter your email to receive a reset link"}
          </p>
        </div>

        {/* Form */}
        <div className="metallic-box p-8 rounded-xl">
          {isSubmitted ? (
            <div className="text-center space-y-6">
              <div className="w-16 h-16 mx-auto bg-emerald-500/20 rounded-full flex items-center justify-center">
                <CheckCircle className="h-8 w-8 text-emerald-400" />
              </div>
              <div className="space-y-2">
                <h2 className="text-xl font-semibold text-white">Check Your Email</h2>
                <p className="text-zinc-400 text-sm">
                  If an account exists with <span className="text-white">{email}</span>, 
                  you'll receive a password reset link shortly.
                </p>
              </div>
              <div className="pt-4">
                <Link href="/login">
                  <Button className="w-full emerald-btn text-white font-semibold">
                    Return to Login
                  </Button>
                </Link>
              </div>
              <p className="text-zinc-500 text-xs">
                Didn't receive an email? Check your spam folder or try again.
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-6">
              {error && (
                <div className="bg-red-500/20 border border-red-500/50 text-red-200 px-4 py-3 rounded-lg text-sm">
                  {error}
                </div>
              )}

              <div className="space-y-2">
                <label htmlFor="email" className="block text-sm font-medium text-zinc-300">
                  Email Address
                </label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-zinc-500" />
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    placeholder="you@example.com"
                    required
                    className="bg-zinc-800/50 border-zinc-700 text-white placeholder:text-zinc-500 pl-10"
                  />
                </div>
              </div>

              <Button
                type="submit"
                disabled={isLoading}
                className="w-full emerald-btn text-white font-semibold py-3"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Sending...
                  </>
                ) : (
                  "Send Reset Link"
                )}
              </Button>
            </form>
          )}

          {!isSubmitted && (
            <div className="mt-6 text-center">
              <Link href="/login" className="text-emerald-400 hover:text-emerald-300 font-medium inline-flex items-center gap-2">
                <ArrowLeft className="h-4 w-4" />
                Back to Login
              </Link>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="mt-8 text-center">
          <p className="text-zinc-500 text-sm">
            © 2025 REOP Solutions. All rights reserved.
          </p>
        </div>
      </div>
    </div>
  );
}
