/**
 * Authentication Routes for ReUnity
 * Built by REOP Solutions - Custom email/password authentication
 */

import { COOKIE_NAME, ONE_YEAR_MS } from "@shared/const";
import type { Express, Request, Response } from "express";
import { getSessionCookieOptions } from "./cookies";
import { sdk } from "./sdk";
import { 
  registerUser, 
  loginUser, 
  logoutUser, 
  requestPasswordReset, 
  resetPassword 
} from "../auth";

export function registerOAuthRoutes(app: Express) {
  // Register new user
  app.post("/api/auth/register", async (req: Request, res: Response) => {
    const { email, password, name } = req.body;

    if (!email || !password) {
      res.status(400).json({ error: "Email and password are required" });
      return;
    }

    try {
      const result = await registerUser(email, password, name);

      if (!result.success) {
        res.status(400).json({ error: result.error });
        return;
      }

      if (result.token && result.user) {
        const sessionToken = await sdk.createSessionToken(
          result.user.id,
          result.user.email,
          {
            name: result.user.name || "",
            expiresInMs: ONE_YEAR_MS,
          }
        );

        const cookieOptions = getSessionCookieOptions(req);
        res.cookie(COOKIE_NAME, sessionToken, { ...cookieOptions, maxAge: ONE_YEAR_MS });
      }

      res.json({ success: true, user: result.user, token: result.token });
    } catch (error) {
      console.error("[Auth] Registration failed", error);
      res.status(500).json({ error: "Registration failed" });
    }
  });

  // Login user
  app.post("/api/auth/login", async (req: Request, res: Response) => {
    const { email, password } = req.body;

    if (!email || !password) {
      res.status(400).json({ error: "Email and password are required" });
      return;
    }

    try {
      const userAgent = req.headers["user-agent"];
      const ipAddress = req.ip || req.socket.remoteAddress;

      const result = await loginUser(email, password, userAgent, ipAddress);

      if (!result.success) {
        res.status(401).json({ error: result.error });
        return;
      }

      if (result.token && result.user) {
        const sessionToken = await sdk.createSessionToken(
          result.user.id,
          result.user.email,
          {
            name: result.user.name || "",
            expiresInMs: ONE_YEAR_MS,
          }
        );

        const cookieOptions = getSessionCookieOptions(req);
        res.cookie(COOKIE_NAME, sessionToken, { ...cookieOptions, maxAge: ONE_YEAR_MS });
      }

      res.json({ success: true, user: result.user, token: result.token });
    } catch (error) {
      console.error("[Auth] Login failed", error);
      res.status(500).json({ error: "Login failed" });
    }
  });

  // Logout user
  app.post("/api/auth/logout", async (req: Request, res: Response) => {
    try {
      const authHeader = req.headers.authorization;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        const token = authHeader.substring(7);
        await logoutUser(token);
      }

      const cookieOptions = getSessionCookieOptions(req);
      res.clearCookie(COOKIE_NAME, cookieOptions);
      res.json({ success: true });
    } catch (error) {
      console.error("[Auth] Logout failed", error);
      res.status(500).json({ error: "Logout failed" });
    }
  });

  // Request password reset
  app.post("/api/auth/forgot-password", async (req: Request, res: Response) => {
    const { email } = req.body;

    if (!email) {
      res.status(400).json({ error: "Email is required" });
      return;
    }

    try {
      const result = await requestPasswordReset(email);
      // Always return success to prevent email enumeration
      res.json({ success: true, message: "If an account exists, a reset email has been sent" });
    } catch (error) {
      console.error("[Auth] Password reset request failed", error);
      res.status(500).json({ error: "Failed to process request" });
    }
  });

  // Reset password with token
  app.post("/api/auth/reset-password", async (req: Request, res: Response) => {
    const { token, password } = req.body;

    if (!token || !password) {
      res.status(400).json({ error: "Token and password are required" });
      return;
    }

    try {
      const result = await resetPassword(token, password);

      if (!result.success) {
        res.status(400).json({ error: result.error });
        return;
      }

      res.json({ success: true, message: "Password has been reset" });
    } catch (error) {
      console.error("[Auth] Password reset failed", error);
      res.status(500).json({ error: "Failed to reset password" });
    }
  });

  // Legacy OAuth callback - redirect to login page
  app.get("/api/oauth/callback", async (req: Request, res: Response) => {
    res.redirect(302, "/login");
  });
}
