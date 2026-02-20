import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, protectedProcedure, router } from "./_core/trpc";
import { z } from "zod";
import { reunity, EntropyState } from "./reunity";
import { generateSpeech, getAvailableVoices, filterVoices, getSupportedLanguages, OPENAI_VOICES, VoiceId } from "./tts";
import { ocrService } from "./ocr";
import { loginUser, registerUser, requestPasswordReset, resetPassword } from './auth';
// Stripe removed for beta - everything is free tier
import {
  createConversation,
  getConversation,
  getUserConversations,
  getActiveConversation,
  updateConversation,
  endConversation,
  createMessage,
  getConversationMessages,
  getRecentMessages,
  loadRIMEMemory,
  saveRIMEMemory,
  getUserMemories,
  saveUserMemory,
  createSessionAnalytics,
  updateSessionAnalytics,
  // Safety Plan
  getSafetyPlan,
  createSafetyPlan,
  updateSafetyPlan,
  // Peer Support
  getPeerProfile,
  createPeerProfile,
  updatePeerProfile,
  getActivePeerProfiles,
  getPeerConnection,
  getUserPeerConnections,
  createPeerConnection,
  updatePeerConnection,
  getPeerMessages,
  createPeerMessage,
  flagPeerMessage,
  createModerationAction,
  // Journal
  getJournalEntry,
  getUserJournalEntries,
  createJournalEntry,
  updateJournalEntry,
  deleteJournalEntry,
  getUserJournalInsights,
  createJournalInsight,
  dismissJournalInsight,
  // User Preferences
  getUserPreferences,
  updateUserPreferences,
  getOrCreateUserPreferences
} from "./db";

// Helper function to analyze journal content for entropy
function analyzeJournalContent(content: string): { entropy: number; state: string; patterns: string[] } {
  const textLower = content.toLowerCase();
  
  // Crisis keywords
  const crisisKeywords = ['suicide', 'kill myself', 'end it all', 'want to die', 'no reason to live', 'hurt myself'];
  const distressKeywords = ['anxious', 'panic', 'scared', 'terrified', 'overwhelmed', 'can\'t cope', 'falling apart', 'breaking down'];
  const sadKeywords = ['sad', 'depressed', 'hopeless', 'empty', 'numb', 'worthless', 'alone', 'lonely'];
  const positiveKeywords = ['grateful', 'happy', 'hopeful', 'better', 'progress', 'calm', 'peaceful', 'good day'];
  
  // Pattern detection
  const patterns: string[] = [];
  
  // Check for crisis
  if (crisisKeywords.some(k => textLower.includes(k))) {
    patterns.push('crisis_indicators');
    return { entropy: 0.95, state: 'crisis', patterns };
  }
  
  // Check for high distress
  const distressCount = distressKeywords.filter(k => textLower.includes(k)).length;
  if (distressCount >= 2) {
    patterns.push('high_distress');
    return { entropy: 0.75, state: 'high', patterns };
  }
  
  // Check for sadness/depression
  const sadCount = sadKeywords.filter(k => textLower.includes(k)).length;
  if (sadCount >= 2) {
    patterns.push('depressive_indicators');
    return { entropy: 0.6, state: 'moderate', patterns };
  }
  
  // Check for positive indicators
  const positiveCount = positiveKeywords.filter(k => textLower.includes(k)).length;
  if (positiveCount >= 2) {
    return { entropy: 0.2, state: 'stable', patterns: ['positive_progress'] };
  }
  
  // Default moderate state
  return { entropy: 0.45, state: 'moderate', patterns };
}

// Generate HTML for session export
function generateExportHTML(data: {
  title: string;
  date: Date;
  messages: Array<{
    role: string;
    content: string;
    timestamp: Date;
    state?: string;
    patterns?: string[];
    isCrisis?: boolean;
    groundingTechnique?: string;
  }>;
  userName: string;
}): string {
  const formatDate = (d: Date) => new Date(d).toLocaleString('en-US', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit'
  });

  const messagesHTML = data.messages.map(m => {
    const roleLabel = m.role === 'user' ? data.userName : 'ReUnity';
    const roleClass = m.role === 'user' ? 'user-message' : 'assistant-message';
    const crisisTag = m.isCrisis ? '<span class="crisis-tag">Crisis Detected</span>' : '';
    const stateTag = m.state ? `<span class="state-tag state-${m.state}">${m.state}</span>` : '';
    const patternsTag = m.patterns && m.patterns.length > 0 
      ? `<span class="patterns-tag">Patterns: ${m.patterns.join(', ')}</span>` 
      : '';
    const groundingTag = m.groundingTechnique 
      ? `<span class="grounding-tag">Grounding: ${m.groundingTechnique}</span>` 
      : '';
    
    return `
      <div class="message ${roleClass}">
        <div class="message-header">
          <strong>${roleLabel}</strong>
          <span class="timestamp">${formatDate(m.timestamp)}</span>
        </div>
        <div class="message-content">${m.content.replace(/\n/g, '<br>')}</div>
        <div class="message-meta">
          ${crisisTag}${stateTag}${patternsTag}${groundingTag}
        </div>
      </div>
    `;
  }).join('');

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>${data.title} - ReUnity Session Export</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      background: #f5f5f5;
      color: #333;
      line-height: 1.6;
      padding: 2rem;
    }
    .container {
      max-width: 800px;
      margin: 0 auto;
      background: white;
      border-radius: 12px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.1);
      overflow: hidden;
    }
    .header {
      background: linear-gradient(135deg, #0a4035 0%, #1a8a6e 100%);
      color: white;
      padding: 2rem;
      text-align: center;
    }
    .header h1 {
      font-size: 1.5rem;
      margin-bottom: 0.5rem;
    }
    .header .subtitle {
      opacity: 0.8;
      font-size: 0.9rem;
    }
    .disclaimer {
      background: #fff3cd;
      border-left: 4px solid #ffc107;
      padding: 1rem;
      margin: 1rem;
      font-size: 0.85rem;
      color: #856404;
    }
    .messages {
      padding: 1rem;
    }
    .message {
      margin-bottom: 1.5rem;
      padding: 1rem;
      border-radius: 8px;
    }
    .user-message {
      background: #e8f5e9;
      border-left: 4px solid #1a8a6e;
    }
    .assistant-message {
      background: #f5f5f5;
      border-left: 4px solid #666;
    }
    .message-header {
      display: flex;
      justify-content: space-between;
      margin-bottom: 0.5rem;
      font-size: 0.85rem;
    }
    .timestamp {
      color: #666;
    }
    .message-content {
      margin-bottom: 0.5rem;
    }
    .message-meta {
      display: flex;
      flex-wrap: wrap;
      gap: 0.5rem;
      font-size: 0.75rem;
    }
    .crisis-tag {
      background: #dc2626;
      color: white;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .state-tag {
      background: #666;
      color: white;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .state-crisis { background: #dc2626; }
    .state-high { background: #f59e0b; }
    .state-moderate { background: #eab308; }
    .state-low { background: #1a8a6e; }
    .state-stable { background: #22c55e; }
    .patterns-tag {
      background: #f59e0b;
      color: white;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .grounding-tag {
      background: #1a8a6e;
      color: white;
      padding: 2px 8px;
      border-radius: 4px;
    }
    .footer {
      background: #f5f5f5;
      padding: 1.5rem;
      text-align: center;
      font-size: 0.85rem;
      color: #666;
      border-top: 1px solid #ddd;
    }
    .footer a {
      color: #1a8a6e;
      text-decoration: none;
    }
    @media print {
      body { padding: 0; background: white; }
      .container { box-shadow: none; }
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>ReUnity Session Export</h1>
      <div class="subtitle">${data.title} • ${formatDate(data.date)}</div>
    </div>
    
    <div class="disclaimer">
      <strong>Important:</strong> This document is intended for sharing with licensed mental health professionals. 
      ReUnity is not a replacement for professional care. If you are in crisis, please call 988.
    </div>
    
    <div class="messages">
      ${messagesHTML}
    </div>
    
    <div class="footer">
      <p>Generated by ReUnity • <a href="https://entropy-physics-ai.com">entropy-physics-ai.com</a></p>
      <p>© ${new Date().getFullYear()} REOP Solutions. All rights reserved.</p>
    </div>
  </div>
</body>
</html>
  `;
}

export const appRouter = router({
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    
    login: publicProcedure
      .input(z.object({
        email: z.string().email(),
        password: z.string().min(1)
      }))
      .mutation(async ({ ctx, input }) => {
        const userAgent = ctx.req.headers['user-agent'];
        const ipAddress = ctx.req.ip || ctx.req.socket.remoteAddress;
        const result = await loginUser(input.email, input.password, userAgent, ipAddress);
        
        if (result.success && result.token) {
          const cookieOptions = getSessionCookieOptions(ctx.req);
          ctx.res.cookie(COOKIE_NAME, result.token, { ...cookieOptions, maxAge: 30 * 24 * 60 * 60 * 1000 });
        }
        
        return result;
      }),
    
    register: publicProcedure
      .input(z.object({
        email: z.string().email(),
        password: z.string().min(8),
        name: z.string().optional()
      }))
      .mutation(async ({ ctx, input }) => {
        const result = await registerUser(input.email, input.password, input.name);
        
        if (result.success && result.token) {
          const cookieOptions = getSessionCookieOptions(ctx.req);
          ctx.res.cookie(COOKIE_NAME, result.token, { ...cookieOptions, maxAge: 30 * 24 * 60 * 60 * 1000 });
        }
        
        return result;
      }),
    
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
    
    requestPasswordReset: publicProcedure
      .input(z.object({
        email: z.string().email()
      }))
      .mutation(async ({ input }) => {
        return await requestPasswordReset(input.email);
      }),
    
    resetPassword: publicProcedure
      .input(z.object({
        token: z.string(),
        newPassword: z.string().min(8)
      }))
      .mutation(async ({ input }) => {
        return await resetPassword(input.token, input.newPassword);
      }),
  }),

  // ============================================
  // CONVERSATION MANAGEMENT
  // ============================================
  conversations: router({
    /** Create a new conversation */
    create: protectedProcedure
      .input(z.object({
        title: z.string().optional()
      }).optional())
      .mutation(async ({ ctx, input }) => {
        const conversation = await createConversation({
          userId: ctx.user.id,
          title: input?.title ?? "New Session",
          currentState: "stable",
          currentRegime: "normal",
          isActive: true
        });

        if (conversation) {
          // Create session analytics entry
          await createSessionAnalytics({
            conversationId: conversation.id,
            userId: ctx.user.id,
            messageCount: 0,
            crisisCount: 0,
            patternCount: 0,
            groundingCount: 0
          });
        }

        return conversation;
      }),

    /** Get all conversations for the current user */
    list: protectedProcedure
      .input(z.object({
        limit: z.number().min(1).max(100).optional().default(20)
      }).optional())
      .query(async ({ ctx, input }) => {
        return await getUserConversations(ctx.user.id, input?.limit || 20);
      }),

    /** Get a specific conversation */
    get: protectedProcedure
      .input(z.object({
        conversationId: z.number()
      }))
      .query(async ({ ctx, input }) => {
        const conversation = await getConversation(input.conversationId);
        if (conversation && conversation.userId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }
        return conversation;
      }),

    /** Get the active conversation or create one */
    getOrCreateActive: protectedProcedure.mutation(async ({ ctx }) => {
      let conversation = await getActiveConversation(ctx.user.id);
      
      if (!conversation) {
        conversation = await createConversation({
          userId: ctx.user.id,
          title: "New Session",
          currentState: "stable",
          currentRegime: "normal",
          isActive: true
        });

        if (conversation) {
          await createSessionAnalytics({
            conversationId: conversation.id,
            userId: ctx.user.id,
            messageCount: 0,
            crisisCount: 0,
            patternCount: 0,
            groundingCount: 0
          });
        }
      }

      return conversation;
    }),

    /** End a conversation */
    end: protectedProcedure
      .input(z.object({
        conversationId: z.number()
      }))
      .mutation(async ({ ctx, input }) => {
        const conversation = await getConversation(input.conversationId);
        if (conversation && conversation.userId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }
        await endConversation(input.conversationId);
        return { success: true };
      }),

    /** Get messages for a conversation */
    getMessages: protectedProcedure
      .input(z.object({
        conversationId: z.number(),
        limit: z.number().min(1).max(500).optional().default(100)
      }))
      .query(async ({ ctx, input }) => {
        const conversation = await getConversation(input.conversationId);
        if (conversation && conversation.userId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }
        return await getConversationMessages(input.conversationId, input.limit);
      })
  }),

  // ============================================
  // RIME MEMORY MANAGEMENT
  // ============================================
  memory: router({
    /** Load all memories for the current user */
    load: protectedProcedure.query(async ({ ctx }) => {
      return await loadRIMEMemory(ctx.user.id);
    }),

    /** Get all raw memories */
    getAll: protectedProcedure.query(async ({ ctx }) => {
      return await getUserMemories(ctx.user.id);
    }),

    /** Save a specific memory */
    save: protectedProcedure
      .input(z.object({
        memoryType: z.enum(["grounding_anchor", "known_trigger", "safe_place", "name", "preference"]),
        memoryKey: z.string().max(128),
        memoryValue: z.string()
      }))
      .mutation(async ({ ctx, input }) => {
        return await saveUserMemory({
          userId: ctx.user.id,
          memoryType: input.memoryType,
          memoryKey: input.memoryKey,
          memoryValue: input.memoryValue
        });
      }),

    /** Bulk save RIME memory */
    saveRIME: protectedProcedure
      .input(z.object({
        groundingAnchors: z.array(z.string()).optional(),
        knownTriggers: z.array(z.string()).optional(),
        safePlace: z.string().nullable().optional(),
        userName: z.string().nullable().optional(),
        preferences: z.record(z.string(), z.string()).optional()
      }))
      .mutation(async ({ ctx, input }) => {
        await saveRIMEMemory(ctx.user.id, {
          ...input,
          preferences: input.preferences as Record<string, string> | undefined
        });
        return { success: true };
      })
  }),

  reunity: router({
    /**
     * Main chat endpoint - processes messages through the full ReUnity pipeline
     * Now with database persistence for authenticated users
     */
    chat: publicProcedure
      .input(z.object({
        message: z.string().min(1).max(10000),
        conversationId: z.number().optional(),
        conversationHistory: z.array(z.object({
          role: z.enum(["user", "assistant"]),
          content: z.string()
        })).optional().default([])
      }))
      .mutation(async ({ ctx, input }) => {
        // Load RIME memory if user is authenticated
        let rimeMemory = null;
        if (ctx.user) {
          rimeMemory = await loadRIMEMemory(ctx.user.id);
          
          // Inject memory into ReUnity
          if (rimeMemory.userName) {
            reunity.setUserName(rimeMemory.userName);
          }
          if (rimeMemory.safePlaces && rimeMemory.safePlaces.length > 0) {
            reunity.setSafePlace(rimeMemory.safePlaces[0]);
          }
          if (rimeMemory.groundingAnchors.length > 0) {
            reunity.setGroundingAnchors(rimeMemory.groundingAnchors);
          }
          if (rimeMemory.knownTriggers.length > 0) {
            reunity.setKnownTriggers(rimeMemory.knownTriggers);
          }
        }

        // Process message through ReUnity
        const result = await reunity.processMessage(
          input.message,
          input.conversationHistory
        );

        // Save to database if user is authenticated and has a conversation
        if (ctx.user && input.conversationId) {
          // Save user message
          await createMessage({
            conversationId: input.conversationId,
            role: "user",
            content: input.message,
            entropyScore: result.entropy.toString(),
            detectedState: result.state,
            detectedPatterns: result.patterns,
            isCrisis: result.isCrisis
          });

          // Save assistant response
          await createMessage({
            conversationId: input.conversationId,
            role: "assistant",
            content: result.response,
            groundingTechnique: result.groundingTechnique?.name
          });

          // Update conversation state
          await updateConversation(input.conversationId, {
            currentState: result.state,
            currentRegime: result.regime
          });

          // Update session analytics
          const analytics = {
            messageCount: input.conversationHistory.length + 2,
            crisisCount: result.isCrisis ? 1 : 0,
            patternCount: result.patterns.length,
            groundingCount: result.groundingTechnique ? 1 : 0
          };
          await updateSessionAnalytics(input.conversationId, analytics);

          // Save any new memories detected
          if (result.memoryUpdated) {
            const sessionStatus = reunity.getSessionStatus();
            await saveRIMEMemory(ctx.user.id, {
              groundingAnchors: sessionStatus.memory.groundingAnchors,
              knownTriggers: sessionStatus.memory.knownTriggers,
              safePlaces: sessionStatus.memory.safePlace ? [sessionStatus.memory.safePlace] : [],
              userName: sessionStatus.memory.userName
            });
          }
        }
        
        return {
          response: result.response,
          state: result.state,
          entropy: result.entropy,
          patterns: result.patterns,
          groundingTechnique: result.groundingTechnique ? {
            name: result.groundingTechnique.name,
            steps: result.groundingTechnique.steps
          } : undefined,
          isCrisis: result.isCrisis,
          regime: result.regime,
          dissociationDetected: result.dissociationDetected,
          memoryUpdated: result.memoryUpdated,
          resources: result.resources
        };
      }),

    /**
     * OCR endpoint - processes images to extract text for analysis
     * Supports screenshots, journal entries, handwritten notes
     */
    processImage: publicProcedure
      .input(z.object({
        imageUrl: z.string().url(),
        imageType: z.enum(["general", "conversation", "journal"]).optional().default("general")
      }))
      .mutation(async ({ input }) => {
        if (input.imageType === "conversation") {
          const result = await ocrService.analyzeConversation(input.imageUrl);
          return {
            success: result.success,
            type: "conversation" as const,
            extractedText: result.messages.map(m => `${m.sender}: ${m.content}`).join("\n"),
            messages: result.messages,
            patterns: result.patterns,
            concernLevel: result.concernLevel,
            analysis: result.analysis,
            error: result.error
          };
        } else if (input.imageType === "journal") {
          const result = await ocrService.analyzeJournalEntry(input.imageUrl);
          return {
            success: result.success,
            type: "journal" as const,
            extractedText: result.text,
            emotionalState: result.emotionalState,
            themes: result.themes,
            supportNeeded: result.supportNeeded,
            error: result.error
          };
        } else {
          const result = await ocrService.processImage(input.imageUrl);
          return {
            success: result.success,
            type: "general" as const,
            extractedText: result.extractedText,
            contentType: result.contentType,
            emotionalContent: result.emotionalContent,
            suggestedContext: result.suggestedContext,
            confidence: result.confidence,
            error: result.error
          };
        }
      }),

    /**
     * Combined OCR + Chat endpoint
     * Processes an image, extracts text, and sends it through the ReUnity pipeline
     */
    chatWithImage: publicProcedure
      .input(z.object({
        imageUrl: z.string().url(),
        additionalMessage: z.string().optional(),
        conversationId: z.number().optional(),
        conversationHistory: z.array(z.object({
          role: z.enum(["user", "assistant"]),
          content: z.string()
        })).optional().default([])
      }))
      .mutation(async ({ ctx, input }) => {
        // First, process the image
        const ocrResult = await ocrService.processImage(input.imageUrl);
        
        if (!ocrResult.success || !ocrResult.extractedText) {
          return {
            success: false,
            error: ocrResult.error || "Failed to extract text from image",
            ocrResult: null,
            chatResult: null
          };
        }
        
        // Build the message combining OCR result and any additional context
        let message = `[Image content]: ${ocrResult.extractedText}`;
        
        if (ocrResult.suggestedContext) {
          message = `[Context: ${ocrResult.suggestedContext}]\n\n${message}`;
        }
        
        if (input.additionalMessage) {
          message = `${message}\n\n[User's message about this]: ${input.additionalMessage}`;
        }

        // Load RIME memory if authenticated
        if (ctx.user) {
          const rimeMemory = await loadRIMEMemory(ctx.user.id);
          if (rimeMemory.userName) reunity.setUserName(rimeMemory.userName);
          if (rimeMemory.safePlaces && rimeMemory.safePlaces.length > 0) reunity.setSafePlace(rimeMemory.safePlaces[0]);
        }
        
        // Process through ReUnity
        const chatResult = await reunity.processMessage(
          message,
          input.conversationHistory
        );

        // Save to database if authenticated
        if (ctx.user && input.conversationId) {
          await createMessage({
            conversationId: input.conversationId,
            role: "user",
            content: message,
            entropyScore: chatResult.entropy.toString(),
            detectedState: chatResult.state,
            detectedPatterns: chatResult.patterns,
            isCrisis: chatResult.isCrisis
          });

          await createMessage({
            conversationId: input.conversationId,
            role: "assistant",
            content: chatResult.response,
            groundingTechnique: chatResult.groundingTechnique?.name
          });
        }
        
        return {
          success: true,
          ocrResult: {
            extractedText: ocrResult.extractedText,
            contentType: ocrResult.contentType,
            emotionalContent: ocrResult.emotionalContent
          },
          chatResult: {
            response: chatResult.response,
            state: chatResult.state,
            entropy: chatResult.entropy,
            patterns: chatResult.patterns,
            groundingTechnique: chatResult.groundingTechnique,
            isCrisis: chatResult.isCrisis,
            regime: chatResult.regime,
            dissociationDetected: chatResult.dissociationDetected,
            resources: chatResult.resources
          }
        };
      }),

    /**
     * Analyze conversation screenshot specifically for pattern detection
     */
    analyzeConversation: publicProcedure
      .input(z.object({
        imageUrl: z.string().url()
      }))
      .mutation(async ({ input }) => {
        return await ocrService.analyzeConversation(input.imageUrl);
      }),

    /**
     * Analyze journal entry for emotional content
     */
    analyzeJournal: publicProcedure
      .input(z.object({
        imageUrl: z.string().url()
      }))
      .mutation(async ({ input }) => {
        return await ocrService.analyzeJournalEntry(input.imageUrl);
      }),

    /**
     * Get session status - returns current regime and memory state
     */
    getSessionStatus: publicProcedure.query(() => {
      return reunity.getSessionStatus();
    }),

    /**
     * Reset the session - clears memory and resets regime
     */
    resetSession: publicProcedure.mutation(() => {
      reunity.resetSession();
      return { success: true };
    }),

    /**
     * Get available grounding techniques
     */
    getGroundingTechniques: publicProcedure.query(() => {
      return {
        techniques: [
          {
            id: "5_4_3_2_1",
            name: "5-4-3-2-1 Sensory Grounding",
            bestFor: ["dissociation", "anxiety", "panic", "flashback"],
            description: "Uses all five senses to anchor you to the present moment"
          },
          {
            id: "box_breathing",
            name: "Box Breathing",
            bestFor: ["anxiety", "panic", "overwhelm", "racing thoughts"],
            description: "A structured breathing technique to activate your parasympathetic nervous system"
          },
          {
            id: "feet_on_floor",
            name: "Feet on Floor Grounding",
            bestFor: ["dissociation", "floating", "derealization", "disconnection"],
            description: "Physical grounding through connection to the earth"
          },
          {
            id: "cold_water",
            name: "Cold Water Reset",
            bestFor: ["dissociation", "panic", "intense emotion", "crisis"],
            description: "Uses cold sensation to activate the dive reflex and calm your nervous system"
          },
          {
            id: "grounding_statements",
            name: "Grounding Statements",
            bestFor: ["dissociation", "flashback", "confusion", "identity"],
            description: "Verbal affirmations to reconnect with your identity and the present"
          },
          {
            id: "progressive_muscle",
            name: "Progressive Muscle Relaxation",
            bestFor: ["anxiety", "tension", "stress", "insomnia"],
            description: "Systematic tensing and releasing of muscle groups"
          },
          {
            id: "safe_place",
            name: "Safe Place Visualization",
            bestFor: ["anxiety", "trauma", "flashback", "overwhelm"],
            description: "Mental imagery of a safe, calming environment"
          },
          {
            id: "butterfly_hug",
            name: "Butterfly Hug",
            bestFor: ["trauma", "flashback", "anxiety", "distress"],
            description: "Bilateral stimulation through self-administered tapping"
          },
          {
            id: "container",
            name: "Container Technique",
            bestFor: ["intrusive thoughts", "OCD", "rumination", "anxiety"],
            description: "Visualization technique to contain distressing thoughts"
          },
          {
            id: "tipp",
            name: "TIPP Intense Exercise",
            bestFor: ["intense emotion", "crisis", "anger", "panic"],
            description: "Brief intense exercise to change body chemistry"
          },
          {
            id: "opposite_action",
            name: "Opposite Action",
            bestFor: ["depression", "avoidance", "fear", "shame"],
            description: "Acting opposite to the emotion-driven urge"
          },
          {
            id: "stop_skill",
            name: "STOP Skill",
            bestFor: ["impulsivity", "crisis", "splitting", "reactivity"],
            description: "Stop, Take a step back, Observe, Proceed mindfully"
          }
        ]
      };
    }),

    /**
     * Create a new conversation
     */
    createConversation: protectedProcedure
      .input(z.object({
        title: z.string().optional()
      }).optional())
      .mutation(async ({ ctx, input }) => {
        return await createConversation({
          userId: ctx.user.id,
          title: input?.title ?? "New Session",
          currentState: "stable",
          currentRegime: "normal",
          isActive: true
        });
      }),

    /**
     * Get user's conversations
     */
    getConversations: protectedProcedure.query(async ({ ctx }) => {
      return await getUserConversations(ctx.user.id, 50);
    }),

    /**
     * Get messages for a conversation
     */
    getConversationMessages: protectedProcedure
      .input(z.object({
        conversationId: z.number()
      }))
      .mutation(async ({ ctx, input }) => {
        const conversation = await getConversation(input.conversationId);
        if (!conversation || conversation.userId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }
        return await getConversationMessages(input.conversationId);
      }),

    /**
     * Save a message to a conversation
     */
    saveMessage: protectedProcedure
      .input(z.object({
        conversationId: z.number(),
        role: z.enum(["user", "assistant"]),
        content: z.string(),
        entropyScore: z.string().optional(),
        detectedState: z.string().optional(),
        detectedPatterns: z.array(z.string()).optional(),
        groundingTechnique: z.string().optional(),
        isCrisis: z.boolean().optional()
      }))
      .mutation(async ({ ctx, input }) => {
        const conversation = await getConversation(input.conversationId);
        if (!conversation || conversation.userId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }
        return await createMessage(input);
      }),

    /**
     * Load RIME memory for authenticated user
     */
    loadMemory: protectedProcedure.query(async ({ ctx }) => {
      return await loadRIMEMemory(ctx.user.id);
    }),

    /**
     * Save RIME memory for authenticated user
     */
    saveMemory: protectedProcedure
      .input(z.object({
        groundingAnchors: z.array(z.string()).optional(),
        knownTriggers: z.array(z.string()).optional(),
        safePlace: z.string().optional(),
        userName: z.string().optional()
      }))
      .mutation(async ({ ctx, input }) => {
        return await saveRIMEMemory(ctx.user.id, input);
      }),

    /**
     * Export conversation to PDF format
     */
    exportConversation: protectedProcedure
      .input(z.object({
        conversationId: z.number()
      }))
      .mutation(async ({ ctx, input }) => {
        const conversation = await getConversation(input.conversationId);
        if (!conversation || conversation.userId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }
        
        const messages = await getConversationMessages(input.conversationId);
        
        // Format messages for export
        const formattedMessages = messages.map((m: any) => ({
          role: m.role,
          content: m.content,
          timestamp: m.createdAt,
          state: m.detectedState,
          patterns: m.detectedPatterns,
          isCrisis: m.isCrisis,
          groundingTechnique: m.groundingTechnique
        }));
        
        // Generate HTML content for PDF
        const htmlContent = generateExportHTML({
          title: conversation.title || 'ReUnity Session',
          date: conversation.createdAt,
          messages: formattedMessages,
          userName: ctx.user.name || 'Anonymous'
        });
        
        return {
          success: true,
          html: htmlContent,
          filename: `reunity-session-${conversation.id}-${new Date().toISOString().split('T')[0]}.html`
        };
      }),

    /**
     * Get crisis resources
     */
    getCrisisResources: publicProcedure.query(() => {
      return {
        resources: [
          {
            name: "988 Suicide & Crisis Lifeline",
            contact: "988",
            type: "call_or_text",
            available: "24/7",
            description: "Free, confidential support for people in distress"
          },
          {
            name: "Crisis Text Line",
            contact: "Text HOME to 741741",
            type: "text",
            available: "24/7",
            description: "Free crisis counseling via text message"
          },
          {
            name: "National Domestic Violence Hotline",
            contact: "1-800-799-7233",
            type: "call",
            available: "24/7",
            description: "Support for domestic violence survivors"
          },
          {
            name: "RAINN (Sexual Assault)",
            contact: "1-800-656-4673",
            type: "call",
            available: "24/7",
            description: "Support for sexual assault survivors"
          },
          {
            name: "Trans Lifeline",
            contact: "1-877-565-8860",
            type: "call",
            available: "24/7",
            description: "Peer support for transgender people"
          },
          {
            name: "Trevor Project (LGBTQ+ Youth)",
            contact: "1-866-488-7386",
            type: "call",
            available: "24/7",
            description: "Crisis intervention for LGBTQ+ young people"
          },
          {
            name: "SAMHSA National Helpline",
            contact: "1-800-662-4357",
            type: "call",
            available: "24/7",
            description: "Treatment referral for substance abuse and mental health"
          },
          {
            name: "NEDA (Eating Disorders)",
            contact: "1-800-931-2237",
            type: "call",
            available: "Mon-Thu 11am-9pm ET, Fri 11am-5pm ET",
            description: "Support for eating disorder recovery"
          }
        ]
      };
    })
  }),

  // ============================================
  // SAFETY PLAN ROUTER
  // ============================================
  safetyPlan: router({
    /** Get user's safety plan */
    get: protectedProcedure.query(async ({ ctx }) => {
      return await getSafetyPlan(ctx.user.id);
    }),

    /** Create or update safety plan */
    save: protectedProcedure
      .input(z.object({
        planData: z.string(), // JSON stringified plan data
        isComplete: z.boolean().optional(),
        currentStep: z.string().optional()
      }))
      .mutation(async ({ ctx, input }) => {
        const existing = await getSafetyPlan(ctx.user.id);
        
        if (existing) {
          await updateSafetyPlan(ctx.user.id, {
            encryptedData: input.planData,
            isComplete: input.isComplete,
            lastStepId: input.currentStep
          });
          return { success: true, updated: true };
        } else {
          await createSafetyPlan({
            userId: ctx.user.id,
            encryptedData: input.planData,
            isComplete: input.isComplete || false,
            lastStepId: input.currentStep || '0'
          });
          return { success: true, created: true };
        }
      }),

    /** Export safety plan as PDF-ready HTML */
    export: protectedProcedure.mutation(async ({ ctx }) => {
      const plan = await getSafetyPlan(ctx.user.id);
      if (!plan) {
        throw new Error("No safety plan found");
      }

      const planData = JSON.parse(plan.encryptedData);
      const html = generateSafetyPlanHTML(planData);
      
      return {
        success: true,
        html,
        filename: `safety-plan-${new Date().toISOString().split('T')[0]}.html`
      };
    })
  }),

  // ============================================
  // PEER SUPPORT ROUTER
  // ============================================
  peerSupport: router({
    /** Get user's peer profile */
    getProfile: protectedProcedure.query(async ({ ctx }) => {
      return await getPeerProfile(ctx.user.id);
    }),

    /** Create or update peer profile */
    saveProfile: protectedProcedure
      .input(z.object({
        displayName: z.string().max(50),
        experienceTags: z.array(z.string()),
        lookingFor: z.array(z.string()),
        isActive: z.boolean().optional()
      }))
      .mutation(async ({ ctx, input }) => {
        const existing = await getPeerProfile(ctx.user.id);
        
        // Store experiences and preferences as JSON
        const experiences = input.experienceTags;
        const preferences = { lookingFor: input.lookingFor };
        
        if (existing) {
          await updatePeerProfile(ctx.user.id, {
            displayName: input.displayName,
            experiences: experiences,
            preferences: preferences,
            isActive: input.isActive ?? true
          });
          return { success: true, updated: true };
        } else {
          await createPeerProfile({
            userId: ctx.user.id,
            displayName: input.displayName,
            experiences: experiences,
            preferences: preferences,
            isActive: true,
            isBanned: false
          });
          return { success: true, created: true };
        }
      }),

    /** Get potential peer matches */
    getMatches: protectedProcedure.query(async ({ ctx }) => {
      const myProfile = await getPeerProfile(ctx.user.id);
      if (!myProfile) {
        return { matches: [], needsProfile: true };
      }

      const allProfiles = await getActivePeerProfiles(ctx.user.id);
      const myExperiences = Array.isArray(myProfile.experiences) ? myProfile.experiences : [];
      const myPrefs = myProfile.preferences as { lookingFor?: string[] } | null;
      const myLookingFor = myPrefs?.lookingFor || [];

      // Calculate match scores
      const matches = allProfiles.map(profile => {
        const theirExperiences = Array.isArray(profile.experiences) ? profile.experiences : [];
        const theirPrefs = profile.preferences as { lookingFor?: string[] } | null;
        const theirLookingFor = theirPrefs?.lookingFor || [];

        // Score based on overlapping experiences and complementary needs
        const experienceOverlap = myExperiences.filter((t: any) => theirExperiences.includes(t)).length;
        const needsMatch = myLookingFor.filter((t: any) => theirExperiences.includes(t)).length;
        const theyNeedMe = theirLookingFor.filter((t: any) => myExperiences.includes(t)).length;

        const score = experienceOverlap * 2 + needsMatch * 3 + theyNeedMe * 3;

        return {
          id: profile.id,
          displayName: profile.displayName,
          experienceTags: theirExperiences,
          matchScore: score,
          sharedExperiences: myExperiences.filter((t: any) => theirExperiences.includes(t))
        };
      });

      // Sort by match score
      matches.sort((a, b) => b.matchScore - a.matchScore);

      return { matches: matches.slice(0, 20), needsProfile: false };
    }),

    /** Get user's connections */
    getConnections: protectedProcedure.query(async ({ ctx }) => {
      const connections = await getUserPeerConnections(ctx.user.id);
      
      // Get profiles for each connection
      const enrichedConnections = await Promise.all(
        connections.map(async (conn) => {
          const otherUserId = conn.requesterId === ctx.user.id 
            ? conn.responderId 
            : conn.requesterId;
          const profile = await getPeerProfile(otherUserId);
          return {
            ...conn,
            peerProfile: profile
          };
        })
      );

      return enrichedConnections;
    }),

    /** Send connection request */
    requestConnection: protectedProcedure
      .input(z.object({
        targetProfileId: z.number(),
        message: z.string().max(500).optional()
      }))
      .mutation(async ({ ctx, input }) => {
        // Get target profile to find userId
        const allProfiles = await getActivePeerProfiles();
        const targetProfile = allProfiles.find(p => p.id === input.targetProfileId);
        
        if (!targetProfile) {
          throw new Error("Profile not found");
        }

        await createPeerConnection({
          requesterId: ctx.user.id,
          responderId: targetProfile.userId,
          status: 'pending'
        });

        return { success: true };
      }),

    /** Accept or reject connection */
    respondToConnection: protectedProcedure
      .input(z.object({
        connectionId: z.number(),
        accept: z.boolean()
      }))
      .mutation(async ({ ctx, input }) => {
        const connection = await getPeerConnection(input.connectionId);
        if (!connection || connection.responderId !== ctx.user.id) {
          throw new Error("Connection not found or unauthorized");
        }

        await updatePeerConnection(input.connectionId, {
          status: input.accept ? 'accepted' : 'declined'
        });

        return { success: true };
      }),

    /** Get messages for a connection */
    getMessages: protectedProcedure
      .input(z.object({
        connectionId: z.number()
      }))
      .query(async ({ ctx, input }) => {
        const connection = await getPeerConnection(input.connectionId);
        if (!connection) {
          throw new Error("Connection not found");
        }
        if (connection.requesterId !== ctx.user.id && connection.responderId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }

        return await getPeerMessages(input.connectionId);
      }),

    /** Send message to peer */
    sendMessage: protectedProcedure
      .input(z.object({
        connectionId: z.number(),
        content: z.string().max(2000)
      }))
      .mutation(async ({ ctx, input }) => {
        const connection = await getPeerConnection(input.connectionId);
        if (!connection || connection.status !== 'accepted') {
          throw new Error("Connection not found or not accepted");
        }
        if (connection.requesterId !== ctx.user.id && connection.responderId !== ctx.user.id) {
          throw new Error("Unauthorized");
        }

        // Check for crisis content
        const lowerContent = input.content.toLowerCase();
        const crisisKeywords = ['suicide', 'kill myself', 'end it all', 'want to die', 'no reason to live'];
        const isCrisis = crisisKeywords.some(k => lowerContent.includes(k));

        const message = await createPeerMessage({
          connectionId: input.connectionId,
          senderId: ctx.user.id,
          content: input.content,
          crisisDetected: isCrisis
        });

        return { success: true, message, isCrisis };
      }),

    /** Flag a message for moderation */
    flagMessage: protectedProcedure
      .input(z.object({
        messageId: z.number(),
        reason: z.string().max(500)
      }))
      .mutation(async ({ ctx, input }) => {
        await flagPeerMessage(input.messageId, input.reason);
        return { success: true };
      }),

    /** Check for new messages (polling) */
    checkNewMessages: protectedProcedure
      .input(z.object({
        connectionId: z.number(),
        lastMessageId: z.number().optional()
      }))
      .query(async ({ ctx, input }) => {
        const connection = await getPeerConnection(input.connectionId);
        if (!connection) {
          return { hasNew: false, messages: [] };
        }
        if (connection.requesterId !== ctx.user.id && connection.responderId !== ctx.user.id) {
          return { hasNew: false, messages: [] };
        }

        const messages = await getPeerMessages(input.connectionId, 50);
        
        if (input.lastMessageId) {
          const newMessages = messages.filter(m => m.id > input.lastMessageId!);
          return { hasNew: newMessages.length > 0, messages: newMessages };
        }

        return { hasNew: false, messages };
      })
  }),

  // ============================================
  // JOURNAL ROUTER
  // ============================================
  journal: router({
    /** Get user's journal entries */
    getEntries: protectedProcedure
      .input(z.object({
        limit: z.number().optional().default(50)
      }))
      .query(async ({ ctx, input }) => {
        return await getUserJournalEntries(ctx.user.id, input.limit);
      }),

    /** Get a single journal entry */
    getEntry: protectedProcedure
      .input(z.object({
        entryId: z.number()
      }))
      .query(async ({ ctx, input }) => {
        const entry = await getJournalEntry(input.entryId);
        if (!entry || entry.userId !== ctx.user.id) {
          throw new Error("Entry not found");
        }
        return entry;
      }),

    /** Create a new journal entry */
    createEntry: protectedProcedure
      .input(z.object({
        title: z.string().max(200).optional(),
        content: z.string().max(10000),
        moodTags: z.array(z.string()).optional()
      }))
      .mutation(async ({ ctx, input }) => {
        // Analyze content for entropy using a simple analysis
        const analysisResult = analyzeJournalContent(input.content);

        const entry = await createJournalEntry({
          userId: ctx.user.id,
          title: input.title,
          content: input.content,
          moodTags: input.moodTags ? JSON.stringify(input.moodTags) : undefined,
          entropyScore: analysisResult.entropy.toString(),
          entropyState: analysisResult.state,
          detectedStates: analysisResult.patterns ? JSON.stringify(analysisResult.patterns) : undefined
        });

        // Generate insights if patterns detected
        if (analysisResult.patterns && analysisResult.patterns.length > 0) {
          for (const pattern of analysisResult.patterns) {
            await createJournalInsight({
              userId: ctx.user.id,
              insightType: 'pattern',
              title: `Pattern Detected: ${pattern}`,
              description: `Your journal entry shows signs of ${pattern}. Consider reviewing coping strategies.`,
              relatedEntries: entry?.id ? [entry.id] : []
            });
          }
        }

        return { 
          success: true, 
          entry,
          analysis: {
            entropy: analysisResult.entropy,
            state: analysisResult.state,
            patterns: analysisResult.patterns
          }
        };
      }),

    /** Update a journal entry */
    updateEntry: protectedProcedure
      .input(z.object({
        entryId: z.number(),
        title: z.string().max(200).optional(),
        content: z.string().max(10000).optional(),
        moodTags: z.array(z.string()).optional()
      }))
      .mutation(async ({ ctx, input }) => {
        const entry = await getJournalEntry(input.entryId);
        if (!entry || entry.userId !== ctx.user.id) {
          throw new Error("Entry not found");
        }

        const updates: any = {};
        if (input.title !== undefined) updates.title = input.title;
        if (input.content !== undefined) {
          updates.content = input.content;
          // Re-analyze if content changed
          const analysisResult = analyzeJournalContent(input.content);
          updates.entropyScore = analysisResult.entropy.toString();
          updates.entropyState = analysisResult.state;
          updates.detectedStates = analysisResult.patterns ? JSON.stringify(analysisResult.patterns) : undefined;
        }
        if (input.moodTags !== undefined) updates.moodTags = JSON.stringify(input.moodTags);

        await updateJournalEntry(input.entryId, updates);
        return { success: true };
      }),

    /** Delete a journal entry */
    deleteEntry: protectedProcedure
      .input(z.object({
        entryId: z.number()
      }))
      .mutation(async ({ ctx, input }) => {
        await deleteJournalEntry(input.entryId, ctx.user.id);
        return { success: true };
      }),

    /** Get user's journal insights */
    getInsights: protectedProcedure.query(async ({ ctx }) => {
      return await getUserJournalInsights(ctx.user.id);
    }),

    /** Dismiss an insight */
    dismissInsight: protectedProcedure
      .input(z.object({
        insightId: z.number()
      }))
      .mutation(async ({ ctx, input }) => {
        await dismissJournalInsight(input.insightId, ctx.user.id);
        return { success: true };
      }),

    /** Get entropy trajectory data for visualization */
    getTrajectory: protectedProcedure
      .input(z.object({
        days: z.number().optional().default(30)
      }))
      .query(async ({ ctx, input }) => {
        const entries = await getUserJournalEntries(ctx.user.id, input.days);
        
        // Calculate trajectory data
        const trajectoryData = entries.map(entry => ({
          date: entry.createdAt,
          entropy: parseFloat(entry.entropyScore || '0.5'),
          state: entry.entropyState,
          moodTags: Array.isArray(entry.moodTags) ? entry.moodTags : []
        })).reverse(); // Oldest first for trajectory

        // Calculate Vicsek-like predictions
        const predictions = [];
        for (let i = 0; i < trajectoryData.length; i++) {
          const neighbors = trajectoryData.slice(Math.max(0, i - 3), i + 1);
          const avgEntropy = neighbors.reduce((sum, n) => sum + n.entropy, 0) / neighbors.length;
          predictions.push({
            date: trajectoryData[i].date,
            predicted: avgEntropy + (Math.random() - 0.5) * 0.1 // Small noise
          });
        }

        // Calculate overall trend
        const recentEntropy = trajectoryData.slice(-7).map(d => d.entropy);
        const olderEntropy = trajectoryData.slice(-14, -7).map(d => d.entropy);
        const recentAvg = recentEntropy.length > 0 
          ? recentEntropy.reduce((a, b) => a + b, 0) / recentEntropy.length 
          : 0.5;
        const olderAvg = olderEntropy.length > 0 
          ? olderEntropy.reduce((a, b) => a + b, 0) / olderEntropy.length 
          : 0.5;
        const trend = recentAvg < olderAvg ? 'improving' : recentAvg > olderAvg ? 'declining' : 'stable';

        return {
          trajectory: trajectoryData,
          predictions,
          trend,
          averageEntropy: trajectoryData.length > 0 
            ? trajectoryData.reduce((sum, d) => sum + d.entropy, 0) / trajectoryData.length 
            : 0.5
        };
      })
  }),

  // ==================== USER PREFERENCES ====================
  preferences: router({
    get: protectedProcedure
      .query(async ({ ctx }) => {
        return await getOrCreateUserPreferences(ctx.user.id);
      }),

    update: protectedProcedure
      .input(z.object({
        languageCode: z.string().optional(),
        beliefSystem: z.string().optional(),
        voicePersona: z.string().optional(),
        voicePitch: z.string().optional(),
        voiceRate: z.string().optional(),
        autoPlayTTS: z.boolean().optional(),
        preferredGroundingCategory: z.string().optional(),
        culturalContext: z.string().optional(),
        communityContext: z.string().optional(),
        themePreference: z.enum(['dark', 'light', 'system']).optional(),
        fontSize: z.enum(['small', 'medium', 'large', 'xlarge']).optional(),
        reduceMotion: z.boolean().optional(),
        highContrast: z.boolean().optional(),
      }))
      .mutation(async ({ ctx, input }) => {
        return await updateUserPreferences(ctx.user.id, input);
      }),

    // Beta: all users are on free tier, subscription management disabled
    updateSubscription: protectedProcedure
      .input(z.object({
        subscriptionTier: z.enum(['free', 'premium', 'professional']).default('free'),
      }))
      .mutation(async ({ ctx, input }) => {
        // During beta, everyone stays on free tier
        return await updateUserPreferences(ctx.user.id, { subscriptionTier: 'free' });
      })
  }),

  // ==================== TEXT-TO-SPEECH (OpenAI) ====================
  tts: router({
    // Get all available voices
    getVoices: publicProcedure
      .query(() => {
        return getAvailableVoices();
      }),

    // Get voices filtered by criteria
    filterVoices: publicProcedure
      .input(z.object({
        gender: z.enum(['female', 'male', 'neutral']).optional(),
        accent: z.enum(['American', 'British']).optional(),
        tone: z.enum(['warm', 'gentle', 'calm', 'expressive', 'neutral']).optional()
      }))
      .query(({ input }) => {
        return filterVoices(input);
      }),

    // Get supported languages
    getLanguages: publicProcedure
      .query(() => {
        return getSupportedLanguages();
      }),

    // Generate speech from text
    speak: protectedProcedure
      .input(z.object({
        text: z.string().min(1).max(4096),
        voice: z.enum(['nova', 'shimmer', 'echo', 'onyx', 'fable', 'alloy']),
        speed: z.number().min(0.25).max(4.0).optional().default(1.0)
      }))
      .mutation(async ({ input }) => {
        const result = await generateSpeech({
          text: input.text,
          voice: input.voice as VoiceId,
          speed: input.speed
        });
        return result;
      })
  }),

  // Stripe subscription removed for beta testing - all features free
  // subscription router removed - to be re-added when ready for paid tiers
});


// Helper function to generate safety plan PDF HTML
function generateSafetyPlanHTML(planData: any): string {
  const sections = [
    { key: 'warningSignals', title: 'Warning Signals', icon: '⚠️' },
    { key: 'safeContacts', title: 'Safe Contacts', icon: '📞' },
    { key: 'safeLocations', title: 'Safe Locations', icon: '🏠' },
    { key: 'emergencyBag', title: 'Emergency Bag Checklist', icon: '🎒' },
    { key: 'documents', title: 'Important Documents', icon: '📄' },
    { key: 'financialSafety', title: 'Financial Safety', icon: '💰' },
    { key: 'technologySafety', title: 'Technology Safety', icon: '📱' },
    { key: 'childrenSafety', title: 'Children Safety', icon: '👶' },
    { key: 'petSafety', title: 'Pet Safety', icon: '🐾' },
    { key: 'exitStrategy', title: 'Exit Strategy', icon: '🚪' }
  ];

  const sectionsHTML = sections.map(section => {
    const data = planData[section.key];
    if (!data) return '';

    let content = '';
    if (Array.isArray(data)) {
      content = `<ul>${data.map(item => `<li>${item}</li>`).join('')}</ul>`;
    } else if (typeof data === 'object') {
      content = `<ul>${Object.entries(data).map(([k, v]) => `<li><strong>${k}:</strong> ${v}</li>`).join('')}</ul>`;
    } else {
      content = `<p>${data}</p>`;
    }

    return `
      <div class="section">
        <h2>${section.icon} ${section.title}</h2>
        ${content}
      </div>
    `;
  }).join('');

  return `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Safety Plan</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      line-height: 1.6;
      color: #333;
      max-width: 800px;
      margin: 0 auto;
      padding: 40px 20px;
    }
    header {
      text-align: center;
      margin-bottom: 40px;
      padding-bottom: 20px;
      border-bottom: 2px solid #1a8a6e;
    }
    h1 { color: #1a8a6e; margin-bottom: 10px; }
    .section {
      margin-bottom: 30px;
      padding: 20px;
      background: #f8f9fa;
      border-radius: 8px;
      border-left: 4px solid #1a8a6e;
    }
    h2 { color: #1a8a6e; margin-bottom: 15px; font-size: 1.2em; }
    ul { margin-left: 20px; }
    li { margin-bottom: 8px; }
    .crisis-banner {
      background: #dc3545;
      color: white;
      padding: 15px;
      text-align: center;
      border-radius: 8px;
      margin-bottom: 30px;
    }
    .footer {
      margin-top: 40px;
      padding-top: 20px;
      border-top: 1px solid #ddd;
      text-align: center;
      color: #666;
      font-size: 0.9em;
    }
    @media print {
      body { padding: 20px; }
      .section { break-inside: avoid; }
    }
  </style>
</head>
<body>
  <header>
    <h1>🛡️ Personal Safety Plan</h1>
    <p>Created with ReUnity | Keep this document in a safe place</p>
  </header>

  <div class="crisis-banner">
    <strong>In immediate danger?</strong> Call 911 or National DV Hotline: 1-800-799-7233
  </div>

  ${sectionsHTML}

  <div class="footer">
    <p>This plan was created on ${new Date().toLocaleDateString()}</p>
    <p>ReUnity by REOP Solutions | You are not alone</p>
  </div>
</body>
</html>
  `;
}

export type AppRouter = typeof appRouter;
