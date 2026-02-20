import { int, mysqlEnum, mysqlTable, text, timestamp, varchar, json, boolean } from "drizzle-orm/mysql-core";

/**
 * Core user table with custom email/password authentication.
 * Extend this file with additional tables as your product grows.
 * Columns use camelCase to match both database fields and generated types.
 */
export const users = mysqlTable("users", {
  /**
   * Surrogate primary key. Auto-incremented numeric value managed by the database.
   * Use this for relations between tables.
   */
  id: int("id").autoincrement().primaryKey(),
  /** User's email address - used for login */
  email: varchar("email", { length: 320 }).notNull().unique(),
  /** Bcrypt hashed password */
  passwordHash: varchar("passwordHash", { length: 255 }).notNull(),
  /** User's display name */
  name: varchar("name", { length: 255 }),
  /** Whether email has been verified */
  emailVerified: boolean("emailVerified").default(false),
  /** Email verification token */
  verificationToken: varchar("verificationToken", { length: 255 }),
  /** Password reset token */
  resetToken: varchar("resetToken", { length: 255 }),
  /** Password reset token expiry */
  resetTokenExpiry: timestamp("resetTokenExpiry"),
  role: mysqlEnum("role", ["user", "admin"]).default("user").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
  lastSignedIn: timestamp("lastSignedIn").defaultNow().notNull(),
});

export type User = typeof users.$inferSelect;
export type InsertUser = typeof users.$inferInsert;

/**
 * Sessions table - stores active user sessions
 */
export const sessions = mysqlTable("sessions", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  /** Session token (JWT or random string) */
  token: varchar("token", { length: 512 }).notNull().unique(),
  /** User agent string for session tracking */
  userAgent: text("userAgent"),
  /** IP address for session tracking */
  ipAddress: varchar("ipAddress", { length: 45 }),
  /** Session expiry time */
  expiresAt: timestamp("expiresAt").notNull(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Session = typeof sessions.$inferSelect;
export type InsertSession = typeof sessions.$inferInsert;

/**
 * Conversations table - stores chat sessions
 */
export const conversations = mysqlTable("conversations", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  title: varchar("title", { length: 255 }),
  /** Current state of the conversation (crisis, high_distress, moderate, low, stable) */
  currentState: varchar("currentState", { length: 32 }).default("stable"),
  /** Current regime (normal, recovery, crisis) */
  currentRegime: varchar("currentRegime", { length: 32 }).default("normal"),
  /** Whether this conversation is active */
  isActive: boolean("isActive").default(true),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type Conversation = typeof conversations.$inferSelect;
export type InsertConversation = typeof conversations.$inferInsert;

/**
 * Messages table - stores individual messages with entropy metadata
 */
export const messages = mysqlTable("messages", {
  id: int("id").autoincrement().primaryKey(),
  conversationId: int("conversationId").notNull(),
  /** Role: user or assistant */
  role: mysqlEnum("role", ["user", "assistant"]).notNull(),
  content: text("content").notNull(),
  /** Entropy score calculated by EntropyAnalyzer (0-1) */
  entropyScore: varchar("entropyScore", { length: 10 }),
  /** Detected state (crisis, high_distress, moderate, low, stable) */
  detectedState: varchar("detectedState", { length: 32 }),
  /** Detected patterns as JSON array */
  detectedPatterns: json("detectedPatterns"),
  /** Grounding technique delivered (if any) */
  groundingTechnique: varchar("groundingTechnique", { length: 64 }),
  /** Detected location context */
  detectedLocation: varchar("detectedLocation", { length: 64 }),
  /** Whether this message triggered crisis mode */
  isCrisis: boolean("isCrisis").default(false),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type Message = typeof messages.$inferSelect;
export type InsertMessage = typeof messages.$inferInsert;

/**
 * User Memory table - RIME memory persistence
 * Stores grounding anchors, known triggers, and other memory data
 */
export const userMemory = mysqlTable("userMemory", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  /** Memory type: grounding_anchor, known_trigger, safe_place, name, preference */
  memoryType: varchar("memoryType", { length: 32 }).notNull(),
  /** The key/identifier for this memory */
  memoryKey: varchar("memoryKey", { length: 128 }).notNull(),
  /** The value/content of this memory */
  memoryValue: text("memoryValue").notNull(),
  /** Confidence score for this memory (0-1) */
  confidence: varchar("confidence", { length: 10 }).default("1.0"),
  /** How many times this memory has been referenced */
  accessCount: int("accessCount").default(0),
  /** Last time this memory was accessed */
  lastAccessed: timestamp("lastAccessed").defaultNow(),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type UserMemory = typeof userMemory.$inferSelect;
export type InsertUserMemory = typeof userMemory.$inferInsert;

/**
 * Session Analytics - tracks session-level metrics for improvement
 */
export const sessionAnalytics = mysqlTable("sessionAnalytics", {
  id: int("id").autoincrement().primaryKey(),
  conversationId: int("conversationId").notNull(),
  userId: int("userId").notNull(),
  /** Total messages in session */
  messageCount: int("messageCount").default(0),
  /** Number of crisis states detected */
  crisisCount: int("crisisCount").default(0),
  /** Number of patterns detected */
  patternCount: int("patternCount").default(0),
  /** Number of grounding techniques delivered */
  groundingCount: int("groundingCount").default(0),
  /** Average entropy score for session */
  avgEntropyScore: varchar("avgEntropyScore", { length: 10 }),
  /** Session duration in seconds */
  durationSeconds: int("durationSeconds"),
  /** Final state when session ended */
  finalState: varchar("finalState", { length: 32 }),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type SessionAnalytics = typeof sessionAnalytics.$inferSelect;
export type InsertSessionAnalytics = typeof sessionAnalytics.$inferInsert;


/**
 * Safety Plans table - encrypted safety plans for DV survivors
 */
export const safetyPlans = mysqlTable("safetyPlans", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  /** Encrypted plan data (all sensitive info encrypted) */
  encryptedData: text("encryptedData").notNull(),
  /** Completed step IDs as JSON array */
  completedSteps: json("completedSteps"),
  /** Whether the plan is complete */
  isComplete: boolean("isComplete").default(false),
  /** Last step accessed */
  lastStepId: varchar("lastStepId", { length: 64 }),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type SafetyPlan = typeof safetyPlans.$inferSelect;
export type InsertSafetyPlan = typeof safetyPlans.$inferInsert;

/**
 * Peer Profiles table - anonymous peer support profiles
 */
export const peerProfiles = mysqlTable("peerProfiles", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().unique(),
  /** Anonymous display name */
  displayName: varchar("displayName", { length: 64 }).notNull(),
  /** Experience tags as JSON array */
  experiences: json("experiences"),
  /** Support preferences as JSON object */
  preferences: json("preferences"),
  /** Safety settings as JSON object */
  safetySettings: json("safetySettings"),
  /** Whether profile is active */
  isActive: boolean("isActive").default(true),
  /** Last active timestamp */
  lastActive: timestamp("lastActive").defaultNow(),
  /** Whether user is banned */
  isBanned: boolean("isBanned").default(false),
  /** Ban reason if banned */
  banReason: text("banReason"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type PeerProfile = typeof peerProfiles.$inferSelect;
export type InsertPeerProfile = typeof peerProfiles.$inferInsert;

/**
 * Peer Connections table - connections between peer supporters
 */
export const peerConnections = mysqlTable("peerConnections", {
  id: int("id").autoincrement().primaryKey(),
  requesterId: int("requesterId").notNull(),
  responderId: int("responderId").notNull(),
  /** Connection status */
  status: mysqlEnum("status", ["pending", "accepted", "declined", "blocked", "ended"]).default("pending").notNull(),
  /** Match quality score (0-100) */
  matchScore: int("matchScore"),
  /** Shared experiences as JSON array */
  sharedExperiences: json("sharedExperiences"),
  /** Number of chat sessions */
  sessionCount: int("sessionCount").default(0),
  /** Last session timestamp */
  lastSessionAt: timestamp("lastSessionAt"),
  /** Total minutes chatted */
  totalMinutes: int("totalMinutes").default(0),
  /** Whether flagged for review */
  flaggedForReview: boolean("flaggedForReview").default(false),
  /** Flag reason if flagged */
  flagReason: text("flagReason"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type PeerConnection = typeof peerConnections.$inferSelect;
export type InsertPeerConnection = typeof peerConnections.$inferInsert;

/**
 * Peer Messages table - messages between peer supporters
 */
export const peerMessages = mysqlTable("peerMessages", {
  id: int("id").autoincrement().primaryKey(),
  connectionId: int("connectionId").notNull(),
  senderId: int("senderId").notNull(),
  content: text("content").notNull(),
  /** Entropy level of message */
  entropyLevel: varchar("entropyLevel", { length: 10 }),
  /** Whether crisis was detected */
  crisisDetected: boolean("crisisDetected").default(false),
  /** Whether message is flagged */
  flagged: boolean("flagged").default(false),
  /** Flag reason if flagged */
  flagReason: text("flagReason"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type PeerMessage = typeof peerMessages.$inferSelect;
export type InsertPeerMessage = typeof peerMessages.$inferInsert;

/**
 * Moderation Actions table - tracks moderation actions on peer support
 */
export const moderationActions = mysqlTable("moderationActions", {
  id: int("id").autoincrement().primaryKey(),
  targetUserId: int("targetUserId").notNull(),
  reporterId: int("reporterId"),
  /** Action type */
  action: mysqlEnum("action", ["warning", "temporary_ban", "permanent_ban", "review_required"]).notNull(),
  reason: text("reason").notNull(),
  /** When action was resolved */
  resolvedAt: timestamp("resolvedAt"),
  /** Who resolved the action */
  resolvedBy: int("resolvedBy"),
  /** Resolution notes */
  resolution: text("resolution"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type ModerationAction = typeof moderationActions.$inferSelect;
export type InsertModerationAction = typeof moderationActions.$inferInsert;

/**
 * Journal Entries table - user journal entries with entropy tracking
 */
export const journalEntries = mysqlTable("journalEntries", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  /** Entry title (optional) */
  title: varchar("title", { length: 255 }),
  /** Entry content */
  content: text("content").notNull(),
  /** User-selected mood tags as JSON array */
  moodTags: json("moodTags"),
  /** Custom tags as JSON array */
  customTags: json("customTags"),
  /** AI-calculated entropy score (0-1) */
  entropyScore: varchar("entropyScore", { length: 10 }),
  /** Detected entropy state */
  entropyState: varchar("entropyState", { length: 32 }),
  /** Detected conditions as JSON array */
  detectedConditions: json("detectedConditions"),
  /** Detected states as JSON array */
  detectedStates: json("detectedStates"),
  /** Identified triggers as JSON array */
  triggersIdentified: json("triggersIdentified"),
  /** Coping strategies used as JSON array */
  copingUsed: json("copingUsed"),
  /** Whether entry is private */
  isPrivate: boolean("isPrivate").default(true),
  /** Whether content is encrypted */
  isEncrypted: boolean("isEncrypted").default(false),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type JournalEntry = typeof journalEntries.$inferSelect;
export type InsertJournalEntry = typeof journalEntries.$inferInsert;

/**
 * Journal Insights table - AI-generated insights from journal patterns
 */
export const journalInsights = mysqlTable("journalInsights", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull(),
  /** Insight type */
  insightType: mysqlEnum("insightType", ["pattern", "progress", "warning", "suggestion"]).notNull(),
  title: varchar("title", { length: 255 }).notNull(),
  description: text("description").notNull(),
  /** Confidence score (0-1) */
  confidence: varchar("confidence", { length: 10 }),
  /** Related entry IDs as JSON array */
  relatedEntries: json("relatedEntries"),
  /** Whether insight has been dismissed */
  isDismissed: boolean("isDismissed").default(false),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type JournalInsight = typeof journalInsights.$inferSelect;
export type InsertJournalInsight = typeof journalInsights.$inferInsert;


/**
 * Therapist Profiles table - licensed therapist information
 */
export const therapistProfiles = mysqlTable("therapistProfiles", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().unique(),
  /** License number */
  licenseNumber: varchar("licenseNumber", { length: 100 }).notNull(),
  /** License state/jurisdiction */
  licenseState: varchar("licenseState", { length: 100 }).notNull(),
  /** License type (LCSW, LMFT, PhD, etc.) */
  licenseType: varchar("licenseType", { length: 50 }).notNull(),
  /** Specializations as JSON array */
  specializations: json("specializations"),
  /** Practice name */
  practiceName: varchar("practiceName", { length: 255 }),
  /** Contact phone */
  phone: varchar("phone", { length: 20 }),
  /** Whether license is verified */
  isVerified: boolean("isVerified").default(false),
  /** Verification date */
  verifiedAt: timestamp("verifiedAt"),
  /** Whether accepting new clients */
  acceptingClients: boolean("acceptingClients").default(true),
  /** Bio/description */
  bio: text("bio"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type TherapistProfile = typeof therapistProfiles.$inferSelect;
export type InsertTherapistProfile = typeof therapistProfiles.$inferInsert;

/**
 * Therapist-Client Relationships table - consent-based monitoring relationships
 */
export const therapistClientRelationships = mysqlTable("therapistClientRelationships", {
  id: int("id").autoincrement().primaryKey(),
  therapistId: int("therapistId").notNull(),
  clientId: int("clientId").notNull(),
  /** Relationship status */
  status: mysqlEnum("status", ["pending", "active", "paused", "ended"]).default("pending").notNull(),
  /** Client consent timestamp */
  consentedAt: timestamp("consentedAt"),
  /** What data client consents to share */
  consentedDataTypes: json("consentedDataTypes"), // ["entropy", "journal_summary", "crisis_alerts", "check_ins"]
  /** Whether therapist receives crisis alerts */
  crisisAlertsEnabled: boolean("crisisAlertsEnabled").default(true),
  /** Notes from therapist */
  therapistNotes: text("therapistNotes"),
  /** When relationship ended */
  endedAt: timestamp("endedAt"),
  /** Reason for ending */
  endReason: text("endReason"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type TherapistClientRelationship = typeof therapistClientRelationships.$inferSelect;
export type InsertTherapistClientRelationship = typeof therapistClientRelationships.$inferInsert;

/**
 * Therapist Alerts table - crisis alerts sent to therapists
 */
export const therapistAlerts = mysqlTable("therapistAlerts", {
  id: int("id").autoincrement().primaryKey(),
  relationshipId: int("relationshipId").notNull(),
  therapistId: int("therapistId").notNull(),
  clientId: int("clientId").notNull(),
  /** Alert type */
  alertType: mysqlEnum("alertType", ["crisis", "high_entropy", "missed_checkin", "concerning_pattern", "progress"]).notNull(),
  /** Alert severity */
  severity: mysqlEnum("severity", ["low", "medium", "high", "critical"]).default("medium").notNull(),
  /** Alert title */
  title: varchar("title", { length: 255 }).notNull(),
  /** Alert description */
  description: text("description").notNull(),
  /** Related data as JSON */
  relatedData: json("relatedData"),
  /** Whether alert has been viewed */
  isViewed: boolean("isViewed").default(false),
  /** When alert was viewed */
  viewedAt: timestamp("viewedAt"),
  /** Whether alert has been acknowledged */
  isAcknowledged: boolean("isAcknowledged").default(false),
  /** Acknowledgment notes */
  acknowledgmentNotes: text("acknowledgmentNotes"),
  /** When alert was acknowledged */
  acknowledgedAt: timestamp("acknowledgedAt"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type TherapistAlert = typeof therapistAlerts.$inferSelect;
export type InsertTherapistAlert = typeof therapistAlerts.$inferInsert;

/**
 * Client Entropy Snapshots table - periodic entropy data shared with therapists
 */
export const clientEntropySnapshots = mysqlTable("clientEntropySnapshots", {
  id: int("id").autoincrement().primaryKey(),
  relationshipId: int("relationshipId").notNull(),
  clientId: int("clientId").notNull(),
  /** Snapshot date */
  snapshotDate: timestamp("snapshotDate").notNull(),
  /** Average entropy score for period */
  avgEntropyScore: varchar("avgEntropyScore", { length: 10 }),
  /** Entropy trend (improving, stable, declining) */
  entropyTrend: varchar("entropyTrend", { length: 20 }),
  /** Dominant emotional states as JSON array */
  dominantStates: json("dominantStates"),
  /** Detected patterns as JSON array */
  detectedPatterns: json("detectedPatterns"),
  /** Number of journal entries in period */
  journalEntryCount: int("journalEntryCount").default(0),
  /** Number of check-ins completed */
  checkInsCompleted: int("checkInsCompleted").default(0),
  /** Number of check-ins missed */
  checkInsMissed: int("checkInsMissed").default(0),
  /** Crisis events count */
  crisisEventsCount: int("crisisEventsCount").default(0),
  /** AI-generated summary */
  aiSummary: text("aiSummary"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
});

export type ClientEntropySnapshot = typeof clientEntropySnapshots.$inferSelect;
export type InsertClientEntropySnapshot = typeof clientEntropySnapshots.$inferInsert;


/**
 * User Preferences table - stores user's language, belief system, voice persona, and other preferences
 */
export const userPreferences = mysqlTable("userPreferences", {
  id: int("id").autoincrement().primaryKey(),
  userId: int("userId").notNull().unique(),
  /** Preferred language code (e.g., 'en', 'es', 'ar', 'hi', 'navajo') */
  languageCode: varchar("languageCode", { length: 32 }).default("en"),
  /** Preferred belief system (e.g., 'christianity', 'buddhism', 'secular_humanism', 'existentialism') */
  beliefSystem: varchar("beliefSystem", { length: 64 }),
  /** Preferred voice persona for TTS (e.g., 'gentle_woman', 'gentle_man', 'neutral', 'warm_elder', 'calm_friend') */
  voicePersona: varchar("voicePersona", { length: 32 }).default("neutral"),
  /** Voice pitch adjustment (-1 to 1) */
  voicePitch: varchar("voicePitch", { length: 10 }).default("1.0"),
  /** Voice rate adjustment (0.5 to 2) */
  voiceRate: varchar("voiceRate", { length: 10 }).default("1.0"),
  /** Whether to auto-play TTS for AI responses */
  autoPlayTTS: boolean("autoPlayTTS").default(false),
  /** Preferred grounding technique category */
  preferredGroundingCategory: varchar("preferredGroundingCategory", { length: 64 }),
  /** Cultural context for responses */
  culturalContext: varchar("culturalContext", { length: 64 }),
  /** Community context (e.g., 'lgbtq', 'veteran', 'immigrant', 'rural') */
  communityContext: varchar("communityContext", { length: 64 }),
  /** Theme preference (dark/light) */
  themePreference: mysqlEnum("themePreference", ["dark", "light", "system"]).default("dark"),
  /** Font size preference */
  fontSize: mysqlEnum("fontSize", ["small", "medium", "large", "xlarge"]).default("medium"),
  /** Reduce motion preference for accessibility */
  reduceMotion: boolean("reduceMotion").default(false),
  /** High contrast mode for accessibility */
  highContrast: boolean("highContrast").default(false),
  /** Subscription tier (free, premium, professional) */
  subscriptionTier: mysqlEnum("subscriptionTier", ["free", "premium", "professional"]).default("free"),
  /** Stripe customer ID */
  stripeCustomerId: varchar("stripeCustomerId", { length: 255 }),
  /** Stripe subscription ID */
  stripeSubscriptionId: varchar("stripeSubscriptionId", { length: 255 }),
  /** Subscription status */
  subscriptionStatus: varchar("subscriptionStatus", { length: 32 }).default("inactive"),
  /** Subscription end date */
  subscriptionEndDate: timestamp("subscriptionEndDate"),
  createdAt: timestamp("createdAt").defaultNow().notNull(),
  updatedAt: timestamp("updatedAt").defaultNow().onUpdateNow().notNull(),
});

export type UserPreferences = typeof userPreferences.$inferSelect;
export type InsertUserPreferences = typeof userPreferences.$inferInsert;
