/**
 * Database procedures for ReUnity
 * Built by REOP Solutions
 */

import { eq, desc, and } from "drizzle-orm";
import { drizzle } from "drizzle-orm/mysql2";
import { 
  InsertUser, 
  users, 
  conversations, 
  messages, 
  userMemory,
  sessionAnalytics,
  safetyPlans,
  peerProfiles,
  peerConnections,
  peerMessages,
  moderationActions,
  journalEntries,
  journalInsights,
  userPreferences,
  InsertConversation,
  InsertMessage,
  InsertUserMemory,
  InsertSessionAnalytics,
  InsertSafetyPlan,
  InsertPeerProfile,
  InsertPeerConnection,
  InsertPeerMessage,
  InsertModerationAction,
  InsertJournalEntry,
  InsertJournalInsight,
  InsertUserPreferences
} from "../drizzle/schema";

export { drizzle };

let _db: ReturnType<typeof drizzle> | null = null;

// Lazily create the drizzle instance so local tooling can run without a DB.
export async function getDb() {
  if (!_db && process.env.DATABASE_URL) {
    try {
      _db = drizzle(process.env.DATABASE_URL);
    } catch (error) {
      console.warn("[Database] Failed to connect:", error);
      _db = null;
    }
  }
  return _db;
}

// Export db for direct access
export const db = {
  get instance() {
    if (!_db && process.env.DATABASE_URL) {
      _db = drizzle(process.env.DATABASE_URL);
    }
    return _db;
  }
};

// ============================================
// USER PROCEDURES
// ============================================

export async function getUserById(userId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select().from(users).where(eq(users.id, userId)).limit(1);
  return result.length > 0 ? result[0] : null;
}

export async function getUserByEmail(email: string) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select().from(users).where(eq(users.email, email.toLowerCase())).limit(1);
  return result.length > 0 ? result[0] : null;
}

export async function createUser(data: InsertUser) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot create user: database not available");
    return null;
  }

  try {
    const result = await db.insert(users).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create user:", error);
    throw error;
  }
}

export async function updateUser(userId: number, data: Partial<InsertUser>) {
  const db = await getDb();
  if (!db) return;

  await db.update(users).set(data).where(eq(users.id, userId));
}

// ============================================
// CONVERSATION PROCEDURES
// ============================================

export async function createConversation(data: InsertConversation): Promise<{
  id: number;
  userId: number;
  title: string | null;
  currentState: string | null;
  currentRegime: string | null;
  isActive: boolean | null;
  createdAt: Date;
  updatedAt: Date;
} | null> {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot create conversation: database not available");
    return null;
  }

  try {
    const result = await db.insert(conversations).values(data);
    const now = new Date();
    return {
      id: result[0].insertId,
      userId: data.userId,
      title: data.title ?? null,
      currentState: data.currentState ?? "stable",
      currentRegime: data.currentRegime ?? "normal",
      isActive: data.isActive ?? true,
      createdAt: now,
      updatedAt: now
    };
  } catch (error) {
    console.error("[Database] Failed to create conversation:", error);
    throw error;
  }
}

export async function getConversation(conversationId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select().from(conversations).where(eq(conversations.id, conversationId)).limit(1);
  return result.length > 0 ? result[0] : null;
}

export async function getUserConversations(userId: number, limit = 20) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(conversations)
    .where(eq(conversations.userId, userId))
    .orderBy(desc(conversations.updatedAt))
    .limit(limit);
}

export async function getActiveConversation(userId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(conversations)
    .where(and(eq(conversations.userId, userId), eq(conversations.isActive, true)))
    .orderBy(desc(conversations.updatedAt))
    .limit(1);
  
  return result.length > 0 ? result[0] : null;
}

export async function updateConversation(conversationId: number, data: Partial<InsertConversation>) {
  const db = await getDb();
  if (!db) return;

  await db.update(conversations)
    .set(data)
    .where(eq(conversations.id, conversationId));
}

export async function endConversation(conversationId: number) {
  const db = await getDb();
  if (!db) return;

  await db.update(conversations)
    .set({ isActive: false })
    .where(eq(conversations.id, conversationId));
}

// ============================================
// MESSAGE PROCEDURES
// ============================================

export async function createMessage(data: InsertMessage) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot create message: database not available");
    return null;
  }

  try {
    const result = await db.insert(messages).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create message:", error);
    throw error;
  }
}

export async function getConversationMessages(conversationId: number, limit = 100) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(messages)
    .where(eq(messages.conversationId, conversationId))
    .orderBy(messages.createdAt)
    .limit(limit);
}

export async function getRecentMessages(conversationId: number, limit = 10) {
  const db = await getDb();
  if (!db) return [];

  const result = await db.select()
    .from(messages)
    .where(eq(messages.conversationId, conversationId))
    .orderBy(desc(messages.createdAt))
    .limit(limit);
  
  return result.reverse();
}

// ============================================
// USER MEMORY (RIME) PROCEDURES
// ============================================

export async function saveUserMemory(data: InsertUserMemory) {
  const db = await getDb();
  if (!db) {
    console.warn("[Database] Cannot save memory: database not available");
    return null;
  }

  try {
    // Check if memory already exists
    const existing = await db.select()
      .from(userMemory)
      .where(and(
        eq(userMemory.userId, data.userId),
        eq(userMemory.memoryType, data.memoryType),
        eq(userMemory.memoryKey, data.memoryKey)
      ))
      .limit(1);

    if (existing.length > 0) {
      // Update existing memory
      await db.update(userMemory)
        .set({
          memoryValue: data.memoryValue,
          confidence: data.confidence,
          accessCount: (existing[0].accessCount || 0) + 1,
          lastAccessed: new Date()
        })
        .where(eq(userMemory.id, existing[0].id));
      return existing[0];
    } else {
      // Create new memory
      const result = await db.insert(userMemory).values(data);
      return { id: result[0].insertId, ...data };
    }
  } catch (error) {
    console.error("[Database] Failed to save memory:", error);
    throw error;
  }
}

export async function getUserMemories(userId: number) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(userMemory)
    .where(eq(userMemory.userId, userId))
    .orderBy(desc(userMemory.lastAccessed));
}

export async function getUserMemoriesByType(userId: number, memoryType: string) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(userMemory)
    .where(and(
      eq(userMemory.userId, userId),
      eq(userMemory.memoryType, memoryType)
    ))
    .orderBy(desc(userMemory.lastAccessed));
}

export async function getMemoryByKey(userId: number, memoryType: string, memoryKey: string) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(userMemory)
    .where(and(
      eq(userMemory.userId, userId),
      eq(userMemory.memoryType, memoryType),
      eq(userMemory.memoryKey, memoryKey)
    ))
    .limit(1);

  if (result.length > 0) {
    // Update access count
    await db.update(userMemory)
      .set({
        accessCount: (result[0].accessCount || 0) + 1,
        lastAccessed: new Date()
      })
      .where(eq(userMemory.id, result[0].id));
  }

  return result.length > 0 ? result[0] : null;
}

export async function deleteUserMemory(userId: number, memoryType: string, memoryKey: string) {
  const db = await getDb();
  if (!db) return;

  await db.delete(userMemory)
    .where(and(
      eq(userMemory.userId, userId),
      eq(userMemory.memoryType, memoryType),
      eq(userMemory.memoryKey, memoryKey)
    ));
}

// ============================================
// SESSION ANALYTICS PROCEDURES
// ============================================

export async function createSessionAnalytics(data: InsertSessionAnalytics) {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db.insert(sessionAnalytics).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create session analytics:", error);
    throw error;
  }
}

export async function updateSessionAnalytics(conversationId: number, data: Partial<InsertSessionAnalytics>) {
  const db = await getDb();
  if (!db) return;

  await db.update(sessionAnalytics)
    .set(data)
    .where(eq(sessionAnalytics.conversationId, conversationId));
}

export async function getSessionAnalytics(conversationId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(sessionAnalytics)
    .where(eq(sessionAnalytics.conversationId, conversationId))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

// ============================================
// RIME MEMORY LOAD/SAVE FOR REUNITY
// ============================================

export interface RIMEMemoryData {
  groundingAnchors: string[];
  knownTriggers: string[];
  safePlaces: string[];
  userName: string | null;
  preferences: Record<string, string>;
}

export async function loadRIMEMemory(userId: number): Promise<RIMEMemoryData> {
  const memories = await getUserMemories(userId);
  
  const result: RIMEMemoryData = {
    groundingAnchors: [],
    knownTriggers: [],
    safePlaces: [],
    userName: null,
    preferences: {}
  };

  for (const memory of memories) {
    switch (memory.memoryType) {
      case 'grounding_anchor':
        result.groundingAnchors.push(memory.memoryValue);
        break;
      case 'known_trigger':
        result.knownTriggers.push(memory.memoryValue);
        break;
      case 'safe_place':
        result.safePlaces.push(memory.memoryValue);
        break;
      case 'name':
        result.userName = memory.memoryValue;
        break;
      case 'preference':
        result.preferences[memory.memoryKey] = memory.memoryValue;
        break;
    }
  }

  return result;
}

export async function saveRIMEMemory(userId: number, data: Partial<RIMEMemoryData>): Promise<void> {
  if (data.groundingAnchors) {
    for (const anchor of data.groundingAnchors) {
      await saveUserMemory({
        userId,
        memoryType: 'grounding_anchor',
        memoryKey: anchor.substring(0, 128),
        memoryValue: anchor
      });
    }
  }

  if (data.knownTriggers) {
    for (const trigger of data.knownTriggers) {
      await saveUserMemory({
        userId,
        memoryType: 'known_trigger',
        memoryKey: trigger.substring(0, 128),
        memoryValue: trigger
      });
    }
  }

  if (data.safePlaces) {
    for (const place of data.safePlaces) {
      await saveUserMemory({
        userId,
        memoryType: 'safe_place',
        memoryKey: place.substring(0, 128),
        memoryValue: place
      });
    }
  }

  if (data.userName) {
    await saveUserMemory({
      userId,
      memoryType: 'name',
      memoryKey: 'user_name',
      memoryValue: data.userName
    });
  }

  if (data.preferences) {
    for (const [key, value] of Object.entries(data.preferences)) {
      await saveUserMemory({
        userId,
        memoryType: 'preference',
        memoryKey: key,
        memoryValue: value
      });
    }
  }
}

// ============================================
// SAFETY PLAN PROCEDURES
// ============================================

export async function getSafetyPlan(userId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(safetyPlans)
    .where(eq(safetyPlans.userId, userId))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

export async function createSafetyPlan(data: InsertSafetyPlan) {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db.insert(safetyPlans).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create safety plan:", error);
    throw error;
  }
}

export async function updateSafetyPlan(userId: number, data: Partial<InsertSafetyPlan>) {
  const db = await getDb();
  if (!db) return;

  await db.update(safetyPlans)
    .set(data)
    .where(eq(safetyPlans.userId, userId));
}

// ============================================
// PEER PROFILE PROCEDURES
// ============================================

export async function getPeerProfile(userId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(peerProfiles)
    .where(eq(peerProfiles.userId, userId))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

export async function createPeerProfile(data: InsertPeerProfile) {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db.insert(peerProfiles).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create peer profile:", error);
    throw error;
  }
}

export async function updatePeerProfile(userId: number, data: Partial<InsertPeerProfile>) {
  const db = await getDb();
  if (!db) return;

  await db.update(peerProfiles)
    .set(data)
    .where(eq(peerProfiles.userId, userId));
}

export async function getActivePeerProfiles(excludeUserId?: number) {
  const db = await getDb();
  if (!db) return [];

  let query = db.select()
    .from(peerProfiles)
    .where(and(
      eq(peerProfiles.isActive, true),
      eq(peerProfiles.isBanned, false)
    ));

  const results = await query;
  
  if (excludeUserId) {
    return results.filter(p => p.userId !== excludeUserId);
  }
  
  return results;
}

// ============================================
// PEER CONNECTION PROCEDURES
// ============================================

export async function getPeerConnection(connectionId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(peerConnections)
    .where(eq(peerConnections.id, connectionId))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

export async function getUserPeerConnections(userId: number) {
  const db = await getDb();
  if (!db) return [];

  // Get connections where user is either requester or responder
  const asRequester = await db.select()
    .from(peerConnections)
    .where(eq(peerConnections.requesterId, userId));

  const asResponder = await db.select()
    .from(peerConnections)
    .where(eq(peerConnections.responderId, userId));

  return [...asRequester, ...asResponder];
}

export async function createPeerConnection(data: InsertPeerConnection) {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db.insert(peerConnections).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create peer connection:", error);
    throw error;
  }
}

export async function updatePeerConnection(connectionId: number, data: Partial<InsertPeerConnection>) {
  const db = await getDb();
  if (!db) return;

  await db.update(peerConnections)
    .set(data)
    .where(eq(peerConnections.id, connectionId));
}

// ============================================
// PEER MESSAGE PROCEDURES
// ============================================

export async function getPeerMessages(connectionId: number, limit: number = 100) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(peerMessages)
    .where(eq(peerMessages.connectionId, connectionId))
    .orderBy(desc(peerMessages.createdAt))
    .limit(limit);
}

export async function createPeerMessage(data: InsertPeerMessage) {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db.insert(peerMessages).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create peer message:", error);
    throw error;
  }
}

export async function flagPeerMessage(messageId: number, reason: string) {
  const db = await getDb();
  if (!db) return;

  await db.update(peerMessages)
    .set({ flagged: true, flagReason: reason })
    .where(eq(peerMessages.id, messageId));
}

// ============================================
// MODERATION PROCEDURES
// ============================================

export async function createModerationAction(data: InsertModerationAction) {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db.insert(moderationActions).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create moderation action:", error);
    throw error;
  }
}

export async function getUserModerationHistory(userId: number) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(moderationActions)
    .where(eq(moderationActions.targetUserId, userId))
    .orderBy(desc(moderationActions.createdAt));
}

// ============================================
// JOURNAL ENTRY PROCEDURES
// ============================================

export async function getJournalEntry(entryId: number) {
  const db = await getDb();
  if (!db) return null;

  const result = await db.select()
    .from(journalEntries)
    .where(eq(journalEntries.id, entryId))
    .limit(1);

  return result.length > 0 ? result[0] : null;
}

export async function getUserJournalEntries(userId: number, limit: number = 100) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(journalEntries)
    .where(eq(journalEntries.userId, userId))
    .orderBy(desc(journalEntries.createdAt))
    .limit(limit);
}

export async function createJournalEntry(data: InsertJournalEntry) {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db.insert(journalEntries).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create journal entry:", error);
    throw error;
  }
}

export async function updateJournalEntry(entryId: number, data: Partial<InsertJournalEntry>) {
  const db = await getDb();
  if (!db) return;

  await db.update(journalEntries)
    .set(data)
    .where(eq(journalEntries.id, entryId));
}

export async function deleteJournalEntry(entryId: number, userId: number) {
  const db = await getDb();
  if (!db) return;

  await db.delete(journalEntries)
    .where(and(
      eq(journalEntries.id, entryId),
      eq(journalEntries.userId, userId)
    ));
}

// ============================================
// JOURNAL INSIGHT PROCEDURES
// ============================================

export async function getUserJournalInsights(userId: number) {
  const db = await getDb();
  if (!db) return [];

  return await db.select()
    .from(journalInsights)
    .where(and(
      eq(journalInsights.userId, userId),
      eq(journalInsights.isDismissed, false)
    ))
    .orderBy(desc(journalInsights.createdAt));
}

export async function createJournalInsight(data: InsertJournalInsight) {
  const db = await getDb();
  if (!db) return null;

  try {
    const result = await db.insert(journalInsights).values(data);
    return { id: result[0].insertId, ...data };
  } catch (error) {
    console.error("[Database] Failed to create journal insight:", error);
    throw error;
  }
}

export async function dismissJournalInsight(insightId: number, userId: number) {
  const db = await getDb();
  if (!db) return;

  await db.update(journalInsights)
    .set({ isDismissed: true })
    .where(and(
      eq(journalInsights.id, insightId),
      eq(journalInsights.userId, userId)
    ));
}


// ==================== USER PREFERENCES ====================

export async function getUserPreferences(userId: number) {
  const db = await getDb();
  if (!db) return null;

  const [prefs] = await db.select()
    .from(userPreferences)
    .where(eq(userPreferences.userId, userId))
    .limit(1);
  
  return prefs || null;
}

export async function createUserPreferences(userId: number, data: Partial<InsertUserPreferences> = {}) {
  const db = await getDb();
  if (!db) return null;

  try {
    await db.insert(userPreferences).values({
      userId,
      ...data,
    });
    return getUserPreferences(userId);
  } catch (error) {
    console.error("[Database] Failed to create user preferences:", error);
    throw error;
  }
}

export async function updateUserPreferences(userId: number, data: Partial<InsertUserPreferences>) {
  const db = await getDb();
  if (!db) return null;

  try {
    // Check if preferences exist
    const existing = await getUserPreferences(userId);
    if (!existing) {
      // Create new preferences
      return createUserPreferences(userId, data);
    }

    // Update existing preferences
    await db.update(userPreferences)
      .set(data)
      .where(eq(userPreferences.userId, userId));
    
    return getUserPreferences(userId);
  } catch (error) {
    console.error("[Database] Failed to update user preferences:", error);
    throw error;
  }
}

export async function getOrCreateUserPreferences(userId: number) {
  const db = await getDb();
  if (!db) return null;

  let prefs = await getUserPreferences(userId);
  if (!prefs) {
    prefs = await createUserPreferences(userId);
  }
  return prefs;
}
