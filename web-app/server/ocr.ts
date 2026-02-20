/**
 * OCR Service for ReUnity
 * Processes images to extract text for analysis
 * Supports screenshots, handwritten notes, and text images
 * 
 * Created by Christopher Ezernack, REOP Solutions
 */

import { invokeLLM } from "./_core/llm";

export interface OCRResult {
  success: boolean;
  extractedText: string;
  confidence: number;
  contentType: "text" | "handwritten" | "screenshot" | "mixed" | "unknown";
  emotionalContent: boolean;
  suggestedContext: string | null;
  error?: string;
}

export interface ImageAnalysisResult {
  extractedText: string;
  emotionalIndicators: string[];
  contextualNotes: string;
  isJournalEntry: boolean;
  isConversationScreenshot: boolean;
  containsCrisisContent: boolean;
}

/**
 * OCR Service using vision capabilities
 * Extracts text from images and provides contextual analysis
 */
export class OCRService {
  /**
   * Process an image URL and extract text with context
   */
  async processImage(imageUrl: string): Promise<OCRResult> {
    try {
      const response = await invokeLLM({
        messages: [
          {
            role: "system",
            content: `You are an OCR and image analysis system for ReUnity, a trauma-aware mental health support application.

Your task is to:
1. Extract ALL text visible in the image accurately
2. Identify the type of content (text message screenshot, journal entry, handwritten note, etc.)
3. Note any emotional indicators or concerning content
4. Provide context that would help a supportive AI respond appropriately

IMPORTANT: 
- Extract text EXACTLY as written, including typos and informal language
- For conversation screenshots, clearly indicate who said what
- Flag any crisis-related content (self-harm, suicidal ideation, abuse mentions)
- Be sensitive - this content may be deeply personal

Respond in this JSON format:
{
  "extractedText": "The full text extracted from the image",
  "contentType": "text_message" | "journal" | "handwritten" | "screenshot" | "document" | "other",
  "emotionalIndicators": ["list", "of", "emotional", "words", "detected"],
  "isConversationScreenshot": true/false,
  "conversationParticipants": ["Person A", "Person B"] if applicable,
  "containsCrisisContent": true/false,
  "crisisIndicators": ["list if any"],
  "contextualNotes": "Brief notes about the content that would help provide supportive response"
}`
          },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Please analyze this image and extract all text content. Provide the analysis in the specified JSON format."
              },
              {
                type: "image_url",
                image_url: {
                  url: imageUrl,
                  detail: "high"
                }
              }
            ]
          }
        ]
      });

      const content = response.choices[0]?.message?.content;
      
      if (typeof content !== "string") {
        return {
          success: false,
          extractedText: "",
          confidence: 0,
          contentType: "unknown",
          emotionalContent: false,
          suggestedContext: null,
          error: "Failed to get response from vision model"
        };
      }

      // Parse the JSON response
      try {
        // Extract JSON from the response (handle markdown code blocks)
        let jsonStr = content;
        const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (jsonMatch) {
          jsonStr = jsonMatch[1];
        }
        
        const parsed = JSON.parse(jsonStr);
        
        return {
          success: true,
          extractedText: parsed.extractedText || "",
          confidence: 0.9,
          contentType: this.mapContentType(parsed.contentType),
          emotionalContent: (parsed.emotionalIndicators?.length || 0) > 0,
          suggestedContext: this.buildContext(parsed)
        };
      } catch (parseError) {
        // If JSON parsing fails, treat the whole response as extracted text
        return {
          success: true,
          extractedText: content,
          confidence: 0.7,
          contentType: "unknown",
          emotionalContent: false,
          suggestedContext: null
        };
      }
    } catch (error) {
      console.error("OCR Error:", error);
      return {
        success: false,
        extractedText: "",
        confidence: 0,
        contentType: "unknown",
        emotionalContent: false,
        suggestedContext: null,
        error: error instanceof Error ? error.message : "Unknown error during OCR processing"
      };
    }
  }

  /**
   * Analyze a conversation screenshot specifically
   */
  async analyzeConversation(imageUrl: string): Promise<{
    success: boolean;
    messages: Array<{ sender: string; content: string }>;
    patterns: string[];
    concernLevel: "low" | "moderate" | "high" | "crisis";
    analysis: string;
    error?: string;
  }> {
    try {
      const response = await invokeLLM({
        messages: [
          {
            role: "system",
            content: `You are analyzing a conversation screenshot for ReUnity, a trauma-aware mental health application.

Your task is to:
1. Extract each message with the sender identified
2. Identify any harmful relationship patterns (gaslighting, manipulation, abuse, control)
3. Assess the concern level of the conversation
4. Provide analysis that would help support the person sharing this

Look for these patterns:
- Gaslighting: "You're imagining things", "That never happened", "You're crazy"
- Love bombing: Excessive flattery, moving too fast, "soulmate" language
- Isolation: Discouraging other relationships, "You don't need them"
- Control: Monitoring, rules, punishment, "You have to ask permission"
- Threats: Any mention of harm, intimidation

Respond in JSON format:
{
  "messages": [
    {"sender": "Them", "content": "message text"},
    {"sender": "User", "content": "message text"}
  ],
  "detectedPatterns": ["pattern1", "pattern2"],
  "concernLevel": "low" | "moderate" | "high" | "crisis",
  "analysis": "Brief supportive analysis of what's happening in this conversation"
}`
          },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Please analyze this conversation screenshot. Extract all messages and identify any concerning patterns."
              },
              {
                type: "image_url",
                image_url: {
                  url: imageUrl,
                  detail: "high"
                }
              }
            ]
          }
        ]
      });

      const content = response.choices[0]?.message?.content;
      
      if (typeof content !== "string") {
        return {
          success: false,
          messages: [],
          patterns: [],
          concernLevel: "low",
          analysis: "",
          error: "Failed to analyze conversation"
        };
      }

      try {
        let jsonStr = content;
        const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (jsonMatch) {
          jsonStr = jsonMatch[1];
        }
        
        const parsed = JSON.parse(jsonStr);
        
        return {
          success: true,
          messages: parsed.messages || [],
          patterns: parsed.detectedPatterns || [],
          concernLevel: parsed.concernLevel || "low",
          analysis: parsed.analysis || ""
        };
      } catch (parseError) {
        return {
          success: false,
          messages: [],
          patterns: [],
          concernLevel: "low",
          analysis: content,
          error: "Failed to parse analysis"
        };
      }
    } catch (error) {
      return {
        success: false,
        messages: [],
        patterns: [],
        concernLevel: "low",
        analysis: "",
        error: error instanceof Error ? error.message : "Unknown error"
      };
    }
  }

  /**
   * Analyze a journal entry or personal writing
   */
  async analyzeJournalEntry(imageUrl: string): Promise<{
    success: boolean;
    text: string;
    emotionalState: string;
    themes: string[];
    supportNeeded: string;
    error?: string;
  }> {
    try {
      const response = await invokeLLM({
        messages: [
          {
            role: "system",
            content: `You are analyzing a journal entry or personal writing for ReUnity, a trauma-aware mental health application.

Your task is to:
1. Transcribe the text exactly as written (preserve original spelling/grammar)
2. Identify the emotional state expressed
3. Note key themes or concerns
4. Suggest what kind of support might be helpful

Be compassionate and non-judgmental. This is someone's private thoughts.

Respond in JSON format:
{
  "transcribedText": "The full text of the journal entry",
  "emotionalState": "Primary emotional state detected (e.g., 'anxious', 'sad', 'confused', 'hopeful')",
  "themes": ["theme1", "theme2"],
  "crisisIndicators": ["any crisis content detected"],
  "supportNeeded": "Brief note on what kind of support might help"
}`
          },
          {
            role: "user",
            content: [
              {
                type: "text",
                text: "Please transcribe and analyze this journal entry or personal writing."
              },
              {
                type: "image_url",
                image_url: {
                  url: imageUrl,
                  detail: "high"
                }
              }
            ]
          }
        ]
      });

      const content = response.choices[0]?.message?.content;
      
      if (typeof content !== "string") {
        return {
          success: false,
          text: "",
          emotionalState: "",
          themes: [],
          supportNeeded: "",
          error: "Failed to analyze journal entry"
        };
      }

      try {
        let jsonStr = content;
        const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)```/);
        if (jsonMatch) {
          jsonStr = jsonMatch[1];
        }
        
        const parsed = JSON.parse(jsonStr);
        
        return {
          success: true,
          text: parsed.transcribedText || "",
          emotionalState: parsed.emotionalState || "unknown",
          themes: parsed.themes || [],
          supportNeeded: parsed.supportNeeded || ""
        };
      } catch (parseError) {
        return {
          success: true,
          text: content,
          emotionalState: "unknown",
          themes: [],
          supportNeeded: ""
        };
      }
    } catch (error) {
      return {
        success: false,
        text: "",
        emotionalState: "",
        themes: [],
        supportNeeded: "",
        error: error instanceof Error ? error.message : "Unknown error"
      };
    }
  }

  private mapContentType(type: string): OCRResult["contentType"] {
    const mapping: Record<string, OCRResult["contentType"]> = {
      "text_message": "screenshot",
      "journal": "text",
      "handwritten": "handwritten",
      "screenshot": "screenshot",
      "document": "text",
      "other": "mixed"
    };
    return mapping[type] || "unknown";
  }

  private buildContext(parsed: any): string | null {
    const parts: string[] = [];
    
    if (parsed.isConversationScreenshot) {
      parts.push("This is a conversation screenshot.");
      if (parsed.conversationParticipants?.length) {
        parts.push(`Participants: ${parsed.conversationParticipants.join(", ")}`);
      }
    }
    
    if (parsed.containsCrisisContent) {
      parts.push("⚠️ Crisis content detected.");
      if (parsed.crisisIndicators?.length) {
        parts.push(`Indicators: ${parsed.crisisIndicators.join(", ")}`);
      }
    }
    
    if (parsed.emotionalIndicators?.length) {
      parts.push(`Emotional content: ${parsed.emotionalIndicators.join(", ")}`);
    }
    
    if (parsed.contextualNotes) {
      parts.push(parsed.contextualNotes);
    }
    
    return parts.length > 0 ? parts.join(" ") : null;
  }
}

// Export singleton instance
export const ocrService = new OCRService();
