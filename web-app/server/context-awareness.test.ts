import { describe, it, expect } from "vitest";
import { 
  detectEnvironment, 
  detectCulturalContext, 
  detectCommunityContext, 
  detectSocioeconomicContext,
  analyzeContext,
  formatContextResources
} from "./context-awareness";

describe("Context Awareness Module", () => {
  describe("Environment Detection", () => {
    it("should detect urban environment", () => {
      const result = detectEnvironment("I live in downtown Chicago near the subway");
      expect(result).not.toBeNull();
      expect(result?.type).toBe("urban");
    });

    it("should detect suburban environment", () => {
      const result = detectEnvironment("I live in a suburb with a long commute to work");
      expect(result).not.toBeNull();
      expect(result?.type).toBe("suburban");
    });

    it("should detect rural environment", () => {
      const result = detectEnvironment("I live on a farm in rural Montana, hours away from any therapist");
      expect(result).not.toBeNull();
      expect(result?.type).toBe("rural");
    });

    it("should detect remote environment", () => {
      const result = detectEnvironment("I live off-grid in the wilderness with no cell service");
      expect(result).not.toBeNull();
      expect(result?.type).toBe("remote");
    });

    it("should return null for no environment indicators", () => {
      const result = detectEnvironment("I feel sad today");
      expect(result).toBeNull();
    });
  });

  describe("Cultural Context Detection", () => {
    it("should detect Latino/Hispanic cultural context", () => {
      const result = detectCulturalContext("My familia doesn't understand mental health");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].culture).toBe("latinx");
    });

    it("should detect Black/African American cultural context", () => {
      const result = detectCulturalContext("As a Black woman, I face discrimination at work");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].culture).toBe("blackAfrican");
    });

    it("should detect Asian cultural context", () => {
      const result = detectCulturalContext("My Asian parents expect me to be perfect");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].culture).toBe("asian");
    });

    it("should detect Indigenous cultural context", () => {
      const result = detectCulturalContext("I'm Native American and live on the reservation");
      expect(result.length).toBeGreaterThan(0);
      expect(result.some(c => c.culture === "indigenous")).toBe(true);
    });

    it("should return empty array for no cultural indicators", () => {
      const result = detectCulturalContext("I feel anxious");
      expect(result.length).toBe(0);
    });
  });

  describe("Community Context Detection", () => {
    it("should detect LGBTQ+ community context", () => {
      const result = detectCommunityContext("I'm transgender and struggling with coming out");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].community).toBe("lgbtq");
    });

    it("should detect veteran community context", () => {
      const result = detectCommunityContext("I'm a veteran dealing with PTSD from my deployment");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].community).toBe("veteran");
    });

    it("should detect immigrant community context", () => {
      const result = detectCommunityContext("As an immigrant, I worry about my visa status");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].community).toBe("immigrant");
    });

    it("should detect elderly community context", () => {
      const result = detectCommunityContext("I'm a retired senior living alone in a nursing home");
      expect(result.length).toBeGreaterThan(0);
      // May detect multiple communities, check that elderly is among them
      expect(result.some(c => c.community === "elderly")).toBe(true);
    });

    it("should detect youth community context", () => {
      const result = detectCommunityContext("I'm a teenager in high school dealing with bullying");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].community).toBe("youth");
    });

    it("should detect religious community context - Christianity", () => {
      const result = detectCommunityContext("I go to church every Sunday and pray for guidance");
      expect(result.length).toBeGreaterThan(0);
      expect(result.some(c => c.community.includes("christianity"))).toBe(true);
    });

    it("should detect disability community context", () => {
      const result = detectCommunityContext("I'm disabled and use a wheelchair, dealing with chronic pain");
      expect(result.length).toBeGreaterThan(0);
      expect(result[0].community).toBe("disability");
    });
  });

  describe("Socioeconomic Context Detection", () => {
    it("should detect poverty context", () => {
      const result = detectSocioeconomicContext("I can't afford therapy, I'm broke and on food stamps");
      expect(result).toContain("poverty");
    });

    it("should detect housing insecurity context", () => {
      const result = detectSocioeconomicContext("I'm facing eviction and might be homeless soon");
      expect(result).toContain("housingSecurity");
    });

    it("should return empty array for no socioeconomic indicators", () => {
      const result = detectSocioeconomicContext("I feel stressed about work");
      expect(result.length).toBe(0);
    });
  });

  describe("Full Context Analysis", () => {
    it("should analyze comprehensive context from message", () => {
      const result = analyzeContext(
        "I'm a Black transgender veteran living in rural Montana on food stamps",
        []
      );
      
      expect(result.environment?.type).toBe("rural");
      expect(result.cultural.length).toBeGreaterThan(0);
      expect(result.community.length).toBeGreaterThan(0);
      expect(result.socioeconomic).toContain("poverty");
      expect(result.contextualGuidance).toBeTruthy();
    });

    it("should include conversation history in analysis", () => {
      const history = ["I live on a farm in rural area", "Mi familia es de Mexico"];
      const result = analyzeContext("I feel anxious today", history);
      
      // Should detect rural from history
      expect(result.environment?.type).toBe("rural");
      // Should detect latinx from history (using familia indicator)
      expect(result.cultural.some(c => c.culture === "latinx")).toBe(true);
    });

    it("should gather additional resources based on context", () => {
      const result = analyzeContext("I'm a veteran struggling with PTSD", []);
      expect(result.additionalResources.length).toBeGreaterThan(0);
    });
  });

  describe("Format Context Resources", () => {
    it("should format resources for response", () => {
      const analysis = analyzeContext("I'm a veteran", []);
      const formatted = formatContextResources(analysis);
      
      if (analysis.additionalResources.length > 0) {
        expect(formatted).toContain("Community-Specific Support");
      }
    });

    it("should deduplicate resources", () => {
      const analysis = analyzeContext("I'm a Black veteran in the military", []);
      const formatted = formatContextResources(analysis);
      
      // Check that resources are not duplicated
      const lines = formatted.split("\n").filter(l => l.startsWith("•"));
      const names = lines.map(l => l.split(":")[0]);
      const uniqueNames = [...new Set(names)];
      expect(names.length).toBe(uniqueNames.length);
    });
  });
});
