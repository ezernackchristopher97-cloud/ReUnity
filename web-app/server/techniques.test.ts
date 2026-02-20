import { describe, it, expect } from "vitest";
import {
  detectConditions,
  detectStates,
  selectTechniques,
  getValidationStatements,
  getCrisisProtocol,
  getTailoredIntervention,
  getConditionInfo,
  getAllCategories,
  getTechniquesByCategory
} from "./techniques";

describe("Techniques Module", () => {
  describe("detectConditions", () => {
    it("should detect PTSD from keywords", () => {
      const conditions = detectConditions("I keep having flashbacks and nightmares about the trauma", []);
      expect(conditions).toContain("ptsd");
    });

    it("should detect BPD from keywords", () => {
      const conditions = detectConditions("everyone leaves me, I feel so empty inside", []);
      expect(conditions).toContain("bpd");
    });

    it("should detect depression from keywords", () => {
      const conditions = detectConditions("I feel hopeless and worthless, nothing matters anymore", []);
      expect(conditions).toContain("depression");
    });

    it("should detect DID/OSDD from keywords", () => {
      const conditions = detectConditions("we have different parts and sometimes we lose time", []);
      expect(conditions).toContain("did_osdd");
    });

    it("should detect multiple conditions", () => {
      const conditions = detectConditions("I have flashbacks from trauma and everyone leaves me", []);
      expect(conditions.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe("detectStates", () => {
    it("should detect dissociation state", () => {
      const states = detectStates("I feel like I'm not real, floating outside my body", 0.5, "emotional");
      expect(states).toContain("dissociation");
    });

    it("should detect panic state", () => {
      const states = detectStates("I can't breathe, my heart is racing, I'm going to die", 0.5, "emotional");
      expect(states).toContain("panic");
    });

    it("should detect suicidal ideation", () => {
      const states = detectStates("I want to die, I can't do this anymore", 0.5, "emotional");
      expect(states).toContain("suicidal_ideation");
    });

    it("should add overwhelm for high entropy", () => {
      const states = detectStates("I feel okay", 0.8, "emotional");
      expect(states).toContain("overwhelm");
    });

    it("should add panic for crisis regime", () => {
      const states = detectStates("I feel okay", 0.5, "crisis");
      expect(states).toContain("panic");
    });
  });

  describe("selectTechniques", () => {
    it("should select techniques for dissociation", () => {
      const techniques = selectTechniques(["dissociation"], [], "medium");
      expect(techniques.length).toBeGreaterThan(0);
      expect(techniques.some(t => t.id === "cold_water_reset" || t.id === "texture_anchoring")).toBe(true);
    });

    it("should select techniques for panic", () => {
      const techniques = selectTechniques(["panic"], [], "medium");
      expect(techniques.length).toBeGreaterThan(0);
    });

    it("should prioritize condition-specific techniques", () => {
      const techniques = selectTechniques(["anxiety"], ["ptsd"], "medium");
      expect(techniques.length).toBeGreaterThan(0);
    });

    it("should respect contraindications", () => {
      const techniques = selectTechniques(["dissociation"], [], "medium", ["heart_conditions"]);
      // Cold water reset has heart_conditions as contraindication
      const hasColdWater = techniques.some(t => t.id === "cold_water_reset");
      expect(hasColdWater).toBe(false);
    });
  });

  describe("getValidationStatements", () => {
    it("should return validation statements for conditions", () => {
      const statements = getValidationStatements(["ptsd"]);
      expect(statements.length).toBeGreaterThan(0);
    });

    it("should return empty array for no conditions", () => {
      const statements = getValidationStatements([]);
      expect(statements).toEqual([]);
    });
  });

  describe("getCrisisProtocol", () => {
    it("should return crisis protocol for suicidal ideation", () => {
      const protocol = getCrisisProtocol(["suicidal_ideation"]);
      expect(protocol).not.toBeNull();
      expect(protocol?.priority).toBe("HIGHEST");
    });

    it("should return crisis protocol for panic", () => {
      const protocol = getCrisisProtocol(["panic"]);
      expect(protocol).not.toBeNull();
      expect(protocol?.priority).toBe("MEDIUM");
    });

    it("should return null for non-crisis states", () => {
      const protocol = getCrisisProtocol(["anxiety"]);
      expect(protocol).toBeNull();
    });
  });

  describe("getTailoredIntervention", () => {
    it("should return complete intervention object", () => {
      const intervention = getTailoredIntervention(
        "I'm having a flashback and can't breathe",
        [],
        0.7,
        "crisis"
      );
      
      expect(intervention).toHaveProperty("techniques");
      expect(intervention).toHaveProperty("validationStatements");
      expect(intervention).toHaveProperty("crisisProtocol");
      expect(intervention).toHaveProperty("detectedConditions");
      expect(intervention).toHaveProperty("detectedStates");
      expect(intervention).toHaveProperty("promptGuidance");
    });

    it("should detect conditions and states from message", () => {
      const intervention = getTailoredIntervention(
        "I keep having flashbacks and I feel like I'm dissociating",
        [],
        0.6,
        "emotional"
      );
      
      expect(intervention.detectedConditions).toContain("ptsd");
      expect(intervention.detectedStates).toContain("dissociation");
    });

    it("should include prompt guidance with techniques", () => {
      const intervention = getTailoredIntervention(
        "I'm panicking and can't breathe",
        [],
        0.8,
        "crisis"
      );
      
      expect(intervention.promptGuidance).toContain("GROUNDING TECHNIQUES");
      expect(intervention.promptGuidance).toContain("Do NOT default to generic 5-4-3-2-1");
    });
  });

  describe("Utility Functions", () => {
    it("should get condition info", () => {
      const info = getConditionInfo("ptsd");
      expect(info).not.toBeNull();
      expect(info?.name).toBe("Post-Traumatic Stress Disorder");
    });

    it("should return null for unknown condition", () => {
      const info = getConditionInfo("unknown_condition");
      expect(info).toBeNull();
    });

    it("should get all categories", () => {
      const categories = getAllCategories();
      expect(categories).toContain("sensory");
      expect(categories).toContain("somatic");
      expect(categories).toContain("cognitive");
      expect(categories).toContain("breathwork");
      expect(categories).toContain("entropy_regulation");
      expect(categories).toContain("fragmentation_restoration");
    });

    it("should get techniques by category", () => {
      const sensoryTechniques = getTechniquesByCategory("sensory");
      expect(sensoryTechniques.length).toBeGreaterThan(0);
      expect(sensoryTechniques.every(t => t.category === "sensory")).toBe(true);
    });
  });
});
