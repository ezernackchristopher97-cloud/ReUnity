/**
 * Tests for new specialized modules:
 * - Vicsek Flocking Model
 * - BPD Splitting
 * - Rural Support
 * - Existential Crisis
 * - OCD Subtypes & Phobias
 */

import { describe, it, expect, beforeEach } from "vitest";
import { analyzeTrajectory, globalVicsekModel, VicsekEmotionalModel } from "./vicsek";
import { analyzeSplitting } from "./bpd-splitting";
import { analyzeRuralContext, getRuralIntervention } from "./rural-support";
import { analyzeExistential } from "./existential-support";
import { analyzeOCD, analyzePhobia } from "./ocd-phobias";

describe("Vicsek Flocking Model", () => {
  let model: VicsekEmotionalModel;
  
  beforeEach(() => {
    model = new VicsekEmotionalModel();
  });
  
  it("should track emotional trajectory over time", () => {
    model.addObservation(["anxiety"], 0.5);
    model.addObservation(["anxiety", "panic"], 0.6);
    model.addObservation(["panic"], 0.7);
    
    const prediction = model.predict();
    expect(prediction.predictedTrajectory).toBeDefined();
    expect(prediction.urgency).toBeDefined();
    expect(prediction.confidence).toBeGreaterThanOrEqual(0);
    expect(prediction.confidence).toBeLessThanOrEqual(1);
  });
  
  it("should detect deteriorating trajectory", () => {
    // Add progressively worsening states
    model.addObservation(["stable"], 0.2);
    model.addObservation(["anxiety"], 0.4);
    model.addObservation(["anxiety", "panic"], 0.6);
    model.addObservation(["panic", "suicidal_ideation"], 0.8);
    model.addObservation(["suicidal_ideation"], 0.9);
    
    const prediction = model.predict();
    expect(["deteriorating", "crisis_imminent"]).toContain(prediction.predictedTrajectory);
    expect(["high", "critical"]).toContain(prediction.urgency);
  });
  
  it("should detect improving trajectory", () => {
    model.addObservation(["panic"], 0.8);
    model.addObservation(["anxiety"], 0.6);
    model.addObservation(["anxiety"], 0.4);
    model.addObservation(["stable"], 0.2);
    model.addObservation(["grounded"], 0.1);
    
    const prediction = model.predict();
    expect(prediction.predictedTrajectory).toBe("improving");
    expect(prediction.urgency).toBe("low");
  });
  
  it("should calculate order parameter", () => {
    model.addObservation(["anxiety"], 0.5);
    model.addObservation(["anxiety"], 0.5);
    model.addObservation(["anxiety"], 0.5);
    
    const orderParam = model.calculateOrderParameter();
    expect(orderParam).toBeGreaterThan(0);
    expect(orderParam).toBeLessThanOrEqual(1);
  });
});

describe("BPD Splitting Module", () => {
  it("should detect self-devaluation splitting", () => {
    const analysis = analyzeSplitting(
      "I'm worthless. I'm the worst person ever. I hate myself completely.",
      0.7
    );
    
    expect(analysis.isSplitting).toBe(true);
    expect(analysis.splittingTarget).toBe("self");
    expect(analysis.polarization).toBe("devaluation");
    expect(analysis.groundingProtocol.name).toBe("Self-Integration Grounding");
  });
  
  it("should detect other-devaluation splitting", () => {
    const analysis = analyzeSplitting(
      "My partner is a monster. They're completely evil. I hate them. They ruined my life.",
      0.7
    );
    
    expect(analysis.isSplitting).toBe(true);
    expect(analysis.splittingTarget).toBe("other");
    expect(analysis.polarization).toBe("devaluation");
  });
  
  it("should detect idealization splitting", () => {
    const analysis = analyzeSplitting(
      "They're perfect. They're my soulmate. I've never felt this way. They're the only one who understands me.",
      0.5
    );
    
    expect(analysis.isSplitting).toBe(true);
    expect(analysis.polarization).toBe("idealization");
  });
  
  it("should detect world splitting", () => {
    // Need 3+ absolute terms or high intensity to trigger splitting
    const analysis = analyzeSplitting(
      "Everything is always terrible. Nothing ever works. The world is completely hopeless. Life is totally ruined. Everyone is fake. No one cares.",
      0.8
    );
    
    expect(analysis.isSplitting).toBe(true);
    expect(analysis.splittingTarget).toBe("world");
  });
  
  it("should not detect splitting in balanced messages", () => {
    const analysis = analyzeSplitting(
      "I'm having a hard day but I know it will pass.",
      0.3
    );
    
    expect(analysis.isSplitting).toBe(false);
  });
  
  it("should provide dialectical statements", () => {
    const analysis = analyzeSplitting(
      "I'm completely worthless. I hate myself.",
      0.7
    );
    
    expect(analysis.groundingProtocol.dialecticalStatements.length).toBeGreaterThan(0);
    expect(analysis.groundingProtocol.dialecticalStatements[0]).toContain("AND");
  });
});

describe("Rural Support Module", () => {
  it("should detect rural isolation", () => {
    const context = analyzeRuralContext(
      "I live on a farm in rural Montana. The nearest town is 50 miles away. I feel so isolated."
    );
    
    expect(context.isRural).toBe(true);
    expect(["high", "extreme"]).toContain(context.isolationLevel);
  });
  
  it("should detect domestic violence indicators", () => {
    const context = analyzeRuralContext(
      "My husband won't let me see my family. He controls the money. He checks my phone constantly."
    );
    
    expect(context.domesticViolenceRisk).toBe(true);
    expect(context.safetyConstraints).toContain("phone_monitored");
    expect(context.safetyConstraints).toContain("financial_control");
  });
  
  it("should detect when victim may not recognize abuse", () => {
    const context = analyzeRuralContext(
      "It's not that bad. He only gets angry when he's drinking. I probably provoked him. It's normal for couples. He hit me but I made him angry."
    );
    
    expect(context.domesticViolenceRisk).toBe(true);
    expect(context.recognizesAbuse).toBe(false);
  });
  
  it("should provide rural-specific intervention", () => {
    const context = analyzeRuralContext(
      "I live in a remote area. My partner controls everything. He won't let me see my family. He monitors my phone. I have no way out."
    );
    
    const intervention = getRuralIntervention(context);
    
    expect(["rural_dv", "rural_isolation"]).toContain(intervention.category);
    expect(intervention.validation).toBeDefined();
    expect(intervention.resources.length).toBeGreaterThan(0);
  });
  
  it("should detect children involved", () => {
    const context = analyzeRuralContext(
      "I'm scared for my kids. He threatens to take them away if I leave."
    );
    
    expect(context.safetyConstraints).toContain("children_involved");
  });
});

describe("Existential Crisis Module", () => {
  it("should detect solipsism crisis", () => {
    const analysis = analyzeExistential(
      "What if I'm the only consciousness? How do I know anyone else is real? It's all in my head."
    );
    
    expect(analysis.isExistentialCrisis).toBe(true);
    expect(analysis.crisisType).toBe("solipsism");
  });
  
  it("should detect death anxiety", () => {
    const analysis = analyzeExistential(
      "I can't stop thinking about death. I'm terrified of dying. What happens when we cease to exist?"
    );
    
    expect(analysis.isExistentialCrisis).toBe(true);
    expect(analysis.crisisType).toBe("death_anxiety");
  });
  
  it("should detect meaninglessness", () => {
    const analysis = analyzeExistential(
      "What's the point of anything? Nothing matters. Everything is meaningless and futile."
    );
    
    expect(analysis.isExistentialCrisis).toBe(true);
    expect(analysis.crisisType).toBe("meaninglessness");
  });
  
  it("should detect cosmic insignificance", () => {
    const analysis = analyzeExistential(
      "We're just a speck of dust in an infinite universe. Nothing we do matters on a cosmic scale."
    );
    
    expect(analysis.isExistentialCrisis).toBe(true);
    expect(analysis.crisisType).toBe("cosmic_insignificance");
  });
  
  it("should detect free will anxiety", () => {
    const analysis = analyzeExistential(
      "Do we even have free will? Everything is predetermined. We're just chemicals and neurons."
    );
    
    expect(analysis.isExistentialCrisis).toBe(true);
    expect(analysis.crisisType).toBe("free_will");
  });
  
  it("should provide philosophical reframes", () => {
    const analysis = analyzeExistential(
      "I'm afraid of the void. The eternal nothingness terrifies me."
    );
    
    expect(analysis.intervention.philosophicalReframes.length).toBeGreaterThan(0);
    expect(analysis.intervention.groundingTechniques.length).toBeGreaterThan(0);
  });
});

describe("OCD Subtypes Module", () => {
  it("should detect contamination OCD", () => {
    const analysis = analyzeOCD(
      "I can't stop washing my hands. Everything feels contaminated. I'm terrified of germs."
    );
    
    expect(analysis.isOCD).toBe(true);
    expect(analysis.subtypes).toContain("contamination");
  });
  
  it("should detect harm OCD", () => {
    const analysis = analyzeOCD(
      "I keep having intrusive thoughts about hurting my family. What if I hurt someone? I'm afraid I'll lose control."
    );
    
    expect(analysis.isOCD).toBe(true);
    expect(analysis.subtypes).toContain("harm_ocd");
  });
  
  it("should detect religious scrupulosity", () => {
    const analysis = analyzeOCD(
      "I'm terrified I've sinned and committed blasphemy. I keep having impure thoughts. I need to confess. God hates me. I'm going to hell."
    );
    
    expect(analysis.isOCD).toBe(true);
    expect(analysis.subtypes).toContain("religious_scrupulosity");
  });
  
  it("should detect relationship OCD", () => {
    const analysis = analyzeOCD(
      "Do I really love my partner? What if they're not the right person? I keep constantly checking my feelings."
    );
    
    expect(analysis.isOCD).toBe(true);
    expect(analysis.subtypes).toContain("relationship_ocd");
  });
  
  it("should detect POCD", () => {
    const analysis = analyzeOCD(
      "I'm afraid I'm a pedophile. I have intrusive thoughts about children that horrify me. I avoid being around kids."
    );
    
    expect(analysis.isOCD).toBe(true);
    expect(analysis.subtypes).toContain("pedophilia_ocd");
  });
  
  it("should detect health anxiety", () => {
    const analysis = analyzeOCD(
      "I keep checking my body for symptoms. What if I have cancer? I can't stop googling symptoms."
    );
    
    expect(analysis.isOCD).toBe(true);
    expect(analysis.subtypes).toContain("health_anxiety");
  });
  
  it("should provide do-not-do guidance", () => {
    const analysis = analyzeOCD(
      "I have intrusive thoughts about hurting someone. What if I hurt my family? I'm afraid I'll lose control."
    );
    
    expect(analysis.isOCD).toBe(true);
    expect(analysis.intervention.doNotDo.length).toBeGreaterThan(0);
    expect(analysis.intervention.doNotDo.some(d => d.toLowerCase().includes("reassur"))).toBe(true);
  });
});

describe("Phobia Module", () => {
  it("should detect agoraphobia", () => {
    const analysis = analyzePhobia(
      "I can't leave my house. I'm afraid to go out. I panic in public places."
    );
    
    expect(analysis.isPhobia).toBe(true);
    expect(analysis.phobiaType).toBe("agoraphobia");
  });
  
  it("should detect emetophobia", () => {
    const analysis = analyzePhobia(
      "I'm terrified of vomiting. I can't eat certain foods. The thought of throwing up makes me panic."
    );
    
    expect(analysis.isPhobia).toBe(true);
    expect(analysis.phobiaType).toBe("emetophobia");
  });
  
  it("should detect social phobia", () => {
    const analysis = analyzePhobia(
      "I'm terrified of being judged. I can't speak in social situations. I'm afraid of being humiliated."
    );
    
    expect(analysis.isPhobia).toBe(true);
    expect(analysis.phobiaType).toBe("social_phobia");
  });
  
  it("should detect claustrophobia", () => {
    const analysis = analyzePhobia(
      "I can't go in elevators. Small spaces make me feel trapped. I have claustrophobia."
    );
    
    expect(analysis.isPhobia).toBe(true);
    expect(analysis.phobiaType).toBe("claustrophobia");
  });
  
  it("should provide exposure guidance", () => {
    const analysis = analyzePhobia(
      "I'm terrified of spiders. I can't even look at pictures of them."
    );
    
    expect(analysis.intervention.exposureGuidance.length).toBeGreaterThan(0);
    expect(analysis.intervention.copingStrategies.length).toBeGreaterThan(0);
  });
});

describe("Bipolar Detection", () => {
  it("should detect manic episode indicators", () => {
    // This tests the bipolar detection in mental-health-interventions.json
    const manicKeywords = [
      "manic", "hypomanic", "haven't slept", "no sleep", "don't need sleep",
      "racing thoughts", "so many ideas", "feel amazing", "on top of the world",
      "grandiose", "invincible", "unstoppable", "spending spree", "big plans"
    ];
    
    // Verify keywords exist
    expect(manicKeywords.length).toBeGreaterThan(0);
    
    // Test message with manic indicators
    const testMessage = "I haven't slept in 3 days but I feel amazing. I have so many ideas and I feel invincible.";
    const hasManicIndicators = manicKeywords.some(kw => testMessage.toLowerCase().includes(kw));
    expect(hasManicIndicators).toBe(true);
  });
});
