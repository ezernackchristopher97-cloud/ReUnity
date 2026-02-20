/**
 * Existential Crisis Support Module
 * 
 * Specialized interventions for existential distress including:
 * - Solipsism and reality questioning
 * - Fear of death and the unknown
 * - Meaninglessness and nihilism
 * - Existential dread and anxiety
 * - Depersonalization from existential causes
 * - Cosmic insignificance
 * - Free will and determinism anxiety
 */

export interface ExistentialAnalysis {
  isExistentialCrisis: boolean;
  crisisType: ExistentialCrisisType | null;
  intensity: number;  // 0-1
  entropyContribution: number;
  intervention: ExistentialIntervention;
}

export type ExistentialCrisisType = 
  | "solipsism"
  | "death_anxiety"
  | "meaninglessness"
  | "cosmic_insignificance"
  | "free_will"
  | "reality_questioning"
  | "existential_isolation"
  | "absurdity"
  | "void_fear";

export interface ExistentialIntervention {
  validation: string;
  normalizing: string[];
  groundingTechniques: string[];
  philosophicalReframes: string[];
  practicalSteps: string[];
  whenToSeekHelp: string;
}

// Detection patterns
const existentialIndicators: Record<ExistentialCrisisType, string[]> = {
  solipsism: [
    "am i real", "is anyone real", "only consciousness", "everyone is fake",
    "simulation", "nothing is real", "all in my head", "no one else exists",
    "can't prove", "how do i know", "what if i'm the only", "alone in universe",
    "brain in a vat", "matrix", "npcs", "philosophical zombie"
  ],
  
  death_anxiety: [
    "afraid of dying", "fear of death", "going to die", "what happens when",
    "after death", "cease to exist", "eternal nothing", "oblivion",
    "mortality", "finite", "running out of time", "death is coming",
    "can't stop thinking about death", "terrified of dying", "inevitable"
  ],
  
  meaninglessness: [
    "no point", "what's the point", "meaningless", "nothing matters",
    "why bother", "futile", "pointless", "no purpose", "no meaning",
    "why do anything", "all for nothing", "doesn't matter", "nihilism",
    "empty", "hollow", "void"
  ],
  
  cosmic_insignificance: [
    "insignificant", "tiny", "universe is so big", "don't matter",
    "speck of dust", "cosmic", "infinite universe", "we're nothing",
    "pale blue dot", "billions of years", "heat death", "entropy",
    "everything will end", "sun will die", "stars will burn out"
  ],
  
  free_will: [
    "free will", "determinism", "no choice", "predetermined", "illusion of choice",
    "no control", "just chemicals", "just neurons", "programmed",
    "no agency", "fate", "destiny", "couldn't have done otherwise"
  ],
  
  reality_questioning: [
    "what is reality", "is this real", "questioning reality", "nothing feels real",
    "dream", "wake up", "can't tell", "losing grip", "reality is",
    "perception", "subjective", "objective reality", "consensus reality"
  ],
  
  existential_isolation: [
    "fundamentally alone", "no one can understand", "trapped in my mind",
    "can't truly connect", "barrier", "isolated consciousness", "alone forever",
    "no one really knows me", "unbridgeable gap", "separate"
  ],
  
  absurdity: [
    "absurd", "ridiculous", "makes no sense", "chaos", "random",
    "no reason", "arbitrary", "cosmic joke", "laughable", "meaningless universe",
    "camus", "sisyphus", "why are we here"
  ],
  
  void_fear: [
    "void", "abyss", "nothingness", "empty", "darkness",
    "staring into", "falling into", "consumed by", "swallowed",
    "infinite darkness", "eternal void", "black hole"
  ]
};

// Interventions for each type
const interventions: Record<ExistentialCrisisType, ExistentialIntervention> = {
  solipsism: {
    validation: "The question of whether other minds exist is one humans have grappled with for millennia. The fact that you're questioning this shows deep thinking, and the distress it causes is real.",
    normalizing: [
      "Solipsism is a philosophical position that's impossible to definitively disprove - and that's okay.",
      "Many philosophers and thinkers have wrestled with this exact question.",
      "The uncertainty you're feeling is a feature of consciousness, not a bug."
    ],
    groundingTechniques: [
      "Focus on sensory experience right now. What do you feel, hear, see?",
      "Engage with something physical - hold an object, feel its weight and texture.",
      "Notice how your body responds to the world around you.",
      "Try having a conversation with someone. Notice how they surprise you - could you really predict everything they say?"
    ],
    philosophicalReframes: [
      "Even if we can't prove other minds exist, we can choose to act as if they do. This is called 'pragmatic belief.'",
      "The fact that other people surprise us, challenge us, and teach us things we didn't know suggests something beyond our own mind.",
      "Whether or not reality is 'real' in some ultimate sense, your experience of it is real to you.",
      "Perhaps the question isn't 'is this real?' but 'how do I want to engage with my experience?'"
    ],
    practicalSteps: [
      "Limit time spent on philosophical rabbit holes when distressed.",
      "Engage in activities that require interaction with the physical world.",
      "Connect with others, even if the connection feels uncertain.",
      "Focus on what you can do and experience, rather than what you can prove."
    ],
    whenToSeekHelp: "If these thoughts are causing significant distress, interfering with daily life, or accompanied by depersonalization/derealization, speaking with a therapist who understands existential concerns can help."
  },
  
  death_anxiety: {
    validation: "Fear of death is one of the most fundamental human experiences. The fact that you're grappling with mortality shows you're engaging with life's biggest questions. This fear is painful, and it's real.",
    normalizing: [
      "Death anxiety is universal - nearly everyone experiences it at some point.",
      "Thinking about death often increases during times of stress, transition, or after loss.",
      "Many people find that confronting death anxiety actually leads to living more fully."
    ],
    groundingTechniques: [
      "Bring your attention to this present moment. Right now, you are alive.",
      "Feel your breath moving in and out. This is life happening.",
      "Notice five things you can see, four you can hear, three you can touch.",
      "Place your hand on your heart. Feel it beating. You are here, now."
    ],
    philosophicalReframes: [
      "The Stoics practiced 'memento mori' - remembering death - not to create fear, but to appreciate life.",
      "What we fear about death is often the unknown. But we also don't remember before we were born, and that doesn't trouble us.",
      "Many who've had near-death experiences report peace, not fear.",
      "Perhaps death is not an ending but a transformation - like deep sleep, or like before birth."
    ],
    practicalSteps: [
      "Write about what specifically frightens you about death. Sometimes naming fears reduces their power.",
      "Consider what would make your life feel meaningful, regardless of its length.",
      "Connect with loved ones. Relationships often ease death anxiety.",
      "Limit late-night rumination - death anxiety often intensifies when tired."
    ],
    whenToSeekHelp: "If death anxiety is constant, interfering with sleep or daily functioning, or causing panic attacks, a therapist specializing in existential concerns or anxiety can provide support."
  },
  
  meaninglessness: {
    validation: "Feeling like nothing matters is one of the most painful experiences. This emptiness is real, and it's exhausting to carry. You're not weak for feeling this way.",
    normalizing: [
      "Questions of meaning have occupied humans throughout history.",
      "Feeling meaningless often comes during depression, transitions, or after loss.",
      "Many people who've felt this way have found their way to meaning again."
    ],
    groundingTechniques: [
      "Notice one small thing that brought even a flicker of interest today.",
      "Do something with your hands - cook, clean, create. Meaning often comes through doing.",
      "Step outside. Notice the sky, the air. You are part of something larger.",
      "Connect with another living being - person, pet, plant."
    ],
    philosophicalReframes: [
      "Perhaps meaning isn't found but created. We are meaning-making creatures.",
      "The universe may not have inherent meaning, but that means we're free to create our own.",
      "Small moments of connection, beauty, or kindness can be meaningful even in an indifferent universe.",
      "Viktor Frankl, who survived the Holocaust, wrote that we can find meaning even in suffering."
    ],
    practicalSteps: [
      "Start very small. What's one thing you could do today that might matter to someone?",
      "Help someone else, even in a tiny way. Meaning often comes through contribution.",
      "Engage with something you used to enjoy, even if it feels pointless. Sometimes action precedes feeling.",
      "Consider what you would want your life to stand for, if you could choose."
    ],
    whenToSeekHelp: "Persistent feelings of meaninglessness, especially with hopelessness, loss of interest, or thoughts of self-harm, warrant professional support. This can be a symptom of depression."
  },
  
  cosmic_insignificance: {
    validation: "Looking at the vastness of the universe and feeling small is a profound experience. The scale is genuinely overwhelming. Your feeling of insignificance in the face of infinity is understandable.",
    normalizing: [
      "This feeling is sometimes called 'cosmic horror' or 'the overview effect' - it's a recognized human experience.",
      "Many astronauts report similar feelings when seeing Earth from space.",
      "Throughout history, humans have grappled with our place in the cosmos."
    ],
    groundingTechniques: [
      "Zoom back in. Focus on what's right in front of you, right now.",
      "Touch something solid. You are here, in this moment, on this planet.",
      "Think of one person who would notice if you weren't here.",
      "Consider: the universe is vast, AND you are experiencing it. That's remarkable."
    ],
    philosophicalReframes: [
      "Scale doesn't determine significance. A mother's love isn't less real because the universe is large.",
      "You are the universe experiencing itself. That's not insignificant - that's extraordinary.",
      "Perhaps significance isn't about size but about experience, connection, and consciousness.",
      "The fact that matter organized itself into something that can contemplate the cosmos is itself cosmic."
    ],
    practicalSteps: [
      "Limit time spent on cosmic-scale content when distressed.",
      "Focus on your immediate sphere of influence - the people and things you can affect.",
      "Engage in something that feels meaningful at a human scale.",
      "Connect with others who share your interests and values."
    ],
    whenToSeekHelp: "If feelings of insignificance lead to hopelessness, depression, or thoughts of self-harm, professional support can help you integrate these big questions in a healthier way."
  },
  
  free_will: {
    validation: "The question of whether we have free will is genuinely unresolved. Scientists and philosophers still debate it. Your distress about this uncertainty is valid.",
    normalizing: [
      "This question has been debated for thousands of years without resolution.",
      "Many people go through periods of questioning free will, especially after learning about neuroscience or determinism.",
      "The uncertainty itself doesn't have to be distressing."
    ],
    groundingTechniques: [
      "Make a small choice right now - move your hand, look somewhere different. Notice the experience of choosing.",
      "Whether or not the choice was 'determined,' you experienced making it.",
      "Focus on what you can do, rather than whether you 'really' chose to do it.",
      "Engage your body - physical activity can quiet philosophical rumination."
    ],
    philosophicalReframes: [
      "Even if determinism is true, we still experience choice. That experience is real.",
      "'Compatibilism' suggests free will and determinism can coexist - we're free when we act according to our own desires, even if those desires were caused.",
      "The question may be less important than how we live. We can act as if we have choice.",
      "Perhaps 'free will' isn't about being uncaused, but about being the kind of cause we want to be."
    ],
    practicalSteps: [
      "Focus on practical decision-making rather than metaphysical questions.",
      "Notice that believing in determinism doesn't actually change how you live.",
      "Limit exposure to deterministic content when it causes distress.",
      "Engage with activities that feel agentic - creating, building, helping."
    ],
    whenToSeekHelp: "If these thoughts are causing significant anxiety, depression, or interfering with daily life, a therapist can help you work through them."
  },
  
  reality_questioning: {
    validation: "Questioning the nature of reality is disorienting and can be frightening. The ground feels like it's shifting under you. This experience is real, even if reality feels uncertain.",
    normalizing: [
      "Philosophers have questioned reality for millennia - you're in good company.",
      "This often happens during stress, sleep deprivation, or after intense experiences.",
      "Many people go through periods of reality questioning and come out the other side."
    ],
    groundingTechniques: [
      "Focus on sensory experience. What can you touch, taste, smell, hear, see right now?",
      "Hold something cold or textured. Focus entirely on the sensation.",
      "Name five things you can see. Four you can hear. Three you can touch.",
      "Splash cold water on your face. Feel it."
    ],
    philosophicalReframes: [
      "Whatever reality 'really' is, your experience of it is real to you.",
      "We can function perfectly well without knowing the ultimate nature of reality.",
      "Perhaps the question isn't 'is this real?' but 'how do I want to engage with my experience?'",
      "Reality is what you interact with. That interaction is undeniable."
    ],
    practicalSteps: [
      "Prioritize sleep, nutrition, and physical health - these affect perception.",
      "Limit substances that alter perception.",
      "Engage in grounding activities - exercise, nature, social connection.",
      "Keep a routine to provide structure and predictability."
    ],
    whenToSeekHelp: "If reality questioning is persistent, accompanied by depersonalization/derealization, or causing significant distress, professional support is important. This can sometimes indicate dissociation or other conditions that respond well to treatment."
  },
  
  existential_isolation: {
    validation: "The feeling that no one can truly know you, that there's an unbridgeable gap between minds, is one of the loneliest experiences. This isolation is painful, and your pain is real.",
    normalizing: [
      "Existential isolation is a recognized human experience - we are, in some sense, alone in our consciousness.",
      "Many people feel this way, especially during difficult times.",
      "Paradoxically, knowing others share this isolation can itself be connecting."
    ],
    groundingTechniques: [
      "Reach out to someone, even with a simple message. Connection doesn't require perfect understanding.",
      "Notice moments of resonance with others - shared laughter, mutual understanding, even brief.",
      "Engage with art, music, or writing that expresses what you feel. Someone else felt this too.",
      "Be present with another living being - person, pet. Notice the connection, however imperfect."
    ],
    philosophicalReframes: [
      "Perfect understanding may be impossible, but meaningful connection isn't.",
      "We can never fully know another mind, but we can share experiences, create together, love.",
      "Perhaps the gap between us is also what makes connection precious.",
      "Every act of communication is a bridge across the isolation. It doesn't have to be perfect to matter."
    ],
    practicalSteps: [
      "Practice vulnerability - share something real with someone you trust.",
      "Seek out others who think about these things - philosophy groups, deep conversations.",
      "Express yourself creatively - art, writing, music can bridge isolation.",
      "Focus on shared experiences rather than perfect understanding."
    ],
    whenToSeekHelp: "If feelings of isolation are persistent and painful, a therapist can provide a space for deep connection and help you build bridges to others."
  },
  
  absurdity: {
    validation: "Seeing the absurdity of existence - the gap between our need for meaning and the universe's silence - is disorienting. This recognition is part of being deeply aware. Your distress is understandable.",
    normalizing: [
      "Albert Camus wrote extensively about the absurd - you're grappling with questions great thinkers have faced.",
      "Many people experience this, especially during transitions or after loss.",
      "Recognizing absurdity doesn't have to lead to despair."
    ],
    groundingTechniques: [
      "Engage in something physical - the body doesn't care about absurdity.",
      "Do something that brings simple pleasure - eat something good, feel the sun.",
      "Connect with someone. Human connection persists despite absurdity.",
      "Create something. The act of creation is a response to absurdity."
    ],
    philosophicalReframes: [
      "Camus concluded we must imagine Sisyphus happy - we can embrace life despite its absurdity.",
      "The absurd is the starting point, not the conclusion. What do you do with it?",
      "Perhaps the response to a meaningless universe is to create meaning anyway, defiantly.",
      "Laughter is a valid response to absurdity. So is love. So is creation."
    ],
    practicalSteps: [
      "Read Camus, Sartre, or other existentialists - they offer ways forward.",
      "Engage in activities that feel meaningful to you, regardless of cosmic meaning.",
      "Connect with others who think about these things.",
      "Create, love, experience - these are responses to absurdity, not solutions."
    ],
    whenToSeekHelp: "If the sense of absurdity leads to hopelessness, depression, or inability to function, a therapist familiar with existential concerns can help you integrate these insights."
  },
  
  void_fear: {
    validation: "The fear of nothingness, of the void, is primal and terrifying. Staring into that darkness takes courage, even when it's involuntary. Your fear is valid.",
    normalizing: [
      "Fear of the void is ancient - humans have always grappled with nothingness.",
      "This fear often intensifies during depression, anxiety, or after loss.",
      "Many people have faced this fear and found their way through."
    ],
    groundingTechniques: [
      "Turn on lights. Surround yourself with warmth and presence.",
      "Focus on something solid, real, present. You are here, not in the void.",
      "Connect with another person or living being. You are not alone.",
      "Engage your senses fully. The void is abstract; sensation is concrete."
    ],
    philosophicalReframes: [
      "The void is a concept, not a place. You cannot actually fall into it.",
      "Nothingness, by definition, is nothing to fear - it has no power.",
      "You are here, conscious, experiencing. That's the opposite of void.",
      "Perhaps what we fear isn't nothingness but the unknown. The unknown isn't necessarily bad."
    ],
    practicalSteps: [
      "Avoid ruminating on the void, especially at night.",
      "Create a comforting environment - light, warmth, familiar objects.",
      "Stay connected to others. Isolation intensifies void fear.",
      "Engage in life-affirming activities - creating, connecting, experiencing."
    ],
    whenToSeekHelp: "If void fear is persistent, causing panic, or interfering with sleep and daily life, professional support can help. This can sometimes be related to depression or anxiety."
  }
};

/**
 * Analyze message for existential crisis indicators
 */
export function analyzeExistential(message: string): ExistentialAnalysis {
  const text = message.toLowerCase();
  
  let maxScore = 0;
  let detectedType: ExistentialCrisisType | null = null;
  
  // Check each type
  for (const [type, indicators] of Object.entries(existentialIndicators)) {
    let score = 0;
    for (const indicator of indicators) {
      if (text.includes(indicator)) score++;
    }
    if (score > maxScore) {
      maxScore = score;
      detectedType = type as ExistentialCrisisType;
    }
  }
  
  const isExistentialCrisis = maxScore >= 2;
  const intensity = Math.min(1, maxScore / 5);
  
  // Get intervention
  const intervention = detectedType ? interventions[detectedType] : interventions.meaninglessness;
  
  return {
    isExistentialCrisis,
    crisisType: isExistentialCrisis ? detectedType : null,
    intensity,
    entropyContribution: intensity * 0.2,
    intervention
  };
}

/**
 * Format existential intervention for LLM context
 */
export function formatExistentialForPrompt(analysis: ExistentialAnalysis): string {
  if (!analysis.isExistentialCrisis) return "";
  
  const intervention = analysis.intervention;
  
  let output = "\n\n[EXISTENTIAL CRISIS DETECTED]\n";
  output += `Type: ${analysis.crisisType}\n`;
  output += `Intensity: ${(analysis.intensity * 100).toFixed(0)}%\n\n`;
  
  output += `VALIDATION: "${intervention.validation}"\n\n`;
  
  output += "NORMALIZING STATEMENTS:\n";
  for (const statement of intervention.normalizing) {
    output += `- ${statement}\n`;
  }
  
  output += "\nGROUNDING TECHNIQUES:\n";
  for (const technique of intervention.groundingTechniques.slice(0, 3)) {
    output += `- ${technique}\n`;
  }
  
  output += "\nPHILOSOPHICAL REFRAMES (use gently, not dismissively):\n";
  for (const reframe of intervention.philosophicalReframes.slice(0, 2)) {
    output += `- ${reframe}\n`;
  }
  
  output += "\nKEY: Validate the distress first. Don't dismiss or argue. ";
  output += "Existential questions are real and important. Help ground while honoring the depth of their thinking.\n";
  
  return output;
}
