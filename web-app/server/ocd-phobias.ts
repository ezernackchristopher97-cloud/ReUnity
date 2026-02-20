/**
 * Expanded OCD Subtypes and Phobias Module
 * 
 * Comprehensive coverage of OCD presentations and specific phobias
 * with tailored interventions for each subtype.
 */

export interface OCDAnalysis {
  isOCD: boolean;
  subtypes: OCDSubtype[];
  primarySubtype: OCDSubtype | null;
  intensity: number;
  intervention: OCDIntervention;
}

export interface PhobiaAnalysis {
  isPhobia: boolean;
  phobiaType: PhobiaType | null;
  intensity: number;
  intervention: PhobiaIntervention;
}

export type OCDSubtype = 
  | "contamination"
  | "harm_ocd"
  | "sexual_intrusive"
  | "religious_scrupulosity"
  | "relationship_ocd"
  | "perfectionism"
  | "symmetry_ordering"
  | "hoarding"
  | "pure_o"
  | "health_anxiety"
  | "existential_ocd"
  | "pedophilia_ocd"  // POCD - fear of being a pedophile
  | "homosexual_ocd"  // HOCD - fear about sexual orientation
  | "real_event_ocd"
  | "sensorimotor"
  | "magical_thinking";

export type PhobiaType =
  | "agoraphobia"
  | "social_phobia"
  | "specific_animal"
  | "specific_natural"
  | "specific_blood_injury"
  | "specific_situational"
  | "emetophobia"
  | "thanatophobia"
  | "trypophobia"
  | "arachnophobia"
  | "claustrophobia"
  | "acrophobia"
  | "nyctophobia"
  | "glossophobia"
  | "tokophobia";

export interface OCDIntervention {
  validation: string;
  psychoeducation: string[];
  erpGuidance: string[];
  cognitiveTechniques: string[];
  doNotDo: string[];
  resources: string[];
}

export interface PhobiaIntervention {
  validation: string;
  psychoeducation: string[];
  exposureGuidance: string[];
  copingStrategies: string[];
  resources: string[];
}

// OCD subtype detection patterns
const ocdPatterns: Record<OCDSubtype, string[]> = {
  contamination: [
    "germs", "dirty", "contaminated", "wash hands", "cleaning", "bacteria",
    "virus", "infection", "touched", "can't stop washing", "feel dirty",
    "need to clean", "sanitize", "bleach", "shower", "won't touch"
  ],
  
  harm_ocd: [
    "hurt someone", "what if i hurt", "violent thoughts", "intrusive thoughts about harming",
    "afraid i'll hurt", "images of hurting", "knife", "push someone", "strangle",
    "what if i snap", "lose control", "dangerous", "harm my family", "hurt my child"
  ],
  
  sexual_intrusive: [
    "sexual thoughts", "intrusive sexual", "inappropriate thoughts", "can't stop thinking about",
    "disgusting thoughts", "unwanted sexual", "images i don't want", "thoughts about",
    "what if i'm attracted to", "disturbing thoughts"
  ],
  
  religious_scrupulosity: [
    "sinned", "blasphemy", "unforgivable sin", "going to hell", "god hates me",
    "religious thoughts", "pray enough", "confess", "impure thoughts", "sacrilege",
    "offended god", "devil", "evil thoughts", "not faithful enough", "scrupulosity"
  ],
  
  relationship_ocd: [
    "do i really love", "right person", "what if i don't love", "relationship anxiety",
    "not attracted enough", "should i break up", "constantly checking", "comparing",
    "rocd", "doubting relationship", "what if there's someone better"
  ],
  
  perfectionism: [
    "has to be perfect", "not good enough", "redo", "start over", "can't finish",
    "perfectionist", "exactly right", "mistake", "error", "flawless", "precise"
  ],
  
  symmetry_ordering: [
    "symmetry", "order", "arrange", "even", "balanced", "lined up", "straight",
    "organized", "just right", "feels wrong", "not right", "have to fix"
  ],
  
  hoarding: [
    "can't throw away", "keep everything", "might need it", "attached to things",
    "collecting", "piles", "clutter", "saving", "can't let go", "hoarding"
  ],
  
  pure_o: [
    "pure o", "only thoughts", "no compulsions", "just in my head", "mental rituals",
    "ruminating", "can't stop thinking", "analyzing", "mental checking"
  ],
  
  health_anxiety: [
    "health anxiety", "what if i have", "cancer", "disease", "symptoms", "dying",
    "checking body", "googling symptoms", "doctor", "test results", "hypochondria",
    "something wrong with me", "illness"
  ],
  
  existential_ocd: [
    "existential ocd", "meaning of life", "why are we here", "what is reality",
    "consciousness", "can't stop thinking about existence", "philosophical questions",
    "stuck on", "ruminating about life"
  ],
  
  pedophilia_ocd: [
    "pocd", "afraid i'm a pedophile", "thoughts about children", "what if i'm attracted",
    "intrusive thoughts about kids", "afraid around children", "avoiding children",
    "disgusted by thoughts", "monster", "what if i hurt a child"
  ],
  
  homosexual_ocd: [
    "hocd", "what if i'm gay", "what if i'm straight", "sexual orientation",
    "checking attraction", "groinal response", "am i attracted", "sexuality ocd",
    "questioning sexuality", "what if i'm in denial"
  ],
  
  real_event_ocd: [
    "real event", "something i did", "past mistake", "did i do something wrong",
    "memory", "can't remember", "what if i", "years ago", "confess", "guilty"
  ],
  
  sensorimotor: [
    "can't stop noticing", "breathing", "blinking", "swallowing", "heartbeat",
    "aware of", "sensorimotor", "body sensation", "can't ignore", "hyperaware"
  ],
  
  magical_thinking: [
    "if i don't", "something bad will happen", "superstition", "lucky number",
    "ritual", "prevent", "cause bad things", "responsible for", "magical thinking",
    "counting", "certain way"
  ]
};

// Phobia detection patterns
const phobiaPatterns: Record<PhobiaType, string[]> = {
  agoraphobia: [
    "can't leave house", "afraid to go out", "panic outside", "trapped",
    "open spaces", "crowded places", "public places", "escape", "agoraphobia"
  ],
  
  social_phobia: [
    "social anxiety", "afraid of people", "judged", "embarrassed", "humiliated",
    "public speaking", "meeting people", "social situations", "what people think"
  ],
  
  specific_animal: [
    "afraid of dogs", "afraid of cats", "afraid of birds", "animal phobia",
    "scared of animals", "can't be near"
  ],
  
  specific_natural: [
    "afraid of storms", "afraid of water", "afraid of heights", "lightning",
    "thunder", "ocean", "drowning", "natural disaster"
  ],
  
  specific_blood_injury: [
    "afraid of blood", "needles", "injections", "medical procedures", "fainting",
    "blood phobia", "can't see blood", "shots", "vaccines"
  ],
  
  specific_situational: [
    "afraid of flying", "elevators", "bridges", "tunnels", "driving",
    "enclosed spaces", "airplanes", "heights"
  ],
  
  emetophobia: [
    "afraid of vomiting", "throwing up", "nausea", "sick", "emetophobia",
    "someone vomiting", "stomach bug", "food poisoning", "can't eat"
  ],
  
  thanatophobia: [
    "afraid of death", "fear of dying", "death phobia", "mortality",
    "going to die", "death anxiety", "thanatophobia"
  ],
  
  trypophobia: [
    "holes", "clusters", "trypophobia", "patterns", "dots", "bumps"
  ],
  
  arachnophobia: [
    "spiders", "afraid of spiders", "arachnophobia", "webs", "tarantula"
  ],
  
  claustrophobia: [
    "claustrophobia", "small spaces", "enclosed", "trapped", "can't breathe",
    "elevator", "mri", "closet"
  ],
  
  acrophobia: [
    "heights", "afraid of heights", "acrophobia", "tall buildings", "balcony",
    "looking down", "vertigo"
  ],
  
  nyctophobia: [
    "afraid of dark", "darkness", "nyctophobia", "night", "can't sleep without light"
  ],
  
  glossophobia: [
    "public speaking", "presentations", "speaking in front", "stage fright",
    "glossophobia", "audience"
  ],
  
  tokophobia: [
    "afraid of pregnancy", "childbirth", "giving birth", "tokophobia",
    "labor", "delivery"
  ]
};

// OCD interventions
const ocdInterventions: Record<OCDSubtype, OCDIntervention> = {
  contamination: {
    validation: "The distress you feel about contamination is real and overwhelming. OCD makes the fear feel urgent and necessary to act on. You're not crazy - your brain is sending false alarms.",
    psychoeducation: [
      "Contamination OCD involves intrusive fears about germs, dirt, or contamination that feel unbearable.",
      "The compulsion to wash or clean provides temporary relief but strengthens the OCD cycle.",
      "Your brain's threat detection system is overactive - it's not your fault."
    ],
    erpGuidance: [
      "ERP involves gradually facing contamination fears without engaging in washing/cleaning compulsions.",
      "Start with lower-anxiety exposures and work up gradually.",
      "The goal is to learn that you can tolerate the anxiety and it will decrease on its own.",
      "Delay washing by increasing amounts of time. Notice the anxiety peak and then decrease."
    ],
    cognitiveTechniques: [
      "Notice the thought: 'I'm having the thought that I'm contaminated.'",
      "Ask: 'Is this thought helpful? Is it OCD talking?'",
      "Remember: Feeling contaminated is not the same as being contaminated."
    ],
    doNotDo: [
      "Don't provide reassurance that they're not contaminated",
      "Don't help them avoid triggers",
      "Don't engage in detailed discussions about contamination likelihood"
    ],
    resources: ["IOCDF.org", "NOCD app", "ERP therapist directory"]
  },
  
  harm_ocd: {
    validation: "Having intrusive thoughts about harming others is terrifying. The fact that these thoughts disturb you so much is actually evidence that you DON'T want to act on them. People who actually want to harm others aren't distressed by these thoughts.",
    psychoeducation: [
      "Harm OCD involves unwanted, intrusive thoughts about hurting others. These are ego-dystonic - they go against your values.",
      "Having a thought is not the same as wanting to act on it or being likely to act on it.",
      "Research shows people with harm OCD are LESS likely to be violent than the general population.",
      "The more you try to suppress these thoughts, the more they come back. This is called the 'white bear effect.'"
    ],
    erpGuidance: [
      "ERP for harm OCD involves accepting the presence of thoughts without engaging in mental rituals.",
      "This might include writing out feared scenarios, being around 'triggers' (like knives) without avoiding.",
      "The goal is not to prove you won't act on thoughts, but to accept uncertainty.",
      "Work with an OCD specialist - this is highly treatable."
    ],
    cognitiveTechniques: [
      "Notice: 'I'm having an intrusive thought about harm. This is OCD.'",
      "Don't argue with the thought or try to prove you won't act on it.",
      "Accept the thought's presence without giving it meaning."
    ],
    doNotDo: [
      "Don't reassure them they won't act on thoughts",
      "Don't help them analyze whether thoughts mean something",
      "Don't treat them as dangerous - they're not"
    ],
    resources: ["IOCDF.org", "NOCD", "Book: 'Overcoming Unwanted Intrusive Thoughts'"]
  },
  
  sexual_intrusive: {
    validation: "Intrusive sexual thoughts are one of the most distressing forms of OCD because they feel so personal and shameful. These thoughts do NOT reflect your desires or character. The distress you feel shows they go against your values.",
    psychoeducation: [
      "Sexual intrusive thoughts are extremely common in OCD and do not reflect desires.",
      "The brain generates all kinds of random thoughts. OCD latches onto the ones that disturb you most.",
      "Having a thought is not the same as wanting it or being likely to act on it.",
      "Many people with this form of OCD suffer in silence due to shame. You're not alone."
    ],
    erpGuidance: [
      "ERP involves accepting the presence of thoughts without mental rituals or avoidance.",
      "This is best done with an OCD specialist who understands this subtype.",
      "The goal is to become bored by the thoughts, not to prove they don't mean anything."
    ],
    cognitiveTechniques: [
      "Label the thought: 'This is an intrusive thought. This is OCD.'",
      "Don't engage with the content or try to figure out what it means.",
      "Allow the thought to be there without pushing it away."
    ],
    doNotDo: [
      "Don't reassure about the meaning of thoughts",
      "Don't help analyze whether thoughts reflect desires",
      "Don't express shock or judgment"
    ],
    resources: ["IOCDF.org", "NOCD", "Book: 'Overcoming Unwanted Intrusive Thoughts'"]
  },
  
  religious_scrupulosity: {
    validation: "Scrupulosity is incredibly painful because it attacks what matters most to you - your faith. The intense fear of sinning or offending God is OCD, not a reflection of your actual spiritual state. Many deeply faithful people struggle with this.",
    psychoeducation: [
      "Scrupulosity is OCD focused on religious or moral themes. It's recognized by religious leaders and mental health professionals.",
      "The excessive guilt and fear are symptoms of OCD, not accurate spiritual perception.",
      "Many saints and religious figures throughout history likely had scrupulosity.",
      "OCD creates false guilt that feels real but isn't based on actual wrongdoing."
    ],
    erpGuidance: [
      "ERP for scrupulosity involves accepting uncertainty about spiritual status.",
      "This might include reducing confession frequency, not seeking reassurance from religious leaders.",
      "Work with a therapist who understands both OCD and respects your faith.",
      "The goal is to live your faith without OCD controlling it."
    ],
    cognitiveTechniques: [
      "Notice: 'This intense guilt might be OCD, not accurate spiritual perception.'",
      "Ask: 'Would a loving God want me to suffer this much over this?'",
      "Remember: OCD lies. It makes small things feel catastrophic."
    ],
    doNotDo: [
      "Don't provide religious reassurance",
      "Don't engage in theological debates about whether they've sinned",
      "Don't dismiss their faith - respect it while addressing OCD"
    ],
    resources: ["IOCDF.org", "Book: 'The Doubting Disease'", "Scrupulous Anonymous"]
  },
  
  relationship_ocd: {
    validation: "The constant doubt about your relationship is exhausting. ROCD makes you question feelings that would otherwise feel natural. This doesn't mean your relationship is wrong - it means OCD has found something important to attack.",
    psychoeducation: [
      "ROCD involves obsessive doubt about relationships - 'Do I really love them?' 'Are they right for me?'",
      "Everyone has some relationship doubts. ROCD makes them constant and unbearable.",
      "Checking feelings, comparing to other couples, and seeking reassurance are compulsions that maintain the cycle.",
      "ROCD attacks what matters to you. It's not a sign the relationship is wrong."
    ],
    erpGuidance: [
      "ERP involves accepting uncertainty about the relationship without checking or seeking reassurance.",
      "This might include not comparing your relationship to others, not analyzing feelings.",
      "The goal is to be in the relationship without OCD running the show.",
      "Work with an OCD specialist who understands ROCD."
    ],
    cognitiveTechniques: [
      "Notice: 'I'm having ROCD thoughts. This is OCD, not relationship insight.'",
      "Don't try to figure out if you 'really' love them - that's a compulsion.",
      "Accept that you can't have 100% certainty about anything, including relationships."
    ],
    doNotDo: [
      "Don't reassure about the relationship",
      "Don't help analyze whether feelings are 'real'",
      "Don't suggest breaking up to 'test' feelings"
    ],
    resources: ["IOCDF.org", "ROCD.net", "NOCD"]
  },
  
  perfectionism: {
    validation: "The need for everything to be perfect is exhausting. OCD perfectionism isn't about high standards - it's about unbearable anxiety when things aren't 'just right.' You're not being difficult; you're suffering.",
    psychoeducation: [
      "OCD perfectionism involves compulsive need for things to be exact, often leading to paralysis or endless redoing.",
      "This is different from healthy striving - it causes distress and impairment.",
      "The 'just right' feeling you're seeking is a moving target that OCD keeps shifting."
    ],
    erpGuidance: [
      "ERP involves intentionally doing things imperfectly and sitting with the discomfort.",
      "Start small - send an email with a typo, leave something slightly crooked.",
      "The goal is to learn you can tolerate imperfection."
    ],
    cognitiveTechniques: [
      "Notice: 'My need for this to be perfect is OCD.'",
      "Ask: 'Is perfect actually possible? Is it necessary?'",
      "Practice: 'Good enough is good enough.'"
    ],
    doNotDo: [
      "Don't help them achieve perfection",
      "Don't reassure that something is perfect",
      "Don't enable redoing or checking"
    ],
    resources: ["IOCDF.org", "NOCD"]
  },
  
  symmetry_ordering: {
    validation: "The need for things to be symmetrical or 'just right' creates constant discomfort. This isn't being picky - it's OCD creating unbearable feelings when things are out of order.",
    psychoeducation: [
      "Symmetry/ordering OCD involves compulsive need to arrange things until they feel 'right.'",
      "The 'not right' feeling is OCD, not accurate perception.",
      "Arranging provides temporary relief but strengthens the OCD."
    ],
    erpGuidance: [
      "ERP involves intentionally leaving things asymmetrical or disordered.",
      "Start with lower-stakes items and work up.",
      "Sit with the 'not right' feeling without fixing it."
    ],
    cognitiveTechniques: [
      "Notice: 'The urge to fix this is OCD.'",
      "Label the feeling: 'This is the OCD 'not right' feeling.'",
      "Remind yourself: 'I can tolerate this discomfort.'"
    ],
    doNotDo: [
      "Don't help arrange things",
      "Don't reassure that things look right",
      "Don't accommodate ordering rituals"
    ],
    resources: ["IOCDF.org", "NOCD"]
  },
  
  hoarding: {
    validation: "The difficulty letting go of things isn't about being messy or lazy. Hoarding involves real distress at the thought of discarding items. This is a recognized condition that responds to treatment.",
    psychoeducation: [
      "Hoarding disorder involves difficulty discarding items due to perceived need or emotional attachment.",
      "The distress at discarding is real, not a choice.",
      "This often develops after loss or trauma."
    ],
    erpGuidance: [
      "Treatment involves gradually discarding items while managing distress.",
      "Start with less emotionally significant items.",
      "Work with a therapist who specializes in hoarding."
    ],
    cognitiveTechniques: [
      "Ask: 'Have I used this in the past year? Will I realistically use it?'",
      "Notice: 'The anxiety about discarding is temporary.'",
      "Remember: 'I am not my possessions.'"
    ],
    doNotDo: [
      "Don't force discarding",
      "Don't clean out their space without consent",
      "Don't shame or criticize"
    ],
    resources: ["IOCDF.org", "Buried in Treasures workbook"]
  },
  
  pure_o: {
    validation: "Pure O is incredibly isolating because the battle is invisible. Just because there are no visible compulsions doesn't mean you're not suffering. Mental rituals are just as exhausting as physical ones.",
    psychoeducation: [
      "'Pure O' involves obsessions with mental (rather than visible) compulsions - analyzing, checking, reassurance-seeking in your head.",
      "It's not actually 'pure' obsessions - the compulsions are just internal.",
      "This is just as treatable as other forms of OCD."
    ],
    erpGuidance: [
      "ERP for Pure O involves accepting intrusive thoughts without mental rituals.",
      "This means not analyzing, not mentally checking, not seeking internal reassurance.",
      "The goal is to let thoughts be there without engaging."
    ],
    cognitiveTechniques: [
      "Notice when you're doing mental rituals (analyzing, checking, reassuring yourself).",
      "Practice letting thoughts be there without engaging.",
      "Label: 'This is an intrusive thought. I don't need to figure it out.'"
    ],
    doNotDo: [
      "Don't help analyze thoughts",
      "Don't provide reassurance",
      "Don't engage in philosophical discussions about thought content"
    ],
    resources: ["IOCDF.org", "NOCD", "Book: 'Overcoming Unwanted Intrusive Thoughts'"]
  },
  
  health_anxiety: {
    validation: "The fear that something is wrong with your body is terrifying and consuming. Health anxiety makes every sensation feel like evidence of serious illness. You're not a hypochondriac - you're suffering from a real condition.",
    psychoeducation: [
      "Health anxiety (illness anxiety disorder) involves preoccupation with having or getting a serious illness.",
      "Checking, googling, and seeking reassurance provide temporary relief but maintain the cycle.",
      "The anxiety is the problem, not an underlying illness."
    ],
    erpGuidance: [
      "ERP involves reducing checking behaviors and tolerating uncertainty about health.",
      "This might include limiting doctor visits, not googling symptoms, not body checking.",
      "The goal is to accept that you can't have 100% certainty about health."
    ],
    cognitiveTechniques: [
      "Notice: 'This worry might be health anxiety, not accurate perception.'",
      "Ask: 'Is checking/googling helping or making it worse?'",
      "Accept: 'I cannot have 100% certainty about my health, and that's okay.'"
    ],
    doNotDo: [
      "Don't reassure about health",
      "Don't help research symptoms",
      "Don't encourage unnecessary medical tests"
    ],
    resources: ["IOCDF.org", "Book: 'It's Not All in Your Head'"]
  },
  
  existential_ocd: {
    validation: "Being stuck on existential questions - the meaning of life, the nature of reality - is exhausting. These aren't just philosophical musings; OCD has latched onto them and won't let go. The rumination is the problem, not the questions themselves.",
    psychoeducation: [
      "Existential OCD involves obsessive rumination on philosophical questions that feel urgent to solve.",
      "Unlike normal philosophical interest, this causes significant distress and impairment.",
      "The compulsion is the endless analyzing and trying to 'figure it out.'"
    ],
    erpGuidance: [
      "ERP involves accepting uncertainty about existential questions without ruminating.",
      "Practice noticing the thought and letting it go without engaging.",
      "The goal is not to answer the questions but to stop needing to."
    ],
    cognitiveTechniques: [
      "Notice: 'I'm ruminating on existential questions. This is OCD.'",
      "Accept: 'I don't need to solve the meaning of life right now.'",
      "Redirect: Engage in present-moment activities."
    ],
    doNotDo: [
      "Don't engage in philosophical discussions",
      "Don't try to answer the existential questions",
      "Don't provide reassurance about reality or meaning"
    ],
    resources: ["IOCDF.org", "NOCD"]
  },
  
  pedophilia_ocd: {
    validation: "POCD is one of the most distressing and shameful forms of OCD. The fact that these thoughts horrify you is proof that you are NOT a pedophile. Real pedophiles are not distressed by attraction to children. Your suffering shows these thoughts go against everything you are.",
    psychoeducation: [
      "POCD involves intrusive, unwanted thoughts about children that cause extreme distress.",
      "These thoughts are ego-dystonic - they go against your values and desires.",
      "People with POCD are NOT at risk of harming children. The distress proves the thoughts are unwanted.",
      "This is a recognized form of OCD that responds to treatment."
    ],
    erpGuidance: [
      "ERP involves accepting the presence of thoughts without avoidance or mental rituals.",
      "This is best done with an OCD specialist who understands POCD.",
      "The goal is to become bored by the thoughts, not to prove you're not a pedophile."
    ],
    cognitiveTechniques: [
      "Label: 'This is a POCD intrusive thought. It does not reflect my desires.'",
      "Don't analyze or try to prove you're not attracted to children - that's a compulsion.",
      "Accept the thought's presence without giving it meaning."
    ],
    doNotDo: [
      "Don't reassure them they're not a pedophile",
      "Don't help them analyze thoughts",
      "Don't treat them as dangerous - they're not"
    ],
    resources: ["IOCDF.org", "NOCD", "Book: 'Overcoming Unwanted Intrusive Thoughts'"]
  },
  
  homosexual_ocd: {
    validation: "HOCD creates relentless doubt about your sexual orientation. Whether you're straight, gay, or anywhere in between, OCD can attack your sense of identity. The constant checking and analyzing is exhausting. This is OCD, not genuine self-discovery.",
    psychoeducation: [
      "HOCD involves obsessive doubt about sexual orientation, regardless of actual orientation.",
      "This is different from genuine questioning - HOCD causes distress and compulsive checking.",
      "The compulsions include checking attraction, analyzing past experiences, seeking reassurance.",
      "HOCD can affect people of any sexual orientation."
    ],
    erpGuidance: [
      "ERP involves accepting uncertainty about orientation without checking or analyzing.",
      "This might include not monitoring attraction, not seeking reassurance.",
      "The goal is to accept that you don't need 100% certainty about orientation."
    ],
    cognitiveTechniques: [
      "Notice: 'I'm checking/analyzing my attraction. This is OCD.'",
      "Accept: 'I don't need to figure out my orientation right now.'",
      "Label: 'This doubt is HOCD, not genuine self-discovery.'"
    ],
    doNotDo: [
      "Don't reassure about orientation",
      "Don't help analyze attraction",
      "Don't suggest 'testing' orientation"
    ],
    resources: ["IOCDF.org", "NOCD"]
  },
  
  real_event_ocd: {
    validation: "Real Event OCD is particularly cruel because it attaches to things that actually happened, making it feel like 'real' guilt rather than OCD. But the obsessive rumination, the need to confess, the inability to move on - that's OCD, not proportionate guilt.",
    psychoeducation: [
      "Real Event OCD involves obsessive guilt about past actions, often minor or already addressed.",
      "The guilt is disproportionate to the event and doesn't resolve with confession or analysis.",
      "OCD latches onto past events and won't let go, even when others would have moved on."
    ],
    erpGuidance: [
      "ERP involves accepting uncertainty about past events without confessing or analyzing.",
      "This might include not seeking reassurance, not confessing repeatedly.",
      "The goal is to accept that you can't change the past and don't need to keep analyzing it."
    ],
    cognitiveTechniques: [
      "Notice: 'I'm ruminating on a past event. This is OCD.'",
      "Ask: 'Would most people still be this distressed about this?'",
      "Accept: 'I cannot change the past. I can only move forward.'"
    ],
    doNotDo: [
      "Don't reassure about the past event",
      "Don't help analyze whether it was 'that bad'",
      "Don't encourage confession"
    ],
    resources: ["IOCDF.org", "NOCD"]
  },
  
  sensorimotor: {
    validation: "Being hyperaware of your breathing, blinking, or swallowing is maddening. Once you notice it, you can't stop noticing. This is sensorimotor OCD, and it's incredibly frustrating because the 'trigger' is always there.",
    psychoeducation: [
      "Sensorimotor OCD involves hyperawareness of automatic bodily processes.",
      "The more you try not to notice, the more you notice - this is the OCD trap.",
      "These processes will return to automatic when you stop monitoring them."
    ],
    erpGuidance: [
      "ERP involves accepting the awareness without trying to control or ignore it.",
      "Paradoxically, allowing yourself to notice can reduce the hyperawareness.",
      "The goal is to let the sensation be there without it being a problem."
    ],
    cognitiveTechniques: [
      "Notice: 'I'm hyperaware of [breathing/blinking/etc]. This is sensorimotor OCD.'",
      "Accept: 'I can notice this and still function.'",
      "Don't try to control or ignore - just let it be there."
    ],
    doNotDo: [
      "Don't reassure that they'll stop noticing",
      "Don't suggest distraction techniques",
      "Don't help them control the sensation"
    ],
    resources: ["IOCDF.org", "NOCD"]
  },
  
  magical_thinking: {
    validation: "The belief that your thoughts or actions can prevent bad things from happening is exhausting. Magical thinking OCD creates an unbearable sense of responsibility. You're not superstitious - you're suffering from OCD.",
    psychoeducation: [
      "Magical thinking OCD involves beliefs that thoughts or rituals can prevent harm.",
      "This creates an impossible burden of responsibility.",
      "The rituals provide temporary relief but strengthen the OCD."
    ],
    erpGuidance: [
      "ERP involves not performing rituals and accepting the anxiety.",
      "This might include intentionally having 'bad' thoughts without neutralizing.",
      "The goal is to learn that thoughts don't cause events."
    ],
    cognitiveTechniques: [
      "Notice: 'I'm doing a ritual to prevent something. This is OCD.'",
      "Ask: 'Is there any real evidence my thoughts affect reality?'",
      "Accept: 'I cannot control the world with my thoughts.'"
    ],
    doNotDo: [
      "Don't reassure that bad things won't happen",
      "Don't accommodate rituals",
      "Don't help them 'prevent' feared outcomes"
    ],
    resources: ["IOCDF.org", "NOCD"]
  }
};

// Phobia interventions
const phobiaInterventions: Record<PhobiaType, PhobiaIntervention> = {
  agoraphobia: {
    validation: "The fear of leaving your safe space is real and overwhelming. Agoraphobia isn't about being lazy or dramatic - it's about genuine terror. Your world has shrunk, and that's painful.",
    psychoeducation: [
      "Agoraphobia involves fear of situations where escape might be difficult or help unavailable.",
      "It often develops after panic attacks but can occur independently.",
      "Avoidance maintains the fear. Gradual exposure is the most effective treatment."
    ],
    exposureGuidance: [
      "Start with small steps outside your comfort zone.",
      "Build a hierarchy from least to most anxiety-provoking situations.",
      "Stay in situations until anxiety decreases naturally.",
      "Work with a therapist for structured exposure."
    ],
    copingStrategies: [
      "Practice grounding techniques before and during exposure.",
      "Use breathing exercises to manage panic symptoms.",
      "Have a support person initially, then gradually increase independence.",
      "Celebrate small victories."
    ],
    resources: ["ADAA.org", "Anxiety Canada", "Therapist specializing in anxiety"]
  },
  
  social_phobia: {
    validation: "The fear of judgment and embarrassment in social situations is paralyzing. Social anxiety isn't shyness - it's intense fear that affects your life. You're not being dramatic; this is real suffering.",
    psychoeducation: [
      "Social anxiety involves fear of negative evaluation by others.",
      "It often includes overestimating how much others notice and judge us.",
      "Avoidance and safety behaviors maintain the fear."
    ],
    exposureGuidance: [
      "Gradually face feared social situations.",
      "Drop safety behaviors (like avoiding eye contact, rehearsing everything).",
      "Test predictions - do the feared outcomes actually happen?",
      "Focus outward on others rather than inward on yourself."
    ],
    copingStrategies: [
      "Challenge thoughts about being judged.",
      "Practice self-compassion - everyone feels awkward sometimes.",
      "Focus on the other person, not on yourself.",
      "Accept that some discomfort is normal."
    ],
    resources: ["ADAA.org", "Social Anxiety Institute", "CBT therapist"]
  },
  
  specific_animal: {
    validation: "Fear of animals, even common ones, can be intense and limiting. This isn't silly - phobias are real and distressing. You're not overreacting.",
    psychoeducation: [
      "Specific phobias involve intense fear of particular objects or situations.",
      "The fear is out of proportion to actual danger.",
      "Exposure therapy is highly effective for specific phobias."
    ],
    exposureGuidance: [
      "Build a hierarchy from pictures to videos to proximity to the animal.",
      "Start with the least anxiety-provoking step.",
      "Stay with each step until anxiety decreases.",
      "Progress gradually - don't rush."
    ],
    copingStrategies: [
      "Use relaxation techniques during exposure.",
      "Challenge catastrophic thoughts.",
      "Reward yourself for facing fears.",
      "Remember: anxiety peaks and then decreases."
    ],
    resources: ["ADAA.org", "Exposure therapist"]
  },
  
  specific_natural: {
    validation: "Fear of natural phenomena like storms, water, or heights is primal and powerful. These fears made sense for our ancestors. Your fear is real, even if the danger is often low.",
    psychoeducation: [
      "Natural environment phobias involve fear of natural phenomena.",
      "These fears may have evolutionary roots.",
      "Gradual exposure is effective treatment."
    ],
    exposureGuidance: [
      "Create a hierarchy of feared situations.",
      "Use virtual reality or videos as initial steps.",
      "Gradually increase real-world exposure.",
      "Stay in situations until anxiety decreases."
    ],
    copingStrategies: [
      "Learn facts about the feared phenomenon.",
      "Practice relaxation techniques.",
      "Use grounding during exposure.",
      "Challenge probability overestimation."
    ],
    resources: ["ADAA.org", "Exposure therapist"]
  },
  
  specific_blood_injury: {
    validation: "Fear of blood, needles, or medical procedures is incredibly common and can be debilitating. The fainting response is real and physical. You're not being weak.",
    psychoeducation: [
      "Blood-injection-injury phobia has a unique physiological response - blood pressure drops, causing fainting.",
      "This is different from other phobias and requires modified treatment.",
      "Applied tension technique can prevent fainting."
    ],
    exposureGuidance: [
      "Learn applied tension technique first (tense muscles to raise blood pressure).",
      "Use applied tension during exposure to blood/needles.",
      "Gradually expose to images, then videos, then real situations.",
      "Work with a therapist who understands this specific phobia."
    ],
    copingStrategies: [
      "Applied tension: Tense arm, leg, and torso muscles for 10-15 seconds, release, repeat.",
      "Lie down during procedures if needed.",
      "Look away initially, then gradually increase exposure.",
      "Inform medical staff about your phobia."
    ],
    resources: ["ADAA.org", "Applied tension resources", "Phobia specialist"]
  },
  
  specific_situational: {
    validation: "Fear of specific situations like flying, elevators, or driving can severely limit your life. These fears feel overwhelming even when you know they're 'irrational.' Your suffering is real.",
    psychoeducation: [
      "Situational phobias involve fear of specific situations.",
      "Avoidance maintains and strengthens the fear.",
      "Exposure therapy is highly effective."
    ],
    exposureGuidance: [
      "Build a hierarchy of feared situations.",
      "Start with less anxiety-provoking versions (e.g., sitting in a parked plane).",
      "Gradually increase exposure intensity.",
      "Stay in situations until anxiety decreases."
    ],
    copingStrategies: [
      "Use breathing techniques to manage anxiety.",
      "Challenge catastrophic thoughts.",
      "Focus on the present moment, not 'what ifs.'",
      "Celebrate progress."
    ],
    resources: ["ADAA.org", "Fear of Flying programs", "Exposure therapist"]
  },
  
  emetophobia: {
    validation: "Fear of vomiting is one of the most limiting phobias. It affects what you eat, where you go, and how you live. This isn't being picky - it's genuine terror. You're not alone in this.",
    psychoeducation: [
      "Emetophobia is fear of vomiting - yourself or others.",
      "It often leads to food restriction, avoidance of places, and constant anxiety.",
      "This is one of the most common phobias and is very treatable."
    ],
    exposureGuidance: [
      "Exposure involves gradually facing vomit-related stimuli.",
      "This might include words, sounds, images, videos, and eventually real situations.",
      "Work with a therapist who specializes in emetophobia.",
      "The goal is to reduce the fear response, not to make you vomit."
    ],
    copingStrategies: [
      "Challenge beliefs about vomiting (it's unpleasant but not dangerous).",
      "Gradually expand food choices.",
      "Reduce safety behaviors (like checking food, avoiding certain places).",
      "Practice tolerating nausea without panic."
    ],
    resources: ["ADAA.org", "Emetophobia.org", "Specialist therapist"]
  },
  
  thanatophobia: {
    validation: "Fear of death is one of the most fundamental human experiences. When it becomes overwhelming, it can consume your thoughts and limit your life. This fear is understandable and treatable.",
    psychoeducation: [
      "Thanatophobia is excessive fear of death or dying.",
      "Some death awareness is normal; phobia involves constant, distressing preoccupation.",
      "This often responds to therapy, particularly existential and CBT approaches."
    ],
    exposureGuidance: [
      "Gradual exposure to death-related topics and situations.",
      "This might include discussing death, visiting cemeteries, writing about mortality.",
      "The goal is to reduce avoidance and accept mortality.",
      "Work with a therapist who understands existential concerns."
    ],
    copingStrategies: [
      "Focus on living fully rather than avoiding death.",
      "Practice mindfulness and present-moment awareness.",
      "Explore what makes life meaningful to you.",
      "Connect with others about these fears."
    ],
    resources: ["ADAA.org", "Existential therapist", "Death Cafe movement"]
  },
  
  trypophobia: {
    validation: "The intense discomfort or disgust at clusters of holes or bumps is real. Trypophobia isn't officially recognized as a phobia, but the distress it causes is genuine.",
    psychoeducation: [
      "Trypophobia involves aversion to clusters of small holes or bumps.",
      "It may be related to disgust response rather than fear.",
      "The reaction is involuntary and not a choice."
    ],
    exposureGuidance: [
      "Gradual exposure to triggering images can reduce the response.",
      "Start with less triggering images and work up.",
      "The goal is habituation - reduced response over time."
    ],
    copingStrategies: [
      "Avoid unnecessary exposure to triggers.",
      "Use grounding techniques when triggered.",
      "Remember the response will pass.",
      "Don't shame yourself for the reaction."
    ],
    resources: ["ADAA.org", "General anxiety resources"]
  },
  
  arachnophobia: {
    validation: "Fear of spiders is one of the most common phobias. The terror is real, even when you know the spider probably can't hurt you. You're not being silly.",
    psychoeducation: [
      "Arachnophobia is intense fear of spiders.",
      "It may have evolutionary roots but is often out of proportion to actual danger.",
      "Exposure therapy is highly effective."
    ],
    exposureGuidance: [
      "Build a hierarchy: pictures, videos, toy spiders, real spiders at distance, closer proximity.",
      "Stay with each step until anxiety decreases.",
      "Don't rush - gradual progress is key.",
      "Virtual reality exposure can be helpful."
    ],
    copingStrategies: [
      "Learn facts about local spiders (most are harmless).",
      "Use relaxation techniques during exposure.",
      "Challenge catastrophic thoughts.",
      "Celebrate progress."
    ],
    resources: ["ADAA.org", "Exposure therapist"]
  },
  
  claustrophobia: {
    validation: "The terror of enclosed spaces is suffocating. Claustrophobia makes everyday situations - elevators, MRIs, crowded rooms - feel life-threatening. Your fear is real.",
    psychoeducation: [
      "Claustrophobia is fear of enclosed or confined spaces.",
      "It often involves fear of being trapped or unable to escape.",
      "Exposure therapy is very effective."
    ],
    exposureGuidance: [
      "Build a hierarchy of enclosed spaces from least to most anxiety-provoking.",
      "Start with brief exposure to less triggering spaces.",
      "Gradually increase duration and intensity.",
      "Practice staying until anxiety decreases."
    ],
    copingStrategies: [
      "Use breathing techniques to manage panic.",
      "Focus on the fact that you CAN leave if needed.",
      "Challenge thoughts about being trapped.",
      "Practice in safe, controlled settings first."
    ],
    resources: ["ADAA.org", "Exposure therapist"]
  },
  
  acrophobia: {
    validation: "Fear of heights can be paralyzing. The dizziness, the racing heart, the urge to get down immediately - it's all real. Acrophobia limits where you can go and what you can do.",
    psychoeducation: [
      "Acrophobia is fear of heights.",
      "It often involves fear of falling or losing control.",
      "Exposure therapy, including virtual reality, is effective."
    ],
    exposureGuidance: [
      "Build a hierarchy from low heights to higher ones.",
      "Use virtual reality as an initial step.",
      "Gradually expose to real heights.",
      "Stay at each level until anxiety decreases."
    ],
    copingStrategies: [
      "Use grounding techniques (focus on solid surfaces).",
      "Challenge thoughts about falling.",
      "Don't look down initially; gradually increase.",
      "Use safety features (railings) without over-relying on them."
    ],
    resources: ["ADAA.org", "Virtual reality exposure programs", "Exposure therapist"]
  },
  
  nyctophobia: {
    validation: "Fear of the dark isn't just for children. Adult nyctophobia is real and can significantly impact sleep and daily life. Your fear is valid.",
    psychoeducation: [
      "Nyctophobia is fear of darkness.",
      "It often involves fear of what might be in the dark rather than darkness itself.",
      "Gradual exposure is effective treatment."
    ],
    exposureGuidance: [
      "Gradually reduce light levels over time.",
      "Start with dim light, progress to darkness.",
      "Practice being in dark rooms for increasing periods.",
      "Challenge beliefs about what's in the dark."
    ],
    copingStrategies: [
      "Use relaxation techniques.",
      "Challenge catastrophic thoughts about the dark.",
      "Gradually reduce reliance on night lights.",
      "Create a calming bedtime routine."
    ],
    resources: ["ADAA.org", "Sleep specialist", "Anxiety therapist"]
  },
  
  glossophobia: {
    validation: "Fear of public speaking is incredibly common and can be career-limiting. The physical symptoms - shaking, sweating, mind going blank - are real and distressing. You're not alone.",
    psychoeducation: [
      "Glossophobia is fear of public speaking.",
      "It's one of the most common fears.",
      "Exposure and skills training are effective treatments."
    ],
    exposureGuidance: [
      "Start with speaking to small, supportive groups.",
      "Gradually increase audience size and formality.",
      "Practice with video recording to desensitize.",
      "Join groups like Toastmasters for regular practice."
    ],
    copingStrategies: [
      "Prepare thoroughly but don't over-rehearse.",
      "Focus on the message, not on yourself.",
      "Accept some nervousness as normal.",
      "Use breathing techniques before speaking."
    ],
    resources: ["ADAA.org", "Toastmasters", "Public speaking coach"]
  },
  
  tokophobia: {
    validation: "Fear of pregnancy and childbirth is real and can affect major life decisions. Tokophobia isn't being dramatic - it's genuine terror about something that is, objectively, a significant physical event.",
    psychoeducation: [
      "Tokophobia is fear of pregnancy and/or childbirth.",
      "It can be primary (never pregnant) or secondary (after traumatic birth).",
      "This is a recognized condition that can be treated."
    ],
    exposureGuidance: [
      "Gradual exposure to pregnancy/birth-related information.",
      "Work with a therapist who understands reproductive anxiety.",
      "Address any underlying trauma.",
      "Consider working with a supportive OB/GYN."
    ],
    copingStrategies: [
      "Educate yourself about modern birth options and pain management.",
      "Connect with others who've experienced tokophobia.",
      "Explore your specific fears and address them.",
      "Know that you have choices about if and how you give birth."
    ],
    resources: ["ADAA.org", "Tokophobia support groups", "Perinatal mental health specialist"]
  }
};

/**
 * Analyze message for OCD indicators
 */
export function analyzeOCD(message: string): OCDAnalysis {
  const text = message.toLowerCase();
  
  const detectedSubtypes: OCDSubtype[] = [];
  let maxScore = 0;
  let primarySubtype: OCDSubtype | null = null;
  
  for (const [subtype, patterns] of Object.entries(ocdPatterns)) {
    let score = 0;
    for (const pattern of patterns) {
      if (text.includes(pattern)) score++;
    }
    if (score >= 2) {
      detectedSubtypes.push(subtype as OCDSubtype);
      if (score > maxScore) {
        maxScore = score;
        primarySubtype = subtype as OCDSubtype;
      }
    }
  }
  
  const isOCD = detectedSubtypes.length > 0;
  const intensity = Math.min(1, maxScore / 5);
  
  const intervention = primarySubtype ? 
    ocdInterventions[primarySubtype] : 
    ocdInterventions.pure_o;
  
  return {
    isOCD,
    subtypes: detectedSubtypes,
    primarySubtype,
    intensity,
    intervention
  };
}

/**
 * Analyze message for phobia indicators
 */
export function analyzePhobia(message: string): PhobiaAnalysis {
  const text = message.toLowerCase();
  
  let maxScore = 0;
  let detectedPhobia: PhobiaType | null = null;
  
  for (const [phobia, patterns] of Object.entries(phobiaPatterns)) {
    let score = 0;
    for (const pattern of patterns) {
      if (text.includes(pattern)) score++;
    }
    if (score > maxScore) {
      maxScore = score;
      detectedPhobia = phobia as PhobiaType;
    }
  }
  
  const isPhobia = maxScore >= 2;
  const intensity = Math.min(1, maxScore / 4);
  
  const intervention = detectedPhobia && isPhobia ? 
    phobiaInterventions[detectedPhobia] : 
    phobiaInterventions.specific_situational;
  
  return {
    isPhobia,
    phobiaType: isPhobia ? detectedPhobia : null,
    intensity,
    intervention
  };
}

/**
 * Format OCD analysis for LLM context
 */
export function formatOCDForPrompt(analysis: OCDAnalysis): string {
  if (!analysis.isOCD) return "";
  
  const intervention = analysis.intervention;
  
  let output = "\n\n[OCD DETECTED]\n";
  output += `Subtypes: ${analysis.subtypes.join(", ")}\n`;
  output += `Primary: ${analysis.primarySubtype}\n`;
  output += `Intensity: ${(analysis.intensity * 100).toFixed(0)}%\n\n`;
  
  output += `VALIDATION: "${intervention.validation}"\n\n`;
  
  output += "PSYCHOEDUCATION:\n";
  for (const edu of intervention.psychoeducation) {
    output += `- ${edu}\n`;
  }
  
  output += "\nDO NOT:\n";
  for (const dont of intervention.doNotDo) {
    output += `- ${dont}\n`;
  }
  
  output += "\nKEY: Validate distress. Don't provide reassurance. Don't engage with OCD content. ";
  output += "Encourage professional treatment with OCD specialist.\n";
  
  return output;
}

/**
 * Format phobia analysis for LLM context
 */
export function formatPhobiaForPrompt(analysis: PhobiaAnalysis): string {
  if (!analysis.isPhobia) return "";
  
  const intervention = analysis.intervention;
  
  let output = "\n\n[PHOBIA DETECTED]\n";
  output += `Type: ${analysis.phobiaType}\n`;
  output += `Intensity: ${(analysis.intensity * 100).toFixed(0)}%\n\n`;
  
  output += `VALIDATION: "${intervention.validation}"\n\n`;
  
  output += "PSYCHOEDUCATION:\n";
  for (const edu of intervention.psychoeducation) {
    output += `- ${edu}\n`;
  }
  
  output += "\nCOPING STRATEGIES:\n";
  for (const strategy of intervention.copingStrategies.slice(0, 3)) {
    output += `- ${strategy}\n`;
  }
  
  return output;
}
