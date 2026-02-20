/**
 * Immigrant and Refugee Support Module for ReUnity
 * 
 * Provides grounding, reassurance, and practical support for:
 * - Immigrants and refugees experiencing anxiety about policies
 * - People worried about family members' immigration status
 * - Those affected by media coverage and uncertainty
 * - Anyone experiencing fear related to their background
 * 
 * This module is NON-POLITICAL and focuses on:
 * - Emotional grounding and calming techniques
 * - Practical systems analysis (understanding how things actually work)
 * - Media literacy (news isn't always peer-reviewed)
 * - Reassurance without dismissing valid concerns
 * - Connection to actual resources and facts
 * 
 * The approach is CALMING, AGREEABLE, and NON-COMBATIVE.
 * We don't argue - we ground, reassure, and provide perspective.
 */

export interface GroundingTechnique {
  id: string;
  name: string;
  description: string;
  steps: string[];
  duration: string;
  bestFor: string[];
}

export interface ReassuranceMessage {
  id: string;
  situation: string;
  message: string;
  perspective: string;
  practicalSteps?: string[];
}

export interface MediaLiteracyTip {
  id: string;
  title: string;
  explanation: string;
  questions: string[];
}

export interface SystemsAnalysis {
  id: string;
  topic: string;
  reality: string;
  perspective: string;
  groundingPoints: string[];
}

// === GROUNDING TECHNIQUES FOR ANXIETY ===
export const groundingTechniques: GroundingTechnique[] = [
  {
    id: 'present-moment',
    name: 'Present Moment Anchoring',
    description: 'When anxiety about the future overwhelms you, anchor yourself in the present moment.',
    steps: [
      'Notice your feet on the ground right now',
      'Feel the surface beneath you - solid, stable',
      'Take three slow breaths',
      'Ask yourself: "Am I safe right now, in this moment?"',
      'Notice that right now, in this second, you are okay',
      'The future is not here yet. You are here, now.',
    ],
    duration: '2-3 minutes',
    bestFor: ['future anxiety', 'policy fears', 'uncertainty'],
  },
  {
    id: 'five-senses',
    name: '5-4-3-2-1 Grounding',
    description: 'Use your senses to come back to the present when worry takes over.',
    steps: [
      'Name 5 things you can SEE around you',
      'Name 4 things you can TOUCH or feel',
      'Name 3 things you can HEAR',
      'Name 2 things you can SMELL',
      'Name 1 thing you can TASTE',
      'You are here. You are present. You are grounded.',
    ],
    duration: '3-5 minutes',
    bestFor: ['panic', 'overwhelming fear', 'spiraling thoughts'],
  },
  {
    id: 'body-scan',
    name: 'Quick Body Scan',
    description: 'Release tension stored in your body from worry and fear.',
    steps: [
      'Close your eyes if comfortable',
      'Notice your shoulders - let them drop',
      'Unclench your jaw',
      'Relax your hands',
      'Take a deep breath into your belly',
      'Your body is safe. Let it rest.',
    ],
    duration: '2 minutes',
    bestFor: ['physical tension', 'stress', 'fear response'],
  },
  {
    id: 'container',
    name: 'Container Technique',
    description: 'Temporarily set aside overwhelming worries so you can function.',
    steps: [
      'Imagine a strong container - a safe, a box, a vault',
      'Visualize putting your worries inside it for now',
      'Close the container securely',
      'Know that the worries will still be there if you need them',
      'But right now, they are contained',
      'You can deal with them when you are ready and resourced',
    ],
    duration: '3 minutes',
    bestFor: ['overwhelming worry', 'need to function', 'work/family obligations'],
  },
  {
    id: 'roots',
    name: 'Roots and Strength',
    description: 'Connect to your inner strength and the strength of your ancestors.',
    steps: [
      'Feel your feet on the ground',
      'Imagine roots growing from your feet deep into the earth',
      'These roots connect you to everyone who came before you',
      'Your ancestors survived hardship. Their strength is in you.',
      'You come from survivors. You carry their resilience.',
      'Draw that strength up through your roots.',
    ],
    duration: '3-5 minutes',
    bestFor: ['feeling powerless', 'identity fears', 'cultural anxiety'],
  },
];

// === REASSURANCE MESSAGES ===
export const reassuranceMessages: ReassuranceMessage[] = [
  {
    id: 'policy-fear',
    situation: 'Fear about immigration policies',
    message: "I hear that you're worried about what might happen. Those feelings are completely valid. Let's take a breath together and look at this from a grounded place.",
    perspective: "Changes in policy take time to implement. There are legal processes, court reviews, and many steps between an announcement and actual implementation. The system moves slowly - that's frustrating sometimes, but it also means there's time to understand what's actually happening versus what's being reported.",
    practicalSteps: [
      'Stay informed through official government sources, not just news headlines',
      'Connect with local immigrant rights organizations who track actual changes',
      'Know your rights - they exist regardless of status',
      'Have important documents organized and accessible',
      'Build community connections for mutual support',
    ],
  },
  {
    id: 'family-worry',
    situation: 'Worry about family members',
    message: "Worrying about people we love is one of the hardest things. Your care for your family shows your deep love. Let's ground ourselves and think about this clearly.",
    perspective: "Most people go about their daily lives without incident. The news reports exceptional cases because they are newsworthy - meaning they are not the norm. Millions of immigrants live, work, and raise families in the US every day without the dramatic scenarios shown on TV.",
    practicalSteps: [
      'Have a family communication plan',
      'Know emergency contacts and keep them accessible',
      'Connect with community organizations that provide support',
      'Focus on what you can control today',
      'Remember that worry doesn\'t change outcomes, but preparation helps',
    ],
  },
  {
    id: 'news-anxiety',
    situation: 'Anxiety from news and social media',
    message: "The news can be overwhelming, especially when it feels personal. It's okay to step back. Your mental health matters, and constant news consumption isn't helping you or anyone you care about.",
    perspective: "News media is designed to capture attention, which means it often emphasizes the most dramatic, scary, or controversial angles. This doesn't mean it's false, but it does mean it's not the complete picture. The vast majority of daily life continues normally, but 'everything is fine' doesn't make headlines.",
    practicalSteps: [
      'Limit news consumption to specific times (not all day)',
      'Choose 1-2 reliable sources rather than scrolling endlessly',
      'Take breaks from social media when it increases anxiety',
      'Remember: being informed doesn\'t require being overwhelmed',
      'Your wellbeing helps you help others',
    ],
  },
  {
    id: 'conspiracy-anxiety',
    situation: 'Anxiety from conspiracy theories or extreme predictions',
    message: "I understand that when people around us are saying alarming things, it can be really unsettling. Let's take a step back and look at this calmly.",
    perspective: "Throughout history, there have always been predictions of imminent disaster that didn't come true. This doesn't mean we should ignore real concerns, but it does mean we should be thoughtful about what we believe. Most extreme predictions don't account for how complex systems actually work - with checks, balances, and many people involved who have different interests.",
    practicalSteps: [
      'Ask: What is the source of this information?',
      'Ask: What would have to be true for this to happen?',
      'Ask: How many people would need to be involved?',
      'Look for what\'s actually happening, not what might happen',
      'Remember that uncertainty is uncomfortable but normal',
    ],
  },
  {
    id: 'identity-fear',
    situation: 'Fear about being targeted for identity',
    message: "Feeling unsafe because of who you are is deeply painful. Your identity is valid and valuable. You belong here, and your presence matters.",
    perspective: "While discrimination exists and is wrong, the vast majority of people you encounter in daily life are focused on their own lives and concerns. Most interactions are neutral or positive. The hateful voices are loud but they are not the majority. Communities across the country actively support and protect their immigrant neighbors.",
    practicalSteps: [
      'Connect with community groups that share your background',
      'Know your rights in various situations',
      'Build relationships with allies in your community',
      'Document any incidents but don\'t let fear isolate you',
      'Remember that you have value regardless of anyone\'s opinion',
    ],
  },
  {
    id: 'children-worry',
    situation: 'Worry about children and their future',
    message: "Your love for your children and concern for their future shows what a caring parent you are. Children are resilient, and they learn from watching how we handle difficulty.",
    perspective: "Children born in the US are citizens with full rights. Children who have grown up here often have strong community ties and support systems. The future is uncertain for everyone, but that has always been true. What matters most is the love and stability you provide today.",
    practicalSteps: [
      'Maintain routines that provide stability',
      'Be honest with children at age-appropriate levels',
      'Model calm problem-solving rather than panic',
      'Connect children with their cultural heritage as a source of strength',
      'Focus on education and building skills that serve them anywhere',
    ],
  },
];

// === MEDIA LITERACY TIPS ===
export const mediaLiteracyTips: MediaLiteracyTip[] = [
  {
    id: 'headlines',
    title: 'Headlines Are Designed to Get Clicks',
    explanation: "Headlines are written to make you click, not to give you accurate information. The actual article often has more nuance than the headline suggests. Many people share articles based on headlines without reading the full content.",
    questions: [
      'Did I read the full article or just the headline?',
      'Does the article support what the headline implies?',
      'What information might be missing?',
    ],
  },
  {
    id: 'sources',
    title: 'Consider the Source',
    explanation: "Not all sources are equal. Official government websites, established news organizations with editorial standards, and academic institutions are generally more reliable than social media posts, blogs, or partisan websites. Even reliable sources can have bias or make mistakes.",
    questions: [
      'Who published this and what is their reputation?',
      'Is this news reporting or opinion/commentary?',
      'Can I find this information from multiple independent sources?',
    ],
  },
  {
    id: 'emotion',
    title: 'Strong Emotions Are a Warning Sign',
    explanation: "If something makes you feel intense fear, anger, or outrage, that's a signal to slow down and think critically. Content designed to manipulate often targets emotions because emotional reactions bypass critical thinking. This doesn't mean the content is false, but it means you should verify before sharing or acting.",
    questions: [
      'Why might someone want me to feel this way?',
      'Am I reacting emotionally or thinking critically?',
      'What would I think about this if I felt calm?',
    ],
  },
  {
    id: 'predictions',
    title: 'Predictions Are Not Facts',
    explanation: "There's a big difference between 'X happened' and 'X might happen' or 'X could happen.' Predictions, especially about complex social and political situations, are often wrong. Experts frequently disagree, and the future is inherently uncertain.",
    questions: [
      'Is this reporting something that happened or predicting something that might happen?',
      'What is this prediction based on?',
      'Have similar predictions been accurate in the past?',
    ],
  },
  {
    id: 'anecdotes',
    title: 'Individual Stories Are Not Statistics',
    explanation: "A single dramatic story doesn't tell you how common something is. News often reports unusual events precisely because they are unusual. One person's experience, while valid, doesn't represent everyone's experience.",
    questions: [
      'Is this an individual case or a pattern?',
      'How common is this actually?',
      'What would the statistics show?',
    ],
  },
  {
    id: 'updates',
    title: 'First Reports Are Often Wrong',
    explanation: "Breaking news is frequently inaccurate. In the rush to report first, details get wrong, context is missing, and the full picture isn't clear. It's often better to wait for follow-up reporting than to react to initial reports.",
    questions: [
      'Is this breaking news or has it been verified?',
      'Have there been corrections or updates?',
      'Should I wait before forming an opinion?',
    ],
  },
];

// === SYSTEMS ANALYSIS ===
export const systemsAnalysis: SystemsAnalysis[] = [
  {
    id: 'policy-implementation',
    topic: 'How Policy Changes Actually Work',
    reality: "Policy changes in the US go through many steps: announcement, rule-making, legal challenges, court reviews, implementation planning, and actual enforcement. This process typically takes months to years, not days.",
    perspective: "The system is designed to be slow. This can be frustrating, but it also provides time for challenges, adjustments, and preparation. What's announced is often different from what's ultimately implemented.",
    groundingPoints: [
      'Announcements are not the same as implementation',
      'Courts regularly review and modify policies',
      'Enforcement requires resources and priorities',
      'Many policies face legal challenges',
      'Implementation varies by location and circumstance',
    ],
  },
  {
    id: 'enforcement-reality',
    topic: 'How Enforcement Actually Works',
    reality: "Enforcement agencies have limited resources and must prioritize. They cannot and do not pursue everyone. Priorities typically focus on specific categories, not broad sweeps of entire communities.",
    perspective: "While enforcement exists, the dramatic scenarios often portrayed are not representative of typical experiences. Most people in immigrant communities go about their daily lives without direct encounters with enforcement.",
    groundingPoints: [
      'Resources are limited and must be prioritized',
      'Most enforcement focuses on specific priorities',
      'Daily life continues normally for most people',
      'Community relationships and local policies matter',
      'Knowing your rights helps in any situation',
    ],
  },
  {
    id: 'media-incentives',
    topic: 'Why News Seems So Scary',
    reality: "News media operates on attention. Scary, dramatic, and controversial content gets more views, clicks, and shares. This creates an incentive to emphasize the most alarming aspects of any story.",
    perspective: "This doesn't mean news is fake, but it does mean it's not a representative sample of reality. 'Everything is fine' doesn't make headlines. The news you see is selected for impact, not for being typical.",
    groundingPoints: [
      'Attention-grabbing content gets prioritized',
      'Unusual events are newsworthy because they\'re unusual',
      'Positive or neutral events rarely make news',
      'Social media amplifies the most emotional content',
      'Your daily experience is more representative than news',
    ],
  },
  {
    id: 'community-reality',
    topic: 'The Reality of Community Support',
    reality: "Across the country, communities actively support and protect their immigrant neighbors. Churches, schools, local governments, businesses, and neighbors provide real support networks.",
    perspective: "The loud voices of hostility get attention, but they don't represent everyone. Many Americans actively support immigrant communities, even if that support doesn't make headlines.",
    groundingPoints: [
      'Many communities have declared themselves welcoming',
      'Local organizations provide real support',
      'Neighbors help neighbors regardless of status',
      'Businesses value immigrant workers and customers',
      'Faith communities often provide sanctuary and support',
    ],
  },
];

// === HELPER FUNCTIONS ===

/**
 * Get a grounding technique by ID
 */
export function getGroundingTechnique(id: string): GroundingTechnique | undefined {
  return groundingTechniques.find(t => t.id === id);
}

/**
 * Get grounding techniques for a specific situation
 */
export function getGroundingForSituation(situation: string): GroundingTechnique[] {
  const s = situation.toLowerCase();
  return groundingTechniques.filter(t =>
    t.bestFor.some(b => s.includes(b.toLowerCase()) || b.toLowerCase().includes(s))
  );
}

/**
 * Get a reassurance message for a situation
 */
export function getReassurance(situation: string): ReassuranceMessage | undefined {
  const s = situation.toLowerCase();
  return reassuranceMessages.find(r =>
    r.situation.toLowerCase().includes(s) ||
    s.includes(r.situation.toLowerCase()) ||
    r.id.includes(s)
  );
}

/**
 * Get all reassurance messages
 */
export function getAllReassurances(): ReassuranceMessage[] {
  return reassuranceMessages;
}

/**
 * Get media literacy tips
 */
export function getMediaLiteracyTips(): MediaLiteracyTip[] {
  return mediaLiteracyTips;
}

/**
 * Get a specific media literacy tip
 */
export function getMediaLiteracyTip(id: string): MediaLiteracyTip | undefined {
  return mediaLiteracyTips.find(t => t.id === id);
}

/**
 * Get systems analysis by topic
 */
export function getSystemsAnalysis(topic: string): SystemsAnalysis | undefined {
  const t = topic.toLowerCase();
  return systemsAnalysis.find(s =>
    s.topic.toLowerCase().includes(t) ||
    t.includes(s.topic.toLowerCase()) ||
    s.id.includes(t)
  );
}

/**
 * Get all systems analyses
 */
export function getAllSystemsAnalyses(): SystemsAnalysis[] {
  return systemsAnalysis;
}

/**
 * Generate a calming response for immigration-related anxiety
 */
export function generateCalmingResponse(concern: string): string {
  const responses: string[] = [
    "I hear your concern, and it's completely understandable to feel this way. Let's take a breath together and look at this from a grounded place.",
    "Those feelings are valid. Uncertainty is hard. Let's focus on what we know and what you can do right now.",
    "It makes sense that you're worried. Let's separate what's actually happening from what might happen, and focus on the present moment.",
    "Your concern shows you care deeply. That's not weakness - that's love. Let's channel that care into something constructive.",
    "I understand this feels overwhelming. You're not alone in these feelings. Let's ground ourselves and think through this together.",
  ];
  
  return responses[Math.floor(Math.random() * responses.length)];
}

/**
 * Generate guidance for AI responses about immigration concerns
 */
export function getImmigrantSupportGuidance(): string {
  return `When someone expresses anxiety about immigration, policies, or their status:

APPROACH:
- Be calming, not dismissive
- Be agreeable, not combative
- Be grounding, not alarming
- Validate feelings while providing perspective
- Focus on present moment, not worst-case futures

DO:
- Acknowledge their feelings as valid
- Offer grounding techniques
- Provide perspective on how systems actually work
- Encourage media literacy without being preachy
- Focus on what they can control
- Connect them to practical resources
- Remind them of community support

DO NOT:
- Argue about politics
- Dismiss their concerns
- Make predictions about the future
- Take sides on policy debates
- Increase their fear
- Tell them they're wrong to worry
- Engage with conspiracy theories argumentatively

REMEMBER:
- The goal is emotional support, not political education
- Grounding in the present is more helpful than debating the future
- Their mental health matters more than being "right"
- Calm presence is more valuable than information
- You can acknowledge uncertainty without amplifying fear`;
}

/**
 * Detect if a message contains immigration-related anxiety
 */
export function detectImmigrationAnxiety(message: string): boolean {
  const keywords = [
    'immigrant', 'immigration', 'deport', 'deportation', 'ice', 'border',
    'visa', 'status', 'undocumented', 'illegal', 'citizenship', 'green card',
    'refugee', 'asylum', 'migrant', 'foreign', 'policy', 'raid', 'detention',
    'family separation', 'daca', 'dreamer', 'papers', 'documentation',
    'scared about', 'worried about', 'afraid of', 'what if they',
    'will they come', 'are they going to', 'news says', 'heard that',
    'my family', 'my parents', 'my children', 'sent back', 'taken away'
  ];
  
  const m = message.toLowerCase();
  return keywords.some(k => m.includes(k));
}

/**
 * Detect if a message contains conspiracy-related anxiety
 */
export function detectConspiracyAnxiety(message: string): boolean {
  const keywords = [
    'they\'re going to', 'they want to', 'they\'re planning',
    'heard that', 'someone told me', 'read online', 'saw on facebook',
    'the government is', 'they\'re coming for', 'it\'s all planned',
    'wake up', 'sheeple', 'don\'t believe', 'mainstream media',
    'they don\'t want you to know', 'hidden agenda', 'deep state',
    'new world order', 'great reset', 'they\'re tracking', 'microchip',
    'control us', 'take over', 'martial law', 'camps'
  ];
  
  const m = message.toLowerCase();
  return keywords.some(k => m.includes(k));
}

/**
 * Generate a non-combative response to conspiracy anxiety
 */
export function generateConspiracyResponse(): string {
  const responses: string[] = [
    "I can hear that you're worried about what you've been hearing. Those feelings are real, even when the information is uncertain. Let's take a breath and ground ourselves in what we actually know.",
    "It sounds like you've been exposed to some alarming information. That can be really unsettling. Let's focus on the present moment - right now, you're here, you're safe, and we can think through this calmly.",
    "When we hear scary predictions, it's natural to feel afraid. Your feelings make sense. Let's separate what's actually happening from what someone is predicting might happen.",
    "I understand that when people around us are saying alarming things, it's hard not to be affected. You don't have to figure out what's true right now. Let's just focus on grounding and calming your nervous system.",
    "It's okay to feel uncertain. The future is always uncertain - that's just reality. What we can do is focus on today, on what's in front of us, and on taking care of ourselves and the people we love.",
  ];
  
  return responses[Math.floor(Math.random() * responses.length)];
}
