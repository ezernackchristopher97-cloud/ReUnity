/**
 * OpenAI Text-to-Speech Integration
 * Natural-sounding voices with gender, accent, and language options
 */

// OpenAI TTS Voice Options - Natural human voices
export const OPENAI_VOICES = {
  // Female voices
  nova: {
    id: 'nova',
    name: 'Nova',
    gender: 'female',
    description: 'Warm, friendly female voice - great for supportive conversations',
    accent: 'American',
    language: 'en',
    tone: 'warm'
  },
  shimmer: {
    id: 'shimmer',
    name: 'Shimmer',
    gender: 'female', 
    description: 'Soft, gentle female voice - calming and soothing',
    accent: 'American',
    language: 'en',
    tone: 'gentle'
  },
  // Male voices
  echo: {
    id: 'echo',
    name: 'Echo',
    gender: 'male',
    description: 'Deep, calm male voice - reassuring and grounded',
    accent: 'American',
    language: 'en',
    tone: 'calm'
  },
  onyx: {
    id: 'onyx',
    name: 'Onyx',
    gender: 'male',
    description: 'Rich, warm male voice - comforting and steady',
    accent: 'American',
    language: 'en',
    tone: 'warm'
  },
  fable: {
    id: 'fable',
    name: 'Fable',
    gender: 'male',
    description: 'Expressive male voice - engaging and empathetic',
    accent: 'British',
    language: 'en',
    tone: 'expressive'
  },
  // Neutral/Androgynous
  alloy: {
    id: 'alloy',
    name: 'Alloy',
    gender: 'neutral',
    description: 'Balanced, neutral voice - versatile and clear',
    accent: 'American',
    language: 'en',
    tone: 'neutral'
  }
} as const;

export type VoiceId = keyof typeof OPENAI_VOICES;
export type Gender = 'female' | 'male' | 'neutral';
export type Accent = 'American' | 'British';
export type Tone = 'warm' | 'gentle' | 'calm' | 'expressive' | 'neutral';

// Supported languages for TTS (OpenAI supports these natively)
export const SUPPORTED_LANGUAGES = {
  en: { code: 'en', name: 'English', native: 'English' },
  es: { code: 'es', name: 'Spanish', native: 'Español' },
  fr: { code: 'fr', name: 'French', native: 'Français' },
  de: { code: 'de', name: 'German', native: 'Deutsch' },
  it: { code: 'it', name: 'Italian', native: 'Italiano' },
  pt: { code: 'pt', name: 'Portuguese', native: 'Português' },
  pl: { code: 'pl', name: 'Polish', native: 'Polski' },
  ru: { code: 'ru', name: 'Russian', native: 'Русский' },
  nl: { code: 'nl', name: 'Dutch', native: 'Nederlands' },
  ja: { code: 'ja', name: 'Japanese', native: '日本語' },
  zh: { code: 'zh', name: 'Chinese', native: '中文' },
  ko: { code: 'ko', name: 'Korean', native: '한국어' },
  ar: { code: 'ar', name: 'Arabic', native: 'العربية' },
  hi: { code: 'hi', name: 'Hindi', native: 'हिन्दी' },
  tr: { code: 'tr', name: 'Turkish', native: 'Türkçe' },
  vi: { code: 'vi', name: 'Vietnamese', native: 'Tiếng Việt' },
  th: { code: 'th', name: 'Thai', native: 'ไทย' },
  uk: { code: 'uk', name: 'Ukrainian', native: 'Українська' },
  id: { code: 'id', name: 'Indonesian', native: 'Bahasa Indonesia' },
  ms: { code: 'ms', name: 'Malay', native: 'Bahasa Melayu' },
  tl: { code: 'tl', name: 'Tagalog', native: 'Tagalog' },
  sw: { code: 'sw', name: 'Swahili', native: 'Kiswahili' },
  he: { code: 'he', name: 'Hebrew', native: 'עברית' },
  fa: { code: 'fa', name: 'Persian/Farsi', native: 'فارسی' },
  ur: { code: 'ur', name: 'Urdu', native: 'اردو' },
  bn: { code: 'bn', name: 'Bengali', native: 'বাংলা' },
  pa: { code: 'pa', name: 'Punjabi', native: 'ਪੰਜਾਬੀ' },
  ta: { code: 'ta', name: 'Tamil', native: 'தமிழ்' },
  te: { code: 'te', name: 'Telugu', native: 'తెలుగు' },
  gu: { code: 'gu', name: 'Gujarati', native: 'ગુજરાતી' },
} as const;

export type LanguageCode = keyof typeof SUPPORTED_LANGUAGES;

export interface TTSRequest {
  text: string;
  voice: VoiceId;
  speed?: number; // 0.25 to 4.0, default 1.0
  language?: LanguageCode; // For multilingual text
}

export interface TTSResponse {
  audioUrl: string;
  audioBase64: string;
  voice: VoiceId;
  duration?: number;
}

export interface VoiceFilter {
  gender?: Gender;
  accent?: Accent;
  tone?: Tone;
}

/**
 * Generate speech from text using OpenAI TTS API
 * Returns base64 encoded audio that can be played directly in the browser
 */
export async function generateSpeech(request: TTSRequest): Promise<TTSResponse> {
  const { text, voice, speed = 1.0 } = request;
  
  // Validate voice
  if (!OPENAI_VOICES[voice]) {
    throw new Error(`Invalid voice: ${voice}. Available voices: ${Object.keys(OPENAI_VOICES).join(', ')}`);
  }
  
  // Validate speed
  const clampedSpeed = Math.max(0.25, Math.min(4.0, speed));
  
  try {
    // Use the Manus built-in API for TTS
    const response = await fetch(`${process.env.BUILT_IN_FORGE_API_URL}/v1/audio/speech`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${process.env.BUILT_IN_FORGE_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'tts-1-hd', // High quality model
        input: text,
        voice: voice,
        speed: clampedSpeed,
        response_format: 'mp3'
      })
    });
    
    if (!response.ok) {
      const error = await response.text();
      throw new Error(`TTS API error: ${error}`);
    }
    
    // Get audio as array buffer
    const audioBuffer = await response.arrayBuffer();
    const audioBase64 = Buffer.from(audioBuffer).toString('base64');
    const audioUrl = `data:audio/mp3;base64,${audioBase64}`;
    
    return {
      audioUrl,
      audioBase64,
      voice
    };
  } catch (error) {
    console.error('[TTS] Error generating speech:', error);
    throw error;
  }
}

/**
 * Get all available voices with their metadata
 */
export function getAvailableVoices() {
  return Object.values(OPENAI_VOICES);
}

/**
 * Get voices filtered by criteria
 */
export function filterVoices(filter: VoiceFilter) {
  let voices = Object.values(OPENAI_VOICES);
  
  if (filter.gender) {
    voices = voices.filter(v => v.gender === filter.gender);
  }
  if (filter.accent) {
    voices = voices.filter(v => v.accent === filter.accent);
  }
  if (filter.tone) {
    voices = voices.filter(v => v.tone === filter.tone);
  }
  
  return voices;
}

/**
 * Get voices by gender
 */
export function getVoicesByGender(gender: Gender) {
  return filterVoices({ gender });
}

/**
 * Get voices by accent
 */
export function getVoicesByAccent(accent: Accent) {
  return filterVoices({ accent });
}

/**
 * Get all supported languages
 */
export function getSupportedLanguages() {
  return Object.values(SUPPORTED_LANGUAGES);
}

/**
 * Check if a language is supported
 */
export function isLanguageSupported(code: string): code is LanguageCode {
  return code in SUPPORTED_LANGUAGES;
}
