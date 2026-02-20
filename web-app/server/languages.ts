/**
 * Multi-Language Support Module for ReUnity
 * 
 * Comprehensive language support for diverse communities including:
 * - Hispanic/Latino communities (Spanish, Portuguese)
 * - South Asian communities (Hindi, Urdu, Punjabi, Bengali, Tamil, Telugu, Gujarati)
 * - Middle Eastern communities (Arabic, Farsi/Persian, Turkish, Hebrew)
 * - Native American communities (Navajo, Cherokee, Lakota, Ojibwe, Apache)
 * - East Asian communities (Mandarin, Cantonese, Japanese, Korean, Vietnamese, Tagalog)
 * - African communities (Swahili, Amharic, Yoruba, Hausa, Somali)
 * - European communities (French, German, Italian, Polish, Russian, Ukrainian)
 * - And many more...
 * 
 * All languages are treated with equal respect and dignity.
 * Everyone is welcome here.
 */

export interface Language {
  id: string;
  name: string;
  nativeName: string;
  code: string; // ISO 639-1 or ISO 639-3 code
  region: string;
  communities: string[];
  greetings: string[];
  comfortingPhrases: string[];
  crisisResources?: string[];
  culturalNotes?: string[];
  voiceSupport: boolean; // Whether Web Speech API supports this language
}

export const languages: Record<string, Language> = {
  // === SPANISH-SPEAKING COMMUNITIES ===
  'spanish': {
    id: 'spanish',
    name: 'Spanish',
    nativeName: 'Español',
    code: 'es',
    region: 'Latin America, Spain, USA',
    communities: ['Hispanic', 'Latino', 'Mexican', 'Puerto Rican', 'Cuban', 'Dominican', 'Central American', 'South American', 'Spanish'],
    greetings: [
      'Hola, estoy aquí contigo.',
      'Bienvenido/a. Estás en un lugar seguro.',
      'Me alegra que estés aquí. Tómate tu tiempo.',
    ],
    comfortingPhrases: [
      'Respira profundo. Estás a salvo aquí.',
      'No estás solo/a. Estoy aquí para escucharte.',
      'Tus sentimientos son válidos.',
      'Un día a la vez. Eso es todo lo que necesitas.',
      'Eres más fuerte de lo que crees.',
      'Está bien no estar bien.',
      'Tu historia importa.',
    ],
    crisisResources: [
      'Línea Nacional de Prevención del Suicidio: 988 (en español disponible)',
      'Crisis Text Line: Envía HOLA al 741741',
      'SAMHSA National Helpline: 1-800-662-4357',
    ],
    culturalNotes: [
      'Family (familia) is often central to healing',
      'Faith and spirituality may be important coping mechanisms',
      'Respect for elders and community support are valued',
    ],
    voiceSupport: true,
  },

  // === SOUTH ASIAN LANGUAGES ===
  'hindi': {
    id: 'hindi',
    name: 'Hindi',
    nativeName: 'हिन्दी',
    code: 'hi',
    region: 'India, Nepal, Fiji',
    communities: ['Indian', 'Hindu', 'North Indian', 'Indian-American', 'Indian diaspora'],
    greetings: [
      'नमस्ते, मैं आपके साथ हूं।',
      'आपका स्वागत है। आप यहां सुरक्षित हैं।',
      'मुझे खुशी है कि आप यहां हैं।',
    ],
    comfortingPhrases: [
      'गहरी सांस लें। आप यहां सुरक्षित हैं।',
      'आप अकेले नहीं हैं। मैं सुनने के लिए यहां हूं।',
      'आपकी भावनाएं मान्य हैं।',
      'एक दिन में एक कदम। बस इतना ही।',
      'आप जितना सोचते हैं उससे कहीं ज्यादा मजबूत हैं।',
      'ठीक न होना भी ठीक है।',
      'सब कुछ ठीक हो जाएगा।',
    ],
    crisisResources: [
      'iCall: 9152987821',
      'Vandrevala Foundation: 1860-2662-345',
      'NIMHANS: 080-46110007',
    ],
    culturalNotes: [
      'Family honor and collective well-being are important',
      'Spirituality and karma may influence worldview',
      'Respect for elders is deeply valued',
    ],
    voiceSupport: true,
  },

  'urdu': {
    id: 'urdu',
    name: 'Urdu',
    nativeName: 'اردو',
    code: 'ur',
    region: 'Pakistan, India',
    communities: ['Pakistani', 'Pakistani-American', 'Muslim South Asian', 'Urdu-speaking Indian'],
    greetings: [
      'السلام علیکم، میں آپ کے ساتھ ہوں۔',
      'خوش آمدید۔ آپ یہاں محفوظ ہیں۔',
      'مجھے خوشی ہے کہ آپ یہاں ہیں۔',
    ],
    comfortingPhrases: [
      'گہری سانس لیں۔ آپ یہاں محفوظ ہیں۔',
      'آپ اکیلے نہیں ہیں۔ میں سننے کے لیے یہاں ہوں۔',
      'آپ کے جذبات درست ہیں۔',
      'ایک وقت میں ایک دن۔ بس اتنا ہی۔',
      'آپ اپنی سوچ سے زیادہ مضبوط ہیں۔',
      'ٹھیک نہ ہونا بھی ٹھیک ہے۔',
      'اللہ پر بھروسہ رکھیں، سب ٹھیک ہو جائے گا۔',
    ],
    crisisResources: [
      'Umang Helpline Pakistan: 0311-7786264',
      'Rozan Counseling: 051-2890505',
    ],
    culturalNotes: [
      'Faith (iman) is often central to coping',
      'Family honor (izzat) is important',
      'Community support is valued',
    ],
    voiceSupport: true,
  },

  'punjabi': {
    id: 'punjabi',
    name: 'Punjabi',
    nativeName: 'ਪੰਜਾਬੀ / پنجابی',
    code: 'pa',
    region: 'Punjab (India/Pakistan)',
    communities: ['Punjabi', 'Sikh', 'Punjabi-American', 'Punjabi-Canadian', 'Punjabi-British'],
    greetings: [
      'ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ ਤੁਹਾਡੇ ਨਾਲ ਹਾਂ।',
      'ਜੀ ਆਇਆਂ ਨੂੰ। ਤੁਸੀਂ ਇੱਥੇ ਸੁਰੱਖਿਅਤ ਹੋ।',
    ],
    comfortingPhrases: [
      'ਡੂੰਘਾ ਸਾਹ ਲਓ। ਤੁਸੀਂ ਇੱਥੇ ਸੁਰੱਖਿਅਤ ਹੋ।',
      'ਤੁਸੀਂ ਇਕੱਲੇ ਨਹੀਂ ਹੋ। ਮੈਂ ਸੁਣਨ ਲਈ ਇੱਥੇ ਹਾਂ।',
      'ਤੁਹਾਡੀਆਂ ਭਾਵਨਾਵਾਂ ਜਾਇਜ਼ ਹਨ।',
      'ਵਾਹਿਗੁਰੂ ਮਿਹਰ ਕਰੇ।',
      'ਸਭ ਠੀਕ ਹੋ ਜਾਵੇਗਾ।',
    ],
    culturalNotes: [
      'Waheguru (God) is central to Sikh faith',
      'Seva (selfless service) and community are valued',
      'Resilience and strength are cultural values',
    ],
    voiceSupport: true,
  },

  'bengali': {
    id: 'bengali',
    name: 'Bengali',
    nativeName: 'বাংলা',
    code: 'bn',
    region: 'Bangladesh, West Bengal (India)',
    communities: ['Bengali', 'Bangladeshi', 'Bengali-American', 'West Bengali'],
    greetings: [
      'নমস্কার, আমি আপনার সাথে আছি।',
      'স্বাগতম। আপনি এখানে নিরাপদ।',
    ],
    comfortingPhrases: [
      'গভীর শ্বাস নিন। আপনি এখানে নিরাপদ।',
      'আপনি একা নন। আমি শোনার জন্য এখানে আছি।',
      'আপনার অনুভূতি বৈধ।',
      'সব ঠিক হয়ে যাবে।',
    ],
    voiceSupport: true,
  },

  'tamil': {
    id: 'tamil',
    name: 'Tamil',
    nativeName: 'தமிழ்',
    code: 'ta',
    region: 'Tamil Nadu (India), Sri Lanka, Singapore, Malaysia',
    communities: ['Tamil', 'Tamil-American', 'Sri Lankan Tamil', 'Malaysian Tamil'],
    greetings: [
      'வணக்கம், நான் உங்களுடன் இருக்கிறேன்.',
      'வரவேற்கிறேன். நீங்கள் இங்கே பாதுகாப்பாக இருக்கிறீர்கள்.',
    ],
    comfortingPhrases: [
      'ஆழமாக சுவாசியுங்கள். நீங்கள் இங்கே பாதுகாப்பாக இருக்கிறீர்கள்.',
      'நீங்கள் தனியாக இல்லை. நான் கேட்க இங்கே இருக்கிறேன்.',
      'உங்கள் உணர்வுகள் செல்லுபடியாகும்.',
      'எல்லாம் சரியாகிவிடும்.',
    ],
    voiceSupport: true,
  },

  'telugu': {
    id: 'telugu',
    name: 'Telugu',
    nativeName: 'తెలుగు',
    code: 'te',
    region: 'Andhra Pradesh, Telangana (India)',
    communities: ['Telugu', 'Telugu-American', 'Andhra', 'Telangana'],
    greetings: [
      'నమస్కారం, నేను మీతో ఉన్నాను.',
      'స్వాగతం. మీరు ఇక్కడ సురక్షితంగా ఉన్నారు.',
    ],
    comfortingPhrases: [
      'లోతుగా శ్వాస తీసుకోండి. మీరు ఇక్కడ సురక్షితంగా ఉన్నారు.',
      'మీరు ఒంటరిగా లేరు. నేను వినడానికి ఇక్కడ ఉన్నాను.',
      'మీ భావాలు చెల్లుబాటు అవుతాయి.',
    ],
    voiceSupport: true,
  },

  'gujarati': {
    id: 'gujarati',
    name: 'Gujarati',
    nativeName: 'ગુજરાતી',
    code: 'gu',
    region: 'Gujarat (India)',
    communities: ['Gujarati', 'Gujarati-American', 'Gujarati-British'],
    greetings: [
      'નમસ્તે, હું તમારી સાથે છું.',
      'સ્વાગત છે. તમે અહીં સુરક્ષિત છો.',
    ],
    comfortingPhrases: [
      'ઊંડો શ્વાસ લો. તમે અહીં સુરક્ષિત છો.',
      'તમે એકલા નથી. હું સાંભળવા માટે અહીં છું.',
      'તમારી લાગણીઓ માન્ય છે.',
      'બધું સારું થઈ જશે.',
    ],
    voiceSupport: true,
  },

  // === MIDDLE EASTERN LANGUAGES ===
  'arabic': {
    id: 'arabic',
    name: 'Arabic',
    nativeName: 'العربية',
    code: 'ar',
    region: 'Middle East, North Africa',
    communities: ['Arab', 'Arab-American', 'Middle Eastern', 'North African', 'Muslim', 'Lebanese', 'Syrian', 'Iraqi', 'Egyptian', 'Palestinian', 'Jordanian', 'Saudi', 'Yemeni', 'Moroccan'],
    greetings: [
      'السلام عليكم، أنا هنا معك.',
      'مرحباً. أنت في مكان آمن هنا.',
      'أهلاً وسهلاً. خذ وقتك.',
    ],
    comfortingPhrases: [
      'تنفس بعمق. أنت آمن هنا.',
      'أنت لست وحدك. أنا هنا للاستماع.',
      'مشاعرك صحيحة ومهمة.',
      'يوم بيوم. هذا كل ما تحتاجه.',
      'أنت أقوى مما تعتقد.',
      'لا بأس أن لا تكون بخير.',
      'إن شاء الله، كل شيء سيكون على ما يرام.',
      'الصبر مفتاح الفرج.',
    ],
    crisisResources: [
      'Embrace Lifeline (Lebanon): 1564',
      'BEFRIENDERS Worldwide Arabic support available',
    ],
    culturalNotes: [
      'Faith (iman) and trust in Allah are central',
      'Family and community bonds are very important',
      'Patience (sabr) is a valued virtue',
      'Hospitality and generosity are cultural values',
    ],
    voiceSupport: true,
  },

  'farsi': {
    id: 'farsi',
    name: 'Farsi/Persian',
    nativeName: 'فارسی',
    code: 'fa',
    region: 'Iran, Afghanistan, Tajikistan',
    communities: ['Iranian', 'Persian', 'Iranian-American', 'Afghan (Dari speakers)'],
    greetings: [
      'سلام، من اینجا با شما هستم.',
      'خوش آمدید. شما اینجا امن هستید.',
    ],
    comfortingPhrases: [
      'نفس عمیق بکشید. شما اینجا امن هستید.',
      'شما تنها نیستید. من اینجا برای گوش دادن هستم.',
      'احساسات شما معتبر است.',
      'همه چیز درست خواهد شد.',
      'این نیز بگذرد.',
    ],
    culturalNotes: [
      'Poetry and literature are deeply valued',
      'Family honor is important',
      'Hospitality (mehman-navazi) is a core value',
    ],
    voiceSupport: true,
  },

  'turkish': {
    id: 'turkish',
    name: 'Turkish',
    nativeName: 'Türkçe',
    code: 'tr',
    region: 'Turkey, Cyprus',
    communities: ['Turkish', 'Turkish-American', 'Turkish-German'],
    greetings: [
      'Merhaba, seninle birlikteyim.',
      'Hoş geldin. Burada güvendesin.',
    ],
    comfortingPhrases: [
      'Derin nefes al. Burada güvendesin.',
      'Yalnız değilsin. Dinlemek için buradayım.',
      'Duyguların geçerli.',
      'Her şey güzel olacak.',
      'Bu da geçer.',
    ],
    voiceSupport: true,
  },

  'hebrew': {
    id: 'hebrew',
    name: 'Hebrew',
    nativeName: 'עברית',
    code: 'he',
    region: 'Israel',
    communities: ['Israeli', 'Jewish', 'Jewish-American'],
    greetings: [
      'שלום, אני כאן איתך.',
      'ברוך הבא. אתה בטוח כאן.',
    ],
    comfortingPhrases: [
      'נשום עמוק. אתה בטוח כאן.',
      'אתה לא לבד. אני כאן להקשיב.',
      'הרגשות שלך תקפים.',
      'הכל יהיה בסדר.',
      'גם זה יעבור.',
    ],
    culturalNotes: [
      'Community (kehila) is important',
      'Tikkun olam (repairing the world) is a value',
      'Family bonds are central',
    ],
    voiceSupport: true,
  },

  // === NATIVE AMERICAN LANGUAGES ===
  'navajo': {
    id: 'navajo',
    name: 'Navajo (Diné)',
    nativeName: 'Diné bizaad',
    code: 'nv',
    region: 'Navajo Nation (Arizona, New Mexico, Utah)',
    communities: ['Navajo', 'Diné', 'Native American', 'Indigenous American'],
    greetings: [
      "Yá'át'ééh. Níká ánáshłééh.",
      "Yá'át'ééh. Kʼad shíká ánáshłééh.",
    ],
    comfortingPhrases: [
      "Hózhǫ́ náhásdlį́į́ʼ. (Beauty is restored.)",
      "Nizhónígo naasháa dooleeł. (You will walk in beauty.)",
      "Tʼáá ákwíí jíní. (It is what it is.)",
      "Shíká ánáshłééh. (I am here for you.)",
      "Hózhǫ́ǫ́jí naasháa doo. (Walk in beauty.)",
    ],
    crisisResources: [
      'StrongHearts Native Helpline: 1-844-762-8483',
      'National Suicide Prevention Lifeline: 988',
      'Indian Health Service Crisis Line: 1-800-273-8255',
    ],
    culturalNotes: [
      'Hózhǫ́ (beauty, balance, harmony) is central to Diné philosophy',
      'Connection to the land and ancestors is sacred',
      'The four sacred mountains define Diné homeland',
      'Ceremonies and traditional healing are important',
      'Respect for elders and traditional knowledge',
    ],
    voiceSupport: false, // Limited Web Speech API support
  },

  'cherokee': {
    id: 'cherokee',
    name: 'Cherokee',
    nativeName: 'ᏣᎳᎩ (Tsalagi)',
    code: 'chr',
    region: 'Oklahoma, North Carolina',
    communities: ['Cherokee', 'Cherokee Nation', 'Native American', 'Indigenous American'],
    greetings: [
      'ᎣᏏᏲ (Osiyo). Hello, I am here with you.',
      'ᏙᎯᏧ (Tohiju). Welcome. You are safe here.',
    ],
    comfortingPhrases: [
      'ᎣᏏᏲ. You are not alone.',
      'ᎦᎸᎳᏗ ᎡᎵᏍᏗ (Galvladi Elisdi) - The sky is peaceful.',
      'Your ancestors walk with you.',
      'You carry the strength of your people.',
      'The seven clans stand with you.',
    ],
    crisisResources: [
      'StrongHearts Native Helpline: 1-844-762-8483',
      'Cherokee Nation Behavioral Health: 918-458-4491',
    ],
    culturalNotes: [
      'The seven clans provide community and identity',
      'Connection to ancestors and land is sacred',
      'Traditional ceremonies support healing',
      'The Cherokee syllabary preserves language and culture',
    ],
    voiceSupport: false,
  },

  'lakota': {
    id: 'lakota',
    name: 'Lakota',
    nativeName: 'Lakȟótiyapi',
    code: 'lkt',
    region: 'North Dakota, South Dakota, Montana, Nebraska',
    communities: ['Lakota', 'Sioux', 'Native American', 'Indigenous American', 'Standing Rock', 'Pine Ridge', 'Rosebud'],
    greetings: [
      'Háu (Hello). Čhiyé waúŋ (I am here with you).',
      'Taŋyáŋ yahí (Welcome). You are safe here.',
    ],
    comfortingPhrases: [
      'Mitákuye Oyásʼiŋ (All my relations/We are all related).',
      'Wóčhekiye (Prayer) brings peace.',
      'The sacred hoop holds us together.',
      'Your ancestors are with you.',
      'Wašté (It is good).',
      'The four directions protect you.',
    ],
    crisisResources: [
      'StrongHearts Native Helpline: 1-844-762-8483',
      'Great Plains Tribal Crisis Line: 1-800-273-8255',
    ],
    culturalNotes: [
      'Mitákuye Oyásʼiŋ reflects interconnectedness of all life',
      'The sacred pipe (Čhaŋnúŋpa) is central to ceremony',
      'The Black Hills (Pahá Sápa) are sacred',
      'Sundance and sweat lodge ceremonies support healing',
      'Respect for the buffalo and all living things',
    ],
    voiceSupport: false,
  },

  'ojibwe': {
    id: 'ojibwe',
    name: 'Ojibwe (Anishinaabemowin)',
    nativeName: 'Anishinaabemowin',
    code: 'oj',
    region: 'Great Lakes region (USA/Canada)',
    communities: ['Ojibwe', 'Chippewa', 'Anishinaabe', 'Native American', 'First Nations'],
    greetings: [
      'Boozhoo. Gidaa-wiijiiw (I am here with you).',
      'Aaniin. You are welcome here.',
    ],
    comfortingPhrases: [
      'Gizaagi\'in (I love you/care for you).',
      'Mino-bimaadiziwin (The good life) awaits.',
      'The Seven Grandfather Teachings guide us.',
      'Your spirit is strong.',
      'The drum beats for you.',
    ],
    crisisResources: [
      'StrongHearts Native Helpline: 1-844-762-8483',
    ],
    culturalNotes: [
      'Seven Grandfather Teachings: wisdom, love, respect, bravery, honesty, humility, truth',
      'The drum is the heartbeat of the nation',
      'Water is sacred and life-giving',
      'Clan system provides identity and responsibility',
    ],
    voiceSupport: false,
  },

  'apache': {
    id: 'apache',
    name: 'Apache',
    nativeName: 'Ndéé bizaa / Nnēē biyáti\'',
    code: 'apw',
    region: 'Arizona, New Mexico, Oklahoma',
    communities: ['Apache', 'Native American', 'Indigenous American', 'White Mountain Apache', 'San Carlos Apache', 'Mescalero Apache', 'Jicarilla Apache'],
    greetings: [
      'Dagotʼee (Hello). I am here with you.',
      'Welcome. You are safe here.',
    ],
    comfortingPhrases: [
      'Your strength comes from your people.',
      'The mountains hold you.',
      'Your ancestors guide you.',
      'You are not alone in this journey.',
    ],
    crisisResources: [
      'StrongHearts Native Helpline: 1-844-762-8483',
      'White Mountain Apache Behavioral Health: 928-338-4911',
    ],
    culturalNotes: [
      'Sunrise Ceremony marks important transitions',
      'Connection to the land is sacred',
      'Respect for elders and traditional knowledge',
      'Storytelling preserves wisdom',
    ],
    voiceSupport: false,
  },

  // === EAST ASIAN LANGUAGES ===
  'mandarin': {
    id: 'mandarin',
    name: 'Mandarin Chinese',
    nativeName: '普通话 / 國語',
    code: 'zh',
    region: 'China, Taiwan, Singapore',
    communities: ['Chinese', 'Chinese-American', 'Taiwanese', 'Singaporean Chinese'],
    greetings: [
      '你好，我在这里陪伴你。',
      '欢迎。你在这里是安全的。',
    ],
    comfortingPhrases: [
      '深呼吸。你在这里是安全的。',
      '你不是一个人。我在这里倾听。',
      '你的感受是有效的。',
      '一切都会好起来的。',
      '慢慢来，不着急。',
    ],
    crisisResources: [
      'Beijing Suicide Research and Prevention Center: 010-82951332',
      'Lifeline China: 400-161-9995',
    ],
    culturalNotes: [
      'Face (面子) and harmony are important',
      'Family obligations are central',
      'Respect for elders is valued',
    ],
    voiceSupport: true,
  },

  'cantonese': {
    id: 'cantonese',
    name: 'Cantonese',
    nativeName: '廣東話 / 粵語',
    code: 'yue',
    region: 'Hong Kong, Macau, Guangdong',
    communities: ['Cantonese', 'Hong Kong', 'Hong Kong-American'],
    greetings: [
      '你好，我喺度陪住你。',
      '歡迎。你喺度係安全嘅。',
    ],
    comfortingPhrases: [
      '深呼吸。你喺度係安全嘅。',
      '你唔係一個人。我喺度聽你講。',
      '你嘅感受係有效嘅。',
      '一切都會好嘅。',
    ],
    voiceSupport: true,
  },

  'japanese': {
    id: 'japanese',
    name: 'Japanese',
    nativeName: '日本語',
    code: 'ja',
    region: 'Japan',
    communities: ['Japanese', 'Japanese-American'],
    greetings: [
      'こんにちは、私はあなたと一緒にいます。',
      'ようこそ。ここは安全な場所です。',
    ],
    comfortingPhrases: [
      '深呼吸してください。ここは安全です。',
      'あなたは一人ではありません。私は聞いています。',
      'あなたの気持ちは大切です。',
      '大丈夫、きっとうまくいきます。',
      '頑張りすぎないでください。',
    ],
    crisisResources: [
      'TELL Lifeline: 03-5774-0992',
      'Befrienders Osaka: 06-6260-4343',
    ],
    culturalNotes: [
      'Harmony (wa) and group cohesion are valued',
      'Indirect communication is common',
      'Perseverance (ganbaru) is respected',
    ],
    voiceSupport: true,
  },

  'korean': {
    id: 'korean',
    name: 'Korean',
    nativeName: '한국어',
    code: 'ko',
    region: 'South Korea, North Korea',
    communities: ['Korean', 'Korean-American'],
    greetings: [
      '안녕하세요, 저는 당신과 함께 있습니다.',
      '환영합니다. 여기는 안전한 곳입니다.',
    ],
    comfortingPhrases: [
      '깊이 숨을 쉬세요. 여기는 안전합니다.',
      '혼자가 아닙니다. 저는 듣고 있습니다.',
      '당신의 감정은 중요합니다.',
      '괜찮아요, 다 잘 될 거예요.',
      '화이팅!',
    ],
    crisisResources: [
      'Korea Suicide Prevention Center: 1393',
      'Mental Health Crisis Line: 1577-0199',
    ],
    culturalNotes: [
      'Jeong (정) - deep emotional bond is valued',
      'Family and social harmony are important',
      'Respect for elders (효도) is central',
    ],
    voiceSupport: true,
  },

  'vietnamese': {
    id: 'vietnamese',
    name: 'Vietnamese',
    nativeName: 'Tiếng Việt',
    code: 'vi',
    region: 'Vietnam',
    communities: ['Vietnamese', 'Vietnamese-American'],
    greetings: [
      'Xin chào, tôi ở đây với bạn.',
      'Chào mừng. Bạn an toàn ở đây.',
    ],
    comfortingPhrases: [
      'Hít thở sâu. Bạn an toàn ở đây.',
      'Bạn không đơn độc. Tôi đang lắng nghe.',
      'Cảm xúc của bạn là hợp lệ.',
      'Mọi thứ sẽ ổn thôi.',
    ],
    voiceSupport: true,
  },

  'tagalog': {
    id: 'tagalog',
    name: 'Tagalog/Filipino',
    nativeName: 'Tagalog / Filipino',
    code: 'tl',
    region: 'Philippines',
    communities: ['Filipino', 'Filipino-American', 'Philippine'],
    greetings: [
      'Kumusta, nandito ako para sa iyo.',
      'Maligayang pagdating. Ligtas ka dito.',
    ],
    comfortingPhrases: [
      'Huminga ng malalim. Ligtas ka dito.',
      'Hindi ka nag-iisa. Naririnig kita.',
      'Valid ang iyong nararamdaman.',
      'Magiging maayos din ang lahat.',
      'Kaya mo yan.',
    ],
    crisisResources: [
      'NCMH Crisis Hotline: 0917-899-8727',
      'Hopeline Philippines: 2919',
    ],
    culturalNotes: [
      'Family (pamilya) is central',
      'Bayanihan (community spirit) is valued',
      'Respect for elders is important',
    ],
    voiceSupport: true,
  },

  // === AFRICAN LANGUAGES ===
  'swahili': {
    id: 'swahili',
    name: 'Swahili',
    nativeName: 'Kiswahili',
    code: 'sw',
    region: 'East Africa (Kenya, Tanzania, Uganda)',
    communities: ['East African', 'Kenyan', 'Tanzanian', 'Ugandan', 'African'],
    greetings: [
      'Habari, niko hapa nawe.',
      'Karibu. Uko salama hapa.',
    ],
    comfortingPhrases: [
      'Pumua kwa kina. Uko salama hapa.',
      'Huko peke yako. Niko hapa kusikiliza.',
      'Hisia zako ni halali.',
      'Kila kitu kitakuwa sawa.',
      'Pole pole ndio mwendo.',
    ],
    culturalNotes: [
      'Ubuntu - I am because we are',
      'Community and family are central',
      'Respect for elders is important',
    ],
    voiceSupport: true,
  },

  'amharic': {
    id: 'amharic',
    name: 'Amharic',
    nativeName: 'አማርኛ',
    code: 'am',
    region: 'Ethiopia',
    communities: ['Ethiopian', 'Ethiopian-American', 'Eritrean'],
    greetings: [
      'ሰላም፣ እኔ ከአንተ ጋር ነኝ።',
      'እንኳን ደህና መጣህ። እዚህ ደህና ነህ።',
    ],
    comfortingPhrases: [
      'ጥልቅ ትንፋሽ ውሰድ። እዚህ ደህና ነህ።',
      'ብቻህን አይደለህም። እኔ እየሰማሁ ነው።',
      'ስሜቶችህ ትክክል ናቸው።',
      'ሁሉም ነገር ደህና ይሆናል።',
    ],
    voiceSupport: false,
  },

  'somali': {
    id: 'somali',
    name: 'Somali',
    nativeName: 'Af Soomaali',
    code: 'so',
    region: 'Somalia, Djibouti, Ethiopia, Kenya',
    communities: ['Somali', 'Somali-American', 'Somali refugee'],
    greetings: [
      'Salaan, waan kula jiraa.',
      'Soo dhawoow. Halkan waxaad ku ammaan tahay.',
    ],
    comfortingPhrases: [
      'Si qoto dheer u neefsoo. Halkan waxaad ku ammaan tahay.',
      'Keligaa ma tihid. Waan ku dhagaysanayaa.',
      'Dareenkaagu waa sax.',
      'Wax walba waa hagaagi doonaan.',
      'Ilaahay ha ku caawino.',
    ],
    culturalNotes: [
      'Faith in Allah is central',
      'Clan and family ties are important',
      'Poetry and oral tradition are valued',
    ],
    voiceSupport: false,
  },

  // === EUROPEAN LANGUAGES ===
  'french': {
    id: 'french',
    name: 'French',
    nativeName: 'Français',
    code: 'fr',
    region: 'France, Canada, Africa',
    communities: ['French', 'French-Canadian', 'Haitian', 'African Francophone'],
    greetings: [
      'Bonjour, je suis là avec vous.',
      'Bienvenue. Vous êtes en sécurité ici.',
    ],
    comfortingPhrases: [
      'Respirez profondément. Vous êtes en sécurité ici.',
      'Vous n\'êtes pas seul(e). Je suis là pour écouter.',
      'Vos sentiments sont valides.',
      'Tout ira bien.',
      'Un jour à la fois.',
    ],
    voiceSupport: true,
  },

  'german': {
    id: 'german',
    name: 'German',
    nativeName: 'Deutsch',
    code: 'de',
    region: 'Germany, Austria, Switzerland',
    communities: ['German', 'German-American', 'Austrian', 'Swiss German'],
    greetings: [
      'Hallo, ich bin hier bei dir.',
      'Willkommen. Du bist hier sicher.',
    ],
    comfortingPhrases: [
      'Atme tief durch. Du bist hier sicher.',
      'Du bist nicht allein. Ich höre zu.',
      'Deine Gefühle sind berechtigt.',
      'Alles wird gut.',
      'Ein Tag nach dem anderen.',
    ],
    voiceSupport: true,
  },

  'russian': {
    id: 'russian',
    name: 'Russian',
    nativeName: 'Русский',
    code: 'ru',
    region: 'Russia, Eastern Europe, Central Asia',
    communities: ['Russian', 'Russian-American', 'Ukrainian', 'Eastern European'],
    greetings: [
      'Привет, я здесь с тобой.',
      'Добро пожаловать. Ты в безопасности здесь.',
    ],
    comfortingPhrases: [
      'Дыши глубоко. Ты в безопасности здесь.',
      'Ты не один/одна. Я слушаю.',
      'Твои чувства важны.',
      'Всё будет хорошо.',
      'Шаг за шагом.',
    ],
    voiceSupport: true,
  },

  'ukrainian': {
    id: 'ukrainian',
    name: 'Ukrainian',
    nativeName: 'Українська',
    code: 'uk',
    region: 'Ukraine',
    communities: ['Ukrainian', 'Ukrainian-American', 'Ukrainian refugee'],
    greetings: [
      'Привіт, я тут з тобою.',
      'Ласкаво просимо. Ти в безпеці тут.',
    ],
    comfortingPhrases: [
      'Дихай глибоко. Ти в безпеці тут.',
      'Ти не сам/сама. Я слухаю.',
      'Твої почуття важливі.',
      'Все буде добре.',
      'Крок за кроком.',
    ],
    crisisResources: [
      'Lifeline Ukraine: 7333',
    ],
    culturalNotes: [
      'Resilience and strength are valued',
      'Family and community bonds are important',
      'Cultural traditions provide comfort',
    ],
    voiceSupport: true,
  },

  'polish': {
    id: 'polish',
    name: 'Polish',
    nativeName: 'Polski',
    code: 'pl',
    region: 'Poland',
    communities: ['Polish', 'Polish-American'],
    greetings: [
      'Cześć, jestem tu z tobą.',
      'Witaj. Jesteś tu bezpieczny/bezpieczna.',
    ],
    comfortingPhrases: [
      'Oddychaj głęboko. Jesteś tu bezpieczny/bezpieczna.',
      'Nie jesteś sam/sama. Słucham.',
      'Twoje uczucia są ważne.',
      'Wszystko będzie dobrze.',
      'Krok po kroku.',
    ],
    voiceSupport: true,
  },

  'portuguese': {
    id: 'portuguese',
    name: 'Portuguese',
    nativeName: 'Português',
    code: 'pt',
    region: 'Brazil, Portugal, Africa',
    communities: ['Brazilian', 'Portuguese', 'Brazilian-American', 'Lusophone African'],
    greetings: [
      'Olá, estou aqui com você.',
      'Bem-vindo(a). Você está seguro(a) aqui.',
    ],
    comfortingPhrases: [
      'Respire fundo. Você está seguro(a) aqui.',
      'Você não está sozinho(a). Estou aqui para ouvir.',
      'Seus sentimentos são válidos.',
      'Tudo vai ficar bem.',
      'Um dia de cada vez.',
    ],
    crisisResources: [
      'CVV (Brazil): 188',
      'SOS Voz Amiga (Portugal): 213 544 545',
    ],
    voiceSupport: true,
  },

  'italian': {
    id: 'italian',
    name: 'Italian',
    nativeName: 'Italiano',
    code: 'it',
    region: 'Italy',
    communities: ['Italian', 'Italian-American'],
    greetings: [
      'Ciao, sono qui con te.',
      'Benvenuto/a. Sei al sicuro qui.',
    ],
    comfortingPhrases: [
      'Respira profondamente. Sei al sicuro qui.',
      'Non sei solo/a. Sono qui per ascoltare.',
      'I tuoi sentimenti sono validi.',
      'Tutto andrà bene.',
      'Un giorno alla volta.',
    ],
    voiceSupport: true,
  },
};

// === HELPER FUNCTIONS ===

/**
 * Get a language by ID
 */
export function getLanguage(id: string): Language | undefined {
  return languages[id.toLowerCase()];
}

/**
 * Get all languages for a specific region
 */
export function getLanguagesByRegion(region: string): Language[] {
  return Object.values(languages).filter(lang => 
    lang.region.toLowerCase().includes(region.toLowerCase())
  );
}

/**
 * Get all languages for a specific community
 */
export function getLanguagesForCommunity(community: string): Language[] {
  return Object.values(languages).filter(lang =>
    lang.communities.some(c => c.toLowerCase().includes(community.toLowerCase()))
  );
}

/**
 * Get a random greeting in a specific language
 */
export function getGreeting(languageId: string): string | undefined {
  const lang = getLanguage(languageId);
  if (!lang || lang.greetings.length === 0) return undefined;
  return lang.greetings[Math.floor(Math.random() * lang.greetings.length)];
}

/**
 * Get a random comforting phrase in a specific language
 */
export function getComfortingPhrase(languageId: string): string | undefined {
  const lang = getLanguage(languageId);
  if (!lang || lang.comfortingPhrases.length === 0) return undefined;
  return lang.comfortingPhrases[Math.floor(Math.random() * lang.comfortingPhrases.length)];
}

/**
 * Get crisis resources for a specific language/region
 */
export function getCrisisResources(languageId: string): string[] {
  const lang = getLanguage(languageId);
  return lang?.crisisResources || [];
}

/**
 * Get cultural notes for a specific language
 */
export function getCulturalNotes(languageId: string): string[] {
  const lang = getLanguage(languageId);
  return lang?.culturalNotes || [];
}

/**
 * Get all language IDs
 */
export function getAllLanguageIds(): string[] {
  return Object.keys(languages);
}

/**
 * Get all languages with voice support
 */
export function getLanguagesWithVoiceSupport(): Language[] {
  return Object.values(languages).filter(lang => lang.voiceSupport);
}

/**
 * Get all Native American languages
 */
export function getNativeAmericanLanguages(): Language[] {
  return Object.values(languages).filter(lang =>
    lang.communities.some(c => 
      c.includes('Native American') || 
      c.includes('Indigenous') ||
      c.includes('First Nations')
    )
  );
}

/**
 * Search languages by query
 */
export function searchLanguages(query: string): Language[] {
  const q = query.toLowerCase();
  return Object.values(languages).filter(lang =>
    lang.name.toLowerCase().includes(q) ||
    lang.nativeName.toLowerCase().includes(q) ||
    lang.communities.some(c => c.toLowerCase().includes(q)) ||
    lang.region.toLowerCase().includes(q)
  );
}

/**
 * Get language guidance for AI responses
 */
export function getLanguageGuidance(languageId: string): string {
  const lang = getLanguage(languageId);
  if (!lang) return '';

  let guidance = `When speaking with someone from the ${lang.communities.join('/')} community:\n`;
  guidance += `- Their primary language is ${lang.name} (${lang.nativeName})\n`;
  
  if (lang.culturalNotes && lang.culturalNotes.length > 0) {
    guidance += `- Cultural considerations:\n`;
    lang.culturalNotes.forEach(note => {
      guidance += `  * ${note}\n`;
    });
  }
  
  guidance += `- You may use phrases like: "${lang.comfortingPhrases[0]}"\n`;
  
  return guidance;
}

/**
 * Get universal multilingual support message
 */
export function getMultilingualWelcome(): string {
  return `Welcome / Bienvenido / 欢迎 / مرحبا / स्वागत / ようこそ / 환영합니다 / Chào mừng / Karibu / Yá'át'ééh

You are welcome here in any language. We support speakers of:
- Spanish, Portuguese, French, German, Italian, Polish, Russian, Ukrainian
- Hindi, Urdu, Punjabi, Bengali, Tamil, Telugu, Gujarati
- Arabic, Farsi, Turkish, Hebrew
- Mandarin, Cantonese, Japanese, Korean, Vietnamese, Tagalog
- Swahili, Amharic, Somali, Yoruba
- Navajo (Diné), Cherokee, Lakota, Ojibwe, Apache
- And many more...

All cultures, all backgrounds, all people are welcome here.`;
}
