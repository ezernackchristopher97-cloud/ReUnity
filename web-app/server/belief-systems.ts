/**
 * Belief Systems Module for ReUnity
 * 
 * Comprehensive support for religious, spiritual, philosophical, and secular worldviews.
 * All beliefs are treated with equal respect and dignity.
 * 
 * "All are welcome here."
 */

export type BeliefCategory = 
  | 'religious'
  | 'spiritual'
  | 'philosophical'
  | 'secular'
  | 'indigenous'
  | 'eastern'
  | 'western'
  | 'mystical';

export interface BeliefSystem {
  id: string;
  name: string;
  category: BeliefCategory;
  description: string;
  coreBeliefs: string[];
  copingStrategies: string[];
  comfortingPhrases: string[];
  sacredTexts?: string[];
  practices?: string[];
  holidays?: string[];
  dietaryConsiderations?: string[];
  endOfLifeBeliefs?: string[];
  mentalHealthPerspective?: string;
  crisisSupport?: string[];
}

// ============================================
// ABRAHAMIC RELIGIONS
// ============================================

const christianity: BeliefSystem = {
  id: 'christianity',
  name: 'Christianity',
  category: 'religious',
  description: 'Faith centered on Jesus Christ as the Son of God and savior of humanity',
  coreBeliefs: [
    'God loves you unconditionally',
    'You are never alone - God is always with you',
    'There is hope and redemption through faith',
    'You are created in God\'s image and have inherent worth',
    'Suffering can have meaning and lead to growth',
  ],
  copingStrategies: [
    'Prayer and meditation on scripture',
    'Seeking comfort in the Psalms',
    'Connecting with your faith community',
    'Remembering God\'s promises of peace',
    'Casting your anxieties on God (1 Peter 5:7)',
    'Finding strength through worship',
  ],
  comfortingPhrases: [
    '"The Lord is close to the brokenhearted" - Psalm 34:18',
    '"I can do all things through Christ who strengthens me" - Philippians 4:13',
    '"Come to me, all who are weary, and I will give you rest" - Matthew 11:28',
    '"For I know the plans I have for you, plans to prosper you" - Jeremiah 29:11',
    '"Peace I leave with you; my peace I give you" - John 14:27',
    '"God is our refuge and strength, an ever-present help in trouble" - Psalm 46:1',
  ],
  sacredTexts: ['Bible', 'Old Testament', 'New Testament', 'Psalms', 'Gospels'],
  practices: ['Prayer', 'Church attendance', 'Bible study', 'Communion', 'Baptism', 'Fasting'],
  crisisSupport: [
    'Remember that Jesus understands suffering - he experienced it too',
    'Your church community can provide support',
    'Christian counselors combine faith with professional help',
    'God\'s love for you doesn\'t depend on your mental state',
  ],
};

const islam: BeliefSystem = {
  id: 'islam',
  name: 'Islam',
  category: 'religious',
  description: 'Monotheistic faith following the teachings of Prophet Muhammad (PBUH)',
  coreBeliefs: [
    'Allah is Most Merciful and Most Compassionate',
    'Everything happens according to Allah\'s wisdom',
    'Patience (sabr) during hardship brings reward',
    'You are never given more than you can bear',
    'This life is a test, and ease follows hardship',
  ],
  copingStrategies: [
    'Salah (prayer) five times daily for peace',
    'Dhikr (remembrance of Allah)',
    'Reading and reflecting on Quran',
    'Making dua (supplication) for relief',
    'Seeking support from the Muslim community (ummah)',
    'Practicing sabr (patience) and tawakkul (trust in Allah)',
  ],
  comfortingPhrases: [
    '"Verily, with hardship comes ease" - Quran 94:6',
    '"Allah does not burden a soul beyond that it can bear" - Quran 2:286',
    '"And He is with you wherever you are" - Quran 57:4',
    '"Remember Me, and I will remember you" - Quran 2:152',
    '"Indeed, Allah is with the patient" - Quran 2:153',
    '"Do not despair of the mercy of Allah" - Quran 39:53',
  ],
  sacredTexts: ['Quran', 'Hadith', 'Sunnah'],
  practices: ['Salah', 'Fasting (Sawm)', 'Zakat', 'Hajj', 'Dhikr', 'Dua'],
  crisisSupport: [
    'Seeking help is encouraged in Islam - the Prophet (PBUH) sought counsel',
    'Mental health struggles do not diminish your faith',
    'Muslim counselors can provide culturally sensitive support',
    'Your community (ummah) is there to support you',
  ],
};

const judaism: BeliefSystem = {
  id: 'judaism',
  name: 'Judaism',
  category: 'religious',
  description: 'Ancient monotheistic faith of the Jewish people',
  coreBeliefs: [
    'You are created b\'tselem Elohim (in God\'s image)',
    'Pikuach nefesh - preserving life is paramount',
    'Tikkun olam - we can repair the world together',
    'Community (kehillah) provides strength',
    'There is always hope for renewal and redemption',
  ],
  copingStrategies: [
    'Prayer and meditation',
    'Study of Torah and wisdom texts',
    'Observing Shabbat for rest and renewal',
    'Connecting with Jewish community',
    'Speaking with a rabbi or Jewish counselor',
    'Finding meaning through mitzvot (good deeds)',
  ],
  comfortingPhrases: [
    '"The Lord is my shepherd, I shall not want" - Psalm 23:1',
    '"This too shall pass" - Jewish proverb',
    '"Even in the darkest times, we have the right to expect some illumination" - Elie Wiesel',
    '"Where there is life, there is hope"',
    '"Gam zu l\'tovah" - This too is for good',
  ],
  sacredTexts: ['Torah', 'Talmud', 'Psalms', 'Proverbs', 'Mishnah'],
  practices: ['Shabbat', 'Prayer', 'Torah study', 'Tzedakah', 'Holidays'],
  crisisSupport: [
    'Pikuach nefesh means seeking help for mental health is a mitzvah',
    'Jewish Family Services offers culturally sensitive support',
    'Rabbis are trained to provide pastoral counseling',
    'Your community wants to support you',
  ],
};

// ============================================
// EASTERN RELIGIONS & PHILOSOPHIES
// ============================================

const buddhism: BeliefSystem = {
  id: 'buddhism',
  name: 'Buddhism',
  category: 'eastern',
  description: 'Path to enlightenment through understanding suffering and cultivating compassion',
  coreBeliefs: [
    'Suffering is part of existence, but it can end',
    'Impermanence means this moment will pass',
    'You have Buddha nature within you',
    'Compassion for yourself is as important as compassion for others',
    'The present moment is where peace exists',
  ],
  copingStrategies: [
    'Mindfulness meditation',
    'Loving-kindness (metta) practice',
    'Observing thoughts without attachment',
    'Walking meditation',
    'Breathing exercises',
    'Practicing the Middle Way - balance',
  ],
  comfortingPhrases: [
    '"Pain is inevitable, suffering is optional"',
    '"You yourself deserve your love and affection"',
    '"In the end, only three things matter: how much you loved, how gently you lived, and how gracefully you let go"',
    '"The wound is the place where the Light enters you"',
    '"No mud, no lotus" - Thich Nhat Hanh',
  ],
  sacredTexts: ['Dhammapada', 'Heart Sutra', 'Lotus Sutra', 'Tibetan Book of Living and Dying'],
  practices: ['Meditation', 'Mindfulness', 'Chanting', 'Prostrations', 'Retreats'],
  crisisSupport: [
    'Seeking help is an act of self-compassion',
    'Buddhist teachers often integrate psychology with practice',
    'Meditation can complement professional treatment',
    'Sangha (community) provides support',
  ],
};

const hinduism: BeliefSystem = {
  id: 'hinduism',
  name: 'Hinduism',
  category: 'eastern',
  description: 'Ancient tradition with diverse paths to spiritual liberation',
  coreBeliefs: [
    'Atman (your true self) is eternal and divine',
    'Karma shapes experience, but you can create new karma',
    'Multiple paths lead to the divine',
    'You are connected to all of existence',
    'This life is one chapter in your soul\'s journey',
  ],
  copingStrategies: [
    'Yoga and pranayama (breathing)',
    'Meditation and mantra',
    'Puja (worship) and devotion',
    'Reading sacred texts like Bhagavad Gita',
    'Seva (selfless service)',
    'Connecting with a guru or spiritual teacher',
  ],
  comfortingPhrases: [
    '"You have the right to work, but never to the fruit of work" - Bhagavad Gita',
    '"The soul is neither born, and nor does it die" - Bhagavad Gita',
    '"When meditation is mastered, the mind is unwavering like the flame of a candle in a windless place"',
    '"Yoga is the journey of the self, through the self, to the self"',
  ],
  sacredTexts: ['Bhagavad Gita', 'Upanishads', 'Vedas', 'Ramayana', 'Mahabharata'],
  practices: ['Yoga', 'Meditation', 'Puja', 'Mantra', 'Kirtan', 'Ayurveda'],
  crisisSupport: [
    'Seeking help aligns with dharma (right action)',
    'Hindu counselors understand cultural context',
    'Yoga therapy combines ancient wisdom with modern psychology',
    'Temple communities can provide support',
  ],
};

const sikhism: BeliefSystem = {
  id: 'sikhism',
  name: 'Sikhism',
  category: 'eastern',
  description: 'Faith emphasizing equality, service, and devotion to one God',
  coreBeliefs: [
    'Ik Onkar - There is one God who loves all equally',
    'All humans are equal regardless of background',
    'Seva (selfless service) brings peace',
    'Naam Japna - remembering God brings comfort',
    'Honest living and sharing with others',
  ],
  copingStrategies: [
    'Naam Japna (meditation on God\'s name)',
    'Reading and listening to Gurbani',
    'Seva (service) to others',
    'Visiting the Gurdwara for community',
    'Langar (community meal) for connection',
  ],
  comfortingPhrases: [
    '"There is only one breath; all are made of the same clay"',
    '"Even Kings and emperors with heaps of wealth and vast dominion cannot compare with an ant filled with the love of God"',
    '"Where there is forgiveness, there is God"',
  ],
  sacredTexts: ['Guru Granth Sahib'],
  practices: ['Naam Japna', 'Kirat Karni', 'Vand Chakna', 'Seva', 'Langar'],
  crisisSupport: [
    'Seeking help is part of caring for the gift of life',
    'Gurdwara communities provide support',
    'Sikh counselors understand cultural values',
  ],
};

// ============================================
// PHILOSOPHICAL FRAMEWORKS
// ============================================

const existentialism: BeliefSystem = {
  id: 'existentialism',
  name: 'Existentialism',
  category: 'philosophical',
  description: 'Philosophy emphasizing individual existence, freedom, and choice',
  coreBeliefs: [
    'You create your own meaning in life',
    'Authenticity comes from owning your choices',
    'Anxiety can be a catalyst for growth',
    'You are free to reinvent yourself',
    'Existence precedes essence - you define who you are',
  ],
  copingStrategies: [
    'Embrace radical responsibility for your life',
    'Find or create meaning in your experiences',
    'Practice authentic self-expression',
    'Accept uncertainty as part of freedom',
    'Journal to explore your authentic self',
    'Engage with existentialist literature',
  ],
  comfortingPhrases: [
    '"Man is condemned to be free" - Sartre',
    '"He who has a why to live can bear almost any how" - Nietzsche',
    '"In the midst of winter, I found there was, within me, an invincible summer" - Camus',
    '"The only way to deal with an unfree world is to become so absolutely free that your very existence is an act of rebellion" - Camus',
    '"Life has no meaning. Each of us has meaning and we bring it to life" - Joseph Campbell',
  ],
  sacredTexts: ['Being and Nothingness', 'The Stranger', 'Man\'s Search for Meaning', 'Thus Spoke Zarathustra'],
  crisisSupport: [
    'Existential therapy helps find meaning in suffering',
    'Your struggle can become a source of strength',
    'Seeking help is an authentic choice',
  ],
};

const stoicism: BeliefSystem = {
  id: 'stoicism',
  name: 'Stoicism',
  category: 'philosophical',
  description: 'Ancient philosophy focused on virtue, wisdom, and emotional resilience',
  coreBeliefs: [
    'You control your responses, not external events',
    'Virtue is the highest good',
    'Obstacles can become opportunities',
    'The present moment is all we truly have',
    'Reason can guide us through any hardship',
  ],
  copingStrategies: [
    'Practice the dichotomy of control',
    'Negative visualization (premeditatio malorum)',
    'Morning and evening reflection',
    'Journaling like Marcus Aurelius',
    'View challenges as training for virtue',
    'Practice voluntary discomfort',
  ],
  comfortingPhrases: [
    '"The impediment to action advances action. What stands in the way becomes the way" - Marcus Aurelius',
    '"We suffer more in imagination than in reality" - Seneca',
    '"It\'s not what happens to you, but how you react to it that matters" - Epictetus',
    '"Difficulties strengthen the mind, as labor does the body" - Seneca',
    '"You have power over your mind - not outside events. Realize this, and you will find strength" - Marcus Aurelius',
  ],
  sacredTexts: ['Meditations', 'Letters from a Stoic', 'Enchiridion', 'Discourses'],
  crisisSupport: [
    'Seeking help is a rational choice aligned with wisdom',
    'Even Stoics had mentors and teachers',
    'CBT has roots in Stoic philosophy',
  ],
};

const nihilism: BeliefSystem = {
  id: 'nihilism',
  name: 'Nihilism',
  category: 'philosophical',
  description: 'Philosophy questioning inherent meaning, values, and purpose',
  coreBeliefs: [
    'The absence of inherent meaning can be liberating',
    'You are free from imposed expectations',
    'Without cosmic meaning, you can create your own',
    'Nothing is predetermined - you are truly free',
    'The universe\'s indifference means your choices matter more',
  ],
  copingStrategies: [
    'Find freedom in the absence of cosmic judgment',
    'Create personal meaning and values',
    'Embrace the absurd with humor',
    'Focus on immediate, tangible experiences',
    'Build connections that matter to you',
  ],
  comfortingPhrases: [
    '"The literal meaning of life is whatever you\'re doing that prevents you from killing yourself" - Camus',
    '"One must imagine Sisyphus happy" - Camus',
    '"To live is to suffer, to survive is to find some meaning in the suffering" - Nietzsche',
    '"Without music, life would be a mistake" - Nietzsche',
  ],
  crisisSupport: [
    'Even without cosmic meaning, your life has value to those around you',
    'Therapy can help navigate existential questions',
    'Connection with others creates meaning',
  ],
};

const absurdism: BeliefSystem = {
  id: 'absurdism',
  name: 'Absurdism',
  category: 'philosophical',
  description: 'Philosophy embracing the conflict between human desire for meaning and the universe\'s silence',
  coreBeliefs: [
    'The search for meaning in a meaningless universe is absurd - and that\'s okay',
    'We can revolt against absurdity by living fully',
    'Embrace life despite its contradictions',
    'Find joy in the struggle itself',
    'Create meaning through passionate engagement with life',
  ],
  copingStrategies: [
    'Embrace the absurd with defiant joy',
    'Live passionately in the present',
    'Find humor in life\'s contradictions',
    'Create art, love, and connection',
    'Revolt against despair through living fully',
  ],
  comfortingPhrases: [
    '"In the depth of winter, I finally learned that within me there lay an invincible summer" - Camus',
    '"Should I kill myself, or have a cup of coffee?" - Camus',
    '"The struggle itself toward the heights is enough to fill a man\'s heart" - Camus',
    '"Live to the point of tears" - Camus',
  ],
  crisisSupport: [
    'The absurd hero chooses to keep living',
    'Seeking help is an act of revolt against despair',
    'Connection with others is meaningful rebellion',
  ],
};

const solipsism: BeliefSystem = {
  id: 'solipsism',
  name: 'Solipsism',
  category: 'philosophical',
  description: 'Philosophy that only one\'s own mind is sure to exist',
  coreBeliefs: [
    'Your experience is the only certainty',
    'Your perception shapes your reality',
    'The mind has immense creative power',
    'Self-knowledge is the most direct knowledge',
    'Your inner world deserves exploration',
  ],
  copingStrategies: [
    'Deep self-reflection and introspection',
    'Meditation on consciousness',
    'Journaling to explore inner experience',
    'Creative expression of inner states',
    'Mindfulness of present experience',
  ],
  comfortingPhrases: [
    '"I think, therefore I am" - Descartes',
    '"The only true wisdom is in knowing you know nothing" - Socrates',
    '"Know thyself"',
    '"Your vision will become clear only when you can look into your own heart" - Jung',
  ],
  crisisSupport: [
    'Even if only your mind exists, your suffering is real and deserves care',
    'Exploring consciousness can include working with a therapist',
    'Your experience of connection, even if uncertain, has value',
  ],
};

// ============================================
// SECULAR & HUMANIST PERSPECTIVES
// ============================================

const atheism: BeliefSystem = {
  id: 'atheism',
  name: 'Atheism',
  category: 'secular',
  description: 'Worldview without belief in deities, often emphasizing reason and evidence',
  coreBeliefs: [
    'This life is precious because it\'s the only one we have',
    'Human connection and love are real and meaningful',
    'We can create our own purpose and meaning',
    'Ethics come from human empathy and reason',
    'Science and reason can guide us through difficulties',
  ],
  copingStrategies: [
    'Evidence-based approaches like CBT',
    'Building strong human connections',
    'Finding meaning through contribution to others',
    'Engaging with nature and the natural world',
    'Pursuing knowledge and understanding',
    'Creating legacy through positive impact',
  ],
  comfortingPhrases: [
    '"We are all connected; to each other, biologically. To the earth, chemically. To the rest of the universe atomically" - Neil deGrasse Tyson',
    '"The good thing about science is that it\'s true whether or not you believe in it" - Neil deGrasse Tyson',
    '"For small creatures such as we, the vastness is bearable only through love" - Carl Sagan',
    '"We are a way for the cosmos to know itself" - Carl Sagan',
  ],
  crisisSupport: [
    'Secular therapy provides evidence-based support',
    'Human connection is real and healing',
    'Your life has value to those who love you',
  ],
};

const agnosticism: BeliefSystem = {
  id: 'agnosticism',
  name: 'Agnosticism',
  category: 'secular',
  description: 'Position that the existence of deities is unknown or unknowable',
  coreBeliefs: [
    'Uncertainty is honest and okay',
    'We can live meaningful lives without certainty',
    'Questions are as valuable as answers',
    'Human experience has value regardless of cosmic truth',
    'Openness to possibility is a strength',
  ],
  copingStrategies: [
    'Embrace uncertainty as part of the human condition',
    'Focus on what can be known - your experience',
    'Build meaning through relationships and actions',
    'Explore various wisdom traditions without commitment',
    'Practice mindfulness and presence',
  ],
  comfortingPhrases: [
    '"I would rather have questions that can\'t be answered than answers that can\'t be questioned" - Richard Feynman',
    '"The important thing is not to stop questioning" - Einstein',
    '"Doubt is not a pleasant condition, but certainty is absurd" - Voltaire',
  ],
  crisisSupport: [
    'Therapy doesn\'t require belief or disbelief',
    'Your uncertainty doesn\'t diminish your worth',
    'Support is available regardless of metaphysical views',
  ],
};

const secularHumanism: BeliefSystem = {
  id: 'secular-humanism',
  name: 'Secular Humanism',
  category: 'secular',
  description: 'Philosophy emphasizing human reason, ethics, and justice without supernatural beliefs',
  coreBeliefs: [
    'Human beings have inherent dignity and worth',
    'Ethics are based on human welfare and happiness',
    'Reason and compassion guide moral decisions',
    'We are responsible for our own lives and each other',
    'Progress is possible through human effort',
  ],
  copingStrategies: [
    'Connect with humanist communities',
    'Engage in service to others',
    'Use reason to problem-solve',
    'Build meaningful relationships',
    'Contribute to human progress',
    'Practice evidence-based self-care',
  ],
  comfortingPhrases: [
    '"No one is free until we are all free" - Martin Luther King Jr.',
    '"The arc of the moral universe is long, but it bends toward justice"',
    '"We are all in this together"',
    '"Compassion is the basis of morality" - Schopenhauer',
  ],
  crisisSupport: [
    'Humanist counselors provide secular support',
    'Your worth is inherent, not earned',
    'Community support is available',
  ],
};

// ============================================
// SPIRITUAL & MYSTICAL TRADITIONS
// ============================================

const paganism: BeliefSystem = {
  id: 'paganism',
  name: 'Paganism',
  category: 'spiritual',
  description: 'Earth-centered spiritual traditions honoring nature and multiple deities',
  coreBeliefs: [
    'Nature is sacred and healing',
    'The divine manifests in many forms',
    'Cycles of nature mirror our inner cycles',
    'You are connected to the earth and cosmos',
    'Magic and intention can create change',
  ],
  copingStrategies: [
    'Spend time in nature',
    'Work with the cycles of the moon and seasons',
    'Create rituals for healing and release',
    'Connect with deity or spirit guides',
    'Use herbs, crystals, or other tools mindfully',
    'Join a coven or pagan community',
  ],
  comfortingPhrases: [
    '"We are the universe experiencing itself"',
    '"As above, so below; as within, so without"',
    '"The Goddess is alive and magic is afoot"',
    '"From the earth we come, to the earth we return"',
  ],
  practices: ['Ritual', 'Spellwork', 'Sabbat celebrations', 'Nature worship', 'Divination'],
  crisisSupport: [
    'Pagan-friendly therapists exist',
    'Your spiritual path is valid',
    'Community support is available',
  ],
};

const wicca: BeliefSystem = {
  id: 'wicca',
  name: 'Wicca',
  category: 'spiritual',
  description: 'Modern pagan religion emphasizing nature, the Goddess and God, and magical practice',
  coreBeliefs: [
    'The divine feminine and masculine are both honored',
    'Harm none, do what you will',
    'What you send out returns threefold',
    'Nature is sacred and teaches us',
    'Magic is natural and available to all',
  ],
  copingStrategies: [
    'Cast a circle for protection and healing',
    'Work with the elements for balance',
    'Create healing spells or rituals',
    'Connect with the Goddess for comfort',
    'Use the Wheel of the Year for perspective',
    'Journal in a Book of Shadows',
  ],
  comfortingPhrases: [
    '"An it harm none, do what ye will"',
    '"We are the flow, we are the ebb, we are the weavers, we are the web"',
    '"The Goddess is alive and magic is afoot"',
  ],
  practices: ['Circle casting', 'Sabbats', 'Esbats', 'Spellwork', 'Divination'],
  crisisSupport: [
    'Wiccan-friendly counselors understand your path',
    'Your coven or community can support you',
    'Seeking help is wise, not weak',
  ],
};

const newAge: BeliefSystem = {
  id: 'new-age',
  name: 'New Age Spirituality',
  category: 'spiritual',
  description: 'Eclectic spiritual movement emphasizing personal transformation and holistic wellness',
  coreBeliefs: [
    'You are a spiritual being having a human experience',
    'Everything is energy and interconnected',
    'You can manifest positive change',
    'Your soul chose this experience for growth',
    'Higher guidance is available to you',
  ],
  copingStrategies: [
    'Meditation and visualization',
    'Energy healing (Reiki, chakra work)',
    'Crystal healing',
    'Affirmations and manifestation',
    'Connect with spirit guides or angels',
    'Past life exploration for understanding',
  ],
  comfortingPhrases: [
    '"You are exactly where you need to be"',
    '"Everything happens for a reason"',
    '"You are a powerful creator"',
    '"The universe has your back"',
    '"This too shall pass, and you will emerge stronger"',
  ],
  practices: ['Meditation', 'Energy healing', 'Channeling', 'Astrology', 'Tarot'],
  crisisSupport: [
    'Holistic therapists integrate spiritual and psychological approaches',
    'Your spiritual experiences are valid',
    'Grounding practices can complement professional help',
  ],
};

const shamanism: BeliefSystem = {
  id: 'shamanism',
  name: 'Shamanism',
  category: 'indigenous',
  description: 'Ancient spiritual practices involving connection with spirit worlds and nature',
  coreBeliefs: [
    'Everything has spirit and is alive',
    'Healing happens on multiple levels',
    'Ancestors and spirits can guide us',
    'Nature is our greatest teacher',
    'Soul retrieval can restore wholeness',
  ],
  copingStrategies: [
    'Journeying for guidance',
    'Working with power animals',
    'Nature immersion and earth connection',
    'Drumming and rhythmic practices',
    'Ceremony and ritual',
    'Connecting with ancestors',
  ],
  comfortingPhrases: [
    '"We are all related" - Lakota prayer',
    '"The earth does not belong to us; we belong to the earth"',
    '"Walk in beauty"',
    '"Every step is a prayer"',
  ],
  practices: ['Journeying', 'Drumming', 'Ceremony', 'Plant medicine', 'Vision quests'],
  crisisSupport: [
    'Shamanic practitioners can work alongside therapists',
    'Soul retrieval addresses spiritual aspects of trauma',
    'Community ceremony provides support',
  ],
};

const animism: BeliefSystem = {
  id: 'animism',
  name: 'Animism',
  category: 'indigenous',
  description: 'Belief that all things - animals, plants, rocks, rivers - possess a spirit',
  coreBeliefs: [
    'Everything is alive with spirit',
    'We are in relationship with all beings',
    'Nature communicates with us',
    'Respect for all life brings balance',
    'We are never truly alone',
  ],
  copingStrategies: [
    'Spend time with trees, water, or stones',
    'Listen to what nature is telling you',
    'Create offerings of gratitude',
    'Ask plants or animals for guidance',
    'Honor the spirits of place',
  ],
  comfortingPhrases: [
    '"The earth has music for those who listen"',
    '"In every walk with nature, one receives far more than one seeks" - John Muir',
    '"We are all connected in the great web of life"',
  ],
  crisisSupport: [
    'Nature-based therapy aligns with animist views',
    'The natural world can be a source of comfort',
    'Ecotherapy combines nature with professional support',
  ],
};

// ============================================
// ADDITIONAL PHILOSOPHICAL FRAMEWORKS
// ============================================

const taoism: BeliefSystem = {
  id: 'taoism',
  name: 'Taoism',
  category: 'eastern',
  description: 'Chinese philosophy emphasizing living in harmony with the Tao (the Way)',
  coreBeliefs: [
    'Go with the flow of life (wu wei)',
    'Balance and harmony are natural states',
    'Simplicity brings peace',
    'Opposites complement each other (yin/yang)',
    'Nature shows us the way',
  ],
  copingStrategies: [
    'Practice non-action (wu wei) - don\'t force',
    'Tai chi or qigong for balance',
    'Meditation on emptiness',
    'Simplify your life',
    'Observe nature for wisdom',
    'Accept the natural flow of emotions',
  ],
  comfortingPhrases: [
    '"Nature does not hurry, yet everything is accomplished" - Lao Tzu',
    '"When I let go of what I am, I become what I might be" - Lao Tzu',
    '"The journey of a thousand miles begins with a single step" - Lao Tzu',
    '"Be content with what you have; rejoice in the way things are" - Lao Tzu',
  ],
  sacredTexts: ['Tao Te Ching', 'Zhuangzi', 'I Ching'],
  practices: ['Meditation', 'Tai Chi', 'Qigong', 'Feng Shui'],
  crisisSupport: [
    'Taoist principles align well with acceptance-based therapies',
    'Going with the flow includes accepting help',
    'Balance includes caring for yourself',
  ],
};

const confucianism: BeliefSystem = {
  id: 'confucianism',
  name: 'Confucianism',
  category: 'eastern',
  description: 'Chinese philosophy emphasizing ethics, family, and social harmony',
  coreBeliefs: [
    'Relationships and community matter',
    'Self-cultivation leads to harmony',
    'Virtue can be developed through practice',
    'Respect for elders and tradition provides stability',
    'Education and learning are lifelong pursuits',
  ],
  copingStrategies: [
    'Strengthen family and community bonds',
    'Practice self-reflection and improvement',
    'Seek wisdom from elders and teachers',
    'Fulfill your roles with integrity',
    'Study and learn continuously',
  ],
  comfortingPhrases: [
    '"It does not matter how slowly you go as long as you do not stop" - Confucius',
    '"Our greatest glory is not in never falling, but in rising every time we fall" - Confucius',
    '"The man who moves a mountain begins by carrying away small stones" - Confucius',
  ],
  sacredTexts: ['Analects', 'Five Classics'],
  crisisSupport: [
    'Family and community support is central',
    'Seeking guidance from wise teachers is valued',
    'Self-improvement includes mental health care',
  ],
};

const epicureanism: BeliefSystem = {
  id: 'epicureanism',
  name: 'Epicureanism',
  category: 'philosophical',
  description: 'Philosophy focused on achieving happiness through simple pleasures and friendship',
  coreBeliefs: [
    'Pleasure (especially mental tranquility) is the highest good',
    'Simple pleasures are most sustainable',
    'Friendship is essential to happiness',
    'Fear of death is unnecessary',
    'Live modestly and cultivate gratitude',
  ],
  copingStrategies: [
    'Focus on simple, sustainable pleasures',
    'Cultivate deep friendships',
    'Reduce unnecessary desires',
    'Practice gratitude for what you have',
    'Avoid things that cause anxiety',
    'Enjoy the present moment',
  ],
  comfortingPhrases: [
    '"Do not spoil what you have by desiring what you have not"',
    '"Of all the means to ensure happiness, friendship is the most important"',
    '"He who is not satisfied with a little is satisfied with nothing"',
  ],
  crisisSupport: [
    'Friendship and community are healing',
    'Reducing anxiety is a worthy goal',
    'Professional help can restore tranquility',
  ],
};

const universalism: BeliefSystem = {
  id: 'universalism',
  name: 'Unitarian Universalism',
  category: 'spiritual',
  description: 'Liberal religion embracing diverse beliefs and emphasizing individual spiritual journeys',
  coreBeliefs: [
    'Every person has inherent worth and dignity',
    'Truth can be found in many sources',
    'Spiritual growth is a lifelong journey',
    'Justice and compassion guide our actions',
    'All beliefs are welcome and respected',
  ],
  copingStrategies: [
    'Draw from multiple wisdom traditions',
    'Connect with UU community',
    'Engage in social justice work',
    'Explore your own spiritual path',
    'Practice compassion for self and others',
  ],
  comfortingPhrases: [
    '"We need not think alike to love alike"',
    '"Standing on the side of love"',
    '"The arc of the moral universe bends toward justice"',
  ],
  crisisSupport: [
    'UU communities are welcoming and supportive',
    'Pastoral care is available',
    'Your unique path is honored',
  ],
};

// ============================================
// BELIEF SYSTEMS DATABASE
// ============================================

export const beliefSystems: Record<string, BeliefSystem> = {
  // Abrahamic
  christianity,
  islam,
  judaism,
  
  // Eastern
  buddhism,
  hinduism,
  sikhism,
  taoism,
  confucianism,
  
  // Philosophical
  existentialism,
  stoicism,
  nihilism,
  absurdism,
  solipsism,
  epicureanism,
  
  // Secular
  atheism,
  agnosticism,
  'secular-humanism': secularHumanism,
  
  // Spiritual/Mystical
  paganism,
  wicca,
  'new-age': newAge,
  shamanism,
  animism,
  universalism,
};

// ============================================
// HELPER FUNCTIONS
// ============================================

/**
 * Get a belief system by ID
 */
export function getBeliefSystem(id: string): BeliefSystem | undefined {
  return beliefSystems[id.toLowerCase()];
}

/**
 * Get all belief systems in a category
 */
export function getBeliefsByCategory(category: BeliefCategory): BeliefSystem[] {
  return Object.values(beliefSystems).filter(b => b.category === category);
}

/**
 * Get a random comforting phrase from a belief system
 */
export function getComfortingPhrase(beliefId: string): string | undefined {
  const belief = getBeliefSystem(beliefId);
  if (!belief || belief.comfortingPhrases.length === 0) return undefined;
  return belief.comfortingPhrases[Math.floor(Math.random() * belief.comfortingPhrases.length)];
}

/**
 * Get coping strategies for a belief system
 */
export function getCopingStrategies(beliefId: string): string[] {
  const belief = getBeliefSystem(beliefId);
  return belief?.copingStrategies || [];
}

/**
 * Get crisis support information for a belief system
 */
export function getCrisisSupport(beliefId: string): string[] {
  const belief = getBeliefSystem(beliefId);
  return belief?.crisisSupport || [];
}

/**
 * Search belief systems by keyword
 */
export function searchBeliefSystems(query: string): BeliefSystem[] {
  const lowerQuery = query.toLowerCase();
  return Object.values(beliefSystems).filter(belief => 
    belief.name.toLowerCase().includes(lowerQuery) ||
    belief.description.toLowerCase().includes(lowerQuery) ||
    belief.coreBeliefs.some(b => b.toLowerCase().includes(lowerQuery))
  );
}

/**
 * Get all available belief system IDs
 */
export function getAllBeliefIds(): string[] {
  return Object.keys(beliefSystems);
}

/**
 * Get belief-appropriate response modifier
 * Returns guidance for how to frame responses based on belief system
 */
export function getResponseGuidance(beliefId: string): string {
  const belief = getBeliefSystem(beliefId);
  if (!belief) return '';
  
  const guidanceMap: Record<string, string> = {
    christianity: 'Frame responses with hope, God\'s love, and biblical wisdom when appropriate. Acknowledge prayer and faith as valid coping tools.',
    islam: 'Respect Islamic values, reference Quranic wisdom when appropriate, acknowledge the importance of sabr (patience) and tawakkul (trust in Allah).',
    judaism: 'Honor Jewish traditions, reference Torah wisdom when appropriate, emphasize community (kehillah) and the value of life (pikuach nefesh).',
    buddhism: 'Use mindfulness language, acknowledge impermanence, emphasize compassion including self-compassion, reference Buddhist concepts when helpful.',
    hinduism: 'Respect Hindu concepts like karma and dharma, acknowledge yoga and meditation as valid practices, reference Bhagavad Gita wisdom when appropriate.',
    sikhism: 'Honor Sikh values of equality and service, acknowledge Naam Japna and seva as coping tools, respect Gurbani.',
    existentialism: 'Emphasize personal meaning-making, acknowledge freedom and responsibility, validate the search for authenticity.',
    stoicism: 'Use Stoic concepts like the dichotomy of control, emphasize rational approaches, reference Stoic philosophers when helpful.',
    nihilism: 'Acknowledge the freedom in meaninglessness, focus on creating personal value, avoid imposing external meaning.',
    absurdism: 'Embrace the absurd with humor and defiance, emphasize living fully despite uncertainty, reference Camus when appropriate.',
    solipsism: 'Honor the primacy of personal experience, focus on inner exploration, validate subjective reality.',
    atheism: 'Use secular, evidence-based language, emphasize human connection and meaning-making, avoid religious references.',
    agnosticism: 'Respect uncertainty, focus on practical coping, draw from multiple wisdom traditions without commitment.',
    'secular-humanism': 'Emphasize human dignity and reason, focus on ethical living and community, use humanist values.',
    paganism: 'Honor nature-based spirituality, acknowledge ritual and magic as valid, respect diverse pagan paths.',
    wicca: 'Respect Wiccan beliefs, acknowledge the Goddess and God, honor the Rede and magical practice.',
    'new-age': 'Use spiritual language about energy and manifestation, acknowledge holistic approaches, respect personal spiritual experiences.',
    shamanism: 'Honor indigenous wisdom, acknowledge spirit work and nature connection, respect ceremonial practices.',
    animism: 'Acknowledge the spirit in all things, emphasize nature connection, respect relationships with non-human beings.',
    taoism: 'Use concepts of flow and balance, emphasize wu wei and harmony, reference Taoist wisdom.',
    confucianism: 'Honor relationships and social harmony, emphasize self-cultivation and virtue, respect tradition.',
    epicureanism: 'Focus on sustainable pleasure and friendship, emphasize simplicity and gratitude, reduce anxiety.',
    universalism: 'Draw from multiple traditions, honor individual spiritual paths, emphasize inherent worth.',
  };
  
  return guidanceMap[beliefId] || '';
}

/**
 * Get a universal message that works across all beliefs
 */
export function getUniversalComfort(): string {
  const universalMessages = [
    "Whatever you believe, you deserve compassion and support.",
    "Your experience is valid, and you don't have to face this alone.",
    "Reaching out for help is a sign of strength, not weakness.",
    "This moment is difficult, but it will pass.",
    "You matter, and your well-being is important.",
    "It's okay to not be okay right now.",
    "You are more than your current struggle.",
    "There is support available for you, whatever your background.",
  ];
  return universalMessages[Math.floor(Math.random() * universalMessages.length)];
}
