import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Sun, Heart, Sparkles, RefreshCw, Bell, Volume2, VolumeX, Star, Moon, Cloud, Zap } from 'lucide-react';

// Comprehensive affirmations database organized by mood/condition
const affirmationsDatabase = {
  // For low mood / depression
  depression: [
    "Even on the darkest days, you are worthy of light and love.",
    "Your feelings are valid, but they don't define your worth.",
    "Small steps forward are still progress. You're doing better than you think.",
    "The heaviness you feel today will not last forever.",
    "You have survived every difficult day so far. You will survive this one too.",
    "It's okay to rest. Healing isn't linear.",
    "Your existence matters, even when you can't feel it.",
    "You are not a burden. You are a human being going through a hard time.",
    "Tomorrow holds possibilities that today cannot see.",
    "You deserve compassion, especially from yourself."
  ],
  // For anxiety
  anxiety: [
    "This moment of anxiety will pass. You have weathered storms before.",
    "Your worries do not control you. You are stronger than your fears.",
    "Breathe in calm, breathe out tension. You are safe right now.",
    "Uncertainty is uncomfortable, but you can handle it.",
    "Your nervous system is trying to protect you. Thank it, then release it.",
    "You don't have to have all the answers today.",
    "Ground yourself in this moment. Right now, you are okay.",
    "Fear is a feeling, not a fact. You can move through it.",
    "You've faced anxiety before and come out the other side.",
    "Peace is available to you, one breath at a time."
  ],
  // For trauma/PTSD
  trauma: [
    "What happened to you was not your fault.",
    "You are more than your past. Your story isn't over.",
    "Healing from trauma takes time, and that's okay.",
    "You survived. That takes incredible strength.",
    "Your triggers don't define you. You are learning to manage them.",
    "Safety is possible. You are building it every day.",
    "The past cannot hurt you in this present moment.",
    "You deserve to take up space and feel safe.",
    "Your body held onto pain to protect you. Now you can release it.",
    "Recovery isn't about forgetting—it's about reclaiming your life."
  ],
  // For grief/loss
  grief: [
    "Grief is love with nowhere to go. It's okay to feel it deeply.",
    "There is no right way to grieve. Your process is valid.",
    "The love you shared doesn't end. It transforms.",
    "Missing someone is a testament to the bond you had.",
    "You can hold grief and hope in the same heart.",
    "Take all the time you need. Grief has no deadline.",
    "It's okay to have moments of joy even while grieving.",
    "Your loved one's impact lives on through you.",
    "Grief comes in waves. You can ride each one.",
    "You are not alone in your sorrow, even when it feels that way."
  ],
  // For isolation/loneliness
  isolation: [
    "You are not as alone as you feel right now.",
    "Connection is possible, even when it feels far away.",
    "Reaching out takes courage. You have that courage within you.",
    "Your presence matters to more people than you realize.",
    "Isolation is a feeling, not a permanent state.",
    "You deserve meaningful connections, and they are possible.",
    "Today, try one small act of connection. It can make a difference.",
    "The world needs your unique light, even when you can't see it.",
    "Loneliness is painful, but it won't last forever.",
    "You are worthy of belonging and being seen."
  ],
  // For self-worth/identity
  selfWorth: [
    "You are enough, exactly as you are right now.",
    "Your worth is not determined by your productivity.",
    "You deserve love and respect, including from yourself.",
    "Imperfection is part of being human. You are beautifully human.",
    "You don't have to earn your place in this world.",
    "Your struggles don't diminish your value.",
    "You are worthy of good things happening to you.",
    "Comparing yourself to others steals your joy. You are uniquely you.",
    "Your journey is valid, even if it looks different from others.",
    "You matter. Your feelings matter. Your life matters."
  ],
  // For recovery/healing
  recovery: [
    "Every day you choose recovery, you choose yourself.",
    "Setbacks are part of the journey, not the end of it.",
    "You are building a life worth living, one day at a time.",
    "Progress isn't always visible, but it's happening.",
    "You have the strength to keep going.",
    "Recovery is an act of courage. You are brave.",
    "Each small victory adds up to transformation.",
    "You are rewriting your story with every healthy choice.",
    "The work you're doing now is creating your future.",
    "You deserve the life you're fighting for."
  ],
  // For morning motivation
  morning: [
    "Today is a new opportunity to be kind to yourself.",
    "You woke up today. That's already an accomplishment.",
    "This day holds possibilities you haven't imagined yet.",
    "You have the power to make today meaningful.",
    "Start where you are. Use what you have. Do what you can.",
    "Today, focus on progress, not perfection.",
    "You are capable of handling whatever today brings.",
    "This morning is a fresh start. Embrace it gently.",
    "Your energy today can be whatever you need it to be.",
    "Today, choose one thing that brings you peace."
  ],
  // For evening/night
  evening: [
    "You made it through today. That's worth celebrating.",
    "Release the day's worries. Tomorrow is a new start.",
    "You did your best today, and that's enough.",
    "Rest is productive. Your body and mind need it.",
    "Let go of what you couldn't control today.",
    "Sleep will restore you. Trust in the healing power of rest.",
    "Tomorrow's challenges can wait. Tonight, be at peace.",
    "You deserve a restful night. Allow yourself to have it.",
    "The night is for healing. Let your mind and body recover.",
    "You are safe. You can let your guard down now."
  ],
  // General positive affirmations
  general: [
    "You are resilient. You have proven this many times.",
    "Your feelings are valid, and so is your need for support.",
    "You are doing the best you can with what you have.",
    "Growth happens outside your comfort zone. You're growing.",
    "You are allowed to take up space in this world.",
    "Your story matters. Your voice matters.",
    "You are not defined by your worst moments.",
    "Change is possible. You are proof of that.",
    "You have survived 100% of your worst days.",
    "You are worthy of all the good things life has to offer."
  ]
};

// Time-based greeting
const getTimeBasedGreeting = () => {
  const hour = new Date().getHours();
  if (hour < 12) return { greeting: "Good morning", icon: Sun, period: 'morning' };
  if (hour < 17) return { greeting: "Good afternoon", icon: Cloud, period: 'afternoon' };
  if (hour < 21) return { greeting: "Good evening", icon: Moon, period: 'evening' };
  return { greeting: "Good night", icon: Star, period: 'night' };
};

// Get affirmations based on mood patterns
const getPersonalizedAffirmations = (recentMoods: number[], conditions: string[]): string[] => {
  const affirmations: string[] = [];
  const timeInfo = getTimeBasedGreeting();
  
  // Add time-based affirmations
  if (timeInfo.period === 'morning') {
    affirmations.push(...affirmationsDatabase.morning.slice(0, 2));
  } else if (timeInfo.period === 'evening' || timeInfo.period === 'night') {
    affirmations.push(...affirmationsDatabase.evening.slice(0, 2));
  }
  
  // Analyze mood patterns
  const avgMood = recentMoods.length > 0 
    ? recentMoods.reduce((a, b) => a + b, 0) / recentMoods.length 
    : 3;
  
  // Low mood pattern detected
  if (avgMood < 2.5) {
    affirmations.push(...affirmationsDatabase.depression.slice(0, 3));
  }
  
  // Add condition-specific affirmations
  conditions.forEach(condition => {
    const conditionKey = condition.toLowerCase();
    if (conditionKey.includes('anxiety') || conditionKey.includes('anxious')) {
      affirmations.push(...affirmationsDatabase.anxiety.slice(0, 2));
    }
    if (conditionKey.includes('trauma') || conditionKey.includes('ptsd')) {
      affirmations.push(...affirmationsDatabase.trauma.slice(0, 2));
    }
    if (conditionKey.includes('grief') || conditionKey.includes('loss')) {
      affirmations.push(...affirmationsDatabase.grief.slice(0, 2));
    }
    if (conditionKey.includes('isolat') || conditionKey.includes('lonely')) {
      affirmations.push(...affirmationsDatabase.isolation.slice(0, 2));
    }
  });
  
  // Add general affirmations
  affirmations.push(...affirmationsDatabase.general.slice(0, 2));
  
  // Add self-worth affirmations
  affirmations.push(...affirmationsDatabase.selfWorth.slice(0, 2));
  
  // Add recovery affirmations
  affirmations.push(...affirmationsDatabase.recovery.slice(0, 2));
  
  // Shuffle and return unique affirmations
  const unique = Array.from(new Set(affirmations));
  return unique.sort(() => Math.random() - 0.5).slice(0, 5);
};

interface DailyAffirmationsEnhancedProps {
  recentMoods?: number[];
  detectedConditions?: string[];
}

export function DailyAffirmationsEnhanced({ 
  recentMoods = [], 
  detectedConditions = [] 
}: DailyAffirmationsEnhancedProps) {
  const [affirmations, setAffirmations] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [favoriteAffirmations, setFavoriteAffirmations] = useState<string[]>([]);
  
  const timeInfo = getTimeBasedGreeting();
  const TimeIcon = timeInfo.icon;
  
  useEffect(() => {
    // Load favorites from localStorage
    const saved = localStorage.getItem('reunity_favorite_affirmations');
    if (saved) {
      setFavoriteAffirmations(JSON.parse(saved));
    }
    
    // Check notification settings
    const notifSetting = localStorage.getItem('reunity_affirmation_notifications');
    setNotificationsEnabled(notifSetting === 'true');
    
    // Generate personalized affirmations
    refreshAffirmations();
  }, [recentMoods, detectedConditions]);
  
  const refreshAffirmations = () => {
    const personalized = getPersonalizedAffirmations(recentMoods, detectedConditions);
    setAffirmations(personalized);
    setCurrentIndex(0);
  };
  
  const nextAffirmation = () => {
    setCurrentIndex((prev) => (prev + 1) % affirmations.length);
  };
  
  const speakAffirmation = () => {
    if ('speechSynthesis' in window) {
      if (isSpeaking) {
        window.speechSynthesis.cancel();
        setIsSpeaking(false);
        return;
      }
      
      const utterance = new SpeechSynthesisUtterance(affirmations[currentIndex]);
      utterance.rate = 0.85;
      utterance.pitch = 1.0;
      
      // Try to use a gentle female voice
      const voices = window.speechSynthesis.getVoices();
      const gentleVoice = voices.find(v => 
        v.name.toLowerCase().includes('samantha') || 
        v.name.toLowerCase().includes('karen') ||
        v.name.toLowerCase().includes('female')
      );
      if (gentleVoice) {
        utterance.voice = gentleVoice;
      }
      
      utterance.onend = () => setIsSpeaking(false);
      utterance.onerror = () => setIsSpeaking(false);
      
      setIsSpeaking(true);
      window.speechSynthesis.speak(utterance);
    }
  };
  
  const toggleFavorite = (affirmation: string) => {
    let updated: string[];
    if (favoriteAffirmations.includes(affirmation)) {
      updated = favoriteAffirmations.filter(a => a !== affirmation);
    } else {
      updated = [...favoriteAffirmations, affirmation];
    }
    setFavoriteAffirmations(updated);
    localStorage.setItem('reunity_favorite_affirmations', JSON.stringify(updated));
  };
  
  const toggleNotifications = async () => {
    if (!notificationsEnabled) {
      if ('Notification' in window) {
        const permission = await Notification.requestPermission();
        if (permission === 'granted') {
          setNotificationsEnabled(true);
          localStorage.setItem('reunity_affirmation_notifications', 'true');
          // Schedule morning notification
          scheduleAffirmationNotification();
        }
      }
    } else {
      setNotificationsEnabled(false);
      localStorage.setItem('reunity_affirmation_notifications', 'false');
    }
  };
  
  const scheduleAffirmationNotification = () => {
    // This would typically use a service worker for background notifications
    // For now, we'll just show a confirmation
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Daily Affirmations Enabled', {
        body: 'You\'ll receive a personalized affirmation each morning at 8 AM.',
        icon: '/icons/icon-192.png'
      });
    }
  };
  
  if (affirmations.length === 0) {
    return null;
  }
  
  const currentAffirmation = affirmations[currentIndex];
  const isFavorite = favoriteAffirmations.includes(currentAffirmation);
  
  return (
    <Card className="bg-gradient-to-br from-amber-900/30 to-orange-900/30 border-amber-700/50">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center justify-between text-amber-200">
          <div className="flex items-center gap-2">
            <TimeIcon className="w-5 h-5" />
            <span>{timeInfo.greeting}</span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleNotifications}
              className={notificationsEnabled ? 'text-amber-400' : 'text-gray-400'}
              title={notificationsEnabled ? 'Notifications enabled' : 'Enable daily notifications'}
            >
              <Bell className="w-4 h-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={refreshAffirmations}
              className="text-amber-400 hover:text-amber-300"
              title="Get new affirmations"
            >
              <RefreshCw className="w-4 h-4" />
            </Button>
          </div>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative">
          <div className="absolute -top-2 -left-2 text-amber-500/30">
            <Sparkles className="w-8 h-8" />
          </div>
          <p className="text-lg text-amber-100 italic pl-6 pr-4 py-2 min-h-[80px] flex items-center">
            "{currentAffirmation}"
          </p>
        </div>
        
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={speakAffirmation}
              className={isSpeaking ? 'text-amber-400' : 'text-gray-400 hover:text-amber-400'}
              title={isSpeaking ? 'Stop speaking' : 'Read aloud'}
            >
              {isSpeaking ? <VolumeX className="w-4 h-4" /> : <Volume2 className="w-4 h-4" />}
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => toggleFavorite(currentAffirmation)}
              className={isFavorite ? 'text-red-400' : 'text-gray-400 hover:text-red-400'}
              title={isFavorite ? 'Remove from favorites' : 'Add to favorites'}
            >
              <Heart className={`w-4 h-4 ${isFavorite ? 'fill-current' : ''}`} />
            </Button>
          </div>
          
          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-400">
              {currentIndex + 1} / {affirmations.length}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={nextAffirmation}
              className="border-amber-700 text-amber-200 hover:bg-amber-900/50"
            >
              Next <Zap className="w-3 h-3 ml-1" />
            </Button>
          </div>
        </div>
        
        {favoriteAffirmations.length > 0 && (
          <div className="pt-2 border-t border-amber-700/30">
            <p className="text-xs text-amber-400/70 flex items-center gap-1">
              <Heart className="w-3 h-3 fill-current" />
              {favoriteAffirmations.length} favorite{favoriteAffirmations.length !== 1 ? 's' : ''} saved
            </p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default DailyAffirmationsEnhanced;
