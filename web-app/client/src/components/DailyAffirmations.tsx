import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Sparkles, RefreshCw, Bell, Heart, Copy, Check } from 'lucide-react';
import { toast } from 'sonner';

const AFFIRMATIONS = [
  // Self-worth
  "I am worthy of love and respect, exactly as I am.",
  "My feelings are valid and deserve to be acknowledged.",
  "I am doing the best I can with what I have.",
  "I deserve to take up space in this world.",
  "My worth is not determined by my productivity.",
  
  // Healing
  "Healing is not linear, and that's okay.",
  "I am allowed to set boundaries that protect my peace.",
  "Every small step forward is still progress.",
  "I am more than my trauma.",
  "I am learning to be gentle with myself.",
  
  // Strength
  "I have survived 100% of my worst days.",
  "I am stronger than I give myself credit for.",
  "My struggles do not define my future.",
  "I am capable of handling whatever comes my way.",
  "I have the power to create positive change in my life.",
  
  // Present moment
  "Right now, in this moment, I am safe.",
  "I choose to focus on what I can control.",
  "This feeling is temporary. It will pass.",
  "I am grounded in the present moment.",
  "I release what no longer serves me.",
  
  // Self-compassion
  "I forgive myself for past mistakes.",
  "I am allowed to ask for help.",
  "I deserve rest and recovery.",
  "I am enough, just as I am.",
  "I treat myself with the kindness I would show a friend.",
  
  // Hope
  "Better days are ahead of me.",
  "I am not alone in my struggles.",
  "I choose hope over fear.",
  "New beginnings are always possible.",
  "I am worthy of happiness and peace.",
  
  // Recovery specific
  "My recovery is my own journey.",
  "I am proud of how far I've come.",
  "I am learning to trust myself again.",
  "I am breaking cycles that no longer serve me.",
  "Every day I am becoming more myself.",
  
  // Grounding
  "I am connected to the present moment.",
  "My breath anchors me to safety.",
  "I am here. I am real. I am okay.",
  "I can find calm within the storm.",
  "I am learning to regulate my nervous system.",
];

const CATEGORIES = {
  'self-worth': AFFIRMATIONS.slice(0, 5),
  'healing': AFFIRMATIONS.slice(5, 10),
  'strength': AFFIRMATIONS.slice(10, 15),
  'present': AFFIRMATIONS.slice(15, 20),
  'compassion': AFFIRMATIONS.slice(20, 25),
  'hope': AFFIRMATIONS.slice(25, 30),
  'recovery': AFFIRMATIONS.slice(30, 35),
  'grounding': AFFIRMATIONS.slice(35, 40),
};

export function DailyAffirmations() {
  const [currentAffirmation, setCurrentAffirmation] = useState('');
  const [notificationsEnabled, setNotificationsEnabled] = useState(false);
  const [copied, setCopied] = useState(false);
  const [favorites, setFavorites] = useState<string[]>([]);
  const [showFavorites, setShowFavorites] = useState(false);

  useEffect(() => {
    // Load saved state
    const savedNotifications = localStorage.getItem('reunity_affirmation_notifications');
    const savedFavorites = localStorage.getItem('reunity_affirmation_favorites');
    
    if (savedNotifications === 'true') {
      setNotificationsEnabled(true);
    }
    
    if (savedFavorites) {
      setFavorites(JSON.parse(savedFavorites));
    }

    // Get daily affirmation based on date
    const today = new Date().toDateString();
    const savedDaily = localStorage.getItem('reunity_daily_affirmation');
    const savedDate = localStorage.getItem('reunity_daily_affirmation_date');

    if (savedDate === today && savedDaily) {
      setCurrentAffirmation(savedDaily);
    } else {
      generateNewAffirmation();
    }
  }, []);

  const generateNewAffirmation = () => {
    const randomIndex = Math.floor(Math.random() * AFFIRMATIONS.length);
    const newAffirmation = AFFIRMATIONS[randomIndex];
    setCurrentAffirmation(newAffirmation);
    
    // Save as daily affirmation
    localStorage.setItem('reunity_daily_affirmation', newAffirmation);
    localStorage.setItem('reunity_daily_affirmation_date', new Date().toDateString());
  };

  const toggleNotifications = async () => {
    if (!notificationsEnabled) {
      // Request permission
      if ('Notification' in window) {
        const permission = await Notification.requestPermission();
        if (permission === 'granted') {
          setNotificationsEnabled(true);
          localStorage.setItem('reunity_affirmation_notifications', 'true');
          toast.success('Daily affirmation notifications enabled');
          
          // Schedule notification (in a real app, this would use a service worker)
          scheduleNotification();
        } else {
          toast.error('Notification permission denied');
        }
      }
    } else {
      setNotificationsEnabled(false);
      localStorage.setItem('reunity_affirmation_notifications', 'false');
      toast.info('Daily affirmation notifications disabled');
    }
  };

  const scheduleNotification = () => {
    // In a real app, this would use a service worker for background notifications
    // For now, we'll show a notification if the user has the app open
    const now = new Date();
    const scheduledTime = new Date();
    scheduledTime.setHours(8, 0, 0, 0); // 8 AM
    
    if (now > scheduledTime) {
      scheduledTime.setDate(scheduledTime.getDate() + 1);
    }
    
    const timeUntilNotification = scheduledTime.getTime() - now.getTime();
    
    setTimeout(() => {
      if (notificationsEnabled && 'Notification' in window) {
        new Notification('ReUnity Daily Affirmation', {
          body: currentAffirmation,
          icon: '/icon-192.png',
        });
      }
    }, timeUntilNotification);
  };

  const copyAffirmation = () => {
    navigator.clipboard.writeText(currentAffirmation);
    setCopied(true);
    toast.success('Affirmation copied to clipboard');
    setTimeout(() => setCopied(false), 2000);
  };

  const toggleFavorite = () => {
    let newFavorites: string[];
    if (favorites.includes(currentAffirmation)) {
      newFavorites = favorites.filter(f => f !== currentAffirmation);
      toast.info('Removed from favorites');
    } else {
      newFavorites = [...favorites, currentAffirmation];
      toast.success('Added to favorites');
    }
    setFavorites(newFavorites);
    localStorage.setItem('reunity_affirmation_favorites', JSON.stringify(newFavorites));
  };

  const isFavorite = favorites.includes(currentAffirmation);

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-emerald-400" />
            Daily Affirmation
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowFavorites(!showFavorites)}
              className={showFavorites ? 'text-emerald-400' : ''}
            >
              <Heart className={`w-4 h-4 ${showFavorites ? 'fill-current' : ''}`} />
            </Button>
            <div className="flex items-center gap-2">
              <Bell className={`w-4 h-4 ${notificationsEnabled ? 'text-emerald-400' : 'text-zinc-500'}`} />
              <Switch
                checked={notificationsEnabled}
                onCheckedChange={toggleNotifications}
              />
            </div>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {showFavorites ? (
          <div className="space-y-3">
            <p className="text-sm text-zinc-400">Your favorite affirmations:</p>
            {favorites.length === 0 ? (
              <p className="text-zinc-500 text-sm italic">No favorites yet. Click the heart to save affirmations.</p>
            ) : (
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {favorites.map((fav, i) => (
                  <div
                    key={i}
                    className="p-2 bg-zinc-800/50 rounded-lg text-sm text-zinc-300 cursor-pointer hover:bg-zinc-800"
                    onClick={() => {
                      setCurrentAffirmation(fav);
                      setShowFavorites(false);
                    }}
                  >
                    "{fav}"
                  </div>
                ))}
              </div>
            )}
          </div>
        ) : (
          <>
            <div className="bg-gradient-to-br from-emerald-900/30 to-teal-900/30 rounded-xl p-6 mb-4 border border-emerald-800/30">
              <p className="text-lg text-center text-white font-medium leading-relaxed">
                "{currentAffirmation}"
              </p>
            </div>

            <div className="flex items-center justify-center gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={generateNewAffirmation}
                className="gap-2"
              >
                <RefreshCw className="w-4 h-4" />
                New Affirmation
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={copyAffirmation}
                className="gap-2"
              >
                {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
                {copied ? 'Copied' : 'Copy'}
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={toggleFavorite}
                className={`gap-2 ${isFavorite ? 'text-red-400 border-red-400/50' : ''}`}
              >
                <Heart className={`w-4 h-4 ${isFavorite ? 'fill-current' : ''}`} />
              </Button>
            </div>

            <p className="text-xs text-zinc-500 text-center mt-4">
              {notificationsEnabled 
                ? 'You\'ll receive a new affirmation every morning at 8 AM'
                : 'Enable notifications to receive daily affirmations'}
            </p>
          </>
        )}
      </CardContent>
    </Card>
  );
}

export default DailyAffirmations;
