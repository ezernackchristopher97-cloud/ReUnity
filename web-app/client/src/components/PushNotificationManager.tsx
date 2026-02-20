import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { 
  Bell, 
  BellOff, 
  AlertTriangle, 
  Clock, 
  Users,
  CheckCircle,
  XCircle
} from 'lucide-react';

interface NotificationPreferences {
  checkInReminders: boolean;
  crisisAlerts: boolean;
  trustedDeviceAlerts: boolean;
  therapistMessages: boolean;
  dailyAffirmations: boolean;
}

const STORAGE_KEY = 'reunity_notification_prefs';

export default function PushNotificationManager() {
  const [permission, setPermission] = useState<NotificationPermission>('default');
  const [isSubscribed, setIsSubscribed] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [preferences, setPreferences] = useState<NotificationPreferences>({
    checkInReminders: true,
    crisisAlerts: true,
    trustedDeviceAlerts: true,
    therapistMessages: true,
    dailyAffirmations: false,
  });

  useEffect(() => {
    if ('Notification' in window) {
      setPermission(Notification.permission);
    }
    checkSubscription();
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        setPreferences(JSON.parse(saved));
      } catch (e) {
        console.error('Failed to load notification preferences:', e);
      }
    }
  }, []);

  const checkSubscription = async () => {
    if ('serviceWorker' in navigator && 'PushManager' in window) {
      try {
        const registration = await navigator.serviceWorker.ready;
        const subscription = await registration.pushManager.getSubscription();
        setIsSubscribed(!!subscription);
      } catch (e) {
        console.error('Error checking subscription:', e);
      }
    }
  };

  const requestPermission = async () => {
    setIsLoading(true);
    try {
      const result = await Notification.requestPermission();
      setPermission(result);
      if (result === 'granted') {
        await subscribeToNotifications();
      }
    } catch (e) {
      console.error('Error requesting permission:', e);
    }
    setIsLoading(false);
  };

  const subscribeToNotifications = async () => {
    if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
      alert('Push notifications are not supported in this browser');
      return;
    }
    try {
      const registration = await navigator.serviceWorker.register('/sw.js');
      await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.subscribe({
        userVisibleOnly: true,
        applicationServerKey: urlBase64ToUint8Array(
          'BEl62iUYgUivxIkv69yViEuiBIa-Ib9-SkvMeAtA3LFgDzkrxZJjSgSnfckjBJuBkr3qBUYIHBQFLXYp5Nksh8U'
        )
      });
      await saveSubscription(subscription);
      setIsSubscribed(true);
    } catch (e) {
      console.error('Error subscribing to notifications:', e);
    }
  };

  const unsubscribeFromNotifications = async () => {
    try {
      const registration = await navigator.serviceWorker.ready;
      const subscription = await registration.pushManager.getSubscription();
      if (subscription) {
        await subscription.unsubscribe();
        setIsSubscribed(false);
      }
    } catch (e) {
      console.error('Error unsubscribing:', e);
    }
  };

  const saveSubscription = async (subscription: PushSubscription) => {
    console.log('Subscription:', JSON.stringify(subscription));
    localStorage.setItem('reunity_push_subscription', JSON.stringify(subscription));
  };

  const updatePreference = (key: keyof NotificationPreferences, value: boolean) => {
    const newPrefs = { ...preferences, [key]: value };
    setPreferences(newPrefs);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newPrefs));
  };

  const sendTestNotification = () => {
    if (permission === 'granted') {
      new Notification('ReUnity Test', {
        body: 'Push notifications are working!',
        icon: '/reop-logo.png',
        tag: 'test-notification'
      });
    }
  };

  function urlBase64ToUint8Array(base64String: string) {
    const padding = '='.repeat((4 - base64String.length % 4) % 4);
    const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/');
    const rawData = window.atob(base64);
    const outputArray = new Uint8Array(rawData.length);
    for (let i = 0; i < rawData.length; ++i) {
      outputArray[i] = rawData.charCodeAt(i);
    }
    return outputArray;
  }

  const getPermissionStatus = () => {
    switch (permission) {
      case 'granted':
        return { icon: <CheckCircle className="w-5 h-5 text-emerald-400" />, text: 'Enabled', color: 'text-emerald-400' };
      case 'denied':
        return { icon: <XCircle className="w-5 h-5 text-red-400" />, text: 'Blocked', color: 'text-red-400' };
      default:
        return { icon: <Bell className="w-5 h-5 text-amber-400" />, text: 'Not Set', color: 'text-amber-400' };
    }
  };

  const status = getPermissionStatus();

  return (
    <Card className="bg-zinc-900/50 border-zinc-800">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-500/20">
              <Bell className="w-5 h-5 text-purple-400" />
            </div>
            <div>
              <CardTitle className="text-lg">Push Notifications</CardTitle>
              <p className="text-xs text-zinc-500">Stay connected even when the app is closed</p>
            </div>
          </div>
          <div className={`flex items-center gap-2 ${status.color}`}>
            {status.icon}
            <span className="text-sm">{status.text}</span>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {permission !== 'granted' ? (
          <div className="p-4 rounded-lg bg-purple-500/10 border border-purple-500/20">
            <p className="text-sm text-zinc-300 mb-3">
              Enable push notifications to receive check-in reminders and crisis alerts even when you're not using the app.
            </p>
            <Button onClick={requestPermission} disabled={isLoading || permission === 'denied'} className="w-full bg-purple-600 hover:bg-purple-700">
              {isLoading ? 'Enabling...' : permission === 'denied' ? 'Blocked in Browser Settings' : 'Enable Notifications'}
            </Button>
            {permission === 'denied' && (
              <p className="text-xs text-zinc-500 mt-2">You've blocked notifications. Please enable them in your browser settings.</p>
            )}
          </div>
        ) : (
          <>
            <div className="space-y-4">
              <h4 className="text-sm font-medium text-zinc-400">Notification Types</h4>
              <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-3">
                  <Clock className="w-5 h-5 text-blue-400" />
                  <div><Label className="text-sm font-medium">Check-in Reminders</Label><p className="text-xs text-zinc-500">Scheduled wellness check-ins</p></div>
                </div>
                <Switch checked={preferences.checkInReminders} onCheckedChange={(checked) => updatePreference('checkInReminders', checked)} />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-400" />
                  <div><Label className="text-sm font-medium">Crisis Alerts</Label><p className="text-xs text-zinc-500">Urgent alerts from trusted contacts</p></div>
                </div>
                <Switch checked={preferences.crisisAlerts} onCheckedChange={(checked) => updatePreference('crisisAlerts', checked)} />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-3">
                  <Users className="w-5 h-5 text-emerald-400" />
                  <div><Label className="text-sm font-medium">Trusted Device Alerts</Label><p className="text-xs text-zinc-500">Updates from paired family devices</p></div>
                </div>
                <Switch checked={preferences.trustedDeviceAlerts} onCheckedChange={(checked) => updatePreference('trustedDeviceAlerts', checked)} />
              </div>
              <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-3">
                  <Bell className="w-5 h-5 text-purple-400" />
                  <div><Label className="text-sm font-medium">Therapist Messages</Label><p className="text-xs text-zinc-500">Messages from your therapist</p></div>
                </div>
                <Switch checked={preferences.therapistMessages} onCheckedChange={(checked) => updatePreference('therapistMessages', checked)} />
              </div>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" className="flex-1 border-zinc-700" onClick={sendTestNotification}>Send Test</Button>
              <Button variant="outline" className="flex-1 border-red-500/30 text-red-400 hover:bg-red-500/10" onClick={unsubscribeFromNotifications}>
                <BellOff className="w-4 h-4 mr-2" />Disable All
              </Button>
            </div>
          </>
        )}
        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <p className="text-xs text-zinc-400"><strong className="text-blue-300">Privacy:</strong> Notification content is encrypted and never stored on our servers.</p>
        </div>
      </CardContent>
    </Card>
  );
}
