import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { 
  Heart, 
  Shield, 
  Bell, 
  Eye, 
  EyeOff, 
  Activity,
  Calendar,
  Clock,
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Minus,
  Phone,
  MessageSquare,
  Settings,
  UserPlus,
  Lock,
  Unlock,
  MapPin,
  Thermometer,
  Moon,
  Flame,
  BookOpen,
  Wind
} from 'lucide-react';

interface LovedOne {
  id: string;
  name: string;
  relationship: string;
  linkedDate: string;
  lastActive: string;
  privacySettings: PrivacySettings;
  wellnessData: WellnessData;
  alerts: Alert[];
}

interface PrivacySettings {
  shareCheckIns: boolean;
  shareMoodData: boolean;
  shareLocation: boolean;
  shareCrisisAlerts: boolean;
  shareJournalSummary: boolean;
  shareStreaks: boolean;
  shareSleepData: boolean;
}

interface WellnessData {
  currentMood: 'great' | 'good' | 'okay' | 'struggling' | 'crisis';
  moodTrend: 'improving' | 'stable' | 'declining';
  lastCheckIn: string;
  checkInStreak: number;
  journalStreak: number;
  sleepQuality: number; // 0-100
  entropyScore: number; // 0-100
  riskLevel: 'low' | 'moderate' | 'elevated' | 'high';
}

interface Alert {
  id: string;
  type: 'crisis' | 'missed_checkin' | 'mood_decline' | 'high_risk' | 'location';
  message: string;
  timestamp: string;
  isRead: boolean;
  severity: 'info' | 'warning' | 'critical';
}

interface CaregiverDashboardProps {
  caregiverId?: string;
  caregiverName?: string;
}

const MOOD_COLORS = {
  great: 'bg-emerald-500/20 text-emerald-400',
  good: 'bg-green-500/20 text-green-400',
  okay: 'bg-yellow-500/20 text-yellow-400',
  struggling: 'bg-orange-500/20 text-orange-400',
  crisis: 'bg-red-500/20 text-red-400',
};

const MOOD_LABELS = {
  great: 'Feeling Great',
  good: 'Doing Good',
  okay: 'Okay',
  struggling: 'Struggling',
  crisis: 'In Crisis',
};

const RISK_COLORS = {
  low: 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30',
  moderate: 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30',
  elevated: 'bg-orange-500/20 text-orange-400 border-orange-500/30',
  high: 'bg-red-500/20 text-red-400 border-red-500/30',
};

export default function CaregiverDashboard({ caregiverId = 'caregiver1', caregiverName = 'Caregiver' }: CaregiverDashboardProps) {
  const [lovedOnes, setLovedOnes] = useState<LovedOne[]>([]);
  const [selectedPerson, setSelectedPerson] = useState<LovedOne | null>(null);
  const [showAddDialog, setShowAddDialog] = useState(false);
  const [linkCode, setLinkCode] = useState('');
  const [unreadAlerts, setUnreadAlerts] = useState(0);

  useEffect(() => {
    loadCaregiverData();
  }, []);

  useEffect(() => {
    const count = lovedOnes.reduce((sum, person) => 
      sum + person.alerts.filter(a => !a.isRead).length, 0
    );
    setUnreadAlerts(count);
  }, [lovedOnes]);

  const loadCaregiverData = () => {
    const stored = localStorage.getItem(`reunity_caregiver_${caregiverId}`);
    if (stored) {
      setLovedOnes(JSON.parse(stored));
    } else {
      // Demo data
      const demoData: LovedOne[] = [
        {
          id: '1',
          name: 'Alex',
          relationship: 'Child',
          linkedDate: new Date(Date.now() - 2592000000).toISOString(),
          lastActive: new Date(Date.now() - 3600000).toISOString(),
          privacySettings: {
            shareCheckIns: true,
            shareMoodData: true,
            shareLocation: false,
            shareCrisisAlerts: true,
            shareJournalSummary: true,
            shareStreaks: true,
            shareSleepData: true,
          },
          wellnessData: {
            currentMood: 'good',
            moodTrend: 'improving',
            lastCheckIn: new Date(Date.now() - 3600000).toISOString(),
            checkInStreak: 7,
            journalStreak: 3,
            sleepQuality: 72,
            entropyScore: 35,
            riskLevel: 'low',
          },
          alerts: [
            {
              id: '1',
              type: 'mood_decline',
              message: 'Mood has been declining over the past 3 days',
              timestamp: new Date(Date.now() - 86400000).toISOString(),
              isRead: true,
              severity: 'warning',
            },
          ],
        },
        {
          id: '2',
          name: 'Jordan',
          relationship: 'Sibling',
          linkedDate: new Date(Date.now() - 5184000000).toISOString(),
          lastActive: new Date(Date.now() - 7200000).toISOString(),
          privacySettings: {
            shareCheckIns: true,
            shareMoodData: true,
            shareLocation: true,
            shareCrisisAlerts: true,
            shareJournalSummary: false,
            shareStreaks: true,
            shareSleepData: false,
          },
          wellnessData: {
            currentMood: 'struggling',
            moodTrend: 'declining',
            lastCheckIn: new Date(Date.now() - 7200000).toISOString(),
            checkInStreak: 2,
            journalStreak: 0,
            sleepQuality: 45,
            entropyScore: 68,
            riskLevel: 'elevated',
          },
          alerts: [
            {
              id: '2',
              type: 'high_risk',
              message: 'Elevated risk detected based on recent patterns',
              timestamp: new Date(Date.now() - 3600000).toISOString(),
              isRead: false,
              severity: 'critical',
            },
            {
              id: '3',
              type: 'missed_checkin',
              message: 'Missed daily check-in yesterday',
              timestamp: new Date(Date.now() - 86400000).toISOString(),
              isRead: false,
              severity: 'warning',
            },
          ],
        },
      ];
      setLovedOnes(demoData);
      localStorage.setItem(`reunity_caregiver_${caregiverId}`, JSON.stringify(demoData));
    }
  };

  const saveData = (data: LovedOne[]) => {
    setLovedOnes(data);
    localStorage.setItem(`reunity_caregiver_${caregiverId}`, JSON.stringify(data));
  };

  const linkLovedOne = () => {
    if (!linkCode.trim()) return;
    
    // In production, this would verify the code with the server
    const newPerson: LovedOne = {
      id: Date.now().toString(),
      name: 'New Connection',
      relationship: 'Family',
      linkedDate: new Date().toISOString(),
      lastActive: new Date().toISOString(),
      privacySettings: {
        shareCheckIns: true,
        shareMoodData: true,
        shareLocation: false,
        shareCrisisAlerts: true,
        shareJournalSummary: false,
        shareStreaks: true,
        shareSleepData: false,
      },
      wellnessData: {
        currentMood: 'okay',
        moodTrend: 'stable',
        lastCheckIn: new Date().toISOString(),
        checkInStreak: 0,
        journalStreak: 0,
        sleepQuality: 50,
        entropyScore: 50,
        riskLevel: 'low',
      },
      alerts: [],
    };

    saveData([...lovedOnes, newPerson]);
    setShowAddDialog(false);
    setLinkCode('');
  };

  const markAlertRead = (personId: string, alertId: string) => {
    const updated = lovedOnes.map(p => {
      if (p.id === personId) {
        return {
          ...p,
          alerts: p.alerts.map(a => 
            a.id === alertId ? { ...a, isRead: true } : a
          ),
        };
      }
      return p;
    });
    saveData(updated);
  };

  const formatTimeAgo = (dateStr: string) => {
    const diff = Date.now() - new Date(dateStr).getTime();
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (hours < 1) return 'Just now';
    if (hours < 24) return `${hours}h ago`;
    return `${days}d ago`;
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving': return <TrendingUp className="w-4 h-4 text-emerald-400" />;
      case 'declining': return <TrendingDown className="w-4 h-4 text-red-400" />;
      default: return <Minus className="w-4 h-4 text-zinc-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Heart className="w-6 h-6 text-pink-400" />
            Caregiver Dashboard
          </h2>
          <p className="text-zinc-400 mt-1">
            Monitor your loved ones' wellness with their consent
          </p>
        </div>
        <div className="flex items-center gap-3">
          {unreadAlerts > 0 && (
            <Badge className="bg-red-500/20 text-red-400">
              {unreadAlerts} new alerts
            </Badge>
          )}
          <Dialog open={showAddDialog} onOpenChange={setShowAddDialog}>
            <DialogTrigger asChild>
              <Button className="bg-pink-600 hover:bg-pink-700">
                <UserPlus className="w-4 h-4 mr-2" />
                Link Loved One
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-zinc-900 border-zinc-800">
              <DialogHeader>
                <DialogTitle className="text-white">Link a Loved One</DialogTitle>
                <DialogDescription>
                  Enter the sharing code provided by your loved one to connect your accounts.
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 mt-4">
                <div>
                  <Label>Sharing Code</Label>
                  <Input
                    value={linkCode}
                    onChange={(e) => setLinkCode(e.target.value)}
                    placeholder="Enter 6-digit code"
                    className="bg-zinc-800 border-zinc-700 text-center text-2xl tracking-widest"
                    maxLength={6}
                  />
                </div>
                <div className="bg-zinc-800/50 rounded-lg p-4 text-sm text-zinc-400">
                  <Shield className="w-5 h-5 text-emerald-400 mb-2" />
                  <p className="font-medium text-white mb-1">Privacy Protected</p>
                  <p>Your loved one controls exactly what information is shared with you. They can change these settings at any time.</p>
                </div>
                <Button onClick={linkLovedOne} className="w-full bg-pink-600 hover:bg-pink-700">
                  Connect Account
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </div>

      {/* Privacy Notice */}
      <Card className="bg-emerald-500/10 border-emerald-500/30">
        <CardContent className="p-4 flex items-start gap-3">
          <Shield className="w-5 h-5 text-emerald-400 mt-0.5" />
          <div>
            <p className="text-sm text-emerald-300 font-medium">Privacy-First Design</p>
            <p className="text-sm text-emerald-200/70">
              All data sharing is controlled by your loved ones. They choose what to share and can revoke access at any time.
              You will only see information they have explicitly consented to share.
            </p>
          </div>
        </CardContent>
      </Card>

      {/* Loved Ones Grid */}
      {lovedOnes.length === 0 ? (
        <Card className="bg-zinc-900 border-zinc-800">
          <CardContent className="p-12 text-center">
            <Heart className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">No Connections Yet</h3>
            <p className="text-zinc-400 mb-4">
              Link with your loved ones to monitor their wellness journey together.
            </p>
            <Button onClick={() => setShowAddDialog(true)} className="bg-pink-600 hover:bg-pink-700">
              <UserPlus className="w-4 h-4 mr-2" />
              Link Your First Loved One
            </Button>
          </CardContent>
        </Card>
      ) : (
        <div className="grid md:grid-cols-2 gap-6">
          {lovedOnes.map((person) => (
            <Card 
              key={person.id} 
              className={`bg-zinc-900 border-zinc-800 cursor-pointer transition-all hover:border-zinc-700 ${
                person.wellnessData.riskLevel === 'high' || person.wellnessData.riskLevel === 'elevated' 
                  ? 'ring-2 ring-orange-500/50' 
                  : ''
              }`}
              onClick={() => setSelectedPerson(person)}
            >
              <CardHeader className="pb-3">
                <div className="flex items-start justify-between">
                  <div>
                    <CardTitle className="text-lg text-white flex items-center gap-2">
                      {person.name}
                      {person.alerts.filter(a => !a.isRead).length > 0 && (
                        <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                      )}
                    </CardTitle>
                    <CardDescription>{person.relationship}</CardDescription>
                  </div>
                  <Badge className={RISK_COLORS[person.wellnessData.riskLevel]}>
                    {person.wellnessData.riskLevel} risk
                  </Badge>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                {/* Current Status */}
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge className={MOOD_COLORS[person.wellnessData.currentMood]}>
                      {MOOD_LABELS[person.wellnessData.currentMood]}
                    </Badge>
                    {getTrendIcon(person.wellnessData.moodTrend)}
                  </div>
                  <span className="text-sm text-zinc-400">
                    Active {formatTimeAgo(person.lastActive)}
                  </span>
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-3 gap-3">
                  {person.privacySettings.shareStreaks && (
                    <div className="bg-zinc-800/50 rounded-lg p-2 text-center">
                      <Flame className="w-4 h-4 text-orange-400 mx-auto mb-1" />
                      <p className="text-lg font-bold text-white">{person.wellnessData.checkInStreak}</p>
                      <p className="text-xs text-zinc-500">Check-in streak</p>
                    </div>
                  )}
                  {person.privacySettings.shareSleepData && (
                    <div className="bg-zinc-800/50 rounded-lg p-2 text-center">
                      <Moon className="w-4 h-4 text-blue-400 mx-auto mb-1" />
                      <p className="text-lg font-bold text-white">{person.wellnessData.sleepQuality}%</p>
                      <p className="text-xs text-zinc-500">Sleep quality</p>
                    </div>
                  )}
                  {person.privacySettings.shareMoodData && (
                    <div className="bg-zinc-800/50 rounded-lg p-2 text-center">
                      <Activity className="w-4 h-4 text-emerald-400 mx-auto mb-1" />
                      <p className="text-lg font-bold text-white">{person.wellnessData.entropyScore}</p>
                      <p className="text-xs text-zinc-500">Entropy score</p>
                    </div>
                  )}
                </div>

                {/* Alerts Preview */}
                {person.alerts.filter(a => !a.isRead).length > 0 && (
                  <div className="bg-orange-500/10 border border-orange-500/30 rounded-lg p-3">
                    <div className="flex items-center gap-2 text-orange-400 text-sm">
                      <AlertTriangle className="w-4 h-4" />
                      <span>{person.alerts.filter(a => !a.isRead).length} unread alert(s)</span>
                    </div>
                  </div>
                )}

                {/* Privacy Indicators */}
                <div className="flex flex-wrap gap-1">
                  {person.privacySettings.shareCheckIns && (
                    <Badge variant="outline" className="text-xs text-zinc-400">
                      <CheckCircle2 className="w-3 h-3 mr-1" /> Check-ins
                    </Badge>
                  )}
                  {person.privacySettings.shareMoodData && (
                    <Badge variant="outline" className="text-xs text-zinc-400">
                      <Heart className="w-3 h-3 mr-1" /> Mood
                    </Badge>
                  )}
                  {person.privacySettings.shareLocation && (
                    <Badge variant="outline" className="text-xs text-zinc-400">
                      <MapPin className="w-3 h-3 mr-1" /> Location
                    </Badge>
                  )}
                  {person.privacySettings.shareCrisisAlerts && (
                    <Badge variant="outline" className="text-xs text-zinc-400">
                      <Bell className="w-3 h-3 mr-1" /> Alerts
                    </Badge>
                  )}
                </div>

                {/* Quick Actions */}
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" className="flex-1">
                    <Phone className="w-4 h-4 mr-1" />
                    Call
                  </Button>
                  <Button variant="outline" size="sm" className="flex-1">
                    <MessageSquare className="w-4 h-4 mr-1" />
                    Message
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Detailed View Dialog */}
      {selectedPerson && (
        <Dialog open={!!selectedPerson} onOpenChange={() => setSelectedPerson(null)}>
          <DialogContent className="bg-zinc-900 border-zinc-800 max-w-2xl max-h-[90vh] overflow-hidden">
            <DialogHeader>
              <DialogTitle className="text-white flex items-center gap-2">
                {selectedPerson.name}'s Wellness Summary
                <Badge className={RISK_COLORS[selectedPerson.wellnessData.riskLevel]}>
                  {selectedPerson.wellnessData.riskLevel} risk
                </Badge>
              </DialogTitle>
              <DialogDescription>
                {selectedPerson.relationship} • Connected {formatTimeAgo(selectedPerson.linkedDate)}
              </DialogDescription>
            </DialogHeader>
            
            <Tabs defaultValue="overview" className="mt-4">
              <TabsList className="bg-zinc-800">
                <TabsTrigger value="overview">Overview</TabsTrigger>
                <TabsTrigger value="alerts">
                  Alerts
                  {selectedPerson.alerts.filter(a => !a.isRead).length > 0 && (
                    <span className="ml-1 w-2 h-2 rounded-full bg-red-500" />
                  )}
                </TabsTrigger>
                <TabsTrigger value="privacy">Privacy</TabsTrigger>
              </TabsList>

              <ScrollArea className="h-[400px] mt-4">
                <TabsContent value="overview" className="space-y-4 pr-4">
                  {/* Current Mood */}
                  <Card className="bg-zinc-800/50 border-zinc-700">
                    <CardContent className="p-4">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm text-zinc-400">Current Mood</p>
                          <div className="flex items-center gap-2 mt-1">
                            <Badge className={MOOD_COLORS[selectedPerson.wellnessData.currentMood] + ' text-lg px-3 py-1'}>
                              {MOOD_LABELS[selectedPerson.wellnessData.currentMood]}
                            </Badge>
                            {getTrendIcon(selectedPerson.wellnessData.moodTrend)}
                            <span className="text-sm text-zinc-400">
                              {selectedPerson.wellnessData.moodTrend}
                            </span>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm text-zinc-400">Last Check-in</p>
                          <p className="text-white">{formatTimeAgo(selectedPerson.wellnessData.lastCheckIn)}</p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Wellness Metrics */}
                  <div className="grid grid-cols-2 gap-4">
                    {selectedPerson.privacySettings.shareStreaks && (
                      <>
                        <Card className="bg-zinc-800/50 border-zinc-700">
                          <CardContent className="p-4 text-center">
                            <Flame className="w-8 h-8 text-orange-400 mx-auto mb-2" />
                            <p className="text-3xl font-bold text-white">{selectedPerson.wellnessData.checkInStreak}</p>
                            <p className="text-sm text-zinc-400">Day Check-in Streak</p>
                          </CardContent>
                        </Card>
                        <Card className="bg-zinc-800/50 border-zinc-700">
                          <CardContent className="p-4 text-center">
                            <BookOpen className="w-8 h-8 text-blue-400 mx-auto mb-2" />
                            <p className="text-3xl font-bold text-white">{selectedPerson.wellnessData.journalStreak}</p>
                            <p className="text-sm text-zinc-400">Day Journal Streak</p>
                          </CardContent>
                        </Card>
                      </>
                    )}
                    {selectedPerson.privacySettings.shareSleepData && (
                      <Card className="bg-zinc-800/50 border-zinc-700">
                        <CardContent className="p-4 text-center">
                          <Moon className="w-8 h-8 text-purple-400 mx-auto mb-2" />
                          <p className="text-3xl font-bold text-white">{selectedPerson.wellnessData.sleepQuality}%</p>
                          <p className="text-sm text-zinc-400">Sleep Quality</p>
                        </CardContent>
                      </Card>
                    )}
                    {selectedPerson.privacySettings.shareMoodData && (
                      <Card className="bg-zinc-800/50 border-zinc-700">
                        <CardContent className="p-4 text-center">
                          <Activity className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                          <p className="text-3xl font-bold text-white">{selectedPerson.wellnessData.entropyScore}</p>
                          <p className="text-sm text-zinc-400">Entropy Score</p>
                        </CardContent>
                      </Card>
                    )}
                  </div>

                  {/* Journal Summary */}
                  {selectedPerson.privacySettings.shareJournalSummary && (
                    <Card className="bg-zinc-800/50 border-zinc-700">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm text-zinc-400">Recent Journal Themes</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex flex-wrap gap-2">
                          <Badge variant="outline">Self-reflection</Badge>
                          <Badge variant="outline">Work stress</Badge>
                          <Badge variant="outline">Gratitude</Badge>
                          <Badge variant="outline">Family</Badge>
                        </div>
                        <p className="text-sm text-zinc-400 mt-3">
                          Overall sentiment: <span className="text-emerald-400">Mostly positive</span>
                        </p>
                      </CardContent>
                    </Card>
                  )}
                </TabsContent>

                <TabsContent value="alerts" className="space-y-3 pr-4">
                  {selectedPerson.alerts.length === 0 ? (
                    <div className="text-center py-8">
                      <CheckCircle2 className="w-12 h-12 text-emerald-400 mx-auto mb-3" />
                      <p className="text-white">No alerts at this time</p>
                      <p className="text-sm text-zinc-400">Everything looks good!</p>
                    </div>
                  ) : (
                    selectedPerson.alerts.map((alert) => (
                      <Card 
                        key={alert.id} 
                        className={`border ${
                          alert.severity === 'critical' ? 'bg-red-500/10 border-red-500/30' :
                          alert.severity === 'warning' ? 'bg-orange-500/10 border-orange-500/30' :
                          'bg-zinc-800/50 border-zinc-700'
                        } ${!alert.isRead ? 'ring-2 ring-white/20' : ''}`}
                        onClick={() => markAlertRead(selectedPerson.id, alert.id)}
                      >
                        <CardContent className="p-4">
                          <div className="flex items-start gap-3">
                            <div className={`p-2 rounded-full ${
                              alert.severity === 'critical' ? 'bg-red-500/20' :
                              alert.severity === 'warning' ? 'bg-orange-500/20' :
                              'bg-zinc-700'
                            }`}>
                              {alert.type === 'crisis' ? <AlertTriangle className="w-5 h-5 text-red-400" /> :
                               alert.type === 'high_risk' ? <Activity className="w-5 h-5 text-orange-400" /> :
                               alert.type === 'missed_checkin' ? <Clock className="w-5 h-5 text-yellow-400" /> :
                               <Bell className="w-5 h-5 text-zinc-400" />}
                            </div>
                            <div className="flex-1">
                              <div className="flex items-center justify-between">
                                <p className="font-medium text-white">{alert.message}</p>
                                {!alert.isRead && (
                                  <Badge className="bg-blue-500/20 text-blue-400">New</Badge>
                                )}
                              </div>
                              <p className="text-sm text-zinc-400 mt-1">
                                {formatTimeAgo(alert.timestamp)}
                              </p>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))
                  )}
                </TabsContent>

                <TabsContent value="privacy" className="space-y-4 pr-4">
                  <Card className="bg-zinc-800/50 border-zinc-700">
                    <CardHeader>
                      <CardTitle className="text-sm text-white flex items-center gap-2">
                        <Shield className="w-4 h-4 text-emerald-400" />
                        Data Sharing Permissions
                      </CardTitle>
                      <CardDescription>
                        These settings are controlled by {selectedPerson.name}
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {Object.entries(selectedPerson.privacySettings).map(([key, value]) => (
                        <div key={key} className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            {value ? (
                              <Eye className="w-4 h-4 text-emerald-400" />
                            ) : (
                              <EyeOff className="w-4 h-4 text-zinc-500" />
                            )}
                            <span className="text-sm text-zinc-300">
                              {key.replace(/([A-Z])/g, ' $1').replace('share ', 'Share ').trim()}
                            </span>
                          </div>
                          <Badge className={value ? 'bg-emerald-500/20 text-emerald-400' : 'bg-zinc-700 text-zinc-400'}>
                            {value ? 'Shared' : 'Private'}
                          </Badge>
                        </div>
                      ))}
                    </CardContent>
                  </Card>

                  <div className="bg-zinc-800/30 rounded-lg p-4 text-sm text-zinc-400">
                    <Lock className="w-5 h-5 text-zinc-500 mb-2" />
                    <p>
                      {selectedPerson.name} has full control over what data is shared with you. 
                      To request access to additional information, please speak with them directly.
                    </p>
                  </div>
                </TabsContent>
              </ScrollArea>
            </Tabs>
          </DialogContent>
        </Dialog>
      )}
    </div>
  );
}
