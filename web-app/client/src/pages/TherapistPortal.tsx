import { useState, useEffect } from 'react';
import { Link } from 'wouter';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Switch } from '@/components/ui/switch';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from '@/components/ui/dialog';
import { 
  Heart, 
  ArrowLeft, 
  Users, 
  Bell, 
  TrendingUp, 
  TrendingDown,
  Minus,
  AlertTriangle,
  CheckCircle,
  Clock,
  Eye,
  FileText,
  Shield,
  UserPlus,
  Activity,
  Calendar,
  MessageSquare,
  ChevronRight,
  Search,
  Filter,
  BarChart3,
  Video,
  Phone
} from 'lucide-react';
import { useAuth } from '@/contexts/AuthContext';
import { trpc } from '@/lib/trpc';

// Types
interface Client {
  id: number;
  name: string;
  email: string;
  status: 'active' | 'paused' | 'pending';
  consentedAt: string;
  lastActivity: string;
  avgEntropy: number;
  entropyTrend: 'improving' | 'stable' | 'declining';
  crisisAlertsEnabled: boolean;
  recentAlerts: number;
}

interface Alert {
  id: number;
  clientId: number;
  clientName: string;
  type: 'crisis' | 'high_entropy' | 'missed_checkin' | 'concerning_pattern' | 'progress';
  severity: 'low' | 'medium' | 'high' | 'critical';
  title: string;
  description: string;
  createdAt: string;
  isViewed: boolean;
  isAcknowledged: boolean;
}

interface EntropySnapshot {
  date: string;
  avgScore: number;
  trend: string;
  journalCount: number;
  checkInsCompleted: number;
  checkInsMissed: number;
  crisisEvents: number;
}

// Mock data for demo
const mockClients: Client[] = [
  {
    id: 1,
    name: 'Sarah M.',
    email: 's***@email.com',
    status: 'active',
    consentedAt: '2026-01-15T10:00:00Z',
    lastActivity: '2026-01-25T14:30:00Z',
    avgEntropy: 0.45,
    entropyTrend: 'improving',
    crisisAlertsEnabled: true,
    recentAlerts: 0
  },
  {
    id: 2,
    name: 'Michael R.',
    email: 'm***@email.com',
    status: 'active',
    consentedAt: '2026-01-10T09:00:00Z',
    lastActivity: '2026-01-25T11:00:00Z',
    avgEntropy: 0.72,
    entropyTrend: 'declining',
    crisisAlertsEnabled: true,
    recentAlerts: 2
  },
  {
    id: 3,
    name: 'Jennifer L.',
    email: 'j***@email.com',
    status: 'pending',
    consentedAt: '',
    lastActivity: '',
    avgEntropy: 0,
    entropyTrend: 'stable',
    crisisAlertsEnabled: false,
    recentAlerts: 0
  }
];

const mockAlerts: Alert[] = [
  {
    id: 1,
    clientId: 2,
    clientName: 'Michael R.',
    type: 'high_entropy',
    severity: 'high',
    title: 'Elevated Entropy Detected',
    description: 'Client\'s entropy score has increased to 0.82 over the past 3 days, indicating heightened emotional distress.',
    createdAt: '2026-01-25T10:00:00Z',
    isViewed: false,
    isAcknowledged: false
  },
  {
    id: 2,
    clientId: 2,
    clientName: 'Michael R.',
    type: 'missed_checkin',
    severity: 'medium',
    title: 'Missed Check-In',
    description: 'Client missed their scheduled wellness check-in yesterday.',
    createdAt: '2026-01-24T18:00:00Z',
    isViewed: true,
    isAcknowledged: false
  },
  {
    id: 3,
    clientId: 1,
    clientName: 'Sarah M.',
    type: 'progress',
    severity: 'low',
    title: 'Positive Progress',
    description: 'Client\'s entropy score has decreased by 15% this week, showing consistent improvement.',
    createdAt: '2026-01-23T09:00:00Z',
    isViewed: true,
    isAcknowledged: true
  }
];

const mockSnapshots: EntropySnapshot[] = [
  { date: '2026-01-25', avgScore: 0.72, trend: 'declining', journalCount: 2, checkInsCompleted: 1, checkInsMissed: 1, crisisEvents: 0 },
  { date: '2026-01-24', avgScore: 0.68, trend: 'stable', journalCount: 3, checkInsCompleted: 2, checkInsMissed: 0, crisisEvents: 0 },
  { date: '2026-01-23', avgScore: 0.65, trend: 'stable', journalCount: 1, checkInsCompleted: 2, checkInsMissed: 0, crisisEvents: 0 },
  { date: '2026-01-22', avgScore: 0.58, trend: 'improving', journalCount: 4, checkInsCompleted: 2, checkInsMissed: 0, crisisEvents: 0 },
  { date: '2026-01-21', avgScore: 0.55, trend: 'improving', journalCount: 2, checkInsCompleted: 2, checkInsMissed: 0, crisisEvents: 0 },
  { date: '2026-01-20', avgScore: 0.60, trend: 'stable', journalCount: 1, checkInsCompleted: 1, checkInsMissed: 1, crisisEvents: 1 },
  { date: '2026-01-19', avgScore: 0.75, trend: 'declining', journalCount: 0, checkInsCompleted: 0, checkInsMissed: 2, crisisEvents: 0 },
];

export default function TherapistPortal() {
  const { user, isLoading } = useAuth();
  const [activeTab, setActiveTab] = useState('dashboard');
  const [clients, setClients] = useState<Client[]>(mockClients);
  const [alerts, setAlerts] = useState<Alert[]>(mockAlerts);
  const [selectedClient, setSelectedClient] = useState<Client | null>(null);
  const [showClientDetail, setShowClientDetail] = useState(false);
  const [showInviteDialog, setShowInviteDialog] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');

  // Filter clients
  const filteredClients = clients.filter(client => {
    const matchesSearch = client.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          client.email.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesFilter = filterStatus === 'all' || client.status === filterStatus;
    return matchesSearch && matchesFilter;
  });

  // Count unread alerts
  const unreadAlerts = alerts.filter(a => !a.isViewed).length;
  const criticalAlerts = alerts.filter(a => a.severity === 'critical' || a.severity === 'high').length;

  // Get trend icon
  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'improving':
        return <TrendingDown className="h-4 w-4 text-emerald-400" />;
      case 'declining':
        return <TrendingUp className="h-4 w-4 text-red-400" />;
      default:
        return <Minus className="h-4 w-4 text-yellow-400" />;
    }
  };

  // Get severity color
  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-500/20 text-red-400 border-red-500/50';
      case 'high':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/50';
      case 'medium':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/50';
      default:
        return 'bg-blue-500/20 text-blue-400 border-blue-500/50';
    }
  };

  // Get alert type icon
  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'crisis':
        return <AlertTriangle className="h-5 w-5 text-red-400" />;
      case 'high_entropy':
        return <Activity className="h-5 w-5 text-orange-400" />;
      case 'missed_checkin':
        return <Clock className="h-5 w-5 text-yellow-400" />;
      case 'progress':
        return <CheckCircle className="h-5 w-5 text-emerald-400" />;
      default:
        return <Bell className="h-5 w-5 text-blue-400" />;
    }
  };

  // Mark alert as viewed
  const markAlertViewed = (alertId: number) => {
    setAlerts(prev => prev.map(a => 
      a.id === alertId ? { ...a, isViewed: true } : a
    ));
  };

  // Acknowledge alert
  const acknowledgeAlert = (alertId: number) => {
    setAlerts(prev => prev.map(a => 
      a.id === alertId ? { ...a, isAcknowledged: true } : a
    ));
  };

  // Send invite
  const handleSendInvite = () => {
    if (!inviteEmail) return;
    
    // In a real app, this would send an invitation
    const newClient: Client = {
      id: clients.length + 1,
      name: 'New Client',
      email: inviteEmail.slice(0, 1) + '***@' + inviteEmail.split('@')[1],
      status: 'pending',
      consentedAt: '',
      lastActivity: '',
      avgEntropy: 0,
      entropyTrend: 'stable',
      crisisAlertsEnabled: false,
      recentAlerts: 0
    };
    
    setClients(prev => [...prev, newClient]);
    setShowInviteDialog(false);
    setInviteEmail('');
  };

  // View client details
  const viewClientDetails = (client: Client) => {
    setSelectedClient(client);
    setShowClientDetail(true);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-emerald-400"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-2">
              <Heart className="h-6 w-6 text-emerald-400" />
              <span className="text-xl font-bold text-white">Therapist Portal</span>
            </Link>
            <Badge variant="outline" className="border-emerald-500/50 text-emerald-400">
              <Shield className="h-3 w-3 mr-1" />
              Licensed Provider
            </Badge>
          </div>
          <div className="flex items-center gap-4">
            <Button 
              variant="ghost" 
              size="sm" 
              className="relative text-slate-400 hover:text-white"
              onClick={() => setActiveTab('alerts')}
            >
              <Bell className="w-5 h-5" />
              {unreadAlerts > 0 && (
                <span className="absolute -top-1 -right-1 w-5 h-5 bg-red-500 rounded-full text-xs flex items-center justify-center text-white">
                  {unreadAlerts}
                </span>
              )}
            </Button>
            <Link href="/">
              <Button variant="ghost" size="sm" className="text-slate-400 hover:text-white">
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Button>
            </Link>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
          <TabsList className="grid grid-cols-3 bg-slate-800/50 p-1 rounded-lg max-w-md">
            <TabsTrigger 
              value="dashboard" 
              className="flex items-center gap-2 data-[state=active]:bg-emerald-600"
            >
              <BarChart3 className="w-4 h-4" />
              Dashboard
            </TabsTrigger>
            <TabsTrigger 
              value="clients"
              className="flex items-center gap-2 data-[state=active]:bg-emerald-600"
            >
              <Users className="w-4 h-4" />
              Clients
            </TabsTrigger>
            <TabsTrigger 
              value="alerts"
              className="flex items-center gap-2 data-[state=active]:bg-emerald-600 relative"
            >
              <Bell className="w-4 h-4" />
              Alerts
              {unreadAlerts > 0 && (
                <span className="w-2 h-2 bg-red-500 rounded-full absolute top-1 right-1"></span>
              )}
            </TabsTrigger>
          </TabsList>

          {/* Dashboard Tab */}
          <TabsContent value="dashboard" className="space-y-6">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <Card className="bg-slate-900/50 border-slate-700">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-400">Active Clients</p>
                      <p className="text-2xl font-bold text-white">
                        {clients.filter(c => c.status === 'active').length}
                      </p>
                    </div>
                    <Users className="h-8 w-8 text-emerald-400 opacity-50" />
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-slate-900/50 border-slate-700">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-400">Pending Invites</p>
                      <p className="text-2xl font-bold text-white">
                        {clients.filter(c => c.status === 'pending').length}
                      </p>
                    </div>
                    <Clock className="h-8 w-8 text-yellow-400 opacity-50" />
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-slate-900/50 border-slate-700">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-400">Unread Alerts</p>
                      <p className="text-2xl font-bold text-white">{unreadAlerts}</p>
                    </div>
                    <Bell className="h-8 w-8 text-blue-400 opacity-50" />
                  </div>
                </CardContent>
              </Card>
              
              <Card className="bg-slate-900/50 border-slate-700">
                <CardContent className="p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-slate-400">Critical Alerts</p>
                      <p className="text-2xl font-bold text-white">{criticalAlerts}</p>
                    </div>
                    <AlertTriangle className="h-8 w-8 text-red-400 opacity-50" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Recent Alerts */}
            <Card className="bg-slate-900/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Bell className="h-5 w-5 text-emerald-400" />
                  Recent Alerts
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {alerts.slice(0, 3).map(alert => (
                    <div 
                      key={alert.id}
                      className={`p-3 rounded-lg border ${getSeverityColor(alert.severity)} flex items-center justify-between cursor-pointer hover:opacity-80 transition-opacity`}
                      onClick={() => {
                        markAlertViewed(alert.id);
                        setActiveTab('alerts');
                      }}
                    >
                      <div className="flex items-center gap-3">
                        {getAlertIcon(alert.type)}
                        <div>
                          <p className="font-medium">{alert.title}</p>
                          <p className="text-xs opacity-70">{alert.clientName} • {new Date(alert.createdAt).toLocaleDateString()}</p>
                        </div>
                      </div>
                      {!alert.isViewed && (
                        <Badge className="bg-blue-500">New</Badge>
                      )}
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Client Overview */}
            <Card className="bg-slate-900/50 border-slate-700">
              <CardHeader>
                <CardTitle className="text-white flex items-center gap-2">
                  <Activity className="h-5 w-5 text-emerald-400" />
                  Client Entropy Overview
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {clients.filter(c => c.status === 'active').map(client => (
                    <div 
                      key={client.id}
                      className="p-3 bg-slate-800/50 rounded-lg flex items-center justify-between cursor-pointer hover:bg-slate-800 transition-colors"
                      onClick={() => viewClientDetails(client)}
                    >
                      <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                          <span className="text-emerald-400 font-medium">
                            {client.name.charAt(0)}
                          </span>
                        </div>
                        <div>
                          <p className="font-medium text-white">{client.name}</p>
                          <p className="text-xs text-slate-400">
                            Last active: {new Date(client.lastActivity).toLocaleDateString()}
                          </p>
                        </div>
                      </div>
                      <div className="flex items-center gap-4">
                        <div className="text-right">
                          <p className="text-sm text-slate-400">Entropy</p>
                          <div className="flex items-center gap-1">
                            <span className={`font-medium ${
                              client.avgEntropy > 0.7 ? 'text-red-400' :
                              client.avgEntropy > 0.5 ? 'text-yellow-400' :
                              'text-emerald-400'
                            }`}>
                              {(client.avgEntropy * 100).toFixed(0)}%
                            </span>
                            {getTrendIcon(client.entropyTrend)}
                          </div>
                        </div>
                        <ChevronRight className="h-5 w-5 text-slate-500" />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          {/* Clients Tab */}
          <TabsContent value="clients" className="space-y-6">
            {/* Search and Filter */}
            <div className="flex gap-4 items-center">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <Input
                  placeholder="Search clients..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10 bg-slate-800 border-slate-600 text-white"
                />
              </div>
              <select
                value={filterStatus}
                onChange={(e) => setFilterStatus(e.target.value)}
                className="bg-slate-800 border border-slate-600 rounded-md px-3 py-2 text-white"
              >
                <option value="all">All Status</option>
                <option value="active">Active</option>
                <option value="pending">Pending</option>
                <option value="paused">Paused</option>
              </select>
              <Button 
                className="bg-emerald-600 hover:bg-emerald-700"
                onClick={() => setShowInviteDialog(true)}
              >
                <UserPlus className="h-4 w-4 mr-2" />
                Invite Client
              </Button>
            </div>

            {/* Clients List */}
            <div className="space-y-3">
              {filteredClients.map(client => (
                <Card 
                  key={client.id} 
                  className="bg-slate-900/50 border-slate-700 cursor-pointer hover:border-emerald-500/50 transition-colors"
                  onClick={() => viewClientDetails(client)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-4">
                        <div className="w-12 h-12 rounded-full bg-emerald-500/20 flex items-center justify-center">
                          <span className="text-emerald-400 font-medium text-lg">
                            {client.name.charAt(0)}
                          </span>
                        </div>
                        <div>
                          <p className="font-medium text-white">{client.name}</p>
                          <p className="text-sm text-slate-400">{client.email}</p>
                        </div>
                      </div>
                      
                      <div className="flex items-center gap-6">
                        <Badge className={
                          client.status === 'active' ? 'bg-emerald-500/20 text-emerald-400' :
                          client.status === 'pending' ? 'bg-yellow-500/20 text-yellow-400' :
                          'bg-slate-500/20 text-slate-400'
                        }>
                          {client.status}
                        </Badge>
                        
                        {client.status === 'active' && (
                          <>
                            <div className="text-right">
                              <p className="text-xs text-slate-400">Entropy</p>
                              <div className="flex items-center gap-1">
                                <span className={`font-medium ${
                                  client.avgEntropy > 0.7 ? 'text-red-400' :
                                  client.avgEntropy > 0.5 ? 'text-yellow-400' :
                                  'text-emerald-400'
                                }`}>
                                  {(client.avgEntropy * 100).toFixed(0)}%
                                </span>
                                {getTrendIcon(client.entropyTrend)}
                              </div>
                            </div>
                            
                            {client.recentAlerts > 0 && (
                              <Badge className="bg-red-500/20 text-red-400">
                                {client.recentAlerts} alerts
                              </Badge>
                            )}
                          </>
                        )}
                        
                        <ChevronRight className="h-5 w-5 text-slate-500" />
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>

          {/* Alerts Tab */}
          <TabsContent value="alerts" className="space-y-6">
            <div className="space-y-3">
              {alerts.map(alert => (
                <Card 
                  key={alert.id} 
                  className={`bg-slate-900/50 border ${getSeverityColor(alert.severity)} ${!alert.isViewed ? 'ring-1 ring-blue-500/50' : ''}`}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start gap-4">
                      <div className="mt-1">
                        {getAlertIcon(alert.type)}
                      </div>
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-1">
                          <h3 className="font-medium text-white">{alert.title}</h3>
                          {!alert.isViewed && (
                            <Badge className="bg-blue-500">New</Badge>
                          )}
                          <Badge className={getSeverityColor(alert.severity)}>
                            {alert.severity}
                          </Badge>
                        </div>
                        <p className="text-sm text-slate-400 mb-2">{alert.description}</p>
                        <div className="flex items-center gap-4 text-xs text-slate-500">
                          <span>{alert.clientName}</span>
                          <span>•</span>
                          <span>{new Date(alert.createdAt).toLocaleString()}</span>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {!alert.isViewed && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => markAlertViewed(alert.id)}
                          >
                            <Eye className="h-4 w-4" />
                          </Button>
                        )}
                        {!alert.isAcknowledged && (
                          <Button
                            variant="outline"
                            size="sm"
                            className="border-emerald-500/50 text-emerald-400"
                            onClick={() => acknowledgeAlert(alert.id)}
                          >
                            <CheckCircle className="h-4 w-4 mr-1" />
                            Acknowledge
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
        </Tabs>
      </main>

      {/* Client Detail Dialog */}
      <Dialog open={showClientDetail} onOpenChange={setShowClientDetail}>
        <DialogContent className="bg-slate-900 border-slate-700 max-w-2xl max-h-[80vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-white flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-emerald-500/20 flex items-center justify-center">
                <span className="text-emerald-400 font-medium">
                  {selectedClient?.name.charAt(0)}
                </span>
              </div>
              {selectedClient?.name}
            </DialogTitle>
            <DialogDescription className="text-slate-400">
              Client monitoring dashboard - data shared with consent
            </DialogDescription>
          </DialogHeader>
          
          {selectedClient && selectedClient.status === 'active' && (
            <div className="space-y-6 mt-4">
              {/* Entropy Overview */}
              <div>
                <h4 className="text-sm font-medium text-slate-400 mb-3">Entropy Trend (7 Days)</h4>
                <div className="bg-slate-800/50 rounded-lg p-4">
                  {/* Simple bar chart */}
                  <div className="flex items-end gap-2 h-32">
                    {mockSnapshots.map((snapshot, i) => (
                      <div key={i} className="flex-1 flex flex-col items-center gap-1">
                        <div 
                          className={`w-full rounded-t transition-all ${
                            snapshot.avgScore > 0.7 ? 'bg-red-500' :
                            snapshot.avgScore > 0.5 ? 'bg-yellow-500' :
                            'bg-emerald-500'
                          }`}
                          style={{ height: `${snapshot.avgScore * 100}%` }}
                        />
                        <span className="text-xs text-slate-500">
                          {new Date(snapshot.date).getDate()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Stats Grid */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-slate-800/50 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-white">
                    {mockSnapshots.reduce((sum, s) => sum + s.journalCount, 0)}
                  </p>
                  <p className="text-xs text-slate-400">Journal Entries</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-emerald-400">
                    {mockSnapshots.reduce((sum, s) => sum + s.checkInsCompleted, 0)}
                  </p>
                  <p className="text-xs text-slate-400">Check-Ins Completed</p>
                </div>
                <div className="bg-slate-800/50 rounded-lg p-3 text-center">
                  <p className="text-2xl font-bold text-red-400">
                    {mockSnapshots.reduce((sum, s) => sum + s.crisisEvents, 0)}
                  </p>
                  <p className="text-xs text-slate-400">Crisis Events</p>
                </div>
              </div>

              {/* Consent Settings */}
              <div>
                <h4 className="text-sm font-medium text-slate-400 mb-3">Data Sharing Consent</h4>
                <div className="space-y-2">
                  <div className="flex items-center justify-between p-2 bg-slate-800/50 rounded">
                    <span className="text-sm text-white">Entropy Data</span>
                    <Badge className="bg-emerald-500/20 text-emerald-400">Consented</Badge>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-slate-800/50 rounded">
                    <span className="text-sm text-white">Journal Summaries</span>
                    <Badge className="bg-emerald-500/20 text-emerald-400">Consented</Badge>
                  </div>
                  <div className="flex items-center justify-between p-2 bg-slate-800/50 rounded">
                    <span className="text-sm text-white">Crisis Alerts</span>
                    <Badge className="bg-emerald-500/20 text-emerald-400">Enabled</Badge>
                  </div>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-3">
                <Link href="/video-call">
                  <Button className="bg-blue-600 hover:bg-blue-700">
                    <Video className="h-4 w-4 mr-2" />
                    Video Call
                  </Button>
                </Link>
                <Button className="flex-1 bg-emerald-600 hover:bg-emerald-700">
                  <MessageSquare className="h-4 w-4 mr-2" />
                  Message
                </Button>
                <Button variant="outline" className="border-slate-600">
                  <FileText className="h-4 w-4 mr-2" />
                  Report
                </Button>
              </div>
            </div>
          )}

          {selectedClient && selectedClient.status === 'pending' && (
            <div className="text-center py-8">
              <Clock className="h-12 w-12 mx-auto mb-4 text-yellow-400 opacity-50" />
              <p className="text-slate-400">Waiting for client to accept invitation</p>
              <p className="text-sm text-slate-500 mt-2">
                Invitation sent to {selectedClient.email}
              </p>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Invite Client Dialog */}
      <Dialog open={showInviteDialog} onOpenChange={setShowInviteDialog}>
        <DialogContent className="bg-slate-900 border-slate-700">
          <DialogHeader>
            <DialogTitle className="text-white">Invite Client</DialogTitle>
            <DialogDescription className="text-slate-400">
              Send an invitation to connect with a client on ReUnity. They will need to consent to share their data with you.
            </DialogDescription>
          </DialogHeader>
          
          <div className="space-y-4 mt-4">
            <div className="space-y-2">
              <Label htmlFor="invite-email" className="text-slate-300">Client Email</Label>
              <Input
                id="invite-email"
                type="email"
                value={inviteEmail}
                onChange={(e) => setInviteEmail(e.target.value)}
                placeholder="client@email.com"
                className="bg-slate-800 border-slate-600 text-white"
              />
            </div>

            <div className="bg-slate-800/50 rounded-lg p-3 text-sm text-slate-400">
              <p className="font-medium text-white mb-2">What happens next:</p>
              <ol className="list-decimal list-inside space-y-1">
                <li>Client receives an email invitation</li>
                <li>They log into ReUnity and review the connection request</li>
                <li>They choose what data to share (entropy, journals, alerts)</li>
                <li>Once consented, you'll see their data in your dashboard</li>
              </ol>
            </div>

            <Button
              className="w-full bg-emerald-600 hover:bg-emerald-700"
              onClick={handleSendInvite}
              disabled={!inviteEmail}
            >
              <UserPlus className="h-4 w-4 mr-2" />
              Send Invitation
            </Button>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
