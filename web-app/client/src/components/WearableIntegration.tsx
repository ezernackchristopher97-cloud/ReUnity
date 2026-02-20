import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { 
  Watch, 
  Heart, 
  Activity, 
  Moon,
  Footprints,
  Zap,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertTriangle,
  TrendingUp,
  TrendingDown
} from 'lucide-react';

interface HealthData {
  heartRate: number;
  heartRateVariability: number;
  steps: number;
  sleepHours: number;
  sleepQuality: number;
  activeMinutes: number;
  stressLevel: number;
  lastSync: Date | null;
}

interface WearableDevice {
  id: string;
  name: string;
  type: 'apple_health' | 'google_fit' | 'fitbit' | 'garmin';
  connected: boolean;
  lastSync: Date | null;
}

const STORAGE_KEY = 'reunity_wearable_data';
const DEVICE_KEY = 'reunity_connected_devices';

export default function WearableIntegration() {
  const [devices, setDevices] = useState<WearableDevice[]>([]);
  const [healthData, setHealthData] = useState<HealthData>({
    heartRate: 0,
    heartRateVariability: 0,
    steps: 0,
    sleepHours: 0,
    sleepQuality: 0,
    activeMinutes: 0,
    stressLevel: 0,
    lastSync: null
  });
  const [isLoading, setIsLoading] = useState(false);
  const [autoSync, setAutoSync] = useState(true);
  const [includeInEntropy, setIncludeInEntropy] = useState(true);

  useEffect(() => {
    loadSavedData();
  }, []);

  const loadSavedData = () => {
    const savedDevices = localStorage.getItem(DEVICE_KEY);
    const savedHealth = localStorage.getItem(STORAGE_KEY);
    
    if (savedDevices) {
      try {
        setDevices(JSON.parse(savedDevices));
      } catch (e) {
        console.error('Failed to load devices:', e);
      }
    }
    
    if (savedHealth) {
      try {
        const data = JSON.parse(savedHealth);
        data.lastSync = data.lastSync ? new Date(data.lastSync) : null;
        setHealthData(data);
      } catch (e) {
        console.error('Failed to load health data:', e);
      }
    }
  };

  const connectAppleHealth = async () => {
    setIsLoading(true);
    
    // Simulate Apple Health authorization
    // In production, this would use HealthKit via a native bridge or PWA API
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const newDevice: WearableDevice = {
      id: 'apple_health_' + Date.now(),
      name: 'Apple Health',
      type: 'apple_health',
      connected: true,
      lastSync: new Date()
    };
    
    const updatedDevices = [...devices.filter(d => d.type !== 'apple_health'), newDevice];
    setDevices(updatedDevices);
    localStorage.setItem(DEVICE_KEY, JSON.stringify(updatedDevices));
    
    // Fetch initial data
    await syncHealthData('apple_health');
    setIsLoading(false);
  };

  const connectGoogleFit = async () => {
    setIsLoading(true);
    
    // Simulate Google Fit OAuth
    // In production, this would use Google Fit REST API
    await new Promise(resolve => setTimeout(resolve, 1500));
    
    const newDevice: WearableDevice = {
      id: 'google_fit_' + Date.now(),
      name: 'Google Fit',
      type: 'google_fit',
      connected: true,
      lastSync: new Date()
    };
    
    const updatedDevices = [...devices.filter(d => d.type !== 'google_fit'), newDevice];
    setDevices(updatedDevices);
    localStorage.setItem(DEVICE_KEY, JSON.stringify(updatedDevices));
    
    await syncHealthData('google_fit');
    setIsLoading(false);
  };

  const disconnectDevice = (deviceId: string) => {
    const updatedDevices = devices.filter(d => d.id !== deviceId);
    setDevices(updatedDevices);
    localStorage.setItem(DEVICE_KEY, JSON.stringify(updatedDevices));
  };

  const syncHealthData = async (source?: string) => {
    setIsLoading(true);
    
    // Simulate fetching health data
    // In production, this would call the respective health APIs
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    // Generate realistic health data
    const newData: HealthData = {
      heartRate: Math.floor(60 + Math.random() * 30),
      heartRateVariability: Math.floor(20 + Math.random() * 60),
      steps: Math.floor(2000 + Math.random() * 8000),
      sleepHours: 5 + Math.random() * 4,
      sleepQuality: Math.floor(50 + Math.random() * 50),
      activeMinutes: Math.floor(10 + Math.random() * 60),
      stressLevel: Math.floor(20 + Math.random() * 60),
      lastSync: new Date()
    };
    
    setHealthData(newData);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(newData));
    
    // Calculate entropy contribution
    if (includeInEntropy) {
      calculateEntropyContribution(newData);
    }
    
    setIsLoading(false);
  };

  const calculateEntropyContribution = (data: HealthData) => {
    // HRV is a key indicator of stress and emotional state
    // Lower HRV often correlates with higher stress
    const hrvScore = data.heartRateVariability / 100;
    
    // Sleep quality affects emotional regulation
    const sleepScore = (data.sleepQuality / 100) * (data.sleepHours / 8);
    
    // Activity level affects mood
    const activityScore = Math.min(data.activeMinutes / 60, 1);
    
    // Combined entropy contribution (0-1 scale)
    const entropyContribution = (hrvScore * 0.4 + sleepScore * 0.35 + activityScore * 0.25);
    
    // Store for use in entropy calculations
    localStorage.setItem('reunity_health_entropy', JSON.stringify({
      contribution: entropyContribution,
      timestamp: new Date().toISOString(),
      factors: { hrvScore, sleepScore, activityScore }
    }));
    
    return entropyContribution;
  };

  const getHRVStatus = () => {
    if (healthData.heartRateVariability >= 50) {
      return { status: 'Good', color: 'text-emerald-400', icon: <TrendingUp className="w-4 h-4" /> };
    } else if (healthData.heartRateVariability >= 30) {
      return { status: 'Moderate', color: 'text-amber-400', icon: <Activity className="w-4 h-4" /> };
    }
    return { status: 'Low', color: 'text-red-400', icon: <TrendingDown className="w-4 h-4" /> };
  };

  const hrvStatus = getHRVStatus();
  const connectedDevices = devices.filter(d => d.connected);

  return (
    <Card className="bg-zinc-900/50 border-zinc-800">
      <CardHeader className="pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-pink-500/20">
              <Watch className="w-5 h-5 text-pink-400" />
            </div>
            <div>
              <CardTitle className="text-lg">Wearable Integration</CardTitle>
              <p className="text-xs text-zinc-500">Connect health devices for better insights</p>
            </div>
          </div>
          {connectedDevices.length > 0 && (
            <Button
              variant="ghost"
              size="sm"
              onClick={() => syncHealthData()}
              disabled={isLoading}
              className="text-zinc-400 hover:text-white"
            >
              <RefreshCw className={`w-4 h-4 mr-2 ${isLoading ? 'animate-spin' : ''}`} />
              Sync
            </Button>
          )}
        </div>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Connected Devices */}
        {connectedDevices.length > 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-zinc-400">Connected Devices</h4>
            {connectedDevices.map(device => (
              <div key={device.id} className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-3">
                  <CheckCircle className="w-5 h-5 text-emerald-400" />
                  <div>
                    <p className="text-sm font-medium">{device.name}</p>
                    <p className="text-xs text-zinc-500">
                      Last sync: {device.lastSync ? new Date(device.lastSync).toLocaleTimeString() : 'Never'}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => disconnectDevice(device.id)}
                  className="text-red-400 hover:text-red-300"
                >
                  Disconnect
                </Button>
              </div>
            ))}
          </div>
        )}

        {/* Connect New Device */}
        {connectedDevices.length === 0 && (
          <div className="space-y-3">
            <h4 className="text-sm font-medium text-zinc-400">Connect a Device</h4>
            <div className="grid gap-2">
              <Button
                variant="outline"
                className="justify-start gap-3 h-auto py-3 border-zinc-700 hover:bg-zinc-800"
                onClick={connectAppleHealth}
                disabled={isLoading}
              >
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-pink-500 to-red-500 flex items-center justify-center">
                  <Heart className="w-4 h-4 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-medium">Apple Health</p>
                  <p className="text-xs text-zinc-500">iPhone & Apple Watch</p>
                </div>
              </Button>
              <Button
                variant="outline"
                className="justify-start gap-3 h-auto py-3 border-zinc-700 hover:bg-zinc-800"
                onClick={connectGoogleFit}
                disabled={isLoading}
              >
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-blue-500 to-green-500 flex items-center justify-center">
                  <Activity className="w-4 h-4 text-white" />
                </div>
                <div className="text-left">
                  <p className="font-medium">Google Fit</p>
                  <p className="text-xs text-zinc-500">Android & Wear OS</p>
                </div>
              </Button>
            </div>
          </div>
        )}

        {/* Health Metrics */}
        {healthData.lastSync && (
          <div className="space-y-4">
            <h4 className="text-sm font-medium text-zinc-400">Today's Metrics</h4>
            
            {/* Heart Rate Variability - Key metric */}
            <div className="p-4 rounded-lg bg-gradient-to-r from-pink-500/10 to-purple-500/10 border border-pink-500/20">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Heart className="w-5 h-5 text-pink-400" />
                  <span className="font-medium">Heart Rate Variability</span>
                </div>
                <div className={`flex items-center gap-1 ${hrvStatus.color}`}>
                  {hrvStatus.icon}
                  <span className="text-sm">{hrvStatus.status}</span>
                </div>
              </div>
              <div className="flex items-end gap-2">
                <span className="text-3xl font-bold">{healthData.heartRateVariability}</span>
                <span className="text-zinc-500 mb-1">ms</span>
              </div>
              <p className="text-xs text-zinc-500 mt-2">
                HRV indicates your body's stress response and recovery capacity
              </p>
            </div>

            {/* Other Metrics Grid */}
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-2 text-zinc-400 mb-1">
                  <Heart className="w-4 h-4" />
                  <span className="text-xs">Heart Rate</span>
                </div>
                <p className="text-xl font-semibold">{healthData.heartRate} <span className="text-sm text-zinc-500">bpm</span></p>
              </div>
              
              <div className="p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-2 text-zinc-400 mb-1">
                  <Footprints className="w-4 h-4" />
                  <span className="text-xs">Steps</span>
                </div>
                <p className="text-xl font-semibold">{healthData.steps.toLocaleString()}</p>
              </div>
              
              <div className="p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-2 text-zinc-400 mb-1">
                  <Moon className="w-4 h-4" />
                  <span className="text-xs">Sleep</span>
                </div>
                <p className="text-xl font-semibold">{healthData.sleepHours.toFixed(1)} <span className="text-sm text-zinc-500">hrs</span></p>
              </div>
              
              <div className="p-3 rounded-lg bg-zinc-800/50">
                <div className="flex items-center gap-2 text-zinc-400 mb-1">
                  <Zap className="w-4 h-4" />
                  <span className="text-xs">Active</span>
                </div>
                <p className="text-xl font-semibold">{healthData.activeMinutes} <span className="text-sm text-zinc-500">min</span></p>
              </div>
            </div>

            {/* Sleep Quality */}
            <div className="p-3 rounded-lg bg-zinc-800/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-zinc-400">Sleep Quality</span>
                <span className="text-sm font-medium">{healthData.sleepQuality}%</span>
              </div>
              <Progress value={healthData.sleepQuality} className="h-2" />
            </div>
          </div>
        )}

        {/* Settings */}
        <div className="space-y-3">
          <h4 className="text-sm font-medium text-zinc-400">Settings</h4>
          
          <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
            <div>
              <Label className="text-sm font-medium">Auto-sync</Label>
              <p className="text-xs text-zinc-500">Sync data every hour</p>
            </div>
            <Switch checked={autoSync} onCheckedChange={setAutoSync} />
          </div>
          
          <div className="flex items-center justify-between p-3 rounded-lg bg-zinc-800/50">
            <div>
              <Label className="text-sm font-medium">Include in Entropy</Label>
              <p className="text-xs text-zinc-500">Use HRV data for wellness insights</p>
            </div>
            <Switch checked={includeInEntropy} onCheckedChange={setIncludeInEntropy} />
          </div>
        </div>

        {/* Privacy Notice */}
        <div className="p-3 rounded-lg bg-blue-500/10 border border-blue-500/20">
          <p className="text-xs text-zinc-400">
            <strong className="text-blue-300">Privacy:</strong> Health data is processed locally and never shared without your consent. 
            Only aggregated insights are used to improve your wellness recommendations.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
