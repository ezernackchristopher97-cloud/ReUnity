import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Bell, BellOff, Pill, Plus, Clock, AlertTriangle, Check, Trash2, Calendar, Package } from 'lucide-react';

interface MedicationSchedule {
  id: string;
  name: string;
  dosage: string;
  times: string[];
  daysOfWeek: number[];
  pillsRemaining: number;
  pillsPerDose: number;
  refillThreshold: number;
  lastTaken: Date | null;
  notificationsEnabled: boolean;
  createdAt: Date;
}

interface TakenLog {
  medicationId: string;
  takenAt: Date;
  scheduledTime: string;
}

export function MedicationReminders() {
  const [medications, setMedications] = useState<MedicationSchedule[]>(() => {
    const saved = localStorage.getItem('reunity-medications');
    return saved ? JSON.parse(saved) : [];
  });
  const [takenLogs, setTakenLogs] = useState<TakenLog[]>(() => {
    const saved = localStorage.getItem('reunity-taken-logs');
    return saved ? JSON.parse(saved) : [];
  });
  const [showAddForm, setShowAddForm] = useState(false);
  const [newMed, setNewMed] = useState({
    name: '',
    dosage: '',
    times: ['08:00'],
    pillsRemaining: 30,
    pillsPerDose: 1,
    refillThreshold: 7,
  });
  const [notificationsSupported, setNotificationsSupported] = useState(false);
  const [notificationsPermission, setNotificationsPermission] = useState<NotificationPermission>('default');

  useEffect(() => {
    if ('Notification' in window) {
      setNotificationsSupported(true);
      setNotificationsPermission(Notification.permission);
    }
  }, []);

  useEffect(() => {
    localStorage.setItem('reunity-medications', JSON.stringify(medications));
  }, [medications]);

  useEffect(() => {
    localStorage.setItem('reunity-taken-logs', JSON.stringify(takenLogs));
  }, [takenLogs]);

  // Check for medication reminders every minute
  useEffect(() => {
    const checkReminders = () => {
      const now = new Date();
      const currentTime = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}`;
      const currentDay = now.getDay();

      medications.forEach(med => {
        if (!med.notificationsEnabled) return;
        if (!med.daysOfWeek.includes(currentDay)) return;

        med.times.forEach(time => {
          if (time === currentTime) {
            // Check if already taken today at this time
            const todayStart = new Date(now);
            todayStart.setHours(0, 0, 0, 0);
            const alreadyTaken = takenLogs.some(log => 
              log.medicationId === med.id && 
              log.scheduledTime === time &&
              new Date(log.takenAt) >= todayStart
            );

            if (!alreadyTaken && notificationsPermission === 'granted') {
              new Notification(`Time to take ${med.name}`, {
                body: `${med.dosage} - ${med.pillsPerDose} pill(s)`,
                icon: '/pill-icon.png',
                tag: `med-${med.id}-${time}`,
              });
            }
          }
        });
      });
    };

    const interval = setInterval(checkReminders, 60000);
    return () => clearInterval(interval);
  }, [medications, takenLogs, notificationsPermission]);

  const requestNotificationPermission = async () => {
    if (!notificationsSupported) return;
    const permission = await Notification.requestPermission();
    setNotificationsPermission(permission);
  };

  const addMedication = () => {
    if (!newMed.name || !newMed.dosage) return;

    const medication: MedicationSchedule = {
      id: Date.now().toString(),
      name: newMed.name,
      dosage: newMed.dosage,
      times: newMed.times,
      daysOfWeek: [0, 1, 2, 3, 4, 5, 6], // Every day by default
      pillsRemaining: newMed.pillsRemaining,
      pillsPerDose: newMed.pillsPerDose,
      refillThreshold: newMed.refillThreshold,
      lastTaken: null,
      notificationsEnabled: notificationsPermission === 'granted',
      createdAt: new Date(),
    };

    setMedications([...medications, medication]);
    setNewMed({ name: '', dosage: '', times: ['08:00'], pillsRemaining: 30, pillsPerDose: 1, refillThreshold: 7 });
    setShowAddForm(false);
  };

  const markAsTaken = (medId: string, scheduledTime: string) => {
    const med = medications.find(m => m.id === medId);
    if (!med) return;

    // Add to taken logs
    const log: TakenLog = {
      medicationId: medId,
      takenAt: new Date(),
      scheduledTime,
    };
    setTakenLogs([...takenLogs, log]);

    // Update pills remaining
    setMedications(medications.map(m => 
      m.id === medId 
        ? { ...m, pillsRemaining: Math.max(0, m.pillsRemaining - m.pillsPerDose), lastTaken: new Date() }
        : m
    ));
  };

  const toggleNotifications = (medId: string) => {
    setMedications(medications.map(m =>
      m.id === medId ? { ...m, notificationsEnabled: !m.notificationsEnabled } : m
    ));
  };

  const deleteMedication = (medId: string) => {
    setMedications(medications.filter(m => m.id !== medId));
    setTakenLogs(takenLogs.filter(l => l.medicationId !== medId));
  };

  const refillMedication = (medId: string, amount: number = 30) => {
    setMedications(medications.map(m =>
      m.id === medId ? { ...m, pillsRemaining: m.pillsRemaining + amount } : m
    ));
  };

  const isTakenToday = (medId: string, time: string) => {
    const todayStart = new Date();
    todayStart.setHours(0, 0, 0, 0);
    return takenLogs.some(log =>
      log.medicationId === medId &&
      log.scheduledTime === time &&
      new Date(log.takenAt) >= todayStart
    );
  };

  const needsRefill = (med: MedicationSchedule) => {
    const daysRemaining = Math.floor(med.pillsRemaining / (med.pillsPerDose * med.times.length));
    return daysRemaining <= med.refillThreshold;
  };

  const lowStockMeds = medications.filter(needsRefill);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-purple-500/20 rounded-lg">
            <Pill className="w-6 h-6 text-purple-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-white">Medication Reminders</h2>
            <p className="text-sm text-zinc-400">Track medications and get timely reminders</p>
          </div>
        </div>
        <Button onClick={() => setShowAddForm(true)} className="gap-2 bg-purple-600 hover:bg-purple-700">
          <Plus className="w-4 h-4" />
          Add Medication
        </Button>
      </div>

      {/* Notification Permission Banner */}
      {notificationsSupported && notificationsPermission !== 'granted' && (
        <Card className="bg-amber-900/20 border-amber-800/30">
          <CardContent className="p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Bell className="w-5 h-5 text-amber-400" />
              <p className="text-amber-200">Enable notifications to receive medication reminders</p>
            </div>
            <Button onClick={requestNotificationPermission} variant="outline" size="sm" className="border-amber-600 text-amber-400 hover:bg-amber-900/30">
              Enable
            </Button>
          </CardContent>
        </Card>
      )}

      {/* Low Stock Alerts */}
      {lowStockMeds.length > 0 && (
        <Card className="bg-red-900/20 border-red-800/30">
          <CardHeader className="pb-2">
            <CardTitle className="text-red-400 flex items-center gap-2 text-base">
              <AlertTriangle className="w-5 h-5" />
              Refill Needed
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {lowStockMeds.map(med => {
              const daysRemaining = Math.floor(med.pillsRemaining / (med.pillsPerDose * med.times.length));
              return (
                <div key={med.id} className="flex items-center justify-between p-2 bg-red-900/10 rounded-lg">
                  <div>
                    <p className="text-white font-medium">{med.name}</p>
                    <p className="text-sm text-red-300">{med.pillsRemaining} pills left (~{daysRemaining} days)</p>
                  </div>
                  <Button onClick={() => refillMedication(med.id)} size="sm" variant="outline" className="border-red-600 text-red-400 hover:bg-red-900/30">
                    <Package className="w-4 h-4 mr-1" />
                    Refill
                  </Button>
                </div>
              );
            })}
          </CardContent>
        </Card>
      )}

      {/* Add Medication Form */}
      {showAddForm && (
        <Card className="bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="text-white">Add New Medication</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label className="text-zinc-300">Medication Name</Label>
                <Input
                  value={newMed.name}
                  onChange={e => setNewMed({ ...newMed, name: e.target.value })}
                  placeholder="e.g., Sertraline"
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-zinc-300">Dosage</Label>
                <Input
                  value={newMed.dosage}
                  onChange={e => setNewMed({ ...newMed, dosage: e.target.value })}
                  placeholder="e.g., 50mg"
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label className="text-zinc-300">Time</Label>
                <Input
                  type="time"
                  value={newMed.times[0]}
                  onChange={e => setNewMed({ ...newMed, times: [e.target.value] })}
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-zinc-300">Pills Remaining</Label>
                <Input
                  type="number"
                  value={newMed.pillsRemaining}
                  onChange={e => setNewMed({ ...newMed, pillsRemaining: parseInt(e.target.value) || 0 })}
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
              <div className="space-y-2">
                <Label className="text-zinc-300">Pills Per Dose</Label>
                <Input
                  type="number"
                  value={newMed.pillsPerDose}
                  onChange={e => setNewMed({ ...newMed, pillsPerDose: parseInt(e.target.value) || 1 })}
                  className="bg-zinc-800 border-zinc-700"
                />
              </div>
            </div>
            <div className="flex gap-2 justify-end">
              <Button variant="ghost" onClick={() => setShowAddForm(false)}>Cancel</Button>
              <Button onClick={addMedication} className="bg-purple-600 hover:bg-purple-700">Add Medication</Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Medication List */}
      <div className="space-y-4">
        {medications.length === 0 ? (
          <Card className="bg-zinc-900/50 border-zinc-800">
            <CardContent className="p-8 text-center">
              <Pill className="w-12 h-12 text-zinc-600 mx-auto mb-4" />
              <p className="text-zinc-400">No medications added yet</p>
              <p className="text-sm text-zinc-500 mt-1">Add your medications to get reminders and track refills</p>
            </CardContent>
          </Card>
        ) : (
          medications.map(med => {
            const daysRemaining = Math.floor(med.pillsRemaining / (med.pillsPerDose * med.times.length));
            return (
              <Card key={med.id} className="bg-zinc-900/50 border-zinc-800">
                <CardContent className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3">
                        <h3 className="text-lg font-medium text-white">{med.name}</h3>
                        <span className="px-2 py-0.5 bg-purple-500/20 text-purple-300 text-sm rounded">{med.dosage}</span>
                        {needsRefill(med) && (
                          <span className="px-2 py-0.5 bg-red-500/20 text-red-300 text-sm rounded flex items-center gap-1">
                            <AlertTriangle className="w-3 h-3" />
                            Low Stock
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-4 mt-2 text-sm text-zinc-400">
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {med.times.join(', ')}
                        </span>
                        <span className="flex items-center gap-1">
                          <Package className="w-4 h-4" />
                          {med.pillsRemaining} pills (~{daysRemaining} days)
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => toggleNotifications(med.id)}
                        className={med.notificationsEnabled ? 'text-purple-400' : 'text-zinc-500'}
                      >
                        {med.notificationsEnabled ? <Bell className="w-5 h-5" /> : <BellOff className="w-5 h-5" />}
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => deleteMedication(med.id)}
                        className="text-zinc-500 hover:text-red-400"
                      >
                        <Trash2 className="w-5 h-5" />
                      </Button>
                    </div>
                  </div>

                  {/* Today's Schedule */}
                  <div className="mt-4 pt-4 border-t border-zinc-800">
                    <p className="text-sm text-zinc-400 mb-2">Today's Schedule</p>
                    <div className="flex gap-2">
                      {med.times.map(time => {
                        const taken = isTakenToday(med.id, time);
                        return (
                          <Button
                            key={time}
                            variant={taken ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => !taken && markAsTaken(med.id, time)}
                            disabled={taken}
                            className={taken ? 'bg-green-600 hover:bg-green-600' : 'border-zinc-700'}
                          >
                            {taken ? <Check className="w-4 h-4 mr-1" /> : <Clock className="w-4 h-4 mr-1" />}
                            {time}
                          </Button>
                        );
                      })}
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })
        )}
      </div>
    </div>
  );
}

export default MedicationReminders;
