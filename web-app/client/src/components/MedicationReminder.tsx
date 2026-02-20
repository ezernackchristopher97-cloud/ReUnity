import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Switch } from '@/components/ui/switch';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Pill, Plus, Clock, Check, X, Bell, Trash2, Edit2 } from 'lucide-react';
import { toast } from 'sonner';

interface Medication {
  id: string;
  name: string;
  dosage: string;
  times: string[]; // Array of times like ["08:00", "20:00"]
  notes?: string;
  enabled: boolean;
}

interface MedicationLog {
  medicationId: string;
  timestamp: number;
  taken: boolean;
}

export function MedicationReminder() {
  const [medications, setMedications] = useState<Medication[]>([]);
  const [logs, setLogs] = useState<MedicationLog[]>([]);
  const [isAddingNew, setIsAddingNew] = useState(false);
  const [editingMed, setEditingMed] = useState<Medication | null>(null);
  const [newMed, setNewMed] = useState({
    name: '',
    dosage: '',
    times: ['08:00'],
    notes: '',
  });

  useEffect(() => {
    // Load saved medications and logs
    const savedMeds = localStorage.getItem('reunity_medications');
    const savedLogs = localStorage.getItem('reunity_medication_logs');
    
    if (savedMeds) {
      setMedications(JSON.parse(savedMeds));
    }
    
    if (savedLogs) {
      // Only keep logs from the last 7 days
      const weekAgo = Date.now() - 7 * 24 * 60 * 60 * 1000;
      const recentLogs = JSON.parse(savedLogs).filter((log: MedicationLog) => log.timestamp > weekAgo);
      setLogs(recentLogs);
    }

    // Check for reminders
    checkReminders();
    const interval = setInterval(checkReminders, 60000); // Check every minute
    
    return () => clearInterval(interval);
  }, []);

  const checkReminders = () => {
    const now = new Date();
    const currentTime = `${String(now.getHours()).padStart(2, '0')}:${String(now.getMinutes()).padStart(2, '0')}`;
    
    medications.forEach(med => {
      if (med.enabled && med.times.includes(currentTime)) {
        // Check if already taken today at this time
        const today = new Date().toDateString();
        const alreadyLogged = logs.some(log => 
          log.medicationId === med.id && 
          new Date(log.timestamp).toDateString() === today &&
          new Date(log.timestamp).getHours() === now.getHours()
        );
        
        if (!alreadyLogged) {
          showReminder(med);
        }
      }
    });
  };

  const showReminder = (med: Medication) => {
    if ('Notification' in window && Notification.permission === 'granted') {
      new Notification('Medication Reminder', {
        body: `Time to take ${med.name} (${med.dosage})`,
        icon: '/icon-192.png',
      });
    }
    
    toast.info(`Time to take ${med.name} (${med.dosage})`, {
      duration: 10000,
      action: {
        label: 'Mark Taken',
        onClick: () => logMedication(med.id, true),
      },
    });
  };

  const saveMedications = (meds: Medication[]) => {
    setMedications(meds);
    localStorage.setItem('reunity_medications', JSON.stringify(meds));
  };

  const saveLogs = (newLogs: MedicationLog[]) => {
    setLogs(newLogs);
    localStorage.setItem('reunity_medication_logs', JSON.stringify(newLogs));
  };

  const addMedication = () => {
    if (!newMed.name || !newMed.dosage) {
      toast.error('Please enter medication name and dosage');
      return;
    }

    const medication: Medication = {
      id: Date.now().toString(),
      name: newMed.name,
      dosage: newMed.dosage,
      times: newMed.times,
      notes: newMed.notes,
      enabled: true,
    };

    saveMedications([...medications, medication]);
    setNewMed({ name: '', dosage: '', times: ['08:00'], notes: '' });
    setIsAddingNew(false);
    toast.success('Medication added');

    // Request notification permission
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  };

  const updateMedication = () => {
    if (!editingMed) return;
    
    const updated = medications.map(m => 
      m.id === editingMed.id ? editingMed : m
    );
    saveMedications(updated);
    setEditingMed(null);
    toast.success('Medication updated');
  };

  const deleteMedication = (id: string) => {
    saveMedications(medications.filter(m => m.id !== id));
    toast.info('Medication removed');
  };

  const toggleMedication = (id: string) => {
    const updated = medications.map(m => 
      m.id === id ? { ...m, enabled: !m.enabled } : m
    );
    saveMedications(updated);
  };

  const logMedication = (medicationId: string, taken: boolean) => {
    const log: MedicationLog = {
      medicationId,
      timestamp: Date.now(),
      taken,
    };
    saveLogs([...logs, log]);
    toast.success(taken ? 'Marked as taken' : 'Marked as skipped');
  };

  const addTime = () => {
    if (editingMed) {
      setEditingMed({ ...editingMed, times: [...editingMed.times, '12:00'] });
    } else {
      setNewMed({ ...newMed, times: [...newMed.times, '12:00'] });
    }
  };

  const removeTime = (index: number) => {
    if (editingMed) {
      setEditingMed({ 
        ...editingMed, 
        times: editingMed.times.filter((_, i) => i !== index) 
      });
    } else {
      setNewMed({ 
        ...newMed, 
        times: newMed.times.filter((_, i) => i !== index) 
      });
    }
  };

  const updateTime = (index: number, value: string) => {
    if (editingMed) {
      const times = [...editingMed.times];
      times[index] = value;
      setEditingMed({ ...editingMed, times });
    } else {
      const times = [...newMed.times];
      times[index] = value;
      setNewMed({ ...newMed, times });
    }
  };

  const getTodayLogs = (medicationId: string) => {
    const today = new Date().toDateString();
    return logs.filter(log => 
      log.medicationId === medicationId && 
      new Date(log.timestamp).toDateString() === today
    );
  };

  const getAdherenceRate = (medicationId: string) => {
    const medLogs = logs.filter(log => log.medicationId === medicationId);
    if (medLogs.length === 0) return 0;
    const taken = medLogs.filter(log => log.taken).length;
    return Math.round((taken / medLogs.length) * 100);
  };

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Pill className="w-5 h-5 text-emerald-400" />
            Medication Reminders
          </CardTitle>
          <Dialog open={isAddingNew} onOpenChange={setIsAddingNew}>
            <DialogTrigger asChild>
              <Button variant="outline" size="sm" className="gap-2">
                <Plus className="w-4 h-4" />
                Add
              </Button>
            </DialogTrigger>
            <DialogContent className="bg-zinc-900 border-zinc-800">
              <DialogHeader>
                <DialogTitle>Add Medication</DialogTitle>
              </DialogHeader>
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-zinc-400">Medication Name</label>
                  <Input
                    value={newMed.name}
                    onChange={e => setNewMed({ ...newMed, name: e.target.value })}
                    placeholder="e.g., Sertraline"
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm text-zinc-400">Dosage</label>
                  <Input
                    value={newMed.dosage}
                    onChange={e => setNewMed({ ...newMed, dosage: e.target.value })}
                    placeholder="e.g., 50mg"
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm text-zinc-400">Reminder Times</label>
                  <div className="space-y-2 mt-1">
                    {newMed.times.map((time, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <Input
                          type="time"
                          value={time}
                          onChange={e => updateTime(i, e.target.value)}
                          className="flex-1"
                        />
                        {newMed.times.length > 1 && (
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => removeTime(i)}
                          >
                            <X className="w-4 h-4" />
                          </Button>
                        )}
                      </div>
                    ))}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={addTime}
                      className="w-full"
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      Add Time
                    </Button>
                  </div>
                </div>
                <div>
                  <label className="text-sm text-zinc-400">Notes (optional)</label>
                  <Input
                    value={newMed.notes}
                    onChange={e => setNewMed({ ...newMed, notes: e.target.value })}
                    placeholder="e.g., Take with food"
                    className="mt-1"
                  />
                </div>
                <Button onClick={addMedication} className="w-full">
                  Add Medication
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </CardHeader>
      <CardContent>
        {medications.length === 0 ? (
          <div className="text-center py-8 text-zinc-500">
            <Pill className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No medications added yet</p>
            <p className="text-sm">Add your medications to get reminders</p>
          </div>
        ) : (
          <div className="space-y-3">
            {medications.map(med => {
              const todayLogs = getTodayLogs(med.id);
              const adherence = getAdherenceRate(med.id);
              
              return (
                <div
                  key={med.id}
                  className={`p-4 rounded-xl border ${
                    med.enabled ? 'bg-zinc-800/50 border-zinc-700' : 'bg-zinc-900/50 border-zinc-800 opacity-60'
                  }`}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <h4 className="font-medium text-white">{med.name}</h4>
                        <span className="text-sm text-zinc-400">{med.dosage}</span>
                      </div>
                      <div className="flex items-center gap-2 mt-1 text-sm text-zinc-500">
                        <Clock className="w-3 h-3" />
                        {med.times.join(', ')}
                      </div>
                      {med.notes && (
                        <p className="text-xs text-zinc-500 mt-1">{med.notes}</p>
                      )}
                    </div>
                    <div className="flex items-center gap-2">
                      <Switch
                        checked={med.enabled}
                        onCheckedChange={() => toggleMedication(med.id)}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setEditingMed(med)}
                      >
                        <Edit2 className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => deleteMedication(med.id)}
                      >
                        <Trash2 className="w-4 h-4 text-red-400" />
                      </Button>
                    </div>
                  </div>
                  
                  {/* Today's status */}
                  <div className="mt-3 pt-3 border-t border-zinc-700/50">
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-zinc-500">Today</span>
                      <div className="flex items-center gap-2">
                        {todayLogs.length > 0 ? (
                          todayLogs.map((log, i) => (
                            <span
                              key={i}
                              className={`w-6 h-6 rounded-full flex items-center justify-center ${
                                log.taken ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'
                              }`}
                            >
                              {log.taken ? <Check className="w-3 h-3" /> : <X className="w-3 h-3" />}
                            </span>
                          ))
                        ) : (
                          <span className="text-xs text-zinc-500">Not logged yet</span>
                        )}
                      </div>
                    </div>
                    
                    {/* Quick log buttons */}
                    {med.enabled && todayLogs.length < med.times.length && (
                      <div className="flex gap-2 mt-2">
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1 gap-2 text-emerald-400 border-emerald-400/30"
                          onClick={() => logMedication(med.id, true)}
                        >
                          <Check className="w-4 h-4" />
                          Taken
                        </Button>
                        <Button
                          variant="outline"
                          size="sm"
                          className="flex-1 gap-2 text-zinc-400"
                          onClick={() => logMedication(med.id, false)}
                        >
                          <X className="w-4 h-4" />
                          Skip
                        </Button>
                      </div>
                    )}
                    
                    {/* Adherence rate */}
                    {adherence > 0 && (
                      <div className="mt-2 text-xs text-zinc-500">
                        7-day adherence: <span className={adherence >= 80 ? 'text-emerald-400' : 'text-amber-400'}>{adherence}%</span>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        )}

        {/* Edit dialog */}
        <Dialog open={!!editingMed} onOpenChange={() => setEditingMed(null)}>
          <DialogContent className="bg-zinc-900 border-zinc-800">
            <DialogHeader>
              <DialogTitle>Edit Medication</DialogTitle>
            </DialogHeader>
            {editingMed && (
              <div className="space-y-4">
                <div>
                  <label className="text-sm text-zinc-400">Medication Name</label>
                  <Input
                    value={editingMed.name}
                    onChange={e => setEditingMed({ ...editingMed, name: e.target.value })}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm text-zinc-400">Dosage</label>
                  <Input
                    value={editingMed.dosage}
                    onChange={e => setEditingMed({ ...editingMed, dosage: e.target.value })}
                    className="mt-1"
                  />
                </div>
                <div>
                  <label className="text-sm text-zinc-400">Reminder Times</label>
                  <div className="space-y-2 mt-1">
                    {editingMed.times.map((time, i) => (
                      <div key={i} className="flex items-center gap-2">
                        <Input
                          type="time"
                          value={time}
                          onChange={e => updateTime(i, e.target.value)}
                          className="flex-1"
                        />
                        {editingMed.times.length > 1 && (
                          <Button
                            variant="ghost"
                            size="icon"
                            onClick={() => removeTime(i)}
                          >
                            <X className="w-4 h-4" />
                          </Button>
                        )}
                      </div>
                    ))}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={addTime}
                      className="w-full"
                    >
                      <Plus className="w-4 h-4 mr-2" />
                      Add Time
                    </Button>
                  </div>
                </div>
                <div>
                  <label className="text-sm text-zinc-400">Notes</label>
                  <Input
                    value={editingMed.notes || ''}
                    onChange={e => setEditingMed({ ...editingMed, notes: e.target.value })}
                    className="mt-1"
                  />
                </div>
                <Button onClick={updateMedication} className="w-full">
                  Save Changes
                </Button>
              </div>
            )}
          </DialogContent>
        </Dialog>
      </CardContent>
    </Card>
  );
}

export default MedicationReminder;
