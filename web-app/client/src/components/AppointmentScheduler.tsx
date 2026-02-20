import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Calendar, 
  Clock, 
  Plus, 
  Trash2, 
  Bell, 
  MapPin,
  User,
  Video,
  Phone as PhoneIcon,
  CheckCircle,
  AlertCircle
} from 'lucide-react';
import { toast } from 'sonner';

interface Appointment {
  id: string;
  title: string;
  provider: string;
  date: string;
  time: string;
  type: 'in-person' | 'video' | 'phone';
  location?: string;
  notes?: string;
  reminder: boolean;
  completed: boolean;
}

export function AppointmentScheduler() {
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [isAdding, setIsAdding] = useState(false);
  const [newAppointment, setNewAppointment] = useState({
    title: '',
    provider: '',
    date: '',
    time: '',
    type: 'in-person' as 'in-person' | 'video' | 'phone',
    location: '',
    notes: '',
    reminder: true,
  });

  useEffect(() => {
    const saved = localStorage.getItem('reunity_appointments');
    if (saved) {
      setAppointments(JSON.parse(saved));
    }
  }, []);

  useEffect(() => {
    // Check for upcoming appointments and show reminders
    const checkReminders = () => {
      const now = new Date();
      appointments.forEach(apt => {
        if (!apt.reminder || apt.completed) return;
        
        const aptDate = new Date(`${apt.date}T${apt.time}`);
        const diff = aptDate.getTime() - now.getTime();
        const hoursUntil = diff / (1000 * 60 * 60);
        
        // Remind 24 hours before
        if (hoursUntil > 0 && hoursUntil <= 24 && hoursUntil > 23) {
          if (Notification.permission === 'granted') {
            new Notification('Appointment Tomorrow', {
              body: `${apt.title} with ${apt.provider} at ${apt.time}`,
              icon: '/reop-logo.png',
            });
          }
        }
        
        // Remind 1 hour before
        if (hoursUntil > 0 && hoursUntil <= 1) {
          if (Notification.permission === 'granted') {
            new Notification('Appointment in 1 Hour', {
              body: `${apt.title} with ${apt.provider}`,
              icon: '/reop-logo.png',
            });
          }
        }
      });
    };

    const interval = setInterval(checkReminders, 60000);
    return () => clearInterval(interval);
  }, [appointments]);

  const saveAppointments = (newAppointments: Appointment[]) => {
    setAppointments(newAppointments);
    localStorage.setItem('reunity_appointments', JSON.stringify(newAppointments));
  };

  const addAppointment = () => {
    if (!newAppointment.title || !newAppointment.date || !newAppointment.time) {
      toast.error('Please fill in required fields');
      return;
    }

    const appointment: Appointment = {
      id: Date.now().toString(),
      ...newAppointment,
      completed: false,
    };

    saveAppointments([...appointments, appointment].sort((a, b) => 
      new Date(`${a.date}T${a.time}`).getTime() - new Date(`${b.date}T${b.time}`).getTime()
    ));

    setNewAppointment({
      title: '',
      provider: '',
      date: '',
      time: '',
      type: 'in-person',
      location: '',
      notes: '',
      reminder: true,
    });
    setIsAdding(false);
    toast.success('Appointment scheduled');

    // Request notification permission
    if (newAppointment.reminder && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  };

  const deleteAppointment = (id: string) => {
    saveAppointments(appointments.filter(a => a.id !== id));
    toast.success('Appointment removed');
  };

  const toggleComplete = (id: string) => {
    saveAppointments(
      appointments.map(a =>
        a.id === id ? { ...a, completed: !a.completed } : a
      )
    );
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'video': return <Video className="w-4 h-4" />;
      case 'phone': return <PhoneIcon className="w-4 h-4" />;
      default: return <MapPin className="w-4 h-4" />;
    }
  };

  const formatDate = (dateStr: string): string => {
    const date = new Date(dateStr);
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(tomorrow.getDate() + 1);

    if (date.toDateString() === today.toDateString()) return 'Today';
    if (date.toDateString() === tomorrow.toDateString()) return 'Tomorrow';
    
    return date.toLocaleDateString('en-US', { 
      weekday: 'short', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  const upcomingAppointments = appointments.filter(a => {
    const aptDate = new Date(`${a.date}T${a.time}`);
    return aptDate >= new Date() && !a.completed;
  });

  const pastAppointments = appointments.filter(a => {
    const aptDate = new Date(`${a.date}T${a.time}`);
    return aptDate < new Date() || a.completed;
  });

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Calendar className="w-5 h-5 text-blue-400" />
            Appointments
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAdding(true)}
            className="gap-1"
          >
            <Plus className="w-4 h-4" />
            Schedule
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Add Appointment Form */}
        {isAdding && (
          <div className="mb-4 p-4 bg-zinc-800/50 rounded-xl border border-zinc-700 space-y-3">
            <Input
              value={newAppointment.title}
              onChange={e => setNewAppointment({ ...newAppointment, title: e.target.value })}
              placeholder="Appointment type (e.g., Therapy Session)"
            />
            <Input
              value={newAppointment.provider}
              onChange={e => setNewAppointment({ ...newAppointment, provider: e.target.value })}
              placeholder="Provider name"
            />
            <div className="grid grid-cols-2 gap-3">
              <Input
                type="date"
                value={newAppointment.date}
                onChange={e => setNewAppointment({ ...newAppointment, date: e.target.value })}
              />
              <Input
                type="time"
                value={newAppointment.time}
                onChange={e => setNewAppointment({ ...newAppointment, time: e.target.value })}
              />
            </div>
            <div className="flex gap-2">
              {(['in-person', 'video', 'phone'] as const).map(type => (
                <button
                  key={type}
                  onClick={() => setNewAppointment({ ...newAppointment, type })}
                  className={`
                    flex-1 py-2 rounded-lg text-sm font-medium transition-all flex items-center justify-center gap-2
                    ${newAppointment.type === type
                      ? 'bg-blue-600 text-white'
                      : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}
                  `}
                >
                  {getTypeIcon(type)}
                  {type.charAt(0).toUpperCase() + type.slice(1)}
                </button>
              ))}
            </div>
            {newAppointment.type === 'in-person' && (
              <Input
                value={newAppointment.location}
                onChange={e => setNewAppointment({ ...newAppointment, location: e.target.value })}
                placeholder="Location/Address"
              />
            )}
            <Input
              value={newAppointment.notes}
              onChange={e => setNewAppointment({ ...newAppointment, notes: e.target.value })}
              placeholder="Notes (optional)"
            />
            <label className="flex items-center gap-2 text-sm text-zinc-400">
              <input
                type="checkbox"
                checked={newAppointment.reminder}
                onChange={e => setNewAppointment({ ...newAppointment, reminder: e.target.checked })}
                className="rounded"
              />
              <Bell className="w-4 h-4" />
              Remind me before appointment
            </label>
            <div className="flex gap-2">
              <Button onClick={addAppointment} className="flex-1">
                Schedule
              </Button>
              <Button variant="outline" onClick={() => setIsAdding(false)}>
                Cancel
              </Button>
            </div>
          </div>
        )}

        {/* Upcoming Appointments */}
        {upcomingAppointments.length > 0 ? (
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-zinc-400">Upcoming</h3>
            {upcomingAppointments.map(apt => (
              <div
                key={apt.id}
                className="p-4 bg-zinc-800/50 rounded-xl border border-zinc-700/50 hover:border-blue-600/30 transition-all"
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <h4 className="font-medium text-white">{apt.title}</h4>
                    <div className="flex items-center gap-2 text-sm text-zinc-400 mt-1">
                      <User className="w-3 h-3" />
                      {apt.provider}
                    </div>
                    <div className="flex items-center gap-4 mt-2 text-sm">
                      <span className="flex items-center gap-1 text-blue-400">
                        <Calendar className="w-3 h-3" />
                        {formatDate(apt.date)}
                      </span>
                      <span className="flex items-center gap-1 text-zinc-400">
                        <Clock className="w-3 h-3" />
                        {apt.time}
                      </span>
                      <span className="flex items-center gap-1 text-zinc-400">
                        {getTypeIcon(apt.type)}
                        {apt.type}
                      </span>
                    </div>
                    {apt.location && (
                      <p className="text-sm text-zinc-500 mt-1 flex items-center gap-1">
                        <MapPin className="w-3 h-3" />
                        {apt.location}
                      </p>
                    )}
                    {apt.notes && (
                      <p className="text-sm text-zinc-500 mt-1">{apt.notes}</p>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    {apt.reminder && (
                      <Bell className="w-4 h-4 text-blue-400" />
                    )}
                    <button
                      onClick={() => toggleComplete(apt.id)}
                      className="p-1.5 rounded-lg text-zinc-400 hover:text-emerald-400 hover:bg-zinc-700 transition-colors"
                      title="Mark as completed"
                    >
                      <CheckCircle className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => deleteAppointment(apt.id)}
                      className="p-1.5 rounded-lg text-zinc-400 hover:text-red-400 hover:bg-zinc-700 transition-colors"
                    >
                      <Trash2 className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-zinc-500">
            <Calendar className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No upcoming appointments</p>
            <p className="text-sm">Schedule your next session</p>
          </div>
        )}

        {/* Past/Completed Appointments */}
        {pastAppointments.length > 0 && (
          <div className="mt-4 pt-4 border-t border-zinc-800">
            <h3 className="text-sm font-medium text-zinc-500 mb-2">Past</h3>
            <div className="space-y-2 max-h-32 overflow-y-auto">
              {pastAppointments.slice(0, 5).map(apt => (
                <div
                  key={apt.id}
                  className="flex items-center justify-between p-2 bg-zinc-800/30 rounded-lg text-sm opacity-60"
                >
                  <div className="flex items-center gap-2">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    <span className="text-zinc-400">{apt.title}</span>
                  </div>
                  <span className="text-zinc-500">{formatDate(apt.date)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default AppointmentScheduler;
