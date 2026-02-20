import { useState, useEffect, useMemo } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Calendar, 
  Clock, 
  User, 
  Video, 
  Phone, 
  MapPin,
  ChevronLeft,
  ChevronRight,
  Check,
  X,
  Star,
  MessageSquare,
  Shield,
  AlertCircle
} from 'lucide-react';

interface Therapist {
  id: string;
  name: string;
  title: string;
  specialties: string[];
  rating: number;
  reviewCount: number;
  avatar?: string;
  availableSlots: TimeSlot[];
  sessionTypes: ('video' | 'phone' | 'inPerson')[];
  location?: string;
  bio: string;
  acceptingNew: boolean;
}

interface TimeSlot {
  id: string;
  date: string;
  startTime: string;
  endTime: string;
  available: boolean;
}

interface Appointment {
  id: string;
  therapistId: string;
  therapistName: string;
  date: string;
  startTime: string;
  endTime: string;
  type: 'video' | 'phone' | 'inPerson';
  status: 'scheduled' | 'completed' | 'cancelled';
  notes?: string;
}

interface TherapistSchedulingProps {
  onAppointmentBooked?: (appointment: Appointment) => void;
  clientView?: boolean;
}

// Mock therapist data
const mockTherapists: Therapist[] = [
  {
    id: '1',
    name: 'Dr. Sarah Chen',
    title: 'Licensed Clinical Psychologist',
    specialties: ['Anxiety', 'Depression', 'Trauma', 'PTSD'],
    rating: 4.9,
    reviewCount: 127,
    sessionTypes: ['video', 'phone', 'inPerson'],
    location: 'San Francisco, CA',
    bio: 'Specializing in evidence-based treatments for anxiety and trauma with 15+ years of experience.',
    acceptingNew: true,
    availableSlots: generateSlots('1'),
  },
  {
    id: '2',
    name: 'Dr. Michael Torres',
    title: 'Licensed Marriage & Family Therapist',
    specialties: ['Relationships', 'Family Therapy', 'Grief', 'Life Transitions'],
    rating: 4.8,
    reviewCount: 89,
    sessionTypes: ['video', 'phone'],
    bio: 'Helping individuals and families navigate life challenges with compassion and expertise.',
    acceptingNew: true,
    availableSlots: generateSlots('2'),
  },
  {
    id: '3',
    name: 'Dr. Emily Watson',
    title: 'Licensed Clinical Social Worker',
    specialties: ['Stress', 'Self-Esteem', 'Career Counseling', 'Mindfulness'],
    rating: 4.7,
    reviewCount: 64,
    sessionTypes: ['video'],
    bio: 'Integrating mindfulness-based approaches with cognitive behavioral therapy.',
    acceptingNew: false,
    availableSlots: generateSlots('3'),
  },
];

function generateSlots(therapistId: string): TimeSlot[] {
  const slots: TimeSlot[] = [];
  const today = new Date();
  
  for (let day = 1; day <= 14; day++) {
    const date = new Date(today);
    date.setDate(today.getDate() + day);
    
    // Skip weekends for some therapists
    if (therapistId === '2' && (date.getDay() === 0 || date.getDay() === 6)) continue;
    
    const dateStr = date.toISOString().split('T')[0];
    const times = ['09:00', '10:00', '11:00', '14:00', '15:00', '16:00'];
    
    times.forEach((time, idx) => {
      // Random availability
      const available = Math.random() > 0.3;
      slots.push({
        id: `${therapistId}-${dateStr}-${time}`,
        date: dateStr,
        startTime: time,
        endTime: `${parseInt(time.split(':')[0]) + 1}:00`,
        available,
      });
    });
  }
  
  return slots;
}

export default function TherapistScheduling({ onAppointmentBooked, clientView = true }: TherapistSchedulingProps) {
  const [selectedTherapist, setSelectedTherapist] = useState<Therapist | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<TimeSlot | null>(null);
  const [selectedType, setSelectedType] = useState<'video' | 'phone' | 'inPerson'>('video');
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [currentWeekStart, setCurrentWeekStart] = useState(new Date());
  const [bookingStep, setBookingStep] = useState<'therapist' | 'datetime' | 'confirm'>('therapist');
  const [bookingNotes, setBookingNotes] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);

  // Load appointments from localStorage
  useEffect(() => {
    const stored = localStorage.getItem('reunity_appointments');
    if (stored) {
      setAppointments(JSON.parse(stored));
    }
  }, []);

  // Get dates for current week view
  const weekDates = useMemo(() => {
    const dates: Date[] = [];
    for (let i = 0; i < 7; i++) {
      const date = new Date(currentWeekStart);
      date.setDate(currentWeekStart.getDate() + i);
      dates.push(date);
    }
    return dates;
  }, [currentWeekStart]);

  const navigateWeek = (direction: 'prev' | 'next') => {
    const newStart = new Date(currentWeekStart);
    newStart.setDate(currentWeekStart.getDate() + (direction === 'next' ? 7 : -7));
    setCurrentWeekStart(newStart);
  };

  const formatDate = (date: Date) => {
    return date.toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' });
  };

  const getSlotsForDate = (therapist: Therapist, dateStr: string) => {
    return therapist.availableSlots.filter(slot => slot.date === dateStr && slot.available);
  };

  const bookAppointment = () => {
    if (!selectedTherapist || !selectedSlot) return;

    const newAppointment: Appointment = {
      id: Date.now().toString(),
      therapistId: selectedTherapist.id,
      therapistName: selectedTherapist.name,
      date: selectedSlot.date,
      startTime: selectedSlot.startTime,
      endTime: selectedSlot.endTime,
      type: selectedType,
      status: 'scheduled',
      notes: bookingNotes,
    };

    const updatedAppointments = [...appointments, newAppointment];
    setAppointments(updatedAppointments);
    localStorage.setItem('reunity_appointments', JSON.stringify(updatedAppointments));

    // Mark slot as unavailable
    selectedTherapist.availableSlots = selectedTherapist.availableSlots.map(slot =>
      slot.id === selectedSlot.id ? { ...slot, available: false } : slot
    );

    onAppointmentBooked?.(newAppointment);
    setShowSuccess(true);
    
    setTimeout(() => {
      setShowSuccess(false);
      resetBooking();
    }, 3000);
  };

  const cancelAppointment = (appointmentId: string) => {
    const updatedAppointments = appointments.map(apt =>
      apt.id === appointmentId ? { ...apt, status: 'cancelled' as const } : apt
    );
    setAppointments(updatedAppointments);
    localStorage.setItem('reunity_appointments', JSON.stringify(updatedAppointments));
  };

  const resetBooking = () => {
    setSelectedTherapist(null);
    setSelectedDate(null);
    setSelectedSlot(null);
    setBookingStep('therapist');
    setBookingNotes('');
  };

  const upcomingAppointments = appointments.filter(
    apt => apt.status === 'scheduled' && new Date(apt.date) >= new Date()
  );

  return (
    <div className="space-y-6">
      {/* Success Message */}
      {showSuccess && (
        <div className="fixed top-4 right-4 z-50 bg-emerald-600 text-white px-6 py-4 rounded-lg shadow-lg flex items-center gap-3 animate-in slide-in-from-right">
          <Check className="w-5 h-5" />
          <div>
            <p className="font-medium">Appointment Booked!</p>
            <p className="text-sm opacity-90">You'll receive a confirmation email shortly.</p>
          </div>
        </div>
      )}

      {/* Upcoming Appointments */}
      {upcomingAppointments.length > 0 && (
        <Card className="bg-emerald-900/20 border-emerald-700/50">
          <CardHeader className="pb-2">
            <div className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-emerald-400" />
              <CardTitle className="text-lg text-emerald-300">Upcoming Appointments</CardTitle>
            </div>
          </CardHeader>
          <CardContent className="space-y-2">
            {upcomingAppointments.map(apt => (
              <div key={apt.id} className="flex items-center justify-between bg-emerald-950/30 rounded-lg p-3">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                    apt.type === 'video' ? 'bg-blue-500/20' : 
                    apt.type === 'phone' ? 'bg-amber-500/20' : 'bg-emerald-500/20'
                  }`}>
                    {apt.type === 'video' ? <Video className="w-5 h-5 text-blue-400" /> :
                     apt.type === 'phone' ? <Phone className="w-5 h-5 text-amber-400" /> :
                     <MapPin className="w-5 h-5 text-emerald-400" />}
                  </div>
                  <div>
                    <p className="font-medium text-white">{apt.therapistName}</p>
                    <p className="text-sm text-emerald-300/70">
                      {new Date(apt.date).toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })} at {apt.startTime}
                    </p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {apt.type === 'video' && (
                    <Button size="sm" className="bg-blue-600 hover:bg-blue-500">
                      <Video className="w-4 h-4 mr-1" />
                      Join
                    </Button>
                  )}
                  <Button 
                    size="sm" 
                    variant="ghost" 
                    className="text-red-400 hover:bg-red-500/20"
                    onClick={() => cancelAppointment(apt.id)}
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              </div>
            ))}
          </CardContent>
        </Card>
      )}

      {/* Booking Flow */}
      <Card className="bg-zinc-900/80 border-zinc-800">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Calendar className="w-5 h-5 text-emerald-400" />
              <CardTitle className="text-lg text-white">Book an Appointment</CardTitle>
            </div>
            {bookingStep !== 'therapist' && (
              <Button variant="ghost" size="sm" onClick={resetBooking} className="text-zinc-400">
                <ChevronLeft className="w-4 h-4 mr-1" />
                Start Over
              </Button>
            )}
          </div>
          <CardDescription className="text-zinc-400">
            {bookingStep === 'therapist' && 'Choose a therapist that fits your needs'}
            {bookingStep === 'datetime' && `Select a time with ${selectedTherapist?.name}`}
            {bookingStep === 'confirm' && 'Review and confirm your appointment'}
          </CardDescription>
        </CardHeader>

        <CardContent>
          {/* Step 1: Select Therapist */}
          {bookingStep === 'therapist' && (
            <div className="space-y-4">
              {mockTherapists.map(therapist => (
                <div
                  key={therapist.id}
                  className={`p-4 rounded-lg border cursor-pointer transition-all ${
                    therapist.acceptingNew
                      ? 'bg-zinc-800/50 border-zinc-700 hover:border-emerald-500/50'
                      : 'bg-zinc-800/30 border-zinc-800 opacity-60'
                  }`}
                  onClick={() => therapist.acceptingNew && (setSelectedTherapist(therapist), setBookingStep('datetime'))}
                >
                  <div className="flex items-start gap-4">
                    <div className="w-14 h-14 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center text-white text-xl font-medium">
                      {therapist.name.split(' ').map(n => n[0]).join('')}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <div>
                          <h3 className="font-medium text-white">{therapist.name}</h3>
                          <p className="text-sm text-zinc-400">{therapist.title}</p>
                        </div>
                        {!therapist.acceptingNew && (
                          <span className="text-xs px-2 py-1 rounded bg-zinc-700 text-zinc-400">
                            Not accepting new clients
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-2 mt-2">
                        <div className="flex items-center gap-1">
                          <Star className="w-4 h-4 text-amber-400 fill-amber-400" />
                          <span className="text-sm text-white">{therapist.rating}</span>
                          <span className="text-sm text-zinc-500">({therapist.reviewCount})</span>
                        </div>
                        <span className="text-zinc-600">•</span>
                        <div className="flex items-center gap-1">
                          {therapist.sessionTypes.includes('video') && <Video className="w-4 h-4 text-blue-400" />}
                          {therapist.sessionTypes.includes('phone') && <Phone className="w-4 h-4 text-amber-400" />}
                          {therapist.sessionTypes.includes('inPerson') && <MapPin className="w-4 h-4 text-emerald-400" />}
                        </div>
                      </div>
                      <div className="flex flex-wrap gap-1 mt-2">
                        {therapist.specialties.slice(0, 4).map(spec => (
                          <span key={spec} className="text-xs px-2 py-0.5 rounded-full bg-zinc-700 text-zinc-300">
                            {spec}
                          </span>
                        ))}
                      </div>
                      <p className="text-sm text-zinc-500 mt-2 line-clamp-2">{therapist.bio}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}

          {/* Step 2: Select Date & Time */}
          {bookingStep === 'datetime' && selectedTherapist && (
            <div className="space-y-4">
              {/* Session Type Selection */}
              <div>
                <label className="text-sm text-zinc-400 mb-2 block">Session Type</label>
                <div className="flex gap-2">
                  {selectedTherapist.sessionTypes.includes('video') && (
                    <Button
                      variant={selectedType === 'video' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setSelectedType('video')}
                      className={selectedType === 'video' ? 'bg-blue-600' : 'border-zinc-700'}
                    >
                      <Video className="w-4 h-4 mr-1" />
                      Video
                    </Button>
                  )}
                  {selectedTherapist.sessionTypes.includes('phone') && (
                    <Button
                      variant={selectedType === 'phone' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setSelectedType('phone')}
                      className={selectedType === 'phone' ? 'bg-amber-600' : 'border-zinc-700'}
                    >
                      <Phone className="w-4 h-4 mr-1" />
                      Phone
                    </Button>
                  )}
                  {selectedTherapist.sessionTypes.includes('inPerson') && (
                    <Button
                      variant={selectedType === 'inPerson' ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setSelectedType('inPerson')}
                      className={selectedType === 'inPerson' ? 'bg-emerald-600' : 'border-zinc-700'}
                    >
                      <MapPin className="w-4 h-4 mr-1" />
                      In Person
                    </Button>
                  )}
                </div>
              </div>

              {/* Week Navigation */}
              <div className="flex items-center justify-between">
                <Button variant="ghost" size="sm" onClick={() => navigateWeek('prev')} className="text-zinc-400">
                  <ChevronLeft className="w-4 h-4" />
                </Button>
                <span className="text-sm text-zinc-300">
                  {formatDate(weekDates[0])} - {formatDate(weekDates[6])}
                </span>
                <Button variant="ghost" size="sm" onClick={() => navigateWeek('next')} className="text-zinc-400">
                  <ChevronRight className="w-4 h-4" />
                </Button>
              </div>

              {/* Date Selection */}
              <div className="grid grid-cols-7 gap-2">
                {weekDates.map(date => {
                  const dateStr = date.toISOString().split('T')[0];
                  const slots = getSlotsForDate(selectedTherapist, dateStr);
                  const isSelected = selectedDate === dateStr;
                  const isPast = date < new Date();
                  
                  return (
                    <button
                      key={dateStr}
                      disabled={isPast || slots.length === 0}
                      onClick={() => setSelectedDate(dateStr)}
                      className={`p-2 rounded-lg text-center transition-all ${
                        isSelected
                          ? 'bg-emerald-600 text-white'
                          : isPast || slots.length === 0
                            ? 'bg-zinc-800/30 text-zinc-600 cursor-not-allowed'
                            : 'bg-zinc-800 text-zinc-300 hover:bg-zinc-700'
                      }`}
                    >
                      <div className="text-xs">{date.toLocaleDateString('en-US', { weekday: 'short' })}</div>
                      <div className="text-lg font-medium">{date.getDate()}</div>
                      {slots.length > 0 && !isPast && (
                        <div className="text-xs text-emerald-400">{slots.length} slots</div>
                      )}
                    </button>
                  );
                })}
              </div>

              {/* Time Slots */}
              {selectedDate && (
                <div>
                  <label className="text-sm text-zinc-400 mb-2 block">Available Times</label>
                  <div className="grid grid-cols-3 gap-2">
                    {getSlotsForDate(selectedTherapist, selectedDate).map(slot => (
                      <Button
                        key={slot.id}
                        variant={selectedSlot?.id === slot.id ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setSelectedSlot(slot)}
                        className={selectedSlot?.id === slot.id ? 'bg-emerald-600' : 'border-zinc-700'}
                      >
                        <Clock className="w-3 h-3 mr-1" />
                        {slot.startTime}
                      </Button>
                    ))}
                  </div>
                </div>
              )}

              {/* Continue Button */}
              {selectedSlot && (
                <Button
                  className="w-full bg-emerald-600 hover:bg-emerald-700"
                  onClick={() => setBookingStep('confirm')}
                >
                  Continue to Confirmation
                </Button>
              )}
            </div>
          )}

          {/* Step 3: Confirm Booking */}
          {bookingStep === 'confirm' && selectedTherapist && selectedSlot && (
            <div className="space-y-4">
              <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
                <div className="flex items-center gap-3">
                  <div className="w-12 h-12 rounded-full bg-gradient-to-br from-emerald-500 to-teal-600 flex items-center justify-center text-white font-medium">
                    {selectedTherapist.name.split(' ').map(n => n[0]).join('')}
                  </div>
                  <div>
                    <h3 className="font-medium text-white">{selectedTherapist.name}</h3>
                    <p className="text-sm text-zinc-400">{selectedTherapist.title}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 pt-2 border-t border-zinc-700">
                  <div>
                    <p className="text-xs text-zinc-500">Date</p>
                    <p className="text-white">
                      {new Date(selectedSlot.date).toLocaleDateString('en-US', { 
                        weekday: 'long', 
                        month: 'long', 
                        day: 'numeric' 
                      })}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500">Time</p>
                    <p className="text-white">{selectedSlot.startTime} - {selectedSlot.endTime}</p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500">Session Type</p>
                    <p className="text-white flex items-center gap-1">
                      {selectedType === 'video' && <><Video className="w-4 h-4 text-blue-400" /> Video Call</>}
                      {selectedType === 'phone' && <><Phone className="w-4 h-4 text-amber-400" /> Phone Call</>}
                      {selectedType === 'inPerson' && <><MapPin className="w-4 h-4 text-emerald-400" /> In Person</>}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-zinc-500">Duration</p>
                    <p className="text-white">60 minutes</p>
                  </div>
                </div>
              </div>

              {/* Notes */}
              <div>
                <label className="text-sm text-zinc-400 mb-2 block">Notes for your therapist (optional)</label>
                <textarea
                  value={bookingNotes}
                  onChange={(e) => setBookingNotes(e.target.value)}
                  placeholder="Anything you'd like your therapist to know before the session..."
                  className="w-full bg-zinc-800 border border-zinc-700 rounded-lg p-3 text-white placeholder:text-zinc-500 resize-none h-24"
                />
              </div>

              {/* Privacy Notice */}
              <div className="flex items-start gap-2 p-3 bg-zinc-800/30 rounded-lg">
                <Shield className="w-4 h-4 text-emerald-400 mt-0.5" />
                <p className="text-xs text-zinc-400">
                  Your appointment details are encrypted and only shared with your therapist. 
                  You can cancel up to 24 hours before without penalty.
                </p>
              </div>

              {/* Confirm Button */}
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  className="flex-1 border-zinc-700"
                  onClick={() => setBookingStep('datetime')}
                >
                  Back
                </Button>
                <Button
                  className="flex-1 bg-emerald-600 hover:bg-emerald-700"
                  onClick={bookAppointment}
                >
                  <Check className="w-4 h-4 mr-1" />
                  Confirm Booking
                </Button>
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
