import React, { useState, useEffect, useMemo } from 'react';
import { 
  View, 
  Text, 
  StyleSheet, 
  TouchableOpacity, 
  TextInput, 
  ScrollView, 
  Alert,
  Modal,
  Linking
} from 'react-native';
import AsyncStorage from '@react-native-async-storage/async-storage';
import { Ionicons } from '@expo/vector-icons';

interface Therapist {
  id: string;
  name: string;
  title: string;
  specialties: string[];
  rating: number;
  reviewCount: number;
  sessionTypes: ('video' | 'phone' | 'inPerson')[];
  bio: string;
  acceptingNew: boolean;
  availableSlots: TimeSlot[];
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
}

function generateSlots(therapistId: string): TimeSlot[] {
  const slots: TimeSlot[] = [];
  const today = new Date();
  
  for (let day = 1; day <= 14; day++) {
    const date = new Date(today);
    date.setDate(today.getDate() + day);
    
    if (therapistId === '2' && (date.getDay() === 0 || date.getDay() === 6)) continue;
    
    const dateStr = date.toISOString().split('T')[0];
    const times = ['09:00', '10:00', '11:00', '14:00', '15:00', '16:00'];
    
    times.forEach((time) => {
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

const mockTherapists: Therapist[] = [
  {
    id: '1',
    name: 'Dr. Sarah Chen',
    title: 'Licensed Clinical Psychologist',
    specialties: ['Anxiety', 'Depression', 'Trauma', 'PTSD'],
    rating: 4.9,
    reviewCount: 127,
    sessionTypes: ['video', 'phone', 'inPerson'],
    bio: 'Specializing in evidence-based treatments for anxiety and trauma with 15+ years of experience.',
    acceptingNew: true,
    availableSlots: generateSlots('1'),
  },
  {
    id: '2',
    name: 'Dr. Michael Torres',
    title: 'Licensed Marriage & Family Therapist',
    specialties: ['Relationships', 'Family Therapy', 'Grief'],
    rating: 4.8,
    reviewCount: 89,
    sessionTypes: ['video', 'phone'],
    bio: 'Helping individuals and families navigate life challenges with compassion.',
    acceptingNew: true,
    availableSlots: generateSlots('2'),
  },
];

export default function TherapistScheduling({ onAppointmentBooked }: TherapistSchedulingProps) {
  const [selectedTherapist, setSelectedTherapist] = useState<Therapist | null>(null);
  const [selectedDate, setSelectedDate] = useState<string | null>(null);
  const [selectedSlot, setSelectedSlot] = useState<TimeSlot | null>(null);
  const [selectedType, setSelectedType] = useState<'video' | 'phone' | 'inPerson'>('video');
  const [appointments, setAppointments] = useState<Appointment[]>([]);
  const [bookingStep, setBookingStep] = useState<'therapist' | 'datetime' | 'confirm'>('therapist');
  const [bookingNotes, setBookingNotes] = useState('');
  const [showSuccess, setShowSuccess] = useState(false);

  useEffect(() => {
    loadAppointments();
  }, []);

  const loadAppointments = async () => {
    try {
      const stored = await AsyncStorage.getItem('reunity_appointments');
      if (stored) {
        setAppointments(JSON.parse(stored));
      }
    } catch (error) {
      console.error('Failed to load appointments:', error);
    }
  };

  const weekDates = useMemo(() => {
    const dates: Date[] = [];
    const today = new Date();
    for (let i = 1; i <= 14; i++) {
      const date = new Date(today);
      date.setDate(today.getDate() + i);
      dates.push(date);
    }
    return dates;
  }, []);

  const getSlotsForDate = (therapist: Therapist, dateStr: string) => {
    return therapist.availableSlots.filter(slot => slot.date === dateStr && slot.available);
  };

  const bookAppointment = async () => {
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
    
    try {
      await AsyncStorage.setItem('reunity_appointments', JSON.stringify(updatedAppointments));
      setAppointments(updatedAppointments);
      onAppointmentBooked?.(newAppointment);
      setShowSuccess(true);
      
      setTimeout(() => {
        setShowSuccess(false);
        resetBooking();
      }, 3000);
    } catch (error) {
      Alert.alert('Error', 'Failed to book appointment');
    }
  };

  const cancelAppointment = async (appointmentId: string) => {
    Alert.alert(
      'Cancel Appointment',
      'Are you sure you want to cancel this appointment?',
      [
        { text: 'Keep', style: 'cancel' },
        { 
          text: 'Cancel', 
          style: 'destructive',
          onPress: async () => {
            const updatedAppointments = appointments.map(apt =>
              apt.id === appointmentId ? { ...apt, status: 'cancelled' as const } : apt
            );
            await AsyncStorage.setItem('reunity_appointments', JSON.stringify(updatedAppointments));
            setAppointments(updatedAppointments);
          }
        },
      ]
    );
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
    <ScrollView style={styles.container}>
      {/* Success Modal */}
      <Modal visible={showSuccess} transparent animationType="fade">
        <View style={styles.successOverlay}>
          <View style={styles.successContent}>
            <Ionicons name="checkmark-circle" size={48} color="#10b981" />
            <Text style={styles.successTitle}>Appointment Booked!</Text>
            <Text style={styles.successText}>You'll receive a confirmation shortly.</Text>
          </View>
        </View>
      </Modal>

      {/* Upcoming Appointments */}
      {upcomingAppointments.length > 0 && (
        <View style={styles.upcomingSection}>
          <View style={styles.sectionHeader}>
            <Ionicons name="calendar" size={20} color="#10b981" />
            <Text style={styles.sectionTitle}>Upcoming Appointments</Text>
          </View>
          
          {upcomingAppointments.map(apt => (
            <View key={apt.id} style={styles.appointmentCard}>
              <View style={styles.appointmentInfo}>
                <View style={[
                  styles.typeIcon,
                  apt.type === 'video' && styles.videoIcon,
                  apt.type === 'phone' && styles.phoneIcon,
                ]}>
                  <Ionicons 
                    name={apt.type === 'video' ? 'videocam' : apt.type === 'phone' ? 'call' : 'location'} 
                    size={20} 
                    color="#fff" 
                  />
                </View>
                <View>
                  <Text style={styles.appointmentName}>{apt.therapistName}</Text>
                  <Text style={styles.appointmentDate}>
                    {new Date(apt.date).toLocaleDateString()} at {apt.startTime}
                  </Text>
                </View>
              </View>
              <View style={styles.appointmentActions}>
                {apt.type === 'video' && (
                  <TouchableOpacity style={styles.joinButton}>
                    <Ionicons name="videocam" size={16} color="#fff" />
                    <Text style={styles.joinText}>Join</Text>
                  </TouchableOpacity>
                )}
                <TouchableOpacity 
                  style={styles.cancelButton}
                  onPress={() => cancelAppointment(apt.id)}
                >
                  <Ionicons name="close" size={16} color="#ef4444" />
                </TouchableOpacity>
              </View>
            </View>
          ))}
        </View>
      )}

      {/* Booking Flow */}
      <View style={styles.bookingSection}>
        <View style={styles.sectionHeader}>
          <Ionicons name="calendar" size={20} color="#10b981" />
          <Text style={styles.sectionTitle}>Book an Appointment</Text>
          {bookingStep !== 'therapist' && (
            <TouchableOpacity onPress={resetBooking}>
              <Text style={styles.resetText}>Start Over</Text>
            </TouchableOpacity>
          )}
        </View>

        {/* Step 1: Select Therapist */}
        {bookingStep === 'therapist' && (
          <View>
            {mockTherapists.map(therapist => (
              <TouchableOpacity
                key={therapist.id}
                style={[
                  styles.therapistCard,
                  !therapist.acceptingNew && styles.therapistDisabled
                ]}
                disabled={!therapist.acceptingNew}
                onPress={() => {
                  setSelectedTherapist(therapist);
                  setBookingStep('datetime');
                }}
              >
                <View style={styles.therapistAvatar}>
                  <Text style={styles.avatarText}>
                    {therapist.name.split(' ').map(n => n[0]).join('')}
                  </Text>
                </View>
                <View style={styles.therapistInfo}>
                  <View style={styles.therapistHeader}>
                    <Text style={styles.therapistName}>{therapist.name}</Text>
                    {!therapist.acceptingNew && (
                      <Text style={styles.notAccepting}>Not accepting</Text>
                    )}
                  </View>
                  <Text style={styles.therapistTitle}>{therapist.title}</Text>
                  <View style={styles.ratingRow}>
                    <Ionicons name="star" size={14} color="#f59e0b" />
                    <Text style={styles.rating}>{therapist.rating}</Text>
                    <Text style={styles.reviews}>({therapist.reviewCount})</Text>
                    <View style={styles.sessionTypes}>
                      {therapist.sessionTypes.includes('video') && (
                        <Ionicons name="videocam" size={14} color="#3b82f6" />
                      )}
                      {therapist.sessionTypes.includes('phone') && (
                        <Ionicons name="call" size={14} color="#f59e0b" />
                      )}
                    </View>
                  </View>
                  <View style={styles.specialties}>
                    {therapist.specialties.slice(0, 3).map(spec => (
                      <View key={spec} style={styles.specialtyBadge}>
                        <Text style={styles.specialtyText}>{spec}</Text>
                      </View>
                    ))}
                  </View>
                </View>
              </TouchableOpacity>
            ))}
          </View>
        )}

        {/* Step 2: Select Date & Time */}
        {bookingStep === 'datetime' && selectedTherapist && (
          <View>
            {/* Session Type */}
            <Text style={styles.label}>Session Type</Text>
            <View style={styles.typeRow}>
              {selectedTherapist.sessionTypes.includes('video') && (
                <TouchableOpacity
                  style={[styles.typeButton, selectedType === 'video' && styles.typeButtonActive]}
                  onPress={() => setSelectedType('video')}
                >
                  <Ionicons name="videocam" size={16} color={selectedType === 'video' ? '#fff' : '#3b82f6'} />
                  <Text style={[styles.typeText, selectedType === 'video' && styles.typeTextActive]}>Video</Text>
                </TouchableOpacity>
              )}
              {selectedTherapist.sessionTypes.includes('phone') && (
                <TouchableOpacity
                  style={[styles.typeButton, selectedType === 'phone' && styles.typeButtonActive]}
                  onPress={() => setSelectedType('phone')}
                >
                  <Ionicons name="call" size={16} color={selectedType === 'phone' ? '#fff' : '#f59e0b'} />
                  <Text style={[styles.typeText, selectedType === 'phone' && styles.typeTextActive]}>Phone</Text>
                </TouchableOpacity>
              )}
            </View>

            {/* Date Selection */}
            <Text style={styles.label}>Select Date</Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.dateScroll}>
              {weekDates.map(date => {
                const dateStr = date.toISOString().split('T')[0];
                const slots = getSlotsForDate(selectedTherapist, dateStr);
                const isSelected = selectedDate === dateStr;
                
                return (
                  <TouchableOpacity
                    key={dateStr}
                    style={[
                      styles.dateButton,
                      isSelected && styles.dateButtonActive,
                      slots.length === 0 && styles.dateButtonDisabled
                    ]}
                    disabled={slots.length === 0}
                    onPress={() => setSelectedDate(dateStr)}
                  >
                    <Text style={[styles.dateDay, isSelected && styles.dateDayActive]}>
                      {date.toLocaleDateString('en-US', { weekday: 'short' })}
                    </Text>
                    <Text style={[styles.dateNum, isSelected && styles.dateNumActive]}>
                      {date.getDate()}
                    </Text>
                    {slots.length > 0 && (
                      <Text style={styles.slotsCount}>{slots.length}</Text>
                    )}
                  </TouchableOpacity>
                );
              })}
            </ScrollView>

            {/* Time Slots */}
            {selectedDate && (
              <>
                <Text style={styles.label}>Available Times</Text>
                <View style={styles.slotsGrid}>
                  {getSlotsForDate(selectedTherapist, selectedDate).map(slot => (
                    <TouchableOpacity
                      key={slot.id}
                      style={[
                        styles.slotButton,
                        selectedSlot?.id === slot.id && styles.slotButtonActive
                      ]}
                      onPress={() => setSelectedSlot(slot)}
                    >
                      <Ionicons 
                        name="time" 
                        size={14} 
                        color={selectedSlot?.id === slot.id ? '#fff' : '#10b981'} 
                      />
                      <Text style={[
                        styles.slotText,
                        selectedSlot?.id === slot.id && styles.slotTextActive
                      ]}>
                        {slot.startTime}
                      </Text>
                    </TouchableOpacity>
                  ))}
                </View>
              </>
            )}

            {/* Continue Button */}
            {selectedSlot && (
              <TouchableOpacity
                style={styles.continueButton}
                onPress={() => setBookingStep('confirm')}
              >
                <Text style={styles.continueText}>Continue to Confirmation</Text>
              </TouchableOpacity>
            )}
          </View>
        )}

        {/* Step 3: Confirm */}
        {bookingStep === 'confirm' && selectedTherapist && selectedSlot && (
          <View>
            <View style={styles.confirmCard}>
              <View style={styles.confirmHeader}>
                <View style={styles.therapistAvatar}>
                  <Text style={styles.avatarText}>
                    {selectedTherapist.name.split(' ').map(n => n[0]).join('')}
                  </Text>
                </View>
                <View>
                  <Text style={styles.therapistName}>{selectedTherapist.name}</Text>
                  <Text style={styles.therapistTitle}>{selectedTherapist.title}</Text>
                </View>
              </View>
              
              <View style={styles.confirmDetails}>
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Date</Text>
                  <Text style={styles.detailValue}>
                    {new Date(selectedSlot.date).toLocaleDateString('en-US', { 
                      weekday: 'long', 
                      month: 'long', 
                      day: 'numeric' 
                    })}
                  </Text>
                </View>
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Time</Text>
                  <Text style={styles.detailValue}>{selectedSlot.startTime} - {selectedSlot.endTime}</Text>
                </View>
                <View style={styles.detailRow}>
                  <Text style={styles.detailLabel}>Type</Text>
                  <Text style={styles.detailValue}>
                    {selectedType === 'video' ? 'Video Call' : selectedType === 'phone' ? 'Phone Call' : 'In Person'}
                  </Text>
                </View>
              </View>
            </View>

            <TextInput
              style={styles.notesInput}
              placeholder="Notes for your therapist (optional)"
              placeholderTextColor="#71717a"
              multiline
              value={bookingNotes}
              onChangeText={setBookingNotes}
            />

            <View style={styles.confirmButtons}>
              <TouchableOpacity
                style={styles.backButton}
                onPress={() => setBookingStep('datetime')}
              >
                <Text style={styles.backText}>Back</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={styles.confirmButton}
                onPress={bookAppointment}
              >
                <Ionicons name="checkmark" size={20} color="#fff" />
                <Text style={styles.confirmText}>Confirm Booking</Text>
              </TouchableOpacity>
            </View>
          </View>
        )}
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
  },
  upcomingSection: {
    margin: 16,
    padding: 16,
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(16, 185, 129, 0.2)',
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    gap: 8,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    flex: 1,
  },
  resetText: {
    color: '#71717a',
    fontSize: 14,
  },
  appointmentCard: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(16, 185, 129, 0.1)',
    borderRadius: 8,
    padding: 12,
    marginBottom: 8,
  },
  appointmentInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 12,
  },
  typeIcon: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: '#10b981',
    justifyContent: 'center',
    alignItems: 'center',
  },
  videoIcon: {
    backgroundColor: '#3b82f6',
  },
  phoneIcon: {
    backgroundColor: '#f59e0b',
  },
  appointmentName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  appointmentDate: {
    fontSize: 14,
    color: '#a1a1aa',
  },
  appointmentActions: {
    flexDirection: 'row',
    gap: 8,
  },
  joinButton: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#3b82f6',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    gap: 4,
  },
  joinText: {
    color: '#fff',
    fontWeight: '600',
  },
  cancelButton: {
    padding: 8,
    backgroundColor: 'rgba(239, 68, 68, 0.2)',
    borderRadius: 8,
  },
  bookingSection: {
    margin: 16,
    padding: 16,
    backgroundColor: '#18181b',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#27272a',
  },
  therapistCard: {
    flexDirection: 'row',
    backgroundColor: '#27272a',
    borderRadius: 12,
    padding: 16,
    marginBottom: 12,
  },
  therapistDisabled: {
    opacity: 0.5,
  },
  therapistAvatar: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: '#10b981',
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  avatarText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '600',
  },
  therapistInfo: {
    flex: 1,
  },
  therapistHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  therapistName: {
    fontSize: 16,
    fontWeight: '600',
    color: '#fff',
  },
  notAccepting: {
    fontSize: 10,
    color: '#71717a',
    backgroundColor: '#3f3f46',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 4,
  },
  therapistTitle: {
    fontSize: 12,
    color: '#a1a1aa',
    marginBottom: 4,
  },
  ratingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
    marginBottom: 8,
  },
  rating: {
    fontSize: 14,
    color: '#fff',
  },
  reviews: {
    fontSize: 12,
    color: '#71717a',
  },
  sessionTypes: {
    flexDirection: 'row',
    gap: 8,
    marginLeft: 8,
  },
  specialties: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 4,
  },
  specialtyBadge: {
    backgroundColor: '#3f3f46',
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 12,
  },
  specialtyText: {
    fontSize: 10,
    color: '#d4d4d8',
  },
  label: {
    fontSize: 14,
    color: '#a1a1aa',
    marginBottom: 8,
    marginTop: 16,
  },
  typeRow: {
    flexDirection: 'row',
    gap: 8,
  },
  typeButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#3f3f46',
    gap: 8,
  },
  typeButtonActive: {
    backgroundColor: '#3f3f46',
    borderColor: '#52525b',
  },
  typeText: {
    color: '#a1a1aa',
  },
  typeTextActive: {
    color: '#fff',
  },
  dateScroll: {
    marginBottom: 16,
  },
  dateButton: {
    width: 60,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#27272a',
    alignItems: 'center',
    marginRight: 8,
  },
  dateButtonActive: {
    backgroundColor: '#10b981',
  },
  dateButtonDisabled: {
    opacity: 0.4,
  },
  dateDay: {
    fontSize: 12,
    color: '#a1a1aa',
  },
  dateDayActive: {
    color: '#fff',
  },
  dateNum: {
    fontSize: 18,
    fontWeight: '600',
    color: '#fff',
    marginVertical: 4,
  },
  dateNumActive: {
    color: '#fff',
  },
  slotsCount: {
    fontSize: 10,
    color: '#10b981',
  },
  slotsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  slotButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#3f3f46',
    gap: 4,
  },
  slotButtonActive: {
    backgroundColor: '#10b981',
    borderColor: '#10b981',
  },
  slotText: {
    color: '#d4d4d8',
  },
  slotTextActive: {
    color: '#fff',
  },
  continueButton: {
    backgroundColor: '#10b981',
    padding: 16,
    borderRadius: 8,
    alignItems: 'center',
    marginTop: 24,
  },
  continueText: {
    color: '#fff',
    fontWeight: '600',
    fontSize: 16,
  },
  confirmCard: {
    backgroundColor: '#27272a',
    borderRadius: 12,
    padding: 16,
  },
  confirmHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#3f3f46',
  },
  confirmDetails: {
    gap: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  detailLabel: {
    color: '#71717a',
  },
  detailValue: {
    color: '#fff',
  },
  notesInput: {
    backgroundColor: '#27272a',
    borderWidth: 1,
    borderColor: '#3f3f46',
    borderRadius: 8,
    padding: 12,
    color: '#fff',
    height: 80,
    textAlignVertical: 'top',
    marginTop: 16,
  },
  confirmButtons: {
    flexDirection: 'row',
    gap: 12,
    marginTop: 16,
  },
  backButton: {
    flex: 1,
    padding: 16,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#3f3f46',
    alignItems: 'center',
  },
  backText: {
    color: '#a1a1aa',
    fontWeight: '600',
  },
  confirmButton: {
    flex: 2,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#10b981',
    padding: 16,
    borderRadius: 8,
    gap: 8,
  },
  confirmText: {
    color: '#fff',
    fontWeight: '600',
  },
  successOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  successContent: {
    backgroundColor: '#18181b',
    borderRadius: 16,
    padding: 32,
    alignItems: 'center',
  },
  successTitle: {
    fontSize: 20,
    fontWeight: '700',
    color: '#fff',
    marginTop: 16,
  },
  successText: {
    color: '#a1a1aa',
    marginTop: 8,
  },
});
