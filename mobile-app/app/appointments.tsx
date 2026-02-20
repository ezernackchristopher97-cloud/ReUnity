import React from 'react';
import { SafeAreaView, StyleSheet, StatusBar } from 'react-native';
import TherapistScheduling from '../components/TherapistScheduling';

export default function AppointmentsScreen() {
  const handleAppointmentBooked = (appointment: any) => {
    console.log('Appointment booked:', appointment);
    // Could sync with web app here
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#09090b" />
      <TherapistScheduling onAppointmentBooked={handleAppointmentBooked} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#09090b',
  },
});
