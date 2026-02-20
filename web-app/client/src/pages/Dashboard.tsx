import { useState } from 'react';
import { Link } from 'wouter';
import { Button } from '@/components/ui/button';
import { MoodCalendar } from '@/components/MoodCalendar';
import { DailyAffirmations } from '@/components/DailyAffirmations';
import { ProgressBadges } from '@/components/ProgressBadges';
import { MedicationReminder } from '@/components/MedicationReminder';
import { CheckInSystem } from '@/components/CheckInSystem';
import { SleepTracker } from '@/components/SleepTracker';
import { CommunityForum } from '@/components/CommunityForum';
import { ResourceBookmarks } from '@/components/ResourceBookmarks';
import { AppointmentScheduler } from '@/components/AppointmentScheduler';
import WearableIntegration from '@/components/WearableIntegration';
import MoodPrediction from '@/components/MoodPrediction';
import { EmergencyContacts, HighRiskAlertDialog } from '@/components/EmergencyContacts';
import TherapistScheduling from '@/components/TherapistScheduling';
import JournalWithSentiment from '@/components/JournalWithSentiment';
import GroupTherapySessions from '@/components/GroupTherapySessions';
import Gamification from '@/components/Gamification';
import CaregiverDashboard from '@/components/CaregiverDashboard';
import PeerSupportMatching from '@/components/PeerSupportMatching';
import SleepTracking from '@/components/SleepTracking';
import FamilyGroupChat from '@/components/FamilyGroupChat';
import { MedicationInteractionChecker } from '@/components/MedicationInteractionChecker';
import { CrisisInterventionTimeline } from '@/components/CrisisInterventionTimeline';
import { CommunitySupportGroups } from '@/components/CommunitySupportGroups';
import MedicationReminders from '@/components/MedicationReminders';
import WellnessReportExport from '@/components/WellnessReportExport';
import GuidedMeditationLibrary from '@/components/GuidedMeditationLibrary';
import { BreathingExercises } from '@/components/BreathingExercises';
import { TherapistNotesSync } from '@/components/TherapistNotesSync';
import { DailyAffirmationsEnhanced } from '@/components/DailyAffirmationsEnhanced';
import { SymptomTracker } from '@/components/SymptomTracker';
import { SocialConnectionPrompts } from '@/components/SocialConnectionPrompts';
import { 
  LayoutDashboard, 
  MessageCircle, 
  BookOpen, 
  Shield, 
  Users, 
  Wind,
  Phone,
  ArrowLeft,
  Settings,
  TrendingUp,
  Moon,
  Bookmark,
  Watch,
  Activity,
  Trophy,
  Heart,
  Video,
  Pill,
  Clock,
  FileText,
  Thermometer,
  UserPlus
} from 'lucide-react';

export default function Dashboard() {
  const [activeSection, setActiveSection] = useState<'overview' | 'wellness' | 'journal' | 'tools' | 'groups' | 'achievements' | 'caregiver' | 'peers' | 'sleep' | 'family' | 'meds' | 'timeline' | 'community' | 'reminders' | 'reports' | 'meditation' | 'breathing' | 'calendar' | 'notes' | 'symptoms' | 'social'>('overview');
  const [showHighRiskAlert, setShowHighRiskAlert] = useState(false);
  const [emergencyContacts, setEmergencyContacts] = useState<any[]>([]);

  return (
    <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950">
      {/* Header */}
      <header className="sticky top-0 z-40 bg-zinc-950/80 backdrop-blur-lg border-b border-zinc-800">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Link href="/">
                <Button variant="ghost" size="icon">
                  <ArrowLeft className="w-5 h-5" />
                </Button>
              </Link>
              <div className="flex items-center gap-2">
                <LayoutDashboard className="w-6 h-6 text-emerald-400" />
                <h1 className="text-xl font-semibold text-white">My Dashboard</h1>
              </div>
            </div>
            <Link href="/resources">
              <Button variant="outline" size="sm" className="gap-2">
                <Settings className="w-4 h-4" />
                Settings
              </Button>
            </Link>
          </div>

          {/* Section tabs */}
          <div className="flex gap-2 mt-4">
            {[
              { id: 'overview', label: 'Overview', icon: TrendingUp },
              { id: 'wellness', label: 'Wellness', icon: Wind },
              { id: 'journal', label: 'Journal', icon: BookOpen },
              { id: 'tools', label: 'Tools', icon: Shield },
              { id: 'groups', label: 'Groups', icon: Video },
              { id: 'achievements', label: 'Achievements', icon: Trophy },
              { id: 'caregiver', label: 'Caregiver', icon: Heart },
              { id: 'peers', label: 'Peers', icon: Users },
              { id: 'sleep', label: 'Sleep', icon: Moon },
              { id: 'family', label: 'Family', icon: MessageCircle },
              { id: 'meds', label: 'Medications', icon: Pill },
              { id: 'timeline', label: 'Timeline', icon: Clock },
              { id: 'community', label: 'Community', icon: Users },
              { id: 'reminders', label: 'Reminders', icon: Pill },
              { id: 'reports', label: 'Reports', icon: BookOpen },
              { id: 'meditation', label: 'Meditation', icon: Wind },
              { id: 'breathing', label: 'Breathing', icon: Wind },
              { id: 'calendar', label: 'Calendar', icon: Activity },
              { id: 'notes', label: 'Notes', icon: FileText },
              { id: 'symptoms', label: 'Symptoms', icon: Thermometer },
              { id: 'social', label: 'Social', icon: UserPlus },
            ].map(section => (
              <Button
                key={section.id}
                variant={activeSection === section.id ? 'default' : 'ghost'}
                size="sm"
                onClick={() => setActiveSection(section.id as typeof activeSection)}
                className="gap-2"
              >
                <section.icon className="w-4 h-4" />
                {section.label}
              </Button>
            ))}
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {activeSection === 'overview' && (
          <div className="space-y-6">
            {/* Quick actions */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <Link href="/chat">
                <div className="p-4 bg-gradient-to-br from-emerald-900/30 to-teal-900/30 rounded-xl border border-emerald-800/30 hover:border-emerald-600/50 transition-all cursor-pointer">
                  <MessageCircle className="w-8 h-8 text-emerald-400 mb-2" />
                  <h3 className="font-medium text-white">Chat</h3>
                  <p className="text-xs text-zinc-400">Talk to ReUnity</p>
                </div>
              </Link>
              <Link href="/journal">
                <div className="p-4 bg-gradient-to-br from-purple-900/30 to-pink-900/30 rounded-xl border border-purple-800/30 hover:border-purple-600/50 transition-all cursor-pointer">
                  <BookOpen className="w-8 h-8 text-purple-400 mb-2" />
                  <h3 className="font-medium text-white">Journal</h3>
                  <p className="text-xs text-zinc-400">Write your thoughts</p>
                </div>
              </Link>
              <Link href="/grounding">
                <div className="p-4 bg-gradient-to-br from-blue-900/30 to-cyan-900/30 rounded-xl border border-blue-800/30 hover:border-blue-600/50 transition-all cursor-pointer">
                  <Wind className="w-8 h-8 text-blue-400 mb-2" />
                  <h3 className="font-medium text-white">Grounding</h3>
                  <p className="text-xs text-zinc-400">Calming exercises</p>
                </div>
              </Link>
              <Link href="/resources">
                <div className="p-4 bg-gradient-to-br from-amber-900/30 to-orange-900/30 rounded-xl border border-amber-800/30 hover:border-amber-600/50 transition-all cursor-pointer">
                  <Phone className="w-8 h-8 text-amber-400 mb-2" />
                  <h3 className="font-medium text-white">Resources</h3>
                  <p className="text-xs text-zinc-400">Emergency help</p>
                </div>
              </Link>
            </div>

            {/* Gamification Widget */}
            <Gamification compact />

            {/* Social Connection Prompts */}
            <SocialConnectionPrompts />

            {/* Main dashboard grid */}
            <div className="grid md:grid-cols-2 gap-6">
              <MoodPrediction />
              <DailyAffirmationsEnhanced />
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <MoodCalendar />
              <CheckInSystem />
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <ProgressBadges />
              <WearableIntegration />
            </div>

            <div className="grid md:grid-cols-3 gap-4">
              <PeerSupportMatching compact />
              <SleepTracking compact />
              <FamilyGroupChat compact />
            </div>
          </div>
        )}

        {activeSection === 'wellness' && (
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <MoodCalendar />
              <DailyAffirmationsEnhanced />
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <SymptomTracker />
              <SocialConnectionPrompts />
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <SleepTracker />
              <WearableIntegration />
            </div>
            <div className="grid md:grid-cols-2 gap-6">
              <MedicationReminder />
              <AppointmentScheduler />
            </div>
            <ProgressBadges />
          </div>
        )}

        {activeSection === 'tools' && (
          <div className="space-y-6">
            <div className="grid md:grid-cols-2 gap-6">
              <CheckInSystem />
              <MedicationReminder />
            </div>

            {/* Therapist Scheduling */}
            <TherapistScheduling clientView={true} />

            {/* Emergency Contacts */}
            <EmergencyContacts />

            {/* Tool links */}
            <div className="grid md:grid-cols-3 gap-4">
              <Link href="/safety-plan">
                <div className="p-6 bg-zinc-900/80 rounded-xl border border-zinc-800 hover:border-emerald-600/50 transition-all cursor-pointer">
                  <Shield className="w-10 h-10 text-emerald-400 mb-3" />
                  <h3 className="font-medium text-white mb-1">Safety Plan</h3>
                  <p className="text-sm text-zinc-400">Your personalized escape plan with biometric protection</p>
                </div>
              </Link>
              <Link href="/peer-support">
                <div className="p-6 bg-zinc-900/80 rounded-xl border border-zinc-800 hover:border-purple-600/50 transition-all cursor-pointer">
                  <Users className="w-10 h-10 text-purple-400 mb-3" />
                  <h3 className="font-medium text-white mb-1">Peer Support</h3>
                  <p className="text-sm text-zinc-400">Connect with others who understand</p>
                </div>
              </Link>
              <Link href="/therapist">
                <div className="p-6 bg-zinc-900/80 rounded-xl border border-zinc-800 hover:border-blue-600/50 transition-all cursor-pointer">
                  <TrendingUp className="w-10 h-10 text-blue-400 mb-3" />
                  <h3 className="font-medium text-white mb-1">Therapist Portal</h3>
                  <p className="text-sm text-zinc-400">For licensed professionals</p>
                </div>
              </Link>
            </div>
          </div>
        )}

        {activeSection === 'journal' && (
          <div className="space-y-6">
            <JournalWithSentiment 
              onCrisisDetected={() => setShowHighRiskAlert(true)}
            />
          </div>
        )}

        {activeSection === 'groups' && (
          <div className="space-y-6">
            <GroupTherapySessions />
          </div>
        )}

        {activeSection === 'achievements' && (
          <div className="space-y-6">
            <Gamification />
          </div>
        )}

        {activeSection === 'caregiver' && (
          <div className="space-y-6">
            <CaregiverDashboard />
          </div>
        )}

        {activeSection === 'peers' && (
          <div className="space-y-6">
            <PeerSupportMatching />
          </div>
        )}

        {activeSection === 'sleep' && (
          <div className="space-y-6">
            <SleepTracking />
          </div>
        )}

        {activeSection === 'family' && (
          <div className="space-y-6">
            <FamilyGroupChat />
          </div>
        )}

        {activeSection === 'meds' && (
          <div className="space-y-6">
            <MedicationInteractionChecker />
          </div>
        )}

        {activeSection === 'timeline' && (
          <div className="space-y-6">
            <CrisisInterventionTimeline />
          </div>
        )}

        {activeSection === 'community' && (
          <div className="space-y-6">
            <CommunitySupportGroups />
          </div>
        )}

        {activeSection === 'reminders' && (
          <div className="space-y-6">
            <MedicationReminders />
          </div>
        )}

        {activeSection === 'reports' && (
          <div className="space-y-6">
            <WellnessReportExport />
          </div>
        )}

        {activeSection === 'meditation' && (
          <div className="space-y-6">
            <GuidedMeditationLibrary />
          </div>
        )}

        {activeSection === 'breathing' && (
          <div className="space-y-6">
            <BreathingExercises />
          </div>
        )}

        {activeSection === 'calendar' && (
          <div className="space-y-6">
            <MoodCalendar />
          </div>
        )}

        {activeSection === 'notes' && (
          <div className="space-y-6">
            <TherapistNotesSync />
          </div>
        )}

        {activeSection === 'symptoms' && (
          <div className="space-y-6">
            <SymptomTracker />
          </div>
        )}

        {activeSection === 'social' && (
          <div className="space-y-6">
            <SocialConnectionPrompts />
            <PeerSupportMatching />
          </div>
        )}
      </main>

      {/* High Risk Alert Dialog */}
      <HighRiskAlertDialog 
        isOpen={showHighRiskAlert} 
        onClose={() => setShowHighRiskAlert(false)} 
        contacts={emergencyContacts}
      />

      {/* Footer */}
      <footer className="border-t border-zinc-800 py-6 mt-12">
        <div className="container mx-auto px-4 text-center text-sm text-zinc-500">
          <p>ReUnity is not a substitute for professional mental health care.</p>
          <p className="mt-1">If you're in crisis, please call 988 (Suicide & Crisis Lifeline)</p>
          <p className="mt-2">
            <a href="https://entropy-physics-ai.com" target="_blank" rel="noopener noreferrer" className="text-emerald-400 hover:text-emerald-300">entropy-physics-ai.com</a>
          </p>
        </div>
      </footer>
    </div>
  );
}
