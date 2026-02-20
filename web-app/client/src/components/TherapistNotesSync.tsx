import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  FileText, 
  Clock, 
  CheckCircle, 
  AlertCircle, 
  Download,
  Eye,
  EyeOff,
  RefreshCw,
  User,
  Calendar
} from 'lucide-react';

interface TherapistNote {
  id: string;
  therapistId: string;
  therapistName: string;
  sessionDate: string;
  createdAt: string;
  type: 'session' | 'progress' | 'support_plan' | 'crisis';
  title: string;
  content: string;
  isSharedWithClient: boolean;
  includeInReport: boolean;
  tags: string[];
}

const mockNotes: TherapistNote[] = [
  {
    id: '1',
    therapistId: 't1',
    therapistName: 'Dr. Sarah Chen',
    sessionDate: '2026-01-20',
    createdAt: '2026-01-20T15:30:00Z',
    type: 'session',
    title: 'Weekly Check-in Session',
    content: 'Client showed improved coping strategies. Discussed grounding techniques and their application during anxiety episodes. Homework: Practice box breathing twice daily.',
    isSharedWithClient: true,
    includeInReport: true,
    tags: ['anxiety', 'coping', 'progress'],
  },
  {
    id: '2',
    therapistId: 't1',
    therapistName: 'Dr. Sarah Chen',
    sessionDate: '2026-01-13',
    createdAt: '2026-01-13T16:00:00Z',
    type: 'progress',
    title: 'Monthly Progress Review',
    content: 'Significant improvement in mood stability over the past month. Entropy scores trending downward. Sleep quality has improved with new bedtime routine. Continue current support plan.',
    isSharedWithClient: true,
    includeInReport: true,
    tags: ['progress', 'sleep', 'mood'],
  },
  {
    id: '3',
    therapistId: 't1',
    therapistName: 'Dr. Sarah Chen',
    sessionDate: '2026-01-06',
    createdAt: '2026-01-06T14:45:00Z',
    type: 'support_plan',
    title: 'Updated Wellness Goals',
    content: 'Goals for Q1 2026: 1) Reduce anxiety episodes to less than 2 per week. 2) Establish consistent sleep schedule. 3) Build support network through peer connections.',
    isSharedWithClient: true,
    includeInReport: true,
    tags: ['goals', 'treatment', 'planning'],
  },
  {
    id: '4',
    therapistId: 't1',
    therapistName: 'Dr. Sarah Chen',
    sessionDate: '2025-12-28',
    createdAt: '2025-12-28T10:15:00Z',
    type: 'crisis',
    title: 'Crisis Intervention Follow-up',
    content: 'Follow-up after crisis alert on 12/27. Client utilized safety plan effectively. Discussed triggers and added new coping strategies. Emergency contacts verified.',
    isSharedWithClient: false,
    includeInReport: false,
    tags: ['crisis', 'safety', 'intervention'],
  },
];

const noteTypeConfig = {
  session: { label: 'Session Note', color: '#10b981', icon: FileText },
  progress: { label: 'Progress Review', color: '#6366f1', icon: CheckCircle },
  support_plan: { label: 'Support Plan', color: '#8b5cf6', icon: Calendar },
  crisis: { label: 'Crisis Note', color: '#ef4444', icon: AlertCircle },
};

export function TherapistNotesSync({ compact = false }: { compact?: boolean }) {
  const [notes, setNotes] = useState<TherapistNote[]>(mockNotes);
  const [selectedNote, setSelectedNote] = useState<TherapistNote | null>(null);
  const [filter, setFilter] = useState<'all' | 'shared' | 'report'>('all');
  const [syncing, setSyncing] = useState(false);
  const [lastSync, setLastSync] = useState<Date>(new Date());

  useEffect(() => {
    const stored = localStorage.getItem('reunity-therapist-notes');
    if (stored) {
      setNotes(JSON.parse(stored));
    }
  }, []);

  const handleSync = async () => {
    setSyncing(true);
    await new Promise(resolve => setTimeout(resolve, 1500));
    setLastSync(new Date());
    setSyncing(false);
  };

  const toggleIncludeInReport = (noteId: string) => {
    setNotes(prev => {
      const updated = prev.map(note =>
        note.id === noteId ? { ...note, includeInReport: !note.includeInReport } : note
      );
      localStorage.setItem('reunity-therapist-notes', JSON.stringify(updated));
      return updated;
    });
  };

  const filteredNotes = notes.filter(note => {
    if (filter === 'shared') return note.isSharedWithClient;
    if (filter === 'report') return note.includeInReport;
    return true;
  });

  const notesForReport = notes.filter(n => n.includeInReport && n.isSharedWithClient);

  if (compact) {
    return (
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="p-4">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full bg-indigo-500/20 flex items-center justify-center">
              <FileText className="w-6 h-6 text-indigo-400" />
            </div>
            <div className="flex-1">
              <p className="text-sm font-medium text-white">Therapist Notes</p>
              <p className="text-xs text-slate-400">
                {notesForReport.length} notes for report
              </p>
            </div>
            <Button
              size="sm"
              variant="ghost"
              onClick={handleSync}
              disabled={syncing}
              className="text-indigo-400"
            >
              <RefreshCw className={`w-4 h-4 ${syncing ? 'animate-spin' : ''}`} />
            </Button>
          </div>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-full bg-indigo-500/20 flex items-center justify-center">
            <FileText className="w-5 h-5 text-indigo-400" />
          </div>
          <div>
            <h2 className="text-xl font-semibold text-white">Therapist Notes</h2>
            <p className="text-sm text-slate-400">Session notes synced to your wellness reports</p>
          </div>
        </div>
        <Button
          variant="outline"
          size="sm"
          onClick={handleSync}
          disabled={syncing}
          className="border-slate-600"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${syncing ? 'animate-spin' : ''}`} />
          {syncing ? 'Syncing...' : 'Sync Now'}
        </Button>
      </div>

      {/* Sync Status */}
      <Card className="bg-slate-800/50 border-slate-700">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-emerald-400" />
              <div>
                <p className="text-sm font-medium text-white">Notes Synced</p>
                <p className="text-xs text-slate-400">
                  Last sync: {lastSync.toLocaleString()}
                </p>
              </div>
            </div>
            <div className="text-right">
              <p className="text-lg font-bold text-white">{notes.length}</p>
              <p className="text-xs text-slate-400">Total notes</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Filter Tabs */}
      <div className="flex gap-2">
        {[
          { key: 'all', label: 'All Notes' },
          { key: 'shared', label: 'Shared with Me' },
          { key: 'report', label: 'In Report' },
        ].map(({ key, label }) => (
          <button
            key={key}
            onClick={() => setFilter(key as typeof filter)}
            className={`px-4 py-2 rounded-lg text-sm transition-all ${
              filter === key
                ? 'bg-indigo-500 text-white'
                : 'bg-slate-800 text-slate-300 hover:bg-slate-700'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Notes List */}
      <div className="space-y-3">
        {filteredNotes.map((note) => {
          const config = noteTypeConfig[note.type];
          const Icon = config.icon;
          
          return (
            <Card
              key={note.id}
              className={`bg-slate-800/50 border-slate-700 cursor-pointer transition-all hover:border-slate-600 ${
                selectedNote?.id === note.id ? 'ring-2 ring-indigo-500' : ''
              }`}
              onClick={() => setSelectedNote(note)}
            >
              <CardContent className="p-4">
                <div className="flex items-start gap-4">
                  <div
                    className="w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0"
                    style={{ backgroundColor: `${config.color}20` }}
                  >
                    <Icon className="w-5 h-5" style={{ color: config.color }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 mb-1">
                      <span
                        className="px-2 py-0.5 rounded text-xs font-medium"
                        style={{ backgroundColor: `${config.color}20`, color: config.color }}
                      >
                        {config.label}
                      </span>
                      {note.isSharedWithClient ? (
                        <Eye className="w-3 h-3 text-emerald-400" />
                      ) : (
                        <EyeOff className="w-3 h-3 text-slate-500" />
                      )}
                    </div>
                    <h3 className="font-medium text-white truncate">{note.title}</h3>
                    <div className="flex items-center gap-4 mt-1 text-xs text-slate-400">
                      <span className="flex items-center gap-1">
                        <User className="w-3 h-3" />
                        {note.therapistName}
                      </span>
                      <span className="flex items-center gap-1">
                        <Calendar className="w-3 h-3" />
                        {new Date(note.sessionDate).toLocaleDateString()}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant={note.includeInReport ? 'default' : 'outline'}
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleIncludeInReport(note.id);
                      }}
                      disabled={!note.isSharedWithClient}
                      className={note.includeInReport ? 'bg-indigo-500' : 'border-slate-600'}
                    >
                      <Download className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>
          );
        })}
      </div>

      {/* Selected Note Detail */}
      {selectedNote && (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-white">{selectedNote.title}</CardTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedNote(null)}
                className="text-slate-400"
              >
                Close
              </Button>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center gap-4 text-sm text-slate-400">
              <span className="flex items-center gap-1">
                <User className="w-4 h-4" />
                {selectedNote.therapistName}
              </span>
              <span className="flex items-center gap-1">
                <Calendar className="w-4 h-4" />
                {new Date(selectedNote.sessionDate).toLocaleDateString()}
              </span>
              <span className="flex items-center gap-1">
                <Clock className="w-4 h-4" />
                {new Date(selectedNote.createdAt).toLocaleTimeString()}
              </span>
            </div>

            <div className="p-4 bg-slate-900/50 rounded-lg">
              <p className="text-slate-300 whitespace-pre-wrap">{selectedNote.content}</p>
            </div>

            <div className="flex flex-wrap gap-2">
              {selectedNote.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 bg-slate-700 text-slate-300 rounded text-xs"
                >
                  #{tag}
                </span>
              ))}
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-slate-700">
              <div className="flex items-center gap-2">
                {selectedNote.isSharedWithClient ? (
                  <span className="flex items-center gap-1 text-sm text-emerald-400">
                    <Eye className="w-4 h-4" />
                    Shared with you
                  </span>
                ) : (
                  <span className="flex items-center gap-1 text-sm text-slate-500">
                    <EyeOff className="w-4 h-4" />
                    Not shared
                  </span>
                )}
              </div>
              <Button
                variant={selectedNote.includeInReport ? 'default' : 'outline'}
                onClick={() => toggleIncludeInReport(selectedNote.id)}
                disabled={!selectedNote.isSharedWithClient}
                className={selectedNote.includeInReport ? 'bg-indigo-500' : 'border-slate-600'}
              >
                <Download className="w-4 h-4 mr-2" />
                {selectedNote.includeInReport ? 'Included in Report' : 'Add to Report'}
              </Button>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Report Summary */}
      <Card className="bg-indigo-500/10 border-indigo-500/30">
        <CardContent className="p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="font-medium text-white">Wellness Report Integration</p>
              <p className="text-sm text-slate-400">
                {notesForReport.length} therapist notes will be included in your next wellness report export
              </p>
            </div>
            <div className="text-right">
              <p className="text-2xl font-bold text-indigo-400">{notesForReport.length}</p>
              <p className="text-xs text-slate-400">notes selected</p>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
