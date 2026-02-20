import { useState, useRef, useCallback, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Checkbox } from '@/components/ui/checkbox';
import { Label } from '@/components/ui/label';
import { Circle, Square, Download, AlertTriangle, Shield, FileVideo } from 'lucide-react';

interface SessionRecorderProps {
  stream: MediaStream | null;
  sessionId: string;
  clientName: string;
  therapistName: string;
  onRecordingStart?: () => void;
  onRecordingStop?: (blob: Blob, duration: number) => void;
}

interface ConsentState {
  clientConsent: boolean;
  therapistConsent: boolean;
  hipaaAcknowledged: boolean;
  storageAcknowledged: boolean;
}

export default function SessionRecorder({
  stream,
  sessionId,
  clientName,
  therapistName,
  onRecordingStart,
  onRecordingStop,
}: SessionRecorderProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [showConsentDialog, setShowConsentDialog] = useState(false);
  const [consent, setConsent] = useState<ConsentState>({
    clientConsent: false,
    therapistConsent: false,
    hipaaAcknowledged: false,
    storageAcknowledged: false,
  });
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [recordings, setRecordings] = useState<Array<{
    id: string;
    blob: Blob;
    duration: number;
    timestamp: Date;
    clientName: string;
  }>>([]);
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const timerRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);

  const allConsentsGiven = 
    consent.clientConsent && 
    consent.therapistConsent && 
    consent.hipaaAcknowledged && 
    consent.storageAcknowledged;

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
        mediaRecorderRef.current.stop();
      }
    };
  }, []);

  const formatDuration = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const handleStartRecording = useCallback(() => {
    if (!stream) return;
    setShowConsentDialog(true);
  }, [stream]);

  const startRecordingAfterConsent = useCallback(() => {
    if (!stream || !allConsentsGiven) return;

    chunksRef.current = [];
    
    const options = { mimeType: 'video/webm;codecs=vp9,opus' };
    let recorder: MediaRecorder;
    
    try {
      recorder = new MediaRecorder(stream, options);
    } catch {
      // Fallback to default codec
      recorder = new MediaRecorder(stream);
    }

    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        chunksRef.current.push(event.data);
      }
    };

    recorder.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: 'video/webm' });
      const duration = Math.floor((Date.now() - startTimeRef.current) / 1000);
      
      const newRecording = {
        id: `${sessionId}-${Date.now()}`,
        blob,
        duration,
        timestamp: new Date(),
        clientName,
      };
      
      setRecordings(prev => [...prev, newRecording]);
      onRecordingStop?.(blob, duration);
      
      if (timerRef.current) {
        clearInterval(timerRef.current);
        timerRef.current = null;
      }
    };

    mediaRecorderRef.current = recorder;
    recorder.start(1000); // Collect data every second
    startTimeRef.current = Date.now();
    setIsRecording(true);
    setShowConsentDialog(false);
    onRecordingStart?.();

    // Start duration timer
    timerRef.current = setInterval(() => {
      setRecordingDuration(Math.floor((Date.now() - startTimeRef.current) / 1000));
    }, 1000);
  }, [stream, allConsentsGiven, sessionId, clientName, onRecordingStart, onRecordingStop]);

  const handleStopRecording = useCallback(() => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setRecordingDuration(0);
    }
  }, []);

  const downloadRecording = (recording: typeof recordings[0]) => {
    const url = URL.createObjectURL(recording.blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `session-${recording.clientName}-${recording.timestamp.toISOString().split('T')[0]}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-4">
      {/* Recording Controls */}
      <div className="flex items-center gap-4">
        {!isRecording ? (
          <Button
            onClick={handleStartRecording}
            disabled={!stream}
            variant="outline"
            className="gap-2 border-red-500/50 text-red-400 hover:bg-red-500/10"
          >
            <Circle className="w-4 h-4 fill-red-500" />
            Start Recording
          </Button>
        ) : (
          <div className="flex items-center gap-4">
            <Button
              onClick={handleStopRecording}
              variant="destructive"
              className="gap-2"
            >
              <Square className="w-4 h-4 fill-white" />
              Stop Recording
            </Button>
            <div className="flex items-center gap-2 text-red-400">
              <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
              <span className="font-mono">{formatDuration(recordingDuration)}</span>
            </div>
          </div>
        )}
      </div>

      {/* Recording Indicator */}
      {isRecording && (
        <div className="p-3 bg-red-500/10 border border-red-500/30 rounded-lg">
          <div className="flex items-center gap-2 text-red-400">
            <AlertTriangle className="w-4 h-4" />
            <span className="text-sm font-medium">Session is being recorded with consent</span>
          </div>
        </div>
      )}

      {/* Saved Recordings */}
      {recordings.length > 0 && (
        <div className="space-y-2">
          <h4 className="text-sm font-medium text-zinc-300 flex items-center gap-2">
            <FileVideo className="w-4 h-4" />
            Session Recordings
          </h4>
          <div className="space-y-2">
            {recordings.map((recording) => (
              <div
                key={recording.id}
                className="flex items-center justify-between p-3 bg-zinc-800/50 rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <FileVideo className="w-5 h-5 text-blue-400" />
                  <div>
                    <p className="text-sm text-white">
                      Session with {recording.clientName}
                    </p>
                    <p className="text-xs text-zinc-400">
                      {recording.timestamp.toLocaleString()} • {formatDuration(recording.duration)}
                    </p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => downloadRecording(recording)}
                  className="gap-2"
                >
                  <Download className="w-4 h-4" />
                  Download
                </Button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Consent Dialog */}
      <Dialog open={showConsentDialog} onOpenChange={setShowConsentDialog}>
        <DialogContent className="bg-zinc-900 border-zinc-800 max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-white">
              <Shield className="w-5 h-5 text-emerald-400" />
              Recording Consent Required
            </DialogTitle>
            <DialogDescription className="text-zinc-400">
              Before recording this session, all parties must provide explicit consent.
              This recording will be stored securely for clinical documentation purposes.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="p-4 bg-amber-500/10 border border-amber-500/30 rounded-lg">
              <div className="flex items-start gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-400 mt-0.5" />
                <div className="text-sm text-amber-200">
                  <p className="font-medium">Important Notice</p>
                  <p className="text-amber-300/80 mt-1">
                    Recording therapy sessions requires informed consent from all participants.
                    Recordings are subject to HIPAA regulations and must be stored securely.
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <Checkbox
                  id="clientConsent"
                  checked={consent.clientConsent}
                  onCheckedChange={(checked) => 
                    setConsent(prev => ({ ...prev, clientConsent: checked === true }))
                  }
                />
                <Label htmlFor="clientConsent" className="text-sm text-zinc-300 cursor-pointer">
                  <span className="font-medium text-white">{clientName}</span> has verbally consented 
                  to this session being recorded for clinical documentation purposes.
                </Label>
              </div>

              <div className="flex items-start gap-3">
                <Checkbox
                  id="therapistConsent"
                  checked={consent.therapistConsent}
                  onCheckedChange={(checked) => 
                    setConsent(prev => ({ ...prev, therapistConsent: checked === true }))
                  }
                />
                <Label htmlFor="therapistConsent" className="text-sm text-zinc-300 cursor-pointer">
                  I, <span className="font-medium text-white">{therapistName}</span>, consent to 
                  recording this session and take responsibility for its secure storage.
                </Label>
              </div>

              <div className="flex items-start gap-3">
                <Checkbox
                  id="hipaaAcknowledged"
                  checked={consent.hipaaAcknowledged}
                  onCheckedChange={(checked) => 
                    setConsent(prev => ({ ...prev, hipaaAcknowledged: checked === true }))
                  }
                />
                <Label htmlFor="hipaaAcknowledged" className="text-sm text-zinc-300 cursor-pointer">
                  I acknowledge that this recording contains Protected Health Information (PHI) 
                  and will be handled in compliance with HIPAA regulations.
                </Label>
              </div>

              <div className="flex items-start gap-3">
                <Checkbox
                  id="storageAcknowledged"
                  checked={consent.storageAcknowledged}
                  onCheckedChange={(checked) => 
                    setConsent(prev => ({ ...prev, storageAcknowledged: checked === true }))
                  }
                />
                <Label htmlFor="storageAcknowledged" className="text-sm text-zinc-300 cursor-pointer">
                  I understand that this recording will be stored locally and I am responsible 
                  for transferring it to a HIPAA-compliant storage system.
                </Label>
              </div>
            </div>
          </div>

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setShowConsentDialog(false);
                setConsent({
                  clientConsent: false,
                  therapistConsent: false,
                  hipaaAcknowledged: false,
                  storageAcknowledged: false,
                });
              }}
            >
              Cancel
            </Button>
            <Button
              onClick={startRecordingAfterConsent}
              disabled={!allConsentsGiven}
              className="bg-red-600 hover:bg-red-700"
            >
              <Circle className="w-4 h-4 fill-white mr-2" />
              Begin Recording
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
