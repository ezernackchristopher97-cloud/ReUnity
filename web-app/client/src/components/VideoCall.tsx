import { useState, useEffect, useRef, useCallback } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { 
  Video, 
  VideoOff, 
  Mic, 
  MicOff, 
  Phone, 
  PhoneOff,
  Maximize2,
  Minimize2,
  Settings,
  MessageSquare,
  Users,
  Shield,
  Circle
} from 'lucide-react';
import SessionRecorder from './SessionRecorder';

interface VideoCallProps {
  sessionId: string;
  participantName: string;
  isTherapist?: boolean;
  therapistName?: string;
  onEnd?: () => void;
}

type ConnectionState = 'connecting' | 'connected' | 'disconnected' | 'failed';

export default function VideoCall({ sessionId, participantName, isTherapist = false, therapistName = 'Therapist', onEnd }: VideoCallProps) {
  const [isVideoEnabled, setIsVideoEnabled] = useState(true);
  const [isAudioEnabled, setIsAudioEnabled] = useState(true);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting');
  const [callDuration, setCallDuration] = useState(0);
  const [showChat, setShowChat] = useState(false);
  const [showRecorder, setShowRecorder] = useState(false);
  
  const localVideoRef = useRef<HTMLVideoElement>(null);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const peerConnectionRef = useRef<RTCPeerConnection | null>(null);
  const localStreamRef = useRef<MediaStream | null>(null);
  const timerRef = useRef<NodeJS.Timeout | null>(null);

  const initializeMedia = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
      });
      localStreamRef.current = stream;
      
      if (localVideoRef.current) {
        localVideoRef.current.srcObject = stream;
      }

      // Initialize WebRTC peer connection
      const configuration: RTCConfiguration = {
        iceServers: [
          { urls: 'stun:stun.l.google.com:19302' },
          { urls: 'stun:stun1.l.google.com:19302' }
        ]
      };

      const pc = new RTCPeerConnection(configuration);
      peerConnectionRef.current = pc;

      // Add local tracks to peer connection
      stream.getTracks().forEach(track => {
        pc.addTrack(track, stream);
      });

      // Handle incoming tracks
      pc.ontrack = (event) => {
        if (remoteVideoRef.current && event.streams[0]) {
          remoteVideoRef.current.srcObject = event.streams[0];
        }
      };

      // Handle connection state changes
      pc.onconnectionstatechange = () => {
        switch (pc.connectionState) {
          case 'connected':
            setConnectionState('connected');
            startTimer();
            break;
          case 'disconnected':
            setConnectionState('disconnected');
            break;
          case 'failed':
            setConnectionState('failed');
            break;
        }
      };

      // Simulate connection for demo
      setTimeout(() => {
        setConnectionState('connected');
        startTimer();
      }, 2000);

    } catch (error) {
      console.error('Error accessing media devices:', error);
      setConnectionState('failed');
    }
  }, []);

  useEffect(() => {
    initializeMedia();

    return () => {
      // Cleanup
      if (localStreamRef.current) {
        localStreamRef.current.getTracks().forEach(track => track.stop());
      }
      if (peerConnectionRef.current) {
        peerConnectionRef.current.close();
      }
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
    };
  }, [initializeMedia]);

  const startTimer = () => {
    timerRef.current = setInterval(() => {
      setCallDuration(prev => prev + 1);
    }, 1000);
  };

  const toggleVideo = () => {
    if (localStreamRef.current) {
      const videoTrack = localStreamRef.current.getVideoTracks()[0];
      if (videoTrack) {
        videoTrack.enabled = !videoTrack.enabled;
        setIsVideoEnabled(videoTrack.enabled);
      }
    }
  };

  const toggleAudio = () => {
    if (localStreamRef.current) {
      const audioTrack = localStreamRef.current.getAudioTracks()[0];
      if (audioTrack) {
        audioTrack.enabled = !audioTrack.enabled;
        setIsAudioEnabled(audioTrack.enabled);
      }
    }
  };

  const toggleFullscreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen();
      setIsFullscreen(false);
    }
  };

  const endCall = () => {
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach(track => track.stop());
    }
    if (peerConnectionRef.current) {
      peerConnectionRef.current.close();
    }
    if (timerRef.current) {
      clearInterval(timerRef.current);
    }
    setConnectionState('disconnected');
    onEnd?.();
  };

  const formatDuration = (seconds: number) => {
    const hrs = Math.floor(seconds / 3600);
    const mins = Math.floor((seconds % 3600) / 60);
    const secs = seconds % 60;
    if (hrs > 0) {
      return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getConnectionStatusColor = () => {
    switch (connectionState) {
      case 'connected': return 'bg-emerald-500';
      case 'connecting': return 'bg-amber-500 animate-pulse';
      case 'disconnected': return 'bg-zinc-500';
      case 'failed': return 'bg-red-500';
    }
  };

  return (
    <div className={`relative ${isFullscreen ? 'fixed inset-0 z-50 bg-black' : 'rounded-xl overflow-hidden'}`}>
      {/* Remote Video (Main) */}
      <div className="relative aspect-video bg-zinc-900">
        <video
          ref={remoteVideoRef}
          autoPlay
          playsInline
          className="w-full h-full object-cover"
        />
        
        {/* Placeholder when no remote video */}
        {connectionState !== 'connected' && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-900">
            <div className="text-center">
              <div className="w-24 h-24 rounded-full bg-zinc-800 flex items-center justify-center mx-auto mb-4">
                <Users className="w-12 h-12 text-zinc-600" />
              </div>
              <p className="text-zinc-400">
                {connectionState === 'connecting' ? 'Connecting...' : 
                 connectionState === 'failed' ? 'Connection failed' : 
                 'Waiting for participant'}
              </p>
            </div>
          </div>
        )}

        {/* Local Video (Picture-in-Picture) */}
        <div className="absolute bottom-4 right-4 w-48 aspect-video rounded-lg overflow-hidden border-2 border-zinc-700 shadow-xl">
          <video
            ref={localVideoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover"
          />
          {!isVideoEnabled && (
            <div className="absolute inset-0 bg-zinc-800 flex items-center justify-center">
              <VideoOff className="w-8 h-8 text-zinc-500" />
            </div>
          )}
        </div>

        {/* Top Bar */}
        <div className="absolute top-0 left-0 right-0 p-4 bg-gradient-to-b from-black/60 to-transparent">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className={`w-3 h-3 rounded-full ${getConnectionStatusColor()}`} />
              <div>
                <p className="text-white font-medium">{participantName}</p>
                <p className="text-xs text-zinc-400">
                  {isTherapist ? 'Therapist Session' : 'Video Call'}
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <div className="px-3 py-1 rounded-full bg-black/40 text-white text-sm">
                {formatDuration(callDuration)}
              </div>
              {isTherapist && (
                <div className="px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-400 text-xs flex items-center gap-1">
                  <Shield className="w-3 h-3" />
                  Encrypted
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Bottom Controls */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-black/60 to-transparent">
          <div className="flex items-center justify-center gap-4">
            <Button
              variant="outline"
              size="icon"
              onClick={toggleAudio}
              className={`rounded-full w-12 h-12 ${!isAudioEnabled ? 'bg-red-500/20 border-red-500 text-red-400' : 'bg-zinc-800/80 border-zinc-700'}`}
            >
              {isAudioEnabled ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
            </Button>
            
            <Button
              variant="outline"
              size="icon"
              onClick={toggleVideo}
              className={`rounded-full w-12 h-12 ${!isVideoEnabled ? 'bg-red-500/20 border-red-500 text-red-400' : 'bg-zinc-800/80 border-zinc-700'}`}
            >
              {isVideoEnabled ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
            </Button>

            <Button
              onClick={endCall}
              className="rounded-full w-14 h-14 bg-red-600 hover:bg-red-700"
            >
              <PhoneOff className="w-6 h-6" />
            </Button>

            <Button
              variant="outline"
              size="icon"
              onClick={() => setShowChat(!showChat)}
              className="rounded-full w-12 h-12 bg-zinc-800/80 border-zinc-700"
            >
              <MessageSquare className="w-5 h-5" />
            </Button>

            {isTherapist && (
              <Button
                variant="outline"
                size="icon"
                onClick={() => setShowRecorder(!showRecorder)}
                className={`rounded-full w-12 h-12 ${showRecorder ? 'bg-red-500/20 border-red-500 text-red-400' : 'bg-zinc-800/80 border-zinc-700'}`}
              >
                <Circle className="w-5 h-5" />
              </Button>
            )}

            <Button
              variant="outline"
              size="icon"
              onClick={toggleFullscreen}
              className="rounded-full w-12 h-12 bg-zinc-800/80 border-zinc-700"
            >
              {isFullscreen ? <Minimize2 className="w-5 h-5" /> : <Maximize2 className="w-5 h-5" />}
            </Button>
          </div>
        </div>
      </div>

      {/* Recording Panel */}
      {showRecorder && isTherapist && (
        <div className="absolute top-0 right-0 bottom-0 w-96 bg-zinc-900 border-l border-zinc-800 flex flex-col">
          <div className="p-4 border-b border-zinc-800">
            <h3 className="font-medium text-white">Session Recording</h3>
            <p className="text-xs text-zinc-500 mt-1">Record with consent for clinical documentation</p>
          </div>
          <div className="flex-1 p-4 overflow-y-auto">
            <SessionRecorder
              stream={localStreamRef.current}
              sessionId={sessionId}
              clientName={participantName}
              therapistName={therapistName}
              onRecordingStart={() => console.log('Recording started')}
              onRecordingStop={(blob, duration) => console.log('Recording stopped', { size: blob.size, duration })}
            />
          </div>
        </div>
      )}

      {/* Chat Sidebar */}
      {showChat && (
        <div className="absolute top-0 right-0 bottom-0 w-80 bg-zinc-900 border-l border-zinc-800 flex flex-col">
          <div className="p-4 border-b border-zinc-800">
            <h3 className="font-medium">Session Chat</h3>
          </div>
          <div className="flex-1 p-4 overflow-y-auto">
            <p className="text-sm text-zinc-500 text-center">Messages during the session appear here</p>
          </div>
          <div className="p-4 border-t border-zinc-800">
            <input
              type="text"
              placeholder="Type a message..."
              className="w-full px-3 py-2 rounded-lg bg-zinc-800 border border-zinc-700 text-sm"
            />
          </div>
        </div>
      )}
    </div>
  );
}

// Standalone video call page component
export function VideoCallPage() {
  const [inCall, setInCall] = useState(false);
  const [sessionId] = useState(() => Math.random().toString(36).substr(2, 9));

  if (!inCall) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-zinc-950 via-zinc-900 to-zinc-950 flex items-center justify-center p-4">
        <Card className="w-full max-w-md bg-zinc-900/50 border-zinc-800">
          <CardHeader>
            <CardTitle className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-emerald-500/20">
                <Video className="w-5 h-5 text-emerald-400" />
              </div>
              Start Video Session
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-lg bg-zinc-800/50">
              <p className="text-sm text-zinc-400 mb-2">Session ID</p>
              <p className="font-mono text-lg">{sessionId}</p>
            </div>
            <div className="p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
              <p className="text-sm text-zinc-300">
                Your video session is end-to-end encrypted. Only you and your therapist can see and hear the conversation.
              </p>
            </div>
            <Button onClick={() => setInCall(true)} className="w-full bg-emerald-600 hover:bg-emerald-700">
              <Phone className="w-4 h-4 mr-2" />
              Join Session
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black p-4">
      <VideoCall
        sessionId={sessionId}
        participantName="Dr. Sarah Mitchell"
        isTherapist={true}
        onEnd={() => setInCall(false)}
      />
    </div>
  );
}
