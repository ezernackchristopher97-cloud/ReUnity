import { useState, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { trpc } from "@/lib/trpc";
import { Link } from "wouter";
import { Send, ArrowLeft, AlertTriangle, Heart, Wind, Image, X, Camera, FileText, MessageSquare, Mic, MicOff, History, Plus, Download, LogOut } from "lucide-react";
import SessionEndDialog from "@/components/SessionEndDialog";
import { Streamdown } from "streamdown";
import { useAuth } from "@/contexts/AuthContext";
import { VoiceChat } from "@/components/VoiceChat";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  state?: string;
  entropy?: number;
  patterns?: string[];
  groundingTechnique?: {
    name: string;
    steps: string[];
  };
  isCrisis?: boolean;
  regime?: string;
  dissociationDetected?: boolean;
  imageUrl?: string;
  imageType?: "general" | "conversation" | "journal";
}

export default function Chat() {
  const { user } = useAuth();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "I'm here with you. This is a space where you can share what you're experiencing - through words, images, or voice. There's no judgment here, only presence.\n\nYou can type a message, speak using the microphone, or share a screenshot of a conversation or journal entry. What's on your mind?",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageType, setImageType] = useState<"general" | "conversation" | "journal">("general");
  const [showImageOptions, setShowImageOptions] = useState(false);
  const [conversationId, setConversationId] = useState<number | null>(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showSessionEnd, setShowSessionEnd] = useState(false);
  
  // Voice input state
  const [isRecording, setIsRecording] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(false);
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Check for voice support
  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      setVoiceSupported(true);
      recognitionRef.current = new SpeechRecognition();
      recognitionRef.current.continuous = true;
      recognitionRef.current.interimResults = true;
      recognitionRef.current.lang = 'en-US';
      
      recognitionRef.current.onresult = (event) => {
        let transcript = '';
        for (let i = event.resultIndex; i < event.results.length; i++) {
          transcript += event.results[i][0].transcript;
        }
        setInput(prev => prev + transcript);
      };
      
      recognitionRef.current.onerror = (event) => {
        console.error('Speech recognition error:', event.error);
        setIsRecording(false);
      };
      
      recognitionRef.current.onend = () => {
        setIsRecording(false);
      };
    }
  }, []);

  // Load RIME memory on mount if user is logged in
  const { data: rimeMemory } = trpc.reunity.loadMemory.useQuery(undefined, {
    enabled: !!user,
  });

  // Load conversation history
  const { data: conversationHistory, refetch: refetchHistory } = trpc.reunity.getConversations.useQuery(undefined, {
    enabled: !!user && showHistory,
  });

  // Create conversation mutation
  const createConversationMutation = trpc.reunity.createConversation.useMutation({
    onSuccess: (data) => {
      if (data) {
        setConversationId(data.id);
      }
    },
  });

  // Save message mutation
  const saveMessageMutation = trpc.reunity.saveMessage.useMutation();

  // Save memory mutation
  const saveMemoryMutation = trpc.reunity.saveMemory.useMutation();

  // Export conversation mutation
  const exportMutation = trpc.reunity.exportConversation.useMutation({
    onSuccess: (data) => {
      if (data.success && data.html) {
        // Create blob and download
        const blob = new Blob([data.html], { type: 'text/html' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = data.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }
    },
  });

  // Load conversation mutation
  const loadConversationMutation = trpc.reunity.getConversationMessages.useMutation({
    onSuccess: (data) => {
      if (data && data.length > 0) {
        const loadedMessages: Message[] = data.map((m: any) => ({
          id: m.id.toString(),
          role: m.role as "user" | "assistant",
          content: m.content,
          state: m.detectedState || undefined,
          patterns: m.detectedPatterns as string[] || undefined,
          isCrisis: m.isCrisis || false,
        }));
        setMessages([
          {
            id: "welcome",
            role: "assistant",
            content: "Welcome back. I've loaded your previous conversation. Feel free to continue where we left off.",
          },
          ...loadedMessages,
        ]);
      }
      setShowHistory(false);
    },
  });

  const chatMutation = trpc.reunity.chat.useMutation({
    onSuccess: (data) => {
      const newMessage: Message = {
        id: Date.now().toString(),
        role: "assistant",
        content: data.response,
        state: data.state,
        entropy: data.entropy,
        patterns: data.patterns,
        groundingTechnique: data.groundingTechnique,
        isCrisis: data.isCrisis,
        regime: data.regime,
        dissociationDetected: data.dissociationDetected,
      };
      setMessages((prev) => [...prev, newMessage]);
      setIsLoading(false);

      // Save assistant message to database if logged in
      if (user && conversationId) {
        saveMessageMutation.mutate({
          conversationId,
          role: "assistant",
          content: data.response,
          entropyScore: data.entropy?.toString(),
          detectedState: data.state,
          detectedPatterns: data.patterns,
          groundingTechnique: data.groundingTechnique?.name,
          isCrisis: data.isCrisis,
        });
      }
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "I'm having trouble connecting right now. Please try again, or if you're in crisis, call 988.",
        },
      ]);
      setIsLoading(false);
    },
  });

  const chatWithImageMutation = trpc.reunity.chatWithImage.useMutation({
    onSuccess: (data) => {
      if (!data.success || !data.chatResult) {
        setMessages((prev) => [
          ...prev,
          {
            id: Date.now().toString(),
            role: "assistant",
            content: data.error || "I couldn't process that image. Could you try sharing it again, or describe what's in it?",
          },
        ]);
      } else {
        const newMessage: Message = {
          id: Date.now().toString(),
          role: "assistant",
          content: data.chatResult.response,
          state: data.chatResult.state,
          entropy: data.chatResult.entropy,
          patterns: data.chatResult.patterns,
          groundingTechnique: data.chatResult.groundingTechnique,
          isCrisis: data.chatResult.isCrisis,
          regime: data.chatResult.regime,
          dissociationDetected: data.chatResult.dissociationDetected,
        };
        setMessages((prev) => [...prev, newMessage]);

        // Save to database if logged in
        if (user && conversationId && data.chatResult) {
          saveMessageMutation.mutate({
            conversationId,
            role: "assistant",
            content: data.chatResult.response,
            entropyScore: data.chatResult.entropy?.toString(),
            detectedState: data.chatResult.state,
            detectedPatterns: data.chatResult.patterns,
            groundingTechnique: data.chatResult.groundingTechnique?.name,
            isCrisis: data.chatResult.isCrisis,
          });
        }
      }
      setIsLoading(false);
      setSelectedImage(null);
    },
    onError: () => {
      setMessages((prev) => [
        ...prev,
        {
          id: Date.now().toString(),
          role: "assistant",
          content: "I'm having trouble processing that image. Please try again, or if you're in crisis, call 988.",
        },
      ]);
      setIsLoading(false);
      setSelectedImage(null);
    },
  });

  // Create a new conversation on first message if logged in
  const ensureConversation = async () => {
    if (user && !conversationId) {
      const result = await createConversationMutation.mutateAsync({ title: "New Session" });
      return result?.id || null;
    }
    return conversationId;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if ((!input.trim() && !selectedImage) || isLoading) return;

    // Ensure we have a conversation ID
    const currentConversationId = await ensureConversation();

    // If there's an image, use the image chat endpoint
    if (selectedImage) {
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: input.trim() || `[Shared ${imageType === "conversation" ? "a conversation screenshot" : imageType === "journal" ? "a journal entry" : "an image"}]`,
        imageUrl: selectedImage,
        imageType: imageType,
      };
      setMessages((prev) => [...prev, userMessage]);
      
      // Save user message to database
      if (user && currentConversationId) {
        saveMessageMutation.mutate({
          conversationId: currentConversationId,
          role: "user",
          content: userMessage.content,
        });
      }
      
      setInput("");
      setIsLoading(true);

      chatWithImageMutation.mutate({
        imageUrl: selectedImage,
        additionalMessage: input.trim() || undefined,
        conversationHistory: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      });
    } else {
      // Text-only message
      const userMessage: Message = {
        id: Date.now().toString(),
        role: "user",
        content: input,
      };
      setMessages((prev) => [...prev, userMessage]);
      
      // Save user message to database
      if (user && currentConversationId) {
        saveMessageMutation.mutate({
          conversationId: currentConversationId,
          role: "user",
          content: input,
        });
      }
      
      setInput("");
      setIsLoading(true);

      chatMutation.mutate({
        message: input,
        conversationHistory: messages.map((m) => ({
          role: m.role,
          content: m.content,
        })),
      });
    }
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = reader.result as string;
        setSelectedImage(base64);
        setShowImageOptions(true);
      };
      reader.readAsDataURL(file);
    }
  };

  const clearSelectedImage = () => {
    setSelectedImage(null);
    setShowImageOptions(false);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const toggleRecording = () => {
    if (!recognitionRef.current) return;
    
    if (isRecording) {
      recognitionRef.current.stop();
      setIsRecording(false);
    } else {
      setInput(""); // Clear input when starting new recording
      recognitionRef.current.start();
      setIsRecording(true);
    }
  };

  const startNewConversation = () => {
    setMessages([
      {
        id: "welcome",
        role: "assistant",
        content: "I'm here with you. This is a space where you can share what you're experiencing - through words, images, or voice. There's no judgment here, only presence.\n\nYou can type a message, speak using the microphone, or share a screenshot of a conversation or journal entry. What's on your mind?",
      },
    ]);
    setConversationId(null);
    setShowHistory(false);
  };

  const loadConversation = (id: number) => {
    setConversationId(id);
    loadConversationMutation.mutate({ conversationId: id });
  };

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const latestAiMessage = messages.filter((m) => m.role === "assistant").pop();
  const showCrisisBar = latestAiMessage?.isCrisis;
  const showDissociationAlert = latestAiMessage?.dissociationDetected && !latestAiMessage?.isCrisis;

  return (
    <div className="min-h-screen bg-[#0a0a0c] flex flex-col">
      {/* Header */}
      <header className="border-b border-white/10 bg-[#0a0a0c]/80 backdrop-blur-sm">
        <div className="container flex items-center justify-between h-14">
          <div className="flex items-center gap-4">
            <Link href="/">
              <button className="flex items-center gap-2 text-white/70 hover:text-white px-3 py-1.5 rounded-lg hover:bg-white/5 transition-colors">
                <ArrowLeft className="w-4 h-4" />
                Back
              </button>
            </Link>
            <div className="flex items-center gap-2">
              <img src="/reop-logo.png" alt="REOP Solutions" className="w-8 h-8 object-contain" />
              <span className="font-semibold text-white">ReUnity</span>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {/* History & New Chat buttons for logged in users */}
            {user && (
              <>
                <button
                  onClick={() => setShowHistory(!showHistory)}
                  className="flex items-center gap-2 text-white/70 hover:text-white px-3 py-1.5 rounded-lg hover:bg-white/5 transition-colors"
                  title="Conversation history"
                >
                  <History className="w-4 h-4" />
                  <span className="hidden sm:inline">History</span>
                </button>
                <button
                  onClick={startNewConversation}
                  className="flex items-center gap-2 text-white/70 hover:text-white px-3 py-1.5 rounded-lg hover:bg-white/5 transition-colors"
                  title="New conversation"
                >
                  <Plus className="w-4 h-4" />
                  <span className="hidden sm:inline">New</span>
                </button>
                {conversationId && (
                  <button
                    onClick={() => exportMutation.mutate({ conversationId })}
                    disabled={exportMutation.isPending}
                    className="flex items-center gap-2 text-white/70 hover:text-white px-3 py-1.5 rounded-lg hover:bg-white/5 transition-colors disabled:opacity-50"
                    title="Export session for therapist"
                  >
                    <Download className="w-4 h-4" />
                    <span className="hidden sm:inline">{exportMutation.isPending ? 'Exporting...' : 'Export'}</span>
                  </button>
                )}
              </>
            )}
            {/* End Session Button */}
            <button
              onClick={() => setShowSessionEnd(true)}
              className="flex items-center gap-2 text-white/70 hover:text-red-400 px-3 py-1.5 rounded-lg hover:bg-red-500/10 transition-colors"
              title="End session"
            >
              <LogOut className="w-4 h-4" />
              <span className="hidden sm:inline">End Session</span>
            </button>
            {latestAiMessage?.regime && latestAiMessage.regime !== "normal" && (
              <span className={`text-xs px-2 py-1 rounded ${
                latestAiMessage.regime === "crisis" ? "bg-[#dc2626]/20 text-[#dc2626]" :
                "bg-[#f59e0b]/20 text-[#f59e0b]"
              }`}>
                {latestAiMessage.regime === "crisis" ? "Crisis Mode" : "Recovery Mode"}
              </span>
            )}
            {latestAiMessage?.state && (
              <div className="flex items-center gap-2 text-sm text-white/50">
                <span className={`w-2 h-2 rounded-full ${getStateColor(latestAiMessage.state)}`} />
                <span className="hidden sm:inline">{getStateLabel(latestAiMessage.state)}</span>
              </div>
            )}
          </div>
        </div>
      </header>

      {/* Conversation History Sidebar */}
      {showHistory && user && (
        <div className="absolute top-14 right-0 w-80 max-h-96 bg-[#0a0a0c] border border-white/10 rounded-bl-lg shadow-xl z-50 overflow-hidden">
          <div className="p-3 border-b border-white/10">
            <h3 className="font-semibold text-white">Previous Sessions</h3>
          </div>
          <ScrollArea className="max-h-80">
            {conversationHistory && conversationHistory.length > 0 ? (
              <div className="p-2 space-y-1">
                {conversationHistory.map((conv: { id: number; title: string | null; createdAt: Date; currentState: string | null }) => (
                  <button
                    key={conv.id}
                    onClick={() => loadConversation(conv.id)}
                    className={`w-full text-left p-3 rounded-lg hover:bg-white/5 transition-colors ${
                      conversationId === conv.id ? "bg-white/10" : ""
                    }`}
                  >
                    <p className="text-sm text-white font-medium truncate">
                      {conv.title || "Untitled Session"}
                    </p>
                    <p className="text-xs text-white/50 mt-1">
                      {new Date(conv.createdAt).toLocaleDateString()} • {conv.currentState || "stable"}
                    </p>
                  </button>
                ))}
              </div>
            ) : (
              <div className="p-4 text-center text-white/50 text-sm">
                No previous sessions found
              </div>
            )}
          </ScrollArea>
        </div>
      )}

      {/* Crisis Banner */}
      {showCrisisBar && (
        <div className="bg-[#dc2626] text-white py-3 px-4">
          <div className="container flex items-center justify-center gap-3 text-sm">
            <AlertTriangle className="w-5 h-5" />
            <span className="font-medium">
              If you're in immediate danger, please call 988 (Suicide & Crisis Lifeline) or 911
            </span>
          </div>
        </div>
      )}

      {/* Dissociation Alert */}
      {showDissociationAlert && (
        <div className="bg-[#7c3aed] text-white py-3 px-4">
          <div className="container flex items-center justify-center gap-3 text-sm">
            <Heart className="w-5 h-5" />
            <span className="font-medium">
              Dissociation detected. Grounding techniques are available. You are here. You are safe.
            </span>
          </div>
        </div>
      )}

      {/* Voice Chat Panel - Collapsible */}
      <div className="border-b border-white/10 bg-[#0a0a0c]/50">
        <div className="container max-w-3xl mx-auto py-3">
          <VoiceChat
            compact
            onSendMessage={async (text) => {
              // Send voice message through the same chat flow
              const userMessage: Message = {
                id: `user-${Date.now()}`,
                role: "user",
                content: text,
              };
              setMessages(prev => [...prev, userMessage]);
              setIsLoading(true);
              
              try {
                const response = await chatMutation.mutateAsync({
                  message: text,
                  conversationId: conversationId || undefined,
                });
                
                const aiMessage: Message = {
                  id: `ai-${Date.now()}`,
                  role: "assistant",
                  content: response.response,
                  state: response.state,
                  entropy: response.entropy,
                  patterns: response.patterns,
                  groundingTechnique: response.groundingTechnique,
                  isCrisis: response.isCrisis,
                  regime: response.regime,
                  dissociationDetected: response.dissociationDetected,
                };
                setMessages(prev => [...prev, aiMessage]);
                
                // Conversation ID is managed separately
                
                return response.response;
              } catch (error) {
                console.error("Voice chat error:", error);
                return "I'm here with you. Please try again.";
              } finally {
                setIsLoading(false);
              }
            }}
          />
        </div>
      </div>

      {/* Chat Area */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="container max-w-3xl mx-auto space-y-4 pb-4">
          {messages.map((message) => (
            <div key={message.id}>
              {/* Pattern Warning */}
              {message.patterns && message.patterns.length > 0 && (
                <div className="bg-[#f59e0b]/10 border-l-4 border-[#f59e0b] rounded-lg p-4 mb-3">
                  <div className="flex items-start gap-3">
                    <AlertTriangle className="w-5 h-5 text-[#f59e0b] flex-shrink-0 mt-0.5" />
                    <div>
                      <p className="font-medium text-[#f59e0b] mb-1">Pattern Detected</p>
                      <p className="text-sm text-white/70">
                        I noticed some language that may indicate {formatPatterns(message.patterns)}. 
                        Your experience is valid, and recognizing these patterns is an important step.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* Message Bubble */}
              <div className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                    message.role === "user"
                      ? "metallic-silver-box text-white"
                      : "bg-white/5 text-white/90 border border-white/10"
                  } ${message.isCrisis ? "ring-2 ring-[#dc2626]/50" : ""}`}
                >
                  {/* Show image preview if message has image */}
                  {message.imageUrl && (
                    <div className="mb-2">
                      <img 
                        src={message.imageUrl} 
                        alt="Shared content" 
                        className="max-w-full max-h-48 rounded-lg object-contain"
                      />
                      <p className="text-xs text-white/50 mt-1">
                        {message.imageType === "conversation" ? "Conversation screenshot" :
                         message.imageType === "journal" ? "Journal entry" : "Image"}
                      </p>
                    </div>
                  )}
                  <Streamdown>{message.content}</Streamdown>
                </div>
              </div>

              {/* Grounding Technique Card */}
              {message.groundingTechnique && (
                <div className="bg-[#1a8a6e]/10 border border-[#1a8a6e]/30 rounded-xl p-5 mt-3 ml-0">
                  <div className="flex items-center gap-2 mb-3">
                    {message.groundingTechnique.name.toLowerCase().includes("breathing") ? (
                      <Wind className="w-5 h-5 text-[#1a8a6e]" />
                    ) : (
                      <Heart className="w-5 h-5 text-[#1a8a6e]" />
                    )}
                    <h4 className="font-semibold text-white">{message.groundingTechnique.name}</h4>
                  </div>
                  <ol className="space-y-2">
                    {message.groundingTechnique.steps.map((step, i) => (
                      <li key={i} className="flex gap-3 text-sm text-white/80">
                        <span className="flex-shrink-0 w-6 h-6 rounded-full bg-[#1a8a6e]/30 flex items-center justify-center text-xs font-medium text-[#1a8a6e]">
                          {i + 1}
                        </span>
                        <span>{step}</span>
                      </li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white/5 border border-white/10 rounded-2xl px-4 py-3">
                <div className="flex gap-1">
                  <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-2 h-2 bg-white/40 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Image Preview & Type Selection */}
      {selectedImage && (
        <div className="border-t border-white/10 bg-[#0a0a0c]/90 p-4">
          <div className="container max-w-3xl mx-auto">
            <div className="flex items-start gap-4">
              <div className="relative">
                <img 
                  src={selectedImage} 
                  alt="Selected" 
                  className="w-24 h-24 object-cover rounded-lg border border-white/20"
                />
                <button
                  onClick={clearSelectedImage}
                  className="absolute -top-2 -right-2 w-6 h-6 bg-[#dc2626] rounded-full flex items-center justify-center text-white hover:bg-[#dc2626]/80"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="flex-1">
                <p className="text-sm text-white/70 mb-2">What type of content is this?</p>
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setImageType("general")}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                      imageType === "general" 
                        ? "bg-[#1a8a6e] text-white" 
                        : "bg-white/5 text-white/70 hover:bg-white/10"
                    }`}
                  >
                    <Camera className="w-4 h-4" />
                    General Image
                  </button>
                  <button
                    onClick={() => setImageType("conversation")}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                      imageType === "conversation" 
                        ? "bg-[#1a8a6e] text-white" 
                        : "bg-white/5 text-white/70 hover:bg-white/10"
                    }`}
                  >
                    <MessageSquare className="w-4 h-4" />
                    Conversation Screenshot
                  </button>
                  <button
                    onClick={() => setImageType("journal")}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${
                      imageType === "journal" 
                        ? "bg-[#1a8a6e] text-white" 
                        : "bg-white/5 text-white/70 hover:bg-white/10"
                    }`}
                  >
                    <FileText className="w-4 h-4" />
                    Journal Entry
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Voice Recording Indicator */}
      {isRecording && (
        <div className="border-t border-[#dc2626]/30 bg-[#dc2626]/10 py-2 px-4">
          <div className="container max-w-3xl mx-auto flex items-center justify-center gap-2 text-[#dc2626]">
            <span className="w-3 h-3 bg-[#dc2626] rounded-full animate-pulse" />
            <span className="text-sm font-medium">Recording... Tap the microphone to stop</span>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="border-t border-white/10 bg-[#0a0a0c]/80 backdrop-blur-sm p-4">
        <form onSubmit={handleSubmit} className="container max-w-3xl mx-auto">
          <div className="flex gap-3">
            {/* Image Upload Button */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageSelect}
              accept="image/*"
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={isLoading}
              className="px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white/70 hover:text-white hover:bg-white/10 transition-colors disabled:opacity-50"
              title="Share an image"
            >
              <Image className="w-5 h-5" />
            </button>

            {/* Voice Input Button */}
            {voiceSupported && (
              <button
                type="button"
                onClick={toggleRecording}
                disabled={isLoading}
                className={`px-3 py-2 rounded-lg border transition-colors disabled:opacity-50 ${
                  isRecording 
                    ? "bg-[#dc2626] border-[#dc2626] text-white" 
                    : "bg-white/5 border-white/10 text-white/70 hover:text-white hover:bg-white/10"
                }`}
                title={isRecording ? "Stop recording" : "Start voice input"}
              >
                {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </button>
            )}

            <Input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                isRecording 
                  ? "Listening..." 
                  : selectedImage 
                    ? "Add context about this image (optional)..." 
                    : "Share what you're experiencing..."
              }
              className="flex-1 bg-white/5 border-white/10 text-white placeholder:text-white/40 focus:border-[#1a8a6e]"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || (!input.trim() && !selectedImage)}
              className="emerald-btn px-4 py-2 rounded-lg disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <Send className="w-4 h-4" />
            </button>
          </div>
          <p className="text-xs text-white/40 mt-2 text-center">
            ReUnity is not a replacement for professional care. In crisis, call 988. © {new Date().getFullYear()} ReUnity-REOP | <a href="https://entropy-physics-ai.com" target="_blank" rel="noopener noreferrer" className="text-emerald-400 hover:text-emerald-300">entropy-physics-ai.com</a>
          </p>
        </form>
      </div>

      {/* Session End Dialog */}
      <SessionEndDialog
        isOpen={showSessionEnd}
        onClose={() => setShowSessionEnd(false)}
        onConfirmEnd={() => {
          // Reset conversation state
          setMessages([{
            id: "welcome",
            role: "assistant",
            content: "I'm here with you. This is a space where you can share what you're experiencing - through words, images, or voice. There's no judgment here, only presence.\n\nYou can type a message, speak using the microphone, or share a screenshot of a conversation or journal entry. What's on your mind?",
          }]);
          setConversationId(null);
          setInput("");
        }}
        messageCount={messages.length - 1}
      />
    </div>
  );
}

// Add type declarations for Web Speech API
interface SpeechRecognitionEvent extends Event {
  resultIndex: number;
  results: SpeechRecognitionResultList;
}

interface SpeechRecognitionResultList {
  length: number;
  item(index: number): SpeechRecognitionResult;
  [index: number]: SpeechRecognitionResult;
}

interface SpeechRecognitionResult {
  isFinal: boolean;
  length: number;
  item(index: number): SpeechRecognitionAlternative;
  [index: number]: SpeechRecognitionAlternative;
}

interface SpeechRecognitionAlternative {
  transcript: string;
  confidence: number;
}

interface SpeechRecognitionErrorEvent extends Event {
  error: string;
  message: string;
}

interface SpeechRecognition extends EventTarget {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
  abort(): void;
}

interface SpeechRecognitionConstructor {
  new (): SpeechRecognition;
}

declare global {
  interface Window {
    SpeechRecognition: SpeechRecognitionConstructor;
    webkitSpeechRecognition: SpeechRecognitionConstructor;
  }
}

function getStateColor(state: string): string {
  switch (state) {
    case "crisis":
      return "bg-[#dc2626]";
    case "high":
      return "bg-[#f59e0b]";
    case "moderate":
      return "bg-[#eab308]";
    case "low":
      return "bg-[#1a8a6e]";
    case "stable":
      return "bg-[#22c55e]";
    default:
      return "bg-white/40";
  }
}

function getStateLabel(state: string): string {
  switch (state) {
    case "crisis":
      return "High distress detected";
    case "high":
      return "Elevated distress";
    case "moderate":
      return "Processing";
    case "low":
      return "Settling";
    case "stable":
      return "Grounded";
    default:
      return "";
  }
}

function formatPatterns(patterns: string[]): string {
  const formatted = patterns.map(p => p.replace(/_/g, " "));
  if (formatted.length === 1) return formatted[0];
  if (formatted.length === 2) return `${formatted[0]} and ${formatted[1]}`;
  return `${formatted.slice(0, -1).join(", ")}, and ${formatted[formatted.length - 1]}`;
}
