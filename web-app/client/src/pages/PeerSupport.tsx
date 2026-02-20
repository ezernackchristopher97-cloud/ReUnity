import { useState, useEffect, useRef } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { Link } from "wouter";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { ScrollArea } from "@/components/ui/scroll-area";
import { toast } from "sonner";
import { 
  Users, 
  Heart, 
  Shield, 
  MessageCircle, 
  UserPlus, 
  Settings,
  AlertTriangle,
  CheckCircle2,
  Send,
  Flag,
  Phone,
  Loader2,
  RefreshCw
} from "lucide-react";
import { trpc } from "@/lib/trpc";

// Experience tags for matching
const experienceTags = [
  { id: "anxiety", label: "Anxiety", description: "Living with anxiety disorders" },
  { id: "depression", label: "Depression", description: "Experiencing depression" },
  { id: "ptsd", label: "PTSD", description: "Post-traumatic stress" },
  { id: "cptsd", label: "Complex Trauma", description: "Complex PTSD" },
  { id: "bpd", label: "BPD", description: "Borderline personality disorder" },
  { id: "bipolar", label: "Bipolar", description: "Bipolar disorder" },
  { id: "ocd", label: "OCD", description: "Obsessive-compulsive disorder" },
  { id: "eating_disorder", label: "Eating Disorder Recovery", description: "Recovery from eating disorders" },
  { id: "substance_recovery", label: "Substance Recovery", description: "Recovery from substance use" },
  { id: "grief", label: "Grief", description: "Processing grief and loss" },
  { id: "domestic_violence", label: "Domestic Violence", description: "Surviving domestic violence" },
  { id: "childhood_trauma", label: "Childhood Trauma", description: "Processing childhood trauma" },
  { id: "dissociation", label: "Dissociation", description: "Experiencing dissociation" },
  { id: "self_harm_recovery", label: "Self-Harm Recovery", description: "Recovery from self-harm" },
  { id: "lgbtq", label: "LGBTQ+", description: "LGBTQ+ experiences" },
  { id: "neurodivergent", label: "Neurodivergent", description: "Autism, ADHD, etc." },
  { id: "rural_isolation", label: "Rural Isolation", description: "Isolation from rural living" },
  { id: "religious_trauma", label: "Religious Trauma", description: "Trauma from religious experiences" }
];

export default function PeerSupport() {
  const { user, isLoading: authLoading } = useAuth();
  // Using sonner toast directly
  const [activeTab, setActiveTab] = useState("discover");
  const [selectedExperiences, setSelectedExperiences] = useState<string[]>([]);
  const [lookingFor, setLookingFor] = useState<string[]>([]);
  const [displayName, setDisplayName] = useState("");
  const [chatMessage, setChatMessage] = useState("");
  const [showGuidelines, setShowGuidelines] = useState(true);
  const [selectedConnection, setSelectedConnection] = useState<number | null>(null);
  const [lastMessageId, setLastMessageId] = useState<number | undefined>(undefined);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // tRPC queries
  const { data: profile, isLoading: profileLoading, refetch: refetchProfile } = trpc.peerSupport.getProfile.useQuery(
    undefined,
    { enabled: !!user }
  );

  const { data: matchesData, isLoading: matchesLoading, refetch: refetchMatches } = trpc.peerSupport.getMatches.useQuery(
    undefined,
    { enabled: !!user && !!profile }
  );

  const { data: connections, isLoading: connectionsLoading, refetch: refetchConnections } = trpc.peerSupport.getConnections.useQuery(
    undefined,
    { enabled: !!user }
  );

  const { data: messages, refetch: refetchMessages } = trpc.peerSupport.getMessages.useQuery(
    { connectionId: selectedConnection! },
    { enabled: !!selectedConnection }
  );

  // Polling for new messages every 5 seconds
  const { data: newMessagesData } = trpc.peerSupport.checkNewMessages.useQuery(
    { connectionId: selectedConnection!, lastMessageId },
    { 
      enabled: !!selectedConnection && activeTab === "messages",
      refetchInterval: 5000 // Poll every 5 seconds
    }
  );

  // tRPC mutations
  const saveProfileMutation = trpc.peerSupport.saveProfile.useMutation({
    onSuccess: () => {
      toast.success("Profile saved! Your peer support profile has been created.");
      refetchProfile();
      refetchMatches();
    },
    onError: (error) => {
      toast.error(error.message);
    }
  });

  const requestConnectionMutation = trpc.peerSupport.requestConnection.useMutation({
    onSuccess: () => {
      toast.success("Connection request sent! They'll be notified of your request.");
      refetchConnections();
    },
    onError: (error) => {
      toast.error(error.message);
    }
  });

  const respondToConnectionMutation = trpc.peerSupport.respondToConnection.useMutation({
    onSuccess: () => {
      toast.success("Response sent!");
      refetchConnections();
    }
  });

  const sendMessageMutation = trpc.peerSupport.sendMessage.useMutation({
    onSuccess: (data) => {
      setChatMessage("");
      refetchMessages();
      if (data.isCrisis) {
        toast.warning("Crisis resources available: If you or your peer is in crisis, please reach out to 988 or text HOME to 741741.");
      }
    },
    onError: (error) => {
      toast.error(error.message);
    }
  });

  const flagMessageMutation = trpc.peerSupport.flagMessage.useMutation({
    onSuccess: () => {
      toast.success("Message flagged. Our moderators will review this message.");
    }
  });

  // Update lastMessageId when messages change
  useEffect(() => {
    if (messages && messages.length > 0) {
      const maxId = Math.max(...messages.map(m => m.id));
      setLastMessageId(maxId);
    }
  }, [messages]);

  // Handle new messages from polling
  useEffect(() => {
    if (newMessagesData?.hasNew) {
      refetchMessages();
    }
  }, [newMessagesData, refetchMessages]);

  // Scroll to bottom when messages change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const toggleExperience = (id: string) => {
    if (selectedExperiences.includes(id)) {
      setSelectedExperiences(selectedExperiences.filter(e => e !== id));
    } else {
      setSelectedExperiences([...selectedExperiences, id]);
    }
  };

  const toggleLookingFor = (id: string) => {
    if (lookingFor.includes(id)) {
      setLookingFor(lookingFor.filter(e => e !== id));
    } else {
      setLookingFor([...lookingFor, id]);
    }
  };

  const handleCreateProfile = () => {
    if (!displayName || selectedExperiences.length === 0) return;
    
    saveProfileMutation.mutate({
      displayName,
      experienceTags: selectedExperiences,
      lookingFor: lookingFor.length > 0 ? lookingFor : selectedExperiences
    });
  };

  const handleConnect = (profileId: number) => {
    requestConnectionMutation.mutate({ targetProfileId: profileId });
  };

  const handleSendMessage = () => {
    if (!chatMessage.trim() || !selectedConnection) return;
    sendMessageMutation.mutate({
      connectionId: selectedConnection,
      content: chatMessage
    });
  };

  const handleFlagMessage = (messageId: number) => {
    const reason = prompt("Please describe why you're flagging this message:");
    if (reason) {
      flagMessageMutation.mutate({ messageId, reason });
    }
  };

  if (authLoading || profileLoading) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
      </div>
    );
  }

  if (!user) {
    return (
      <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <Card className="bg-slate-800/50 border-slate-700 max-w-md">
          <CardHeader>
            <CardTitle className="text-white">Sign In Required</CardTitle>
            <CardDescription className="text-slate-400">
              Please sign in to access peer support features.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Link href="/login">
              <Button className="w-full bg-emerald-500 hover:bg-emerald-600">
                Sign In
              </Button>
            </Link>
          </CardContent>
        </Card>
      </div>
    );
  }

  const hasProfile = !!profile;
  const pendingConnections = connections?.filter(c => c.status === 'pending' && c.responderId === user.id) || [];
  const acceptedConnections = connections?.filter(c => c.status === 'accepted') || [];

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <header className="border-b border-slate-700/50 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-50">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <Users className="h-6 w-6 text-emerald-400" />
            <span className="text-xl font-bold text-white">Peer Support</span>
          </Link>
          <div className="flex items-center gap-4">
            {pendingConnections.length > 0 && (
              <Badge className="bg-amber-500/20 text-amber-300 border-amber-500/30">
                {pendingConnections.length} pending
              </Badge>
            )}
            <Badge variant="outline" className="border-emerald-500/30 text-emerald-400">
              <Shield className="h-3 w-3 mr-1" />
              Anonymous
            </Badge>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 max-w-6xl">
        {/* Guidelines Modal */}
        {showGuidelines && !hasProfile && (
          <Card className="bg-slate-800/50 border-slate-700 mb-8">
            <CardHeader>
              <CardTitle className="text-white flex items-center gap-2">
                <Heart className="h-5 w-5 text-pink-400" />
                Welcome to Peer Support
              </CardTitle>
              <CardDescription className="text-slate-400">
                Connect with others who understand your experiences
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <Alert className="bg-teal-500/10 border-teal-500/30">
                <Shield className="h-4 w-4 text-teal-400" />
                <AlertTitle className="text-teal-300">What This Is</AlertTitle>
                <AlertDescription className="text-teal-200/80">
                  <ul className="list-disc list-inside space-y-1 mt-2">
                    <li>A place to connect with others who understand</li>
                    <li>A space for mutual support, not professional treatment</li>
                    <li>An anonymous community focused on healing together</li>
                  </ul>
                </AlertDescription>
              </Alert>

              <Alert className="bg-amber-500/10 border-amber-500/30">
                <AlertTriangle className="h-4 w-4 text-amber-400" />
                <AlertTitle className="text-amber-300">Community Rules</AlertTitle>
                <AlertDescription className="text-amber-200/80">
                  <ul className="list-disc list-inside space-y-1 mt-2">
                    <li>Respect boundaries - ask before discussing sensitive topics</li>
                    <li>Maintain anonymity - don't share personal info</li>
                    <li>Support, don't fix - listen more than advise</li>
                    <li>No romantic or sexual content</li>
                    <li>Report concerns to moderators</li>
                  </ul>
                </AlertDescription>
              </Alert>

              <Alert className="bg-red-500/10 border-red-500/30">
                <Phone className="h-4 w-4 text-red-400" />
                <AlertTitle className="text-red-300">Crisis Resources</AlertTitle>
                <AlertDescription className="text-red-200/80">
                  If you or someone you're talking to is in crisis:
                  <ul className="list-disc list-inside space-y-1 mt-2">
                    <li>988 Suicide & Crisis Lifeline: Call or text 988</li>
                    <li>Crisis Text Line: Text HOME to 741741</li>
                  </ul>
                </AlertDescription>
              </Alert>

              <Button 
                onClick={() => setShowGuidelines(false)}
                className="w-full bg-emerald-500 hover:bg-emerald-600"
              >
                Continue to Profile Setup
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Profile Setup */}
        {!showGuidelines && !hasProfile && (
          <Card className="bg-slate-800/50 border-slate-700 mb-8">
            <CardHeader>
              <CardTitle className="text-white">Create Your Anonymous Profile</CardTitle>
              <CardDescription className="text-slate-400">
                Your real identity is never shared. Choose what you're comfortable sharing.
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-4">
                <Label className="text-slate-300">Display Name</Label>
                <Input
                  value={displayName}
                  onChange={(e) => setDisplayName(e.target.value)}
                  placeholder="e.g., GentleRiver42"
                  className="bg-slate-900/50 border-slate-600 text-white"
                />
                <p className="text-sm text-slate-400">
                  This is how others will see you. Choose something anonymous.
                </p>
              </div>

              <div className="space-y-4">
                <Label className="text-slate-300">My Experiences (Select all that apply)</Label>
                <p className="text-sm text-slate-400">
                  This helps us match you with others who understand your experiences.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {experienceTags.map((tag) => (
                    <button
                      key={tag.id}
                      onClick={() => toggleExperience(tag.id)}
                      className={`p-3 rounded-lg text-left transition-colors ${
                        selectedExperiences.includes(tag.id)
                          ? "bg-emerald-500/20 border border-emerald-500/30 text-emerald-300"
                          : "bg-slate-700/50 border border-slate-600 text-slate-300 hover:bg-slate-700"
                      }`}
                    >
                      <div className="font-medium text-sm">{tag.label}</div>
                      <div className="text-xs opacity-70">{tag.description}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <Label className="text-slate-300">Looking For Support With (Optional)</Label>
                <p className="text-sm text-slate-400">
                  Select experiences you'd like to find support for.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {experienceTags.map((tag) => (
                    <button
                      key={tag.id}
                      onClick={() => toggleLookingFor(tag.id)}
                      className={`p-3 rounded-lg text-left transition-colors ${
                        lookingFor.includes(tag.id)
                          ? "bg-purple-500/20 border border-purple-500/30 text-purple-300"
                          : "bg-slate-700/50 border border-slate-600 text-slate-300 hover:bg-slate-700"
                      }`}
                    >
                      <div className="font-medium text-sm">{tag.label}</div>
                    </button>
                  ))}
                </div>
              </div>

              <Button 
                onClick={handleCreateProfile}
                disabled={!displayName || selectedExperiences.length === 0 || saveProfileMutation.isPending}
                className="w-full bg-emerald-500 hover:bg-emerald-600"
              >
                {saveProfileMutation.isPending ? (
                  <><Loader2 className="h-4 w-4 mr-2 animate-spin" /> Creating...</>
                ) : (
                  "Create Profile"
                )}
              </Button>
            </CardContent>
          </Card>
        )}

        {/* Main Interface */}
        {hasProfile && (
          <Tabs value={activeTab} onValueChange={setActiveTab}>
            <TabsList className="bg-slate-800/50 border-slate-700 mb-6">
              <TabsTrigger value="discover" className="data-[state=active]:bg-emerald-500/20">
                <UserPlus className="h-4 w-4 mr-2" />
                Discover
              </TabsTrigger>
              <TabsTrigger value="connections" className="data-[state=active]:bg-emerald-500/20">
                <Users className="h-4 w-4 mr-2" />
                Connections
                {pendingConnections.length > 0 && (
                  <Badge className="ml-2 bg-amber-500/20 text-amber-300 text-xs">
                    {pendingConnections.length}
                  </Badge>
                )}
              </TabsTrigger>
              <TabsTrigger value="messages" className="data-[state=active]:bg-emerald-500/20">
                <MessageCircle className="h-4 w-4 mr-2" />
                Messages
              </TabsTrigger>
              <TabsTrigger value="settings" className="data-[state=active]:bg-emerald-500/20">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </TabsTrigger>
            </TabsList>

            {/* Discover Tab */}
            <TabsContent value="discover">
              <div className="grid gap-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h2 className="text-xl font-semibold text-white">People Who Understand</h2>
                    <p className="text-slate-400">
                      These peers share similar experiences and are looking to connect.
                    </p>
                  </div>
                  <Button 
                    variant="outline" 
                    size="sm"
                    onClick={() => refetchMatches()}
                    className="border-slate-600"
                  >
                    <RefreshCw className="h-4 w-4 mr-2" />
                    Refresh
                  </Button>
                </div>

                {matchesLoading ? (
                  <div className="flex justify-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
                  </div>
                ) : matchesData?.needsProfile ? (
                  <Card className="bg-slate-800/50 border-slate-700">
                    <CardContent className="p-8 text-center">
                      <p className="text-slate-400">Complete your profile to see matches</p>
                    </CardContent>
                  </Card>
                ) : matchesData?.matches.length === 0 ? (
                  <Card className="bg-slate-800/50 border-slate-700">
                    <CardContent className="p-8 text-center">
                      <Users className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                      <p className="text-slate-400">No matches found yet</p>
                      <p className="text-sm text-slate-500 mt-2">
                        Check back later as more people join
                      </p>
                    </CardContent>
                  </Card>
                ) : (
                  matchesData?.matches.map((match) => (
                    <Card key={match.id} className="bg-slate-800/50 border-slate-700">
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-4">
                            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-emerald-400 to-teal-500 flex items-center justify-center text-white font-bold">
                              {match.displayName.charAt(0)}
                            </div>
                            <div>
                              <h3 className="text-lg font-medium text-white">{match.displayName}</h3>
                            </div>
                          </div>
                          <Badge className="bg-emerald-500/20 text-emerald-300 border-emerald-500/30">
                            {match.matchScore}% Match
                          </Badge>
                        </div>

                        <div className="mt-4">
                          <p className="text-sm text-slate-400 mb-2">Shared experiences:</p>
                          <div className="flex flex-wrap gap-2">
                            {match.sharedExperiences.map((exp: string) => (
                              <Badge key={exp} variant="outline" className="border-slate-600 text-slate-300">
                                {experienceTags.find(t => t.id === exp)?.label || exp}
                              </Badge>
                            ))}
                          </div>
                        </div>

                        <div className="mt-4 flex gap-2">
                          <Button 
                            onClick={() => handleConnect(match.id)}
                            disabled={requestConnectionMutation.isPending}
                            className="bg-emerald-500 hover:bg-emerald-600"
                          >
                            <Heart className="h-4 w-4 mr-2" />
                            Connect
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))
                )}
              </div>
            </TabsContent>

            {/* Connections Tab */}
            <TabsContent value="connections">
              <div className="grid gap-4">
                {/* Pending Requests */}
                {pendingConnections.length > 0 && (
                  <>
                    <h2 className="text-xl font-semibold text-white">Pending Requests</h2>
                    {pendingConnections.map((connection) => (
                      <Card key={connection.id} className="bg-amber-500/10 border-amber-500/30">
                        <CardContent className="p-6">
                          <div className="flex items-start justify-between">
                            <div className="flex items-center gap-4">
                              <div className="w-12 h-12 rounded-full bg-gradient-to-br from-amber-400 to-orange-500 flex items-center justify-center text-white font-bold">
                                {connection.peerProfile?.displayName?.charAt(0) || "?"}
                              </div>
                              <div>
                                <h3 className="text-lg font-medium text-white">
                                  {connection.peerProfile?.displayName || "Anonymous"}
                                </h3>
                                <p className="text-sm text-amber-300">Wants to connect</p>
                              </div>
                            </div>
                            <div className="flex gap-2">
                              <Button 
                                onClick={() => respondToConnectionMutation.mutate({ connectionId: connection.id, accept: true })}
                                className="bg-emerald-500 hover:bg-emerald-600"
                                size="sm"
                              >
                                <CheckCircle2 className="h-4 w-4 mr-1" />
                                Accept
                              </Button>
                              <Button 
                                onClick={() => respondToConnectionMutation.mutate({ connectionId: connection.id, accept: false })}
                                variant="outline"
                                className="border-slate-600"
                                size="sm"
                              >
                                Decline
                              </Button>
                            </div>
                          </div>
                        </CardContent>
                      </Card>
                    ))}
                  </>
                )}

                <h2 className="text-xl font-semibold text-white">Your Connections</h2>

                {connectionsLoading ? (
                  <div className="flex justify-center py-8">
                    <Loader2 className="h-8 w-8 animate-spin text-emerald-500" />
                  </div>
                ) : acceptedConnections.length === 0 ? (
                  <Card className="bg-slate-800/50 border-slate-700">
                    <CardContent className="p-8 text-center">
                      <Users className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                      <p className="text-slate-400">No connections yet</p>
                      <p className="text-sm text-slate-500 mt-2">
                        Discover peers who share your experiences
                      </p>
                      <Button 
                        onClick={() => setActiveTab("discover")}
                        className="mt-4 bg-emerald-500 hover:bg-emerald-600"
                      >
                        Find Peers
                      </Button>
                    </CardContent>
                  </Card>
                ) : (
                  acceptedConnections.map((connection) => (
                    <Card key={connection.id} className="bg-slate-800/50 border-slate-700">
                      <CardContent className="p-6">
                        <div className="flex items-start justify-between">
                          <div className="flex items-center gap-4">
                            <div className="w-12 h-12 rounded-full bg-gradient-to-br from-purple-400 to-pink-500 flex items-center justify-center text-white font-bold">
                              {connection.peerProfile?.displayName?.charAt(0) || "?"}
                            </div>
                            <div>
                              <h3 className="text-lg font-medium text-white">
                                {connection.peerProfile?.displayName || "Anonymous"}
                              </h3>
                            </div>
                          </div>
                          <Badge className="bg-green-500/20 text-green-300 border-green-500/30">
                            <CheckCircle2 className="h-3 w-3 mr-1" />
                            Connected
                          </Badge>
                        </div>

                        <div className="mt-4 flex gap-2">
                          <Button 
                            onClick={() => {
                              setSelectedConnection(connection.id);
                              setActiveTab("messages");
                            }}
                            className="bg-emerald-500 hover:bg-emerald-600"
                          >
                            <MessageCircle className="h-4 w-4 mr-2" />
                            Message
                          </Button>
                        </div>
                      </CardContent>
                    </Card>
                  ))
                )}
              </div>
            </TabsContent>

            {/* Messages Tab */}
            <TabsContent value="messages">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 h-[600px]">
                {/* Conversation List */}
                <Card className="bg-slate-800/50 border-slate-700">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-lg text-white">Conversations</CardTitle>
                  </CardHeader>
                  <CardContent className="p-2">
                    {acceptedConnections.length === 0 ? (
                      <p className="text-slate-400 text-sm p-3">No conversations yet</p>
                    ) : (
                      acceptedConnections.map((connection) => (
                        <button
                          key={connection.id}
                          onClick={() => setSelectedConnection(connection.id)}
                          className={`w-full p-3 rounded-lg text-left transition-colors ${
                            selectedConnection === connection.id
                              ? "bg-emerald-500/20"
                              : "hover:bg-slate-700/50"
                          }`}
                        >
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-400 to-pink-500 flex items-center justify-center text-white font-bold text-sm">
                              {connection.peerProfile?.displayName?.charAt(0) || "?"}
                            </div>
                            <div className="flex-1 min-w-0">
                              <p className="text-white font-medium truncate">
                                {connection.peerProfile?.displayName || "Anonymous"}
                              </p>
                            </div>
                          </div>
                        </button>
                      ))
                    )}
                  </CardContent>
                </Card>

                {/* Chat Area */}
                <Card className="bg-slate-800/50 border-slate-700 md:col-span-2 flex flex-col">
                  {selectedConnection ? (
                    <>
                      <CardHeader className="border-b border-slate-700">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-purple-400 to-pink-500 flex items-center justify-center text-white font-bold">
                              {acceptedConnections.find(c => c.id === selectedConnection)?.peerProfile?.displayName?.charAt(0) || "?"}
                            </div>
                            <div>
                              <CardTitle className="text-lg text-white">
                                {acceptedConnections.find(c => c.id === selectedConnection)?.peerProfile?.displayName || "Anonymous"}
                              </CardTitle>
                              <CardDescription className="text-slate-400">
                                {newMessagesData?.hasNew && <span className="text-emerald-400">New messages</span>}
                              </CardDescription>
                            </div>
                          </div>
                        </div>
                      </CardHeader>

                      <ScrollArea className="flex-1 p-4">
                        <div className="space-y-4">
                          {/* Welcome message */}
                          <div className="bg-teal-500/10 border border-teal-500/30 rounded-lg p-4 text-center">
                            <p className="text-teal-300 text-sm">
                              Remember: This is peer support, not professional therapy.
                              If either of you is in crisis, please reach out to 988.
                            </p>
                          </div>

                          {/* Messages */}
                          {messages?.slice().reverse().map((message) => (
                            <div 
                              key={message.id}
                              className={`flex ${message.senderId === user?.id ? "justify-end" : "justify-start"}`}
                            >
                              <div className={`rounded-lg p-3 max-w-[80%] relative group ${
                                message.senderId === user?.id
                                  ? "bg-emerald-500/20"
                                  : "bg-slate-700/50"
                              }`}>
                                <p className="text-white">{message.content}</p>
                                <p className="text-xs text-slate-400 mt-1">
                                  {new Date(message.createdAt).toLocaleTimeString()}
                                </p>
                                {message.senderId !== user?.id && (
                                  <button
                                    onClick={() => handleFlagMessage(message.id)}
                                    className="absolute -right-8 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity"
                                  >
                                    <Flag className="h-4 w-4 text-slate-400 hover:text-red-400" />
                                  </button>
                                )}
                                {message.crisisDetected && (
                                  <div className="mt-2 p-2 bg-red-500/20 rounded text-xs text-red-300">
                                    Crisis resources: 988 or text HOME to 741741
                                  </div>
                                )}
                              </div>
                            </div>
                          ))}
                          <div ref={messagesEndRef} />
                        </div>
                      </ScrollArea>

                      <div className="p-4 border-t border-slate-700">
                        <div className="flex gap-2">
                          <Input
                            value={chatMessage}
                            onChange={(e) => setChatMessage(e.target.value)}
                            onKeyDown={(e) => e.key === "Enter" && handleSendMessage()}
                            placeholder="Type a message..."
                            className="bg-slate-900/50 border-slate-600 text-white"
                          />
                          <Button 
                            onClick={handleSendMessage}
                            disabled={!chatMessage.trim() || sendMessageMutation.isPending}
                            className="bg-emerald-500 hover:bg-emerald-600"
                          >
                            {sendMessageMutation.isPending ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Send className="h-4 w-4" />
                            )}
                          </Button>
                        </div>
                      </div>
                    </>
                  ) : (
                    <CardContent className="flex-1 flex items-center justify-center">
                      <div className="text-center">
                        <MessageCircle className="h-12 w-12 text-slate-500 mx-auto mb-4" />
                        <p className="text-slate-400">Select a conversation to start messaging</p>
                      </div>
                    </CardContent>
                  )}
                </Card>
              </div>
            </TabsContent>

            {/* Settings Tab */}
            <TabsContent value="settings">
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle className="text-white">Profile Settings</CardTitle>
                  <CardDescription className="text-slate-400">
                    Manage your peer support profile and preferences
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="space-y-4">
                    <Label className="text-slate-300">Display Name</Label>
                    <Input
                      defaultValue={profile?.displayName}
                      className="bg-slate-900/50 border-slate-600 text-white"
                    />
                  </div>

                  <div className="space-y-4">
                    <Label className="text-slate-300">Safety Settings</Label>
                    <div className="space-y-2">
                      <div className="flex items-center space-x-2">
                        <Checkbox id="crisis" defaultChecked />
                        <Label htmlFor="crisis" className="text-slate-300">
                          Allow crisis escalation (show resources when crisis detected)
                        </Label>
                      </div>
                    </div>
                  </div>

                  <Button className="bg-emerald-500 hover:bg-emerald-600">
                    Save Settings
                  </Button>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        )}
      </main>
    </div>
  );
}
