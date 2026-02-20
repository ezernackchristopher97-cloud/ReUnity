import { useState } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { 
  Users, 
  MessageCircle, 
  Heart, 
  Shield, 
  Search, 
  Plus, 
  Clock, 
  Star,
  BookOpen,
  Sparkles,
  AlertTriangle,
  CheckCircle,
  Send,
  ThumbsUp,
  Flag,
  Lock
} from "lucide-react";

interface SupportGroup {
  id: string;
  name: string;
  topic: string;
  description: string;
  memberCount: number;
  isModerated: boolean;
  isPrivate: boolean;
  tags: string[];
  lastActive: Date;
  isMember: boolean;
}

interface Discussion {
  id: string;
  groupId: string;
  author: string;
  authorBadge?: string;
  content: string;
  timestamp: Date;
  likes: number;
  replies: number;
  isLiked: boolean;
  isPinned?: boolean;
}

interface Resource {
  id: string;
  title: string;
  type: "article" | "video" | "worksheet" | "guide";
  topic: string;
  description: string;
}

// Mock community groups
const MOCK_GROUPS: SupportGroup[] = [
  {
    id: "1",
    name: "Anxiety Warriors",
    topic: "Anxiety",
    description: "A safe space to share experiences with anxiety, panic attacks, and learn coping strategies together.",
    memberCount: 1247,
    isModerated: true,
    isPrivate: false,
    tags: ["anxiety", "panic", "coping", "support"],
    lastActive: new Date(Date.now() - 1000 * 60 * 5),
    isMember: true
  },
  {
    id: "2",
    name: "Depression Support Circle",
    topic: "Depression",
    description: "Understanding depression together. Share your journey, find hope, and support each other through dark times.",
    memberCount: 2103,
    isModerated: true,
    isPrivate: false,
    tags: ["depression", "hope", "recovery", "support"],
    lastActive: new Date(Date.now() - 1000 * 60 * 15),
    isMember: false
  },
  {
    id: "3",
    name: "PTSD & Trauma Healing",
    topic: "PTSD/Trauma",
    description: "For survivors of trauma. A moderated, trigger-warned space for healing and understanding.",
    memberCount: 856,
    isModerated: true,
    isPrivate: true,
    tags: ["ptsd", "trauma", "healing", "survivors"],
    lastActive: new Date(Date.now() - 1000 * 60 * 30),
    isMember: true
  },
  {
    id: "4",
    name: "BPD Understanding",
    topic: "BPD",
    description: "Living with Borderline Personality Disorder. DBT skills, emotional regulation, and peer support.",
    memberCount: 634,
    isModerated: true,
    isPrivate: false,
    tags: ["bpd", "dbt", "emotions", "identity"],
    lastActive: new Date(Date.now() - 1000 * 60 * 45),
    isMember: false
  },
  {
    id: "5",
    name: "Grief & Loss",
    topic: "Grief",
    description: "Processing loss together. Whether recent or long-ago, all grief is valid here.",
    memberCount: 1089,
    isModerated: true,
    isPrivate: false,
    tags: ["grief", "loss", "bereavement", "healing"],
    lastActive: new Date(Date.now() - 1000 * 60 * 60),
    isMember: false
  },
  {
    id: "6",
    name: "Relationship Recovery",
    topic: "Relationships",
    description: "Healing from toxic relationships, rebuilding trust, and learning healthy relationship patterns.",
    memberCount: 923,
    isModerated: true,
    isPrivate: false,
    tags: ["relationships", "abuse", "healing", "boundaries"],
    lastActive: new Date(Date.now() - 1000 * 60 * 90),
    isMember: true
  }
];

// Mock discussions
const MOCK_DISCUSSIONS: Discussion[] = [
  {
    id: "d1",
    groupId: "1",
    author: "HopefulHeart",
    authorBadge: "Peer Supporter",
    content: "Today I managed to go to the grocery store without having a panic attack. It's been 6 months of work to get here. Small wins matter! 💪",
    timestamp: new Date(Date.now() - 1000 * 60 * 30),
    likes: 47,
    replies: 12,
    isLiked: false,
    isPinned: true
  },
  {
    id: "d2",
    groupId: "1",
    author: "QuietStrength",
    content: "Does anyone else find that their anxiety is worse in the morning? I wake up with my heart racing almost every day.",
    timestamp: new Date(Date.now() - 1000 * 60 * 120),
    likes: 23,
    replies: 8,
    isLiked: true
  },
  {
    id: "d3",
    groupId: "3",
    author: "GentleSoul",
    authorBadge: "Moderator",
    content: "Reminder: This is a safe space. Please use trigger warnings when discussing specific traumatic events. We're all here to support each other. 💚",
    timestamp: new Date(Date.now() - 1000 * 60 * 60),
    likes: 89,
    replies: 3,
    isLiked: true,
    isPinned: true
  }
];

// Mock resources
const MOCK_RESOURCES: Resource[] = [
  { id: "r1", title: "Understanding Panic Attacks", type: "article", topic: "Anxiety", description: "Learn what happens in your body during a panic attack and why they're not dangerous." },
  { id: "r2", title: "Grounding Techniques Video", type: "video", topic: "General", description: "A guided 5-minute grounding exercise you can use anywhere." },
  { id: "r3", title: "Mood Tracking Worksheet", type: "worksheet", topic: "Depression", description: "Daily mood tracking template to identify patterns." },
  { id: "r4", title: "DBT Skills Guide", type: "guide", topic: "BPD", description: "Introduction to Dialectical Behavior Therapy skills for emotional regulation." },
];

export function CommunitySupportGroups({ compact = false }: { compact?: boolean }) {
  const [groups, setGroups] = useState<SupportGroup[]>(MOCK_GROUPS);
  const [discussions] = useState<Discussion[]>(MOCK_DISCUSSIONS);
  const [resources] = useState<Resource[]>(MOCK_RESOURCES);
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedGroup, setSelectedGroup] = useState<SupportGroup | null>(null);
  const [newPost, setNewPost] = useState("");
  const [activeTab, setActiveTab] = useState("discover");

  const filteredGroups = groups.filter(g => 
    g.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    g.topic.toLowerCase().includes(searchTerm.toLowerCase()) ||
    g.tags.some(t => t.toLowerCase().includes(searchTerm.toLowerCase()))
  );

  const myGroups = groups.filter(g => g.isMember);

  const joinGroup = (groupId: string) => {
    setGroups(groups.map(g => 
      g.id === groupId ? { ...g, isMember: true, memberCount: g.memberCount + 1 } : g
    ));
  };

  const leaveGroup = (groupId: string) => {
    setGroups(groups.map(g => 
      g.id === groupId ? { ...g, isMember: false, memberCount: g.memberCount - 1 } : g
    ));
  };

  const getTimeAgo = (date: Date) => {
    const minutes = Math.floor((Date.now() - date.getTime()) / (1000 * 60));
    if (minutes < 60) return `${minutes}m ago`;
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    return `${Math.floor(hours / 24)}d ago`;
  };

  const getResourceIcon = (type: Resource["type"]) => {
    switch (type) {
      case "article": return <BookOpen className="h-4 w-4" />;
      case "video": return <Sparkles className="h-4 w-4" />;
      case "worksheet": return <CheckCircle className="h-4 w-4" />;
      case "guide": return <BookOpen className="h-4 w-4" />;
    }
  };

  if (compact) {
    return (
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader className="pb-2">
          <CardTitle className="text-sm flex items-center gap-2">
            <Users className="h-4 w-4 text-indigo-400" />
            Community Groups
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-between text-xs">
            <div className="text-slate-400">
              {myGroups.length} groups joined
            </div>
            <Badge variant="outline" className="text-xs bg-indigo-500/20 text-indigo-400 border-indigo-500/30">
              {groups.reduce((sum, g) => sum + g.memberCount, 0).toLocaleString()} members
            </Badge>
          </div>
          <div className="mt-2 flex flex-wrap gap-1">
            {myGroups.slice(0, 3).map(g => (
              <Badge key={g.id} variant="outline" className="text-xs bg-slate-700/50">
                {g.name}
              </Badge>
            ))}
          </div>
        </CardContent>
      </Card>
    );
  }

  if (selectedGroup) {
    const groupDiscussions = discussions.filter(d => d.groupId === selectedGroup.id);
    
    return (
      <div className="space-y-4">
        <Card className="bg-slate-800/50 border-slate-700">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <Button 
                  variant="ghost" 
                  size="sm" 
                  onClick={() => setSelectedGroup(null)}
                  className="mb-2 -ml-2"
                >
                  ← Back to Groups
                </Button>
                <CardTitle className="flex items-center gap-2">
                  {selectedGroup.isPrivate && <Lock className="h-4 w-4 text-yellow-400" />}
                  {selectedGroup.name}
                </CardTitle>
                <CardDescription>{selectedGroup.description}</CardDescription>
              </div>
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="bg-slate-700/50">
                  <Users className="h-3 w-3 mr-1" />
                  {selectedGroup.memberCount.toLocaleString()}
                </Badge>
                {selectedGroup.isMember ? (
                  <Button variant="outline" size="sm" onClick={() => leaveGroup(selectedGroup.id)}>
                    Leave
                  </Button>
                ) : (
                  <Button size="sm" onClick={() => joinGroup(selectedGroup.id)}>
                    Join
                  </Button>
                )}
              </div>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* New Post */}
            {selectedGroup.isMember && (
              <div className="bg-slate-900/50 rounded-lg p-4">
                <Textarea
                  placeholder="Share something with the group..."
                  value={newPost}
                  onChange={(e) => setNewPost(e.target.value)}
                  className="bg-slate-800/50 border-slate-600 mb-2"
                  rows={3}
                />
                <div className="flex justify-between items-center">
                  <div className="text-xs text-slate-400">
                    <Shield className="h-3 w-3 inline mr-1" />
                    All posts are moderated
                  </div>
                  <Button size="sm" disabled={!newPost.trim()}>
                    <Send className="h-4 w-4 mr-1" />
                    Post
                  </Button>
                </div>
              </div>
            )}

            {/* Discussions */}
            <div className="space-y-3">
              {groupDiscussions.map((discussion) => (
                <div 
                  key={discussion.id} 
                  className={`bg-slate-900/50 rounded-lg p-4 ${discussion.isPinned ? 'border border-yellow-500/30' : ''}`}
                >
                  {discussion.isPinned && (
                    <Badge variant="outline" className="mb-2 text-xs bg-yellow-500/20 text-yellow-400 border-yellow-500/30">
                      <Star className="h-3 w-3 mr-1" />
                      Pinned
                    </Badge>
                  )}
                  <div className="flex items-center gap-2 mb-2">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-500 flex items-center justify-center text-xs font-bold">
                      {discussion.author[0]}
                    </div>
                    <div>
                      <div className="font-medium text-sm">{discussion.author}</div>
                      {discussion.authorBadge && (
                        <Badge variant="outline" className="text-xs bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                          {discussion.authorBadge}
                        </Badge>
                      )}
                    </div>
                    <div className="ml-auto text-xs text-slate-400">
                      {getTimeAgo(discussion.timestamp)}
                    </div>
                  </div>
                  <p className="text-sm mb-3">{discussion.content}</p>
                  <div className="flex items-center gap-4 text-xs text-slate-400">
                    <button className={`flex items-center gap-1 hover:text-pink-400 ${discussion.isLiked ? 'text-pink-400' : ''}`}>
                      <Heart className={`h-4 w-4 ${discussion.isLiked ? 'fill-current' : ''}`} />
                      {discussion.likes}
                    </button>
                    <button className="flex items-center gap-1 hover:text-blue-400">
                      <MessageCircle className="h-4 w-4" />
                      {discussion.replies} replies
                    </button>
                    <button className="flex items-center gap-1 hover:text-red-400 ml-auto">
                      <Flag className="h-4 w-4" />
                      Report
                    </button>
                  </div>
                </div>
              ))}
            </div>

            {/* Community Guidelines */}
            <div className="bg-indigo-500/10 border border-indigo-500/30 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <Shield className="h-5 w-5 text-indigo-400 mt-0.5" />
                <div>
                  <div className="font-medium text-indigo-400">Community Guidelines</div>
                  <ul className="text-xs text-slate-400 mt-1 space-y-1">
                    <li>• Be kind and supportive to all members</li>
                    <li>• Use trigger warnings when discussing sensitive topics</li>
                    <li>• No medical advice - share experiences, not prescriptions</li>
                    <li>• Report any concerning content to moderators</li>
                  </ul>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <Card className="bg-slate-800/50 border-slate-700">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Users className="h-5 w-5 text-indigo-400" />
            Community Support Groups
          </CardTitle>
          <CardDescription>
            Connect with others who understand. All groups are moderated for safety.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
            <TabsList className="bg-slate-900/50">
              <TabsTrigger value="discover">Discover</TabsTrigger>
              <TabsTrigger value="my-groups">My Groups ({myGroups.length})</TabsTrigger>
              <TabsTrigger value="resources">Resources</TabsTrigger>
            </TabsList>

            <TabsContent value="discover" className="space-y-4">
              {/* Search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
                <Input
                  placeholder="Search groups by topic, name, or tag..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 bg-slate-900/50 border-slate-600"
                />
              </div>

              {/* Group List */}
              <div className="space-y-3">
                {filteredGroups.map((group) => (
                  <div
                    key={group.id}
                    className="bg-slate-900/50 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-colors cursor-pointer"
                    onClick={() => setSelectedGroup(group)}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2">
                          <h4 className="font-medium">{group.name}</h4>
                          {group.isPrivate && (
                            <Lock className="h-3 w-3 text-yellow-400" />
                          )}
                          {group.isModerated && (
                            <Badge variant="outline" className="text-xs bg-emerald-500/20 text-emerald-400 border-emerald-500/30">
                              <Shield className="h-3 w-3 mr-1" />
                              Moderated
                            </Badge>
                          )}
                        </div>
                        <p className="text-sm text-slate-400 mt-1">{group.description}</p>
                        <div className="flex flex-wrap gap-1 mt-2">
                          {group.tags.map((tag) => (
                            <Badge key={tag} variant="outline" className="text-xs bg-slate-700/50">
                              {tag}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <div className="text-right ml-4">
                        <div className="flex items-center gap-1 text-sm text-slate-400">
                          <Users className="h-4 w-4" />
                          {group.memberCount.toLocaleString()}
                        </div>
                        <div className="flex items-center gap-1 text-xs text-slate-500 mt-1">
                          <Clock className="h-3 w-3" />
                          {getTimeAgo(group.lastActive)}
                        </div>
                        {group.isMember ? (
                          <Badge className="mt-2 bg-indigo-500/20 text-indigo-400">
                            Joined
                          </Badge>
                        ) : (
                          <Button 
                            size="sm" 
                            className="mt-2"
                            onClick={(e) => {
                              e.stopPropagation();
                              joinGroup(group.id);
                            }}
                          >
                            <Plus className="h-3 w-3 mr-1" />
                            Join
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </TabsContent>

            <TabsContent value="my-groups" className="space-y-3">
              {myGroups.length === 0 ? (
                <div className="text-center py-8 text-slate-400">
                  <Users className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>You haven't joined any groups yet</p>
                  <Button 
                    variant="outline" 
                    className="mt-3"
                    onClick={() => setActiveTab("discover")}
                  >
                    Discover Groups
                  </Button>
                </div>
              ) : (
                myGroups.map((group) => (
                  <div
                    key={group.id}
                    className="bg-slate-900/50 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-colors cursor-pointer"
                    onClick={() => setSelectedGroup(group)}
                  >
                    <div className="flex items-center justify-between">
                      <div>
                        <h4 className="font-medium">{group.name}</h4>
                        <div className="flex items-center gap-2 text-xs text-slate-400 mt-1">
                          <Users className="h-3 w-3" />
                          {group.memberCount.toLocaleString()} members
                          <span>•</span>
                          <Clock className="h-3 w-3" />
                          Active {getTimeAgo(group.lastActive)}
                        </div>
                      </div>
                      <Button variant="ghost" size="sm">
                        <MessageCircle className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                ))
              )}
            </TabsContent>

            <TabsContent value="resources" className="space-y-3">
              {resources.map((resource) => (
                <div
                  key={resource.id}
                  className="bg-slate-900/50 rounded-lg p-4 border border-slate-700 hover:border-slate-600 transition-colors cursor-pointer"
                >
                  <div className="flex items-start gap-3">
                    <div className="p-2 rounded-lg bg-indigo-500/20 text-indigo-400">
                      {getResourceIcon(resource.type)}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <h4 className="font-medium">{resource.title}</h4>
                        <Badge variant="outline" className="text-xs bg-slate-700/50">
                          {resource.type}
                        </Badge>
                      </div>
                      <p className="text-sm text-slate-400 mt-1">{resource.description}</p>
                      <Badge variant="outline" className="text-xs mt-2 bg-slate-700/50">
                        {resource.topic}
                      </Badge>
                    </div>
                  </div>
                </div>
              ))}
            </TabsContent>
          </Tabs>

          {/* Safety Notice */}
          <div className="mt-4 bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 text-xs">
            <div className="flex items-start gap-2">
              <AlertTriangle className="h-4 w-4 text-yellow-400 mt-0.5" />
              <div>
                <strong className="text-yellow-400">Safety First:</strong>
                <span className="text-slate-400"> Community support is not a substitute for professional help. If you're in crisis, please contact 988 or your local emergency services.</span>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
