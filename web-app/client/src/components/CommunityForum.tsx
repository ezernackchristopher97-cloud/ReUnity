import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  MessageSquare, 
  Heart, 
  Flag, 
  Send, 
  User, 
  Clock, 
  Shield,
  ChevronDown,
  ChevronUp,
  AlertTriangle
} from 'lucide-react';
import { toast } from 'sonner';

interface ForumPost {
  id: string;
  content: string;
  category: string;
  timestamp: number;
  likes: number;
  replies: ForumReply[];
  isAnonymous: boolean;
  authorId: string;
}

interface ForumReply {
  id: string;
  content: string;
  timestamp: number;
  likes: number;
  isAnonymous: boolean;
  authorId: string;
}

const CATEGORIES = [
  { id: 'support', label: 'General Support', color: 'emerald' },
  { id: 'anxiety', label: 'Anxiety', color: 'blue' },
  { id: 'depression', label: 'Depression', color: 'purple' },
  { id: 'trauma', label: 'Trauma Recovery', color: 'amber' },
  { id: 'relationships', label: 'Relationships', color: 'pink' },
  { id: 'wins', label: 'Wins & Progress', color: 'green' },
];

const CRISIS_KEYWORDS = [
  'suicide', 'kill myself', 'end it', 'don\'t want to live', 'hurt myself',
  'self harm', 'cutting', 'overdose', 'no reason to live'
];

export function CommunityForum() {
  const [posts, setPosts] = useState<ForumPost[]>([]);
  const [newPost, setNewPost] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('support');
  const [filterCategory, setFilterCategory] = useState<string | null>(null);
  const [expandedPost, setExpandedPost] = useState<string | null>(null);
  const [replyText, setReplyText] = useState('');
  const [showCrisisWarning, setShowCrisisWarning] = useState(false);

  useEffect(() => {
    const saved = localStorage.getItem('reunity_forum_posts');
    if (saved) {
      setPosts(JSON.parse(saved));
    } else {
      // Sample posts for new users
      setPosts([
        {
          id: '1',
          content: 'Just completed my first week of using the grounding techniques. The box breathing really helps when I feel overwhelmed. Wanted to share in case it helps someone else.',
          category: 'wins',
          timestamp: Date.now() - 86400000,
          likes: 12,
          replies: [
            {
              id: 'r1',
              content: 'That\'s amazing! Keep going, you\'ve got this 💪',
              timestamp: Date.now() - 43200000,
              likes: 3,
              isAnonymous: true,
              authorId: 'anon2'
            }
          ],
          isAnonymous: true,
          authorId: 'anon1'
        },
        {
          id: '2',
          content: 'Does anyone else struggle with feeling like a burden? I know logically I\'m not, but the feeling is so strong sometimes.',
          category: 'support',
          timestamp: Date.now() - 172800000,
          likes: 24,
          replies: [
            {
              id: 'r2',
              content: 'You\'re definitely not alone in this. That feeling is so common but it doesn\'t make it true. The people who care about you want to be there for you.',
              timestamp: Date.now() - 86400000,
              likes: 8,
              isAnonymous: true,
              authorId: 'anon3'
            },
            {
              id: 'r3',
              content: 'I feel this way too sometimes. What helps me is writing down times when people reached out to me - it reminds me they choose to be in my life.',
              timestamp: Date.now() - 43200000,
              likes: 5,
              isAnonymous: true,
              authorId: 'anon4'
            }
          ],
          isAnonymous: true,
          authorId: 'anon5'
        }
      ]);
    }
  }, []);

  const savePosts = (newPosts: ForumPost[]) => {
    setPosts(newPosts);
    localStorage.setItem('reunity_forum_posts', JSON.stringify(newPosts));
  };

  const checkForCrisis = (text: string): boolean => {
    const lowerText = text.toLowerCase();
    return CRISIS_KEYWORDS.some(keyword => lowerText.includes(keyword));
  };

  const submitPost = () => {
    if (!newPost.trim()) return;

    if (checkForCrisis(newPost)) {
      setShowCrisisWarning(true);
      return;
    }

    const post: ForumPost = {
      id: Date.now().toString(),
      content: newPost,
      category: selectedCategory,
      timestamp: Date.now(),
      likes: 0,
      replies: [],
      isAnonymous: true,
      authorId: 'user_' + Math.random().toString(36).substr(2, 9),
    };

    savePosts([post, ...posts]);
    setNewPost('');
    toast.success('Post shared with the community');
  };

  const submitReply = (postId: string) => {
    if (!replyText.trim()) return;

    if (checkForCrisis(replyText)) {
      setShowCrisisWarning(true);
      return;
    }

    const reply: ForumReply = {
      id: Date.now().toString(),
      content: replyText,
      timestamp: Date.now(),
      likes: 0,
      isAnonymous: true,
      authorId: 'user_' + Math.random().toString(36).substr(2, 9),
    };

    const updated = posts.map(p => 
      p.id === postId 
        ? { ...p, replies: [...p.replies, reply] }
        : p
    );
    savePosts(updated);
    setReplyText('');
    toast.success('Reply posted');
  };

  const likePost = (postId: string) => {
    const updated = posts.map(p =>
      p.id === postId ? { ...p, likes: p.likes + 1 } : p
    );
    savePosts(updated);
  };

  const likeReply = (postId: string, replyId: string) => {
    const updated = posts.map(p =>
      p.id === postId
        ? {
            ...p,
            replies: p.replies.map(r =>
              r.id === replyId ? { ...r, likes: r.likes + 1 } : r
            ),
          }
        : p
    );
    savePosts(updated);
  };

  const reportPost = (postId: string) => {
    toast.success('Post reported. Our moderators will review it.');
  };

  const formatTime = (timestamp: number): string => {
    const diff = Date.now() - timestamp;
    const hours = Math.floor(diff / 3600000);
    const days = Math.floor(diff / 86400000);
    
    if (days > 0) return `${days}d ago`;
    if (hours > 0) return `${hours}h ago`;
    return 'Just now';
  };

  const filteredPosts = filterCategory
    ? posts.filter(p => p.category === filterCategory)
    : posts;

  const getCategoryColor = (categoryId: string) => {
    const cat = CATEGORIES.find(c => c.id === categoryId);
    const colors: Record<string, string> = {
      emerald: 'bg-emerald-900/30 text-emerald-400 border-emerald-800/30',
      blue: 'bg-blue-900/30 text-blue-400 border-blue-800/30',
      purple: 'bg-purple-900/30 text-purple-400 border-purple-800/30',
      amber: 'bg-amber-900/30 text-amber-400 border-amber-800/30',
      pink: 'bg-pink-900/30 text-pink-400 border-pink-800/30',
      green: 'bg-green-900/30 text-green-400 border-green-800/30',
    };
    return colors[cat?.color || 'emerald'];
  };

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-purple-400" />
            Community Forum
          </CardTitle>
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-zinc-500" />
            <span className="text-xs text-zinc-500">Anonymous & Moderated</span>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        {/* Crisis Warning Modal */}
        {showCrisisWarning && (
          <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-4">
            <div className="bg-zinc-900 rounded-xl p-6 max-w-md border border-red-800">
              <div className="flex items-center gap-3 mb-4">
                <AlertTriangle className="w-8 h-8 text-red-400" />
                <h3 className="text-xl font-semibold text-white">Are You Okay?</h3>
              </div>
              <p className="text-zinc-300 mb-4">
                Your message suggests you might be going through a really difficult time. 
                We care about you and want to make sure you're safe.
              </p>
              <div className="bg-red-900/20 rounded-lg p-4 mb-4 border border-red-800/30">
                <p className="text-sm font-medium text-red-400 mb-2">If you're in crisis:</p>
                <p className="text-white font-bold">Call 988 (Suicide & Crisis Lifeline)</p>
                <p className="text-sm text-zinc-400 mt-1">Available 24/7, free and confidential</p>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  className="flex-1"
                  onClick={() => {
                    setShowCrisisWarning(false);
                    setNewPost('');
                    setReplyText('');
                  }}
                >
                  I'll Reach Out for Help
                </Button>
                <Button
                  className="flex-1 bg-red-600 hover:bg-red-700"
                  onClick={() => window.open('tel:988')}
                >
                  Call 988 Now
                </Button>
              </div>
            </div>
          </div>
        )}

        {/* New Post */}
        <div className="space-y-3 mb-6">
          <div className="flex flex-wrap gap-2">
            {CATEGORIES.map(cat => (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={`
                  px-3 py-1 rounded-full text-xs font-medium transition-all border
                  ${selectedCategory === cat.id
                    ? getCategoryColor(cat.id)
                    : 'bg-zinc-800 text-zinc-400 border-zinc-700 hover:bg-zinc-700'}
                `}
              >
                {cat.label}
              </button>
            ))}
          </div>
          <div className="flex gap-2">
            <Input
              value={newPost}
              onChange={e => setNewPost(e.target.value)}
              placeholder="Share your thoughts anonymously..."
              className="flex-1"
              onKeyDown={e => e.key === 'Enter' && submitPost()}
            />
            <Button onClick={submitPost} disabled={!newPost.trim()}>
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>

        {/* Filter */}
        <div className="flex gap-2 mb-4 overflow-x-auto pb-2">
          <button
            onClick={() => setFilterCategory(null)}
            className={`
              px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap transition-all
              ${!filterCategory
                ? 'bg-zinc-700 text-white'
                : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}
            `}
          >
            All
          </button>
          {CATEGORIES.map(cat => (
            <button
              key={cat.id}
              onClick={() => setFilterCategory(cat.id)}
              className={`
                px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap transition-all
                ${filterCategory === cat.id
                  ? 'bg-zinc-700 text-white'
                  : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}
              `}
            >
              {cat.label}
            </button>
          ))}
        </div>

        {/* Posts */}
        <div className="space-y-4 max-h-[500px] overflow-y-auto">
          {filteredPosts.map(post => (
            <div
              key={post.id}
              className="bg-zinc-800/50 rounded-xl p-4 border border-zinc-700/50"
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex items-center gap-2">
                  <div className="w-8 h-8 rounded-full bg-zinc-700 flex items-center justify-center">
                    <User className="w-4 h-4 text-zinc-400" />
                  </div>
                  <div>
                    <span className="text-sm text-zinc-400">Anonymous</span>
                    <span className={`ml-2 px-2 py-0.5 rounded-full text-xs border ${getCategoryColor(post.category)}`}>
                      {CATEGORIES.find(c => c.id === post.category)?.label}
                    </span>
                  </div>
                </div>
                <div className="flex items-center gap-2 text-xs text-zinc-500">
                  <Clock className="w-3 h-3" />
                  {formatTime(post.timestamp)}
                </div>
              </div>

              <p className="text-white mb-3">{post.content}</p>

              <div className="flex items-center gap-4">
                <button
                  onClick={() => likePost(post.id)}
                  className="flex items-center gap-1 text-sm text-zinc-400 hover:text-pink-400 transition-colors"
                >
                  <Heart className="w-4 h-4" />
                  {post.likes}
                </button>
                <button
                  onClick={() => setExpandedPost(expandedPost === post.id ? null : post.id)}
                  className="flex items-center gap-1 text-sm text-zinc-400 hover:text-blue-400 transition-colors"
                >
                  <MessageSquare className="w-4 h-4" />
                  {post.replies.length}
                  {expandedPost === post.id ? (
                    <ChevronUp className="w-3 h-3" />
                  ) : (
                    <ChevronDown className="w-3 h-3" />
                  )}
                </button>
                <button
                  onClick={() => reportPost(post.id)}
                  className="flex items-center gap-1 text-sm text-zinc-400 hover:text-red-400 transition-colors ml-auto"
                >
                  <Flag className="w-4 h-4" />
                </button>
              </div>

              {/* Replies */}
              {expandedPost === post.id && (
                <div className="mt-4 pt-4 border-t border-zinc-700/50 space-y-3">
                  {post.replies.map(reply => (
                    <div key={reply.id} className="pl-4 border-l-2 border-zinc-700">
                      <div className="flex items-center gap-2 mb-1">
                        <div className="w-6 h-6 rounded-full bg-zinc-700 flex items-center justify-center">
                          <User className="w-3 h-3 text-zinc-400" />
                        </div>
                        <span className="text-xs text-zinc-400">Anonymous</span>
                        <span className="text-xs text-zinc-500">{formatTime(reply.timestamp)}</span>
                      </div>
                      <p className="text-sm text-zinc-300 mb-2">{reply.content}</p>
                      <button
                        onClick={() => likeReply(post.id, reply.id)}
                        className="flex items-center gap-1 text-xs text-zinc-400 hover:text-pink-400 transition-colors"
                      >
                        <Heart className="w-3 h-3" />
                        {reply.likes}
                      </button>
                    </div>
                  ))}

                  {/* Reply input */}
                  <div className="flex gap-2 mt-3">
                    <Input
                      value={replyText}
                      onChange={e => setReplyText(e.target.value)}
                      placeholder="Write a supportive reply..."
                      className="flex-1 text-sm"
                      onKeyDown={e => e.key === 'Enter' && submitReply(post.id)}
                    />
                    <Button
                      size="sm"
                      onClick={() => submitReply(post.id)}
                      disabled={!replyText.trim()}
                    >
                      Reply
                    </Button>
                  </div>
                </div>
              )}
            </div>
          ))}

          {filteredPosts.length === 0 && (
            <div className="text-center py-8 text-zinc-500">
              <MessageSquare className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p>No posts in this category yet</p>
              <p className="text-sm">Be the first to share</p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export default CommunityForum;
