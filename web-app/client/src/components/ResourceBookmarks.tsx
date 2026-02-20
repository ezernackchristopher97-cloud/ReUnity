import { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { 
  Bookmark, 
  Plus, 
  ExternalLink, 
  Trash2, 
  Search,
  Folder,
  Star,
  Link as LinkIcon
} from 'lucide-react';
import { toast } from 'sonner';

interface BookmarkedResource {
  id: string;
  title: string;
  url: string;
  category: string;
  notes?: string;
  isFavorite: boolean;
  addedAt: number;
}

const CATEGORIES = [
  { id: 'crisis', label: 'Crisis Resources', icon: '🆘' },
  { id: 'therapy', label: 'Therapy & Counseling', icon: '💬' },
  { id: 'articles', label: 'Helpful Articles', icon: '📖' },
  { id: 'videos', label: 'Videos & Podcasts', icon: '🎬' },
  { id: 'apps', label: 'Apps & Tools', icon: '📱' },
  { id: 'support', label: 'Support Groups', icon: '🤝' },
  { id: 'other', label: 'Other', icon: '📌' },
];

// Pre-populated helpful resources
const DEFAULT_RESOURCES: BookmarkedResource[] = [
  {
    id: 'default1',
    title: '988 Suicide & Crisis Lifeline',
    url: 'https://988lifeline.org',
    category: 'crisis',
    notes: 'Call or text 988 - Available 24/7',
    isFavorite: true,
    addedAt: Date.now() - 86400000,
  },
  {
    id: 'default2',
    title: 'National Domestic Violence Hotline',
    url: 'https://www.thehotline.org',
    category: 'crisis',
    notes: '1-800-799-7233 - 24/7 support',
    isFavorite: true,
    addedAt: Date.now() - 86400000,
  },
  {
    id: 'default3',
    title: 'SAMHSA Treatment Locator',
    url: 'https://findtreatment.gov',
    category: 'therapy',
    notes: 'Find mental health treatment near you',
    isFavorite: false,
    addedAt: Date.now() - 86400000,
  },
  {
    id: 'default4',
    title: 'Grounding Techniques Guide',
    url: '/grounding',
    category: 'articles',
    notes: 'ReUnity\'s built-in grounding exercises',
    isFavorite: true,
    addedAt: Date.now() - 86400000,
  },
];

export function ResourceBookmarks() {
  const [resources, setResources] = useState<BookmarkedResource[]>([]);
  const [isAdding, setIsAdding] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [filterCategory, setFilterCategory] = useState<string | null>(null);
  const [newResource, setNewResource] = useState({
    title: '',
    url: '',
    category: 'other',
    notes: '',
  });

  useEffect(() => {
    const saved = localStorage.getItem('reunity_bookmarks');
    if (saved) {
      setResources(JSON.parse(saved));
    } else {
      setResources(DEFAULT_RESOURCES);
      localStorage.setItem('reunity_bookmarks', JSON.stringify(DEFAULT_RESOURCES));
    }
  }, []);

  const saveResources = (newResources: BookmarkedResource[]) => {
    setResources(newResources);
    localStorage.setItem('reunity_bookmarks', JSON.stringify(newResources));
  };

  const addResource = () => {
    if (!newResource.title.trim() || !newResource.url.trim()) {
      toast.error('Please enter a title and URL');
      return;
    }

    const resource: BookmarkedResource = {
      id: Date.now().toString(),
      title: newResource.title,
      url: newResource.url.startsWith('http') ? newResource.url : `https://${newResource.url}`,
      category: newResource.category,
      notes: newResource.notes,
      isFavorite: false,
      addedAt: Date.now(),
    };

    saveResources([resource, ...resources]);
    setNewResource({ title: '', url: '', category: 'other', notes: '' });
    setIsAdding(false);
    toast.success('Resource bookmarked');
  };

  const deleteResource = (id: string) => {
    saveResources(resources.filter(r => r.id !== id));
    toast.success('Bookmark removed');
  };

  const toggleFavorite = (id: string) => {
    saveResources(
      resources.map(r =>
        r.id === id ? { ...r, isFavorite: !r.isFavorite } : r
      )
    );
  };

  const filteredResources = resources.filter(r => {
    const matchesSearch = 
      r.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
      r.notes?.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = !filterCategory || r.category === filterCategory;
    return matchesSearch && matchesCategory;
  });

  const favorites = filteredResources.filter(r => r.isFavorite);
  const others = filteredResources.filter(r => !r.isFavorite);

  return (
    <Card className="bg-zinc-900/80 border-zinc-800">
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between">
          <CardTitle className="text-lg flex items-center gap-2">
            <Bookmark className="w-5 h-5 text-amber-400" />
            Resource Library
          </CardTitle>
          <Button
            variant="outline"
            size="sm"
            onClick={() => setIsAdding(true)}
            className="gap-1"
          >
            <Plus className="w-4 h-4" />
            Add
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        {/* Add Resource Form */}
        {isAdding && (
          <div className="mb-4 p-4 bg-zinc-800/50 rounded-xl border border-zinc-700 space-y-3">
            <Input
              value={newResource.title}
              onChange={e => setNewResource({ ...newResource, title: e.target.value })}
              placeholder="Resource title"
            />
            <Input
              value={newResource.url}
              onChange={e => setNewResource({ ...newResource, url: e.target.value })}
              placeholder="URL (e.g., example.com)"
            />
            <select
              value={newResource.category}
              onChange={e => setNewResource({ ...newResource, category: e.target.value })}
              className="w-full px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-white"
            >
              {CATEGORIES.map(cat => (
                <option key={cat.id} value={cat.id}>
                  {cat.icon} {cat.label}
                </option>
              ))}
            </select>
            <Input
              value={newResource.notes}
              onChange={e => setNewResource({ ...newResource, notes: e.target.value })}
              placeholder="Notes (optional)"
            />
            <div className="flex gap-2">
              <Button onClick={addResource} className="flex-1">
                Save Bookmark
              </Button>
              <Button variant="outline" onClick={() => setIsAdding(false)}>
                Cancel
              </Button>
            </div>
          </div>
        )}

        {/* Search & Filter */}
        <div className="space-y-3 mb-4">
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-zinc-500" />
            <Input
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              placeholder="Search bookmarks..."
              className="pl-10"
            />
          </div>
          <div className="flex gap-2 overflow-x-auto pb-2">
            <button
              onClick={() => setFilterCategory(null)}
              className={`
                px-3 py-1 rounded-full text-xs font-medium whitespace-nowrap transition-all
                ${!filterCategory
                  ? 'bg-amber-600 text-white'
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
                    ? 'bg-amber-600 text-white'
                    : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'}
                `}
              >
                {cat.icon} {cat.label}
              </button>
            ))}
          </div>
        </div>

        {/* Favorites */}
        {favorites.length > 0 && (
          <div className="mb-4">
            <h3 className="text-sm font-medium text-amber-400 flex items-center gap-2 mb-2">
              <Star className="w-4 h-4" />
              Favorites
            </h3>
            <div className="space-y-2">
              {favorites.map(resource => (
                <ResourceItem
                  key={resource.id}
                  resource={resource}
                  onDelete={deleteResource}
                  onToggleFavorite={toggleFavorite}
                />
              ))}
            </div>
          </div>
        )}

        {/* Other Resources */}
        {others.length > 0 && (
          <div>
            {favorites.length > 0 && (
              <h3 className="text-sm font-medium text-zinc-400 flex items-center gap-2 mb-2">
                <Folder className="w-4 h-4" />
                All Resources
              </h3>
            )}
            <div className="space-y-2 max-h-[300px] overflow-y-auto">
              {others.map(resource => (
                <ResourceItem
                  key={resource.id}
                  resource={resource}
                  onDelete={deleteResource}
                  onToggleFavorite={toggleFavorite}
                />
              ))}
            </div>
          </div>
        )}

        {filteredResources.length === 0 && (
          <div className="text-center py-8 text-zinc-500">
            <Bookmark className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p>No bookmarks found</p>
            <p className="text-sm">Add resources you find helpful</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

function ResourceItem({
  resource,
  onDelete,
  onToggleFavorite,
}: {
  resource: BookmarkedResource;
  onDelete: (id: string) => void;
  onToggleFavorite: (id: string) => void;
}) {
  const category = CATEGORIES.find(c => c.id === resource.category);
  const isExternal = resource.url.startsWith('http');

  return (
    <div className="flex items-start gap-3 p-3 bg-zinc-800/50 rounded-lg border border-zinc-700/50 hover:border-zinc-600/50 transition-all group">
      <div className="text-xl">{category?.icon || '📌'}</div>
      <div className="flex-1 min-w-0">
        <a
          href={resource.url}
          target={isExternal ? '_blank' : '_self'}
          rel={isExternal ? 'noopener noreferrer' : undefined}
          className="font-medium text-white hover:text-amber-400 transition-colors flex items-center gap-1"
        >
          {resource.title}
          {isExternal && <ExternalLink className="w-3 h-3" />}
        </a>
        {resource.notes && (
          <p className="text-sm text-zinc-400 truncate">{resource.notes}</p>
        )}
        <p className="text-xs text-zinc-500 flex items-center gap-1 mt-1">
          <LinkIcon className="w-3 h-3" />
          {resource.url.replace(/^https?:\/\//, '').split('/')[0]}
        </p>
      </div>
      <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <button
          onClick={() => onToggleFavorite(resource.id)}
          className={`p-1.5 rounded-lg transition-colors ${
            resource.isFavorite
              ? 'text-amber-400 bg-amber-900/30'
              : 'text-zinc-400 hover:text-amber-400 hover:bg-zinc-700'
          }`}
        >
          <Star className="w-4 h-4" fill={resource.isFavorite ? 'currentColor' : 'none'} />
        </button>
        <button
          onClick={() => onDelete(resource.id)}
          className="p-1.5 rounded-lg text-zinc-400 hover:text-red-400 hover:bg-zinc-700 transition-colors"
        >
          <Trash2 className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}

export default ResourceBookmarks;
