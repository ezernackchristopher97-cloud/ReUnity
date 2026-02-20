// ReUnity Service Worker for Push Notifications
const CACHE_NAME = 'reunity-v1';
const OFFLINE_URLS = [
  '/',
  '/grounding',
  '/offline-crisis.html'
];

// Install event - cache offline resources
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(OFFLINE_URLS);
    })
  );
  self.skipWaiting();
});

// Activate event - clean old caches
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name))
      );
    })
  );
  self.clients.claim();
});

// Push notification handler
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {};
  const options = {
    body: data.body || 'You have a new notification',
    icon: '/reop-logo.png',
    badge: '/reop-logo.png',
    vibrate: [200, 100, 200],
    tag: data.tag || 'reunity-notification',
    requireInteraction: data.urgent || false,
    actions: data.actions || [],
    data: {
      url: data.url || '/',
      type: data.type || 'general'
    }
  };

  // Crisis alerts get special treatment
  if (data.type === 'crisis') {
    options.requireInteraction = true;
    options.vibrate = [500, 200, 500, 200, 500];
    options.tag = 'crisis-alert';
  }

  // Check-in reminders
  if (data.type === 'checkin') {
    options.actions = [
      { action: 'okay', title: "I'm Okay" },
      { action: 'help', title: 'Need Help' }
    ];
  }

  event.waitUntil(
    self.registration.showNotification(data.title || 'ReUnity', options)
  );
});

// Notification click handler
self.addEventListener('notificationclick', (event) => {
  event.notification.close();

  const action = event.action;
  const notificationData = event.notification.data;

  if (action === 'okay') {
    // Quick check-in response
    event.waitUntil(
      fetch('/api/trpc/checkin.quickResponse', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'okay' })
      })
    );
  } else if (action === 'help') {
    // Open app to get help
    event.waitUntil(
      clients.openWindow('/chat?urgent=true')
    );
  } else {
    // Default: open the specified URL
    event.waitUntil(
      clients.openWindow(notificationData.url || '/')
    );
  }
});

// Background sync for offline check-ins
self.addEventListener('sync', (event) => {
  if (event.tag === 'sync-checkins') {
    event.waitUntil(syncCheckins());
  }
});

async function syncCheckins() {
  const db = await openDB();
  const pendingCheckins = await db.getAll('pending-checkins');
  
  for (const checkin of pendingCheckins) {
    try {
      await fetch('/api/trpc/checkin.submit', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(checkin)
      });
      await db.delete('pending-checkins', checkin.id);
    } catch (e) {
      console.error('Failed to sync checkin:', e);
    }
  }
}

function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('reunity-offline', 1);
    request.onerror = () => reject(request.error);
    request.onsuccess = () => resolve(request.result);
    request.onupgradeneeded = (event) => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('pending-checkins')) {
        db.createObjectStore('pending-checkins', { keyPath: 'id' });
      }
    };
  });
}
