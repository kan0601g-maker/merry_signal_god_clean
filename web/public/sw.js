const VERSION = "v1";
const CACHE_NAME = `merry-signal-${VERSION}`;

const APP_SHELL = [
  "/",
  "/index.html",
  "/manifest.webmanifest",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(keys.map((k) => (k === CACHE_NAME ? null : caches.delete(k))));
      await self.clients.claim();
    })()
  );
});

function isJsonEndpoint(url) {
  return url.pathname.endsWith("/god_state.json") || url.pathname.endsWith("/history.json");
}

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // 同一オリジンだけ扱う
  if (url.origin !== self.location.origin) return;

  // JSONはネット優先
  if (isJsonEndpoint(url)) {
    event.respondWith(
      (async () => {
        try {
          const res = await fetch(event.request, { cache: "no-store" });
          const cache = await caches.open(CACHE_NAME);
          cache.put(event.request, res.clone());
          return res;
        } catch (e) {
          const cached = await caches.match(event.request);
          return cached || new Response("offline", { status: 503 });
        }
      })()
    );
    return;
  }

  // それ以外はキャッシュ優先
  event.respondWith(
    (async () => {
      const cached = await caches.match(event.request);
      if (cached) return cached;

      const res = await fetch(event.request);
      const cache = await caches.open(CACHE_NAME);
      cache.put(event.request, res.clone());
      return res;
    })()
  );
});