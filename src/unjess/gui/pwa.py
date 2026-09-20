"""PWA (Progressive Web App) integration module for Unjess.

Provides manifest.json, service worker registration, and mobile head meta tags
so users can install Unjess as a native-feeling mobile app on iOS & Android.
"""

import json
from nicegui import app, ui


_MANIFEST_CONTENT = {
    "name": "Unjess AI",
    "short_name": "Unjess",
    "description": "Full-featured AI Coding Agent — Mobile Companion",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#121212",
    "theme_color": "#121212",
    "orientation": "any",
    "icons": [
        {
            "src": "/pwa-icon.svg",
            "sizes": "192x192 512x512",
            "type": "image/svg+xml",
            "purpose": "any maskable",
        }
    ],
}

_SERVICE_WORKER_JS = """// Unjess PWA Service Worker
self.addEventListener('install', (event) => {
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});
"""

_PWA_HEAD_HTML = """
<link rel="manifest" href="/manifest.json">
<meta name="theme-color" content="#121212">
<meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Unjess">
<meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
<link rel="apple-touch-icon" href="/pwa-icon.svg">
<script>
  if ('serviceWorker' in navigator) {
    window.addEventListener('load', function() {
      navigator.serviceWorker.register('/sw.js').catch(function(err) {
        console.debug('SW registration skipped:', err);
      });
    });
  }
</script>
"""

_SVG_ICON = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
  <rect width="512" height="512" rx="100" fill="#121212"/>
  <circle cx="256" cy="256" r="160" fill="none" stroke="#ffffff" stroke-width="32"/>
  <path d="M170 256 L230 316 L342 196" fill="none" stroke="#e0e0e0" stroke-width="36" stroke-linecap="round" stroke-linejoin="round"/>
</svg>"""


def register_pwa_routes() -> None:
    """Register FastAPI routes for manifest.json, sw.js, and pwa-icon.svg."""

    @app.get("/manifest.json")
    def manifest():
        return _MANIFEST_CONTENT

    @app.get("/sw.js")
    def service_worker():
        from fastapi.responses import Response
        return Response(content=_SERVICE_WORKER_JS, media_type="application/javascript")

    @app.get("/pwa-icon.svg")
    def pwa_icon():
        from fastapi.responses import Response
        return Response(content=_SVG_ICON, media_type="image/svg+xml")


def inject_pwa_meta() -> None:
    """Inject PWA meta tags and manifest links into NiceGUI HTML head."""
    ui.add_head_html(_PWA_HEAD_HTML)
