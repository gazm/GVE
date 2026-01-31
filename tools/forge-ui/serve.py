#!/usr/bin/env python3
"""
Standalone development server for Forge UI (stub data only).

RECOMMENDED: Use the backend API instead for real data:
    cd backend/architect
    uvicorn src.api:app --reload --port 8888

This server provides stub/mock data for UI development when
the backend is not running.

Usage: python serve.py [port]
Default port: 8080
"""

import http.server
import socketserver
import json
import os
from pathlib import Path
from urllib.parse import urlparse, parse_qs

PORT = 8080

# Stub library data for development
STUB_LIBRARY = {
    "geometry": [
        {"id": "geo_001", "name": "AK Receiver Pattern", "tags": ["weapon", "rifle", "receiver", "ak"], "rating": 4.5, "usage_count": 47},
        {"id": "geo_002", "name": "M4 Stock", "tags": ["weapon", "rifle", "stock", "military"], "rating": 5, "usage_count": 89},
        {"id": "geo_003", "name": "Pistol Grip", "tags": ["weapon", "grip", "tactical"], "rating": 3, "usage_count": 15},
        {"id": "geo_004", "name": "Chair Frame (Modern)", "tags": ["furniture", "chair", "modern"], "rating": 4, "usage_count": 34},
        {"id": "geo_005", "name": "Table Leg (Industrial)", "tags": ["furniture", "table", "industrial", "metal"], "rating": 4, "usage_count": 28},
        {"id": "geo_006", "name": "Crate (Military)", "tags": ["prop", "crate", "military", "wood"], "rating": 4, "usage_count": 56},
        {"id": "geo_007", "name": "Barrel (Rusty)", "tags": ["prop", "barrel", "metal", "rusted"], "rating": 3.5, "usage_count": 41},
        {"id": "geo_008", "name": "Door Frame (Standard)", "tags": ["architecture", "door", "wood"], "rating": 4, "usage_count": 67},
    ],
    "materials": [
        {"id": "mat_001", "name": "Steel (Battle-Worn)", "tags": ["metal", "steel", "worn"], "rating": 4.5, "usage_count": 78},
        {"id": "mat_002", "name": "Oak Wood (Natural)", "tags": ["wood", "oak", "natural"], "rating": 4, "usage_count": 52},
        {"id": "mat_003", "name": "Rusted Iron", "tags": ["metal", "iron", "rusted", "damaged"], "rating": 4, "usage_count": 39},
        {"id": "mat_004", "name": "Clean Aluminum", "tags": ["metal", "aluminum", "pristine"], "rating": 4.5, "usage_count": 44},
        {"id": "mat_005", "name": "Weathered Concrete", "tags": ["stone", "concrete", "worn"], "rating": 3.5, "usage_count": 31},
        {"id": "mat_006", "name": "Black ABS Plastic", "tags": ["plastic", "abs", "pristine"], "rating": 4, "usage_count": 27},
    ],
    "textures": [
        {"id": "tex_001", "name": "Rusty Steel - Heavy Wear", "tags": ["metal", "rust", "worn"], "rating": 4, "usage_count": 47},
        {"id": "tex_002", "name": "Oak Wood Grain", "tags": ["wood", "oak", "natural"], "rating": 4.5, "usage_count": 63},
        {"id": "tex_003", "name": "Tactical Black Polymer", "tags": ["plastic", "tactical", "pristine"], "rating": 4, "usage_count": 35},
        {"id": "tex_004", "name": "Cracked Concrete", "tags": ["stone", "concrete", "damaged"], "rating": 3.5, "usage_count": 29},
        {"id": "tex_005", "name": "Worn Leather (Brown)", "tags": ["fabric", "leather", "worn"], "rating": 4, "usage_count": 41},
    ],
    "audio": [
        {"id": "aud_001", "name": "Steel Impact - Heavy", "tags": ["metal", "impact", "heavy"], "rating": 4.5, "usage_count": 89},
        {"id": "aud_002", "name": "Wood Creak", "tags": ["wood", "creak", "ambient"], "rating": 4, "usage_count": 34},
        {"id": "aud_003", "name": "Glass Shatter", "tags": ["glass", "impact", "destruction"], "rating": 4, "usage_count": 56},
        {"id": "aud_004", "name": "Metal Scrape", "tags": ["metal", "scrape", "friction"], "rating": 3.5, "usage_count": 28},
    ],
    "recipes": [
        {"id": "recipe_001", "name": "Standard AK-47 (Worn)", "tags": ["weapon", "rifle", "ak", "military", "worn"], "rating": 5, "usage_count": 47, "cost": 0, "cards": 4},
        {"id": "recipe_002", "name": "Sci-Fi Laser Rifle", "tags": ["weapon", "sci-fi", "energy"], "rating": 4, "usage_count": 12, "cost": 0.08, "cards": 6},
        {"id": "recipe_003", "name": "Office Chair (Modern)", "tags": ["furniture", "chair", "modern"], "rating": 4, "usage_count": 23, "cost": 0, "cards": 3},
        {"id": "recipe_004", "name": "Military Crate Set", "tags": ["prop", "military", "crate"], "rating": 4.5, "usage_count": 67, "cost": 0, "cards": 5},
    ],
}


def render_library_card(item: dict, library_type: str) -> str:
    """Render a single library card HTML."""
    icons = {"geometry": "◇", "materials": "◈", "textures": "▦", "audio": "♫", "recipes": "▣"}
    icon = icons.get(library_type, "◇")
    
    tags_html = "".join(f'<span class="tag-mini">{tag}</span>' for tag in item.get("tags", [])[:3])
    if len(item.get("tags", [])) > 3:
        tags_html += f'<span class="tag-more">+{len(item["tags"]) - 3}</span>'
    
    rating = int(item.get("rating", 0))
    stars_html = "".join(f'<span class="star {"filled" if i < rating else ""}">★</span>' for i in range(5))
    
    cost_html = ""
    if "cost" in item:
        cost_class = "free" if item["cost"] == 0 else ""
        cost_text = "Free" if item["cost"] == 0 else f"${item['cost']:.2f}"
        cost_html = f'<span class="stat cost {cost_class}">{cost_text}</span>'
    
    action_btn = '<button class="btn-use">Use</button>' if library_type == "recipes" else '<button class="btn-add">+</button>'
    
    return f'''
    <div class="library-card" data-id="{item['id']}" data-type="{library_type}">
        <div class="card-preview">
            <div class="preview-{library_type}">{icon}</div>
        </div>
        <div class="card-body">
            <h3 class="card-name">{item['name']}</h3>
            <div class="card-tags">{tags_html}</div>
        </div>
        <div class="card-meta">
            <div class="card-rating">{stars_html}</div>
            <div class="card-stats">
                <span class="stat" title="Times used">{item.get('usage_count', 0)}×</span>
                {cost_html}
            </div>
        </div>
        <div class="card-actions">
            {action_btn}
            <button class="btn-preview">👁</button>
        </div>
    </div>
    '''


def render_library_grid(items: list, library_type: str) -> str:
    """Render the library grid HTML."""
    if not items:
        return '<div class="empty-state"><p>No items found. Try adjusting your search or filters.</p></div>'
    return "".join(render_library_card(item, library_type) for item in items)


class ForgeHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        # Serve from static/ directory
        static_dir = Path(__file__).parent / "static"
        super().__init__(*args, directory=str(static_dir), **kwargs)
    
    def do_GET(self):
        # Route /static/* to root (since we're already in static/)
        if self.path.startswith('/static/'):
            self.path = self.path[7:]  # Remove '/static' prefix
        
        # Serve index.html at root
        if self.path == '/':
            self.path = '/index.html'
        
        # Stub API endpoints
        if self.path.startswith('/api/'):
            self.send_api_response()
            return
        
        return super().do_GET()
    
    def send_api_response(self):
        """Stub API responses for development."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        
        # Library endpoints
        if '/api/library/' in self.path:
            library_type = parsed.path.split('/api/library/')[-1].split('?')[0]
            
            # Search endpoint
            if library_type == 'search':
                query = params.get('q', [''])[0].lower()
                lib_type = params.get('type', ['geometry'])[0]
                tags_param = params.get('tags', [''])[0]
                active_tags = [t for t in tags_param.split(',') if t]
                
                items = STUB_LIBRARY.get(lib_type, [])
                
                # Filter by search query
                if query:
                    items = [i for i in items if query in i['name'].lower() or any(query in t for t in i.get('tags', []))]
                
                # Filter by tags
                if active_tags:
                    items = [i for i in items if any(t in i.get('tags', []) for t in active_tags)]
                
                html = render_library_grid(items, lib_type)
                self.wfile.write(html.encode())
                return
            
            # Direct library type endpoint
            if library_type in STUB_LIBRARY:
                html = render_library_grid(STUB_LIBRARY[library_type], library_type)
                self.wfile.write(html.encode())
                return
        
        # Card chain endpoint
        if 'chain' in self.path:
            html = '''
            <div class="asset-card" data-asset-id="1">
                <div class="card-preview">🧊</div>
                <div class="card-info">
                    <span class="card-name">Test Cube</span>
                    <span class="card-type">Primitive</span>
                </div>
            </div>
            '''
            self.wfile.write(html.encode())
            return
        
        self.wfile.write(b'<p>API stub</p>')
    
    def end_headers(self):
        # Add CORS headers for local development
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        super().end_headers()
    
    def guess_type(self, path):
        """Add WASM MIME type."""
        if path.endswith('.wasm'):
            return 'application/wasm'
        return super().guess_type(path)


if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else PORT
    
    with socketserver.TCPServer(("", port), ForgeHandler) as httpd:
        print(f"🔥 Forge UI server running at http://localhost:{port}")
        print(f"   Serving from: {Path(__file__).parent / 'static'}")
        print(f"   WASM at: /wasm/pkg/gve_wasm.js")
        print(f"\n   Library endpoints:")
        print(f"     GET /api/library/geometry")
        print(f"     GET /api/library/materials")
        print(f"     GET /api/library/textures")
        print(f"     GET /api/library/audio")
        print(f"     GET /api/library/recipes")
        print(f"     GET /api/library/search?q=...&type=...&tags=...")
        print(f"\n   Press Ctrl+C to stop\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Server stopped")
