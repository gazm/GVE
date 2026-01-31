"""
Library API endpoints for forge-ui component browser.

Maps existing assets collection to library categories for the frontend.
"""
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse
from typing import List, Optional
from src.librarian import list_assets, search_assets
from .templates import templates

router = APIRouter()

# Library type icons for rendering
LIBRARY_ICONS = {
    "geometry": "◇",
    "materials": "◈", 
    "textures": "▦",
    "audio": "♫",
    "recipes": "▣",
}

def render_library_card(item: dict, library_type: str) -> str:
    """Render a single library card as HTML."""
    icon = LIBRARY_ICONS.get(library_type, "◇")
    
    tags = item.get("tags", [])
    tags_html = "".join(f'<span class="tag-mini">{tag}</span>' for tag in tags[:3])
    if len(tags) > 3:
        tags_html += f'<span class="tag-more">+{len(tags) - 3}</span>'
    
    rating = int(item.get("rating", 4))
    stars_html = "".join(
        f'<span class="star {"filled" if i < rating else ""}">★</span>' 
        for i in range(5)
    )
    
    usage_count = item.get("usage_count", item.get("version", 1))
    
    cost = item.get("cost", 0)
    cost_class = "free" if cost == 0 else ""
    cost_text = "Free" if cost == 0 else f"${cost:.2f}"
    cost_html = f'<span class="stat cost {cost_class}">{cost_text}</span>'
    
    # Action button based on library type
    if library_type == "recipes":
        action_btn = f'''<button class="btn-use" 
            hx-post="/api/assets/from-recipe/{item["id"]}" 
            hx-target="#card-chain">Use</button>'''
    else:
        aid = item["id"].replace('"', '\\"')
        action_btn = f'''<button class="btn-add" 
            hx-post="/api/assets/chain/fill" 
            hx-vals='{{"asset_id": "{aid}"}}' 
            hx-target="#card-chain" 
            hx-swap="innerHTML"
            title="Add to card chain">+</button>'''
    
    return f'''
    <div class="library-card" data-id="{item["id"]}" data-type="{library_type}">
        <div class="card-preview">
            <div class="preview-{library_type}">{icon}</div>
        </div>
        <div class="card-body">
            <h3 class="card-name">{item["name"]}</h3>
            <div class="card-tags">{tags_html}</div>
        </div>
        <div class="card-meta">
            <div class="card-rating">{stars_html}</div>
            <div class="card-stats">
                <span class="stat" title="Times used">{usage_count}×</span>
                {cost_html}
            </div>
        </div>
        <div class="card-actions">
            {action_btn}
            <button class="btn-preview" 
                hx-get="/api/assets/{item["id"]}/binary" 
                hx-swap="none"
                onclick="window.load_asset && window.load_asset('/api/assets/{item["id"]}/binary', '{item["id"]}')"
                title="Preview in viewport">👁</button>
        </div>
    </div>
    '''


def render_library_grid(items: list, library_type: str) -> str:
    """Render the full library grid HTML."""
    if not items:
        return '''<div class="empty-state">
            <p>No items found. Try adjusting your search or filters.</p>
        </div>'''
    return "".join(render_library_card(item, library_type) for item in items)


async def get_library_items(library_type: str, limit: int = 50) -> list:
    """
    Fetch library items from database.
    
    Currently maps all asset categories to library types.
    Future: separate collections per library type.
    """
    assets = await list_assets(limit=limit)
    
    items = []
    for asset in assets:
        # Map asset to library item format
        category = asset.category.value.lower() if hasattr(asset.category, 'value') else str(asset.category).lower()
        
        # Filter by library type
        if library_type == "geometry":
            # All assets have geometry (SDF trees)
            pass
        elif library_type == "materials":
            # Filter to assets with material tags
            if not any(t in asset.tags for t in ["metal", "wood", "stone", "plastic", "fabric"]):
                continue
        elif library_type == "textures":
            # Filter to assets with texture/pbr tags  
            if not any(t in asset.tags for t in ["texture", "pbr", "albedo", "normal"]):
                continue
        elif library_type == "audio":
            # Filter to assets with audio tags
            if not any(t in asset.tags for t in ["audio", "sound", "impact", "ambient"]):
                continue
        elif library_type == "recipes":
            # Recipes are complete asset templates
            pass
        
        items.append({
            "id": asset.id,
            "name": asset.name,
            "tags": asset.tags,
            "category": category,
            "rating": 4,  # TODO: Add rating field to AssetMetadata
            "usage_count": asset.version,
            "cost": 0,  # Library items are free
        })
    
    return items


@router.get("/geometry", response_class=HTMLResponse)
async def get_geometry_library(limit: int = 50):
    """Get geometry library items (SDF components)."""
    items = await get_library_items("geometry", limit)
    return HTMLResponse(content=render_library_grid(items, "geometry"))


@router.get("/materials", response_class=HTMLResponse)
async def get_materials_library(limit: int = 50):
    """Get materials library items (physics specs)."""
    items = await get_library_items("materials", limit)
    return HTMLResponse(content=render_library_grid(items, "materials"))


@router.get("/textures", response_class=HTMLResponse)
async def get_textures_library(limit: int = 50):
    """Get textures library items (PBR maps)."""
    items = await get_library_items("textures", limit)
    return HTMLResponse(content=render_library_grid(items, "textures"))


@router.get("/audio", response_class=HTMLResponse)
async def get_audio_library(limit: int = 50):
    """Get audio library items (DSP patches)."""
    items = await get_library_items("audio", limit)
    return HTMLResponse(content=render_library_grid(items, "audio"))


@router.get("/recipes", response_class=HTMLResponse)
async def get_recipes_library(limit: int = 50):
    """Get recipe library items (complete asset templates)."""
    items = await get_library_items("recipes", limit)
    return HTMLResponse(content=render_library_grid(items, "recipes"))


@router.get("/search", response_class=HTMLResponse)
async def search_library(
    q: str = Query(default=""),
    type: str = Query(default="geometry"),
    tags: str = Query(default=""),
):
    """
    Search library items by name and tags.
    
    Args:
        q: Search query string
        type: Library type (geometry, materials, textures, audio, recipes)
        tags: Comma-separated tag filters
    """
    # Parse tags
    active_tags = [t.strip() for t in tags.split(",") if t.strip()]
    
    # Get base items
    if q:
        # Search by name
        assets = await search_assets(q)
        items = [{
            "id": a.id,
            "name": a.name,
            "tags": a.tags,
            "category": a.category.value.lower() if hasattr(a.category, 'value') else str(a.category).lower(),
            "rating": 4,
            "usage_count": a.version,
            "cost": 0,
        } for a in assets]
    else:
        items = await get_library_items(type)
    
    # Filter by tags if provided
    if active_tags:
        items = [i for i in items if any(t in i.get("tags", []) for t in active_tags)]
    
    return HTMLResponse(content=render_library_grid(items, type))
