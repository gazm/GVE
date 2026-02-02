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

# Helper functions removed in favor of Jinja2 templates (library_grid.html)


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
async def get_geometry_library(request: Request, limit: int = 50):
    """Get geometry library items (SDF components)."""
    items = await get_library_items("geometry", limit)
    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": "geometry"})


@router.get("/materials", response_class=HTMLResponse)
async def get_materials_library(request: Request, limit: int = 50):
    """Get materials library items (physics specs)."""
    items = await get_library_items("materials", limit)
    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": "materials"})


@router.get("/textures", response_class=HTMLResponse)
async def get_textures_library(request: Request, limit: int = 50):
    """Get textures library items (PBR maps)."""
    items = await get_library_items("textures", limit)
    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": "textures"})


@router.get("/audio", response_class=HTMLResponse)
async def get_audio_library(request: Request, limit: int = 50):
    """Get audio library items (DSP patches)."""
    items = await get_library_items("audio", limit)
    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": "audio"})


@router.get("/recipes", response_class=HTMLResponse)
async def get_recipes_library(request: Request, limit: int = 50):
    """Get recipe library items (complete asset templates)."""
    items = await get_library_items("recipes", limit)
    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": "recipes"})


@router.get("/search", response_class=HTMLResponse)
async def search_library(
    request: Request,
    q: str = Query(default=""),
    type: str = Query(default="geometry"),
    tags: str = Query(default=""),
):
    """
    Search library items by name and tags.
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
            "thumbnail_url": a.thumbnail_url # key fix: ensure thumbnail URL is passed
        } for a in assets]
    else:
        items = await get_library_items(type)
    
    # Filter by tags if provided
    if active_tags:
        items = [i for i in items if any(t in i.get("tags", []) for t in active_tags)]
    
    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": type})
