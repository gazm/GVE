"""
Library API endpoints for forge-ui component browser.

Maps existing assets collection to library categories for the frontend.
Tag filtering pushed to MongoDB via librarian.list_assets_by_tags().
"""
from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from src.librarian import list_assets, list_assets_by_tags, search_assets
from .templates import templates

router = APIRouter()

# Tag sets that define each library category (used for DB-level filtering)
LIBRARY_TAG_FILTERS: dict[str, list[str]] = {
    "materials": ["metal", "wood", "stone", "plastic", "fabric"],
    "textures": ["texture", "pbr", "albedo", "normal"],
    "audio": ["audio", "sound", "impact", "ambient"],
}

# Library categories with descriptions
LIBRARY_CATEGORIES = {
    "geometry": {"icon": "◇", "description": "SDF geometry components"},
    "materials": {"icon": "◈", "description": "Physics material specs"},
    "textures": {"icon": "▦", "description": "PBR texture maps"},
    "audio": {"icon": "♫", "description": "DSP audio patches"},
    "recipes": {"icon": "▣", "description": "Complete asset templates"},
}


def _asset_to_item(asset) -> dict:
    """Convert an AssetMetadata to a library item dict."""
    return {
        "id": asset.id,
        "name": asset.name,
        "tags": asset.tags,
        "category": asset.category.value.lower() if hasattr(asset.category, "value") else str(asset.category).lower(),
        "rating": 4,  # TODO: Add rating field to AssetMetadata
        "usage_count": asset.version,
        "cost": 0,
        "thumbnail_url": getattr(asset, "thumbnail_url", None),
    }


@router.get("/", response_class=JSONResponse)
async def get_library_root():
    """Get library API summary with available categories."""
    return {
        "status": "ok",
        "categories": LIBRARY_CATEGORIES,
        "endpoints": {
            "geometry": "/api/library/geometry",
            "materials": "/api/library/materials",
            "textures": "/api/library/textures",
            "audio": "/api/library/audio",
            "recipes": "/api/library/recipes",
            "search": "/api/library/search?q=<query>&type=<category>",
        }
    }


async def get_library_items(library_type: str, limit: int = 50) -> list[dict]:
    """
    Fetch library items from database.

    For categories with tag filters (materials, textures, audio) the filtering
    is done in MongoDB via ``list_assets_by_tags``.  Geometry and recipes
    return all non-draft assets.
    """
    tag_filter = LIBRARY_TAG_FILTERS.get(library_type)

    if tag_filter:
        assets = await list_assets_by_tags(tags=tag_filter, limit=limit)
    else:
        # geometry / recipes — all assets qualify
        assets = await list_assets(limit=limit)

    return [_asset_to_item(a) for a in assets]


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
    """Search library items by name and/or tags (filtering in MongoDB)."""
    active_tags = [t.strip() for t in tags.split(",") if t.strip()]

    if q and active_tags:
        # Name search + tag filter: search first, then filter client-side
        # (search_assets doesn't support combined tag queries yet)
        assets = await search_assets(q, limit=50)
        items = [_asset_to_item(a) for a in assets]
        items = [i for i in items if any(t in i.get("tags", []) for t in active_tags)]
    elif q:
        assets = await search_assets(q, limit=50)
        items = [_asset_to_item(a) for a in assets]
    elif active_tags:
        assets = await list_assets_by_tags(tags=active_tags, limit=50)
        items = [_asset_to_item(a) for a in assets]
    else:
        items = await get_library_items(type)

    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": type})
