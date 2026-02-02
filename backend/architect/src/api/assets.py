from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from bson import ObjectId
from pathlib import Path

from src.librarian import (
    load_asset, save_asset, delete_asset, list_assets, search_assets, 
    update_asset_field, update_asset_rag, load_asset_doc
)
from src.librarian.cache import resolve_cache_path
from src.ai_pipeline import index_asset
from generated.types import AssetMetadata
from .templates import templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi import Request, Body, BackgroundTasks, Form

router = APIRouter()

# In-memory chain state: 3 bakeable slots (geometry, splat, audio). Key = slot_type, value = asset_id or None.
_chain_slots: dict[str, Optional[str]] = {"geometry": None, "splat": None, "audio": None}

def _get_slot_display(slot_type: str) -> tuple[str, str]:
    """Return (label, description) for an empty slot."""
    labels = {"geometry": ("Geometry", "SDF shape — volume & shell derived"), "splat": ("Splat", "Gaussian splat data"), "audio": ("Audio", "Audio patch")}
    return labels.get(slot_type, (slot_type.title(), ""))

@router.get("/", response_model=List[AssetMetadata])
async def get_assets(limit: int = 50, skip: int = 0):
    return await list_assets(limit=limit, skip=skip)

@router.get("/search", response_model=List[AssetMetadata])
async def find_assets(q: str = Query(..., min_length=1)):
    return await search_assets(q)

# IMPORTANT: More specific routes must come BEFORE less specific ones
@router.get("/{asset_id}/binary")
async def get_asset_binary(asset_id: str):
    """Serve the compiled binary file for an asset."""
    # First try the compiled directory (where compiler outputs)
    binary_path = resolve_cache_path(asset_id)
    
    if not binary_path.exists():
        # Fallback: try loading asset and resolving by metadata
        asset = await load_asset(asset_id)
        if asset:
            binary_path = resolve_cache_path(asset)
    
    if not binary_path.exists():
        raise HTTPException(status_code=404, detail="Binary file not found. Asset may not be compiled yet.")
    
    return FileResponse(
        path=str(binary_path),
        media_type="application/octet-stream",
        filename=binary_path.name
    )

@router.get("/{asset_id}", response_model=AssetMetadata)
async def get_asset(asset_id: str):
    asset = await load_asset(asset_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    return asset

@router.post("/", response_model=str)
async def create_asset(asset: AssetMetadata):
    # Librarian handles versioning and background compile trigger
    return await save_asset(asset)

@router.delete("/{asset_id}")
async def remove_asset(asset_id: str):
    await delete_asset(asset_id)
    return {"status": "deleted"}

@router.post("/{asset_id}/save")
async def save_draft(asset_id: str, background_tasks: BackgroundTasks):
    """
    Finalize a draft asset.
    1. Removes 'is_draft' flag.
    2. Indexes asset for RAG (learning loop).
    """
    # 1. Promote from draft
    success = await update_asset_field(asset_id, {"is_draft": False})
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")
    
    # 2. Trigger RAG Indexing (Learning)
    # Load raw doc to get DNA for indexing
    doc = await load_asset_doc(asset_id)
    if doc and "dna" in doc:
        background_tasks.add_task(index_asset, asset_id, doc["dna"])
        
    return {"status": "saved", "message": "Asset saved to library and indexed for learning"}

@router.post("/{asset_id}/feedback")
async def rate_asset(asset_id: str, background_tasks: BackgroundTasks, rating: int = Body(..., embed=True)):
    """
    Rate an asset (1-5).
    High ratings (>=4) trigger RAG indexing if not already indexed.
    """
    if not (1 <= rating <= 5):
        raise HTTPException(status_code=400, detail="Rating must be 1-5")
        
    # Update rating
    success = await update_asset_field(asset_id, {"rating": rating})
    if not success:
        raise HTTPException(status_code=404, detail="Asset not found")

    # If good rating, ensure it's learnt
    if rating >= 4:
        doc = await load_asset_doc(asset_id)
        if doc and "dna" in doc:
            # We re-index to boost its reinforcement
            background_tasks.add_task(index_asset, asset_id, doc["dna"])
            
    return {"status": "rated", "rating": rating}

# HTMX Partial Endpoints
@router.get("/partials/chain", response_class=HTMLResponse)
async def get_chain_partial(request: Request):
    """Return chain as 3 bakeable slots: geometry, splat, audio. Each slot is either empty or filled with one asset."""
    slots: list[dict] = []
    for slot_type in ("geometry", "splat", "audio"):
        asset_id = _chain_slots.get(slot_type)
        label, desc = _get_slot_display(slot_type)
        if asset_id:
            asset = await load_asset(asset_id)
            if asset:
                slots.append({
                    "slot_type": slot_type,
                    "asset_id": asset_id,
                    "name": asset.name,
                    "description": ", ".join(asset.tags) if asset.tags else desc,
                    "status": "cached",
                    "empty": False,
                })
            else:
                slots.append({"slot_type": slot_type, "asset_id": None, "name": label, "description": desc, "status": "cached", "empty": True})
        else:
            slots.append({"slot_type": slot_type, "asset_id": None, "name": label, "description": desc, "status": "cached", "empty": True})
    return templates.TemplateResponse("card_chain.html", {"request": request, "slots": slots})


@router.post("/chain/slot/{slot_type}", response_class=HTMLResponse)
async def fill_chain_slot(slot_type: str, request: Request, asset_id: Optional[str] = Form(None)):
    """Fill a bakeable slot (geometry, splat, audio) with an asset. Omit asset_id to clear. HTMX can swap the chain partial after."""
    if slot_type not in _chain_slots:
        raise HTTPException(status_code=400, detail="Invalid slot; use geometry, splat, or audio")
    _chain_slots[slot_type] = asset_id if asset_id else None
    return await get_chain_partial(request)


@router.post("/chain/smart-add", response_class=HTMLResponse)
async def smart_add_asset(request: Request, asset_id: str = Form(...), asset_type: Optional[str] = Form(None)):
    """
    Intelligently add an asset to the appropriate slot based on its type.
    - Geometry -> 'geometry' slot
    - Audio -> 'audio' slot
    - Splat -> 'splat' slot
    Returns updated chain partial.
    """
    # If type hint provided from frontend (e.g. from library tab), use it.
    # Otherwise we could load the asset and check its metadata.
    target_slot = None
    
    # Simple mapping from library types to chain slots
    if asset_type:
        type_lower = asset_type.lower()
        if type_lower in ("geometry", "prop", "weapon", "vehicle", "character"):
            target_slot = "geometry"
        elif type_lower == "audio":
            target_slot = "audio"
        elif type_lower == "splat":
            target_slot = "splat"
    
    # Fallback: Load asset to check internal type if needed (omitted for speed if hint is good)
    if not target_slot:
        # Default to geometry if unsure, or specific logic
        target_slot = "geometry"
        
    if target_slot in _chain_slots:
        _chain_slots[target_slot] = asset_id
    
    return await get_chain_partial(request)


@router.post("/chain/fill", response_class=HTMLResponse)
async def fill_first_empty_slot(request: Request, asset_id: str = Form(...)):
    """Fill the first empty slot (geometry, then splat, then audio) with the given asset_id. Returns updated chain partial."""
    for slot_type in ("geometry", "splat", "audio"):
        if _chain_slots[slot_type] is None:
            _chain_slots[slot_type] = asset_id
            break
    return await get_chain_partial(request)


@router.get("/partials/tree", response_class=HTMLResponse)
async def get_tree_partial(request: Request):
    """Hierarchy panel: root Scene + children from chain state (_chain_slots). App is source of truth."""
    
    # 1. Gather slots
    slots: list[dict] = []
    main_asset_name = "New Asset"
    
    for slot_type in ("geometry", "splat", "audio"):
        asset_id = _chain_slots.get(slot_type)
        label, desc = _get_slot_display(slot_type)
        
        if asset_id:
            asset = await load_asset(asset_id)
            if asset:
                # Use geometry name as the main asset name if available
                if slot_type == "geometry":
                    main_asset_name = asset.name
                
                slots.append({
                    "slot_type": slot_type,
                    "asset_id": asset_id,
                    "name": "Geometry" if slot_type == "geometry" else asset.name,
                    "empty": False,
                })
            else:
                slots.append({"slot_type": slot_type, "asset_id": None, "name": label, "empty": True})
        else:
            slots.append({"slot_type": slot_type, "asset_id": None, "name": label, "empty": True})

    # 2. Structure as Scene -> Asset -> Slots
    # We only have one "Asset" in the chain context right now.
    entities = [
        {
            "name": main_asset_name,
            "type": "asset",
            "children": slots
        }
    ]

    return templates.TemplateResponse("tree_viewer.html", {"request": request, "root_name": "Scene", "entities": entities})


@router.get("/partials/search", response_class=HTMLResponse)
async def search_assets_partial(request: Request, q: str = Query("", min_length=0)):
    """Return asset browser grid partial for search query."""
    if not q:
        items = [] # Or return popular/recent
    else:
        items = await search_assets(q)
    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": "geometry"}) # Default type

@router.get("/partials/browser", response_class=HTMLResponse)
async def get_browser_partial(request: Request):
    return templates.TemplateResponse("asset_browser.html", {"request": request})

@router.get("/partials/editor/{card_id}", response_class=HTMLResponse)
async def get_editor_partial(card_id: str, request: Request):
    asset = await load_asset(card_id)
    
    # Map AssetSettings to property groups or return mock for testing
    if card_id == "1":
        card_name = "AK Receiver"
        property_groups = [
            {
                "name": "Transform",
                "properties": [
                    {"name": "pos_x", "label": "Position X", "type": "number", "min": -100, "max": 100, "value": 0},
                    {"name": "pos_y", "label": "Position Y", "type": "number", "min": -100, "max": 100, "value": 0},
                ]
            },
            {
                "name": "Material",
                "properties": [
                    {"name": "roughness", "label": "Roughness", "type": "range", "min": 0, "max": 1, "value": 0.5},
                    {"name": "metalness", "label": "Metalness", "type": "range", "min": 0, "max": 1, "value": 0.8},
                ]
            }
        ]
    else:
        if not asset:
            raise HTTPException(status_code=404, detail="Asset not found")
        card_name = asset.name
        property_groups = [
            {
                "name": "General Settings",
                "properties": [
                    {"name": "name", "label": "Asset Name", "type": "text", "value": asset.name},
                    {"name": "category", "label": "Category", "type": "text", "value": asset.category.value, "readonly": True},
                ]
            },
            {
                "name": "Optimization",
                "properties": [
                    {"name": "lod_count", "label": "LOD Count", "type": "number", "min": 0, "max": 5, "value": asset.settings.lod_count},
                    {"name": "resolution", "label": "Resolution", "type": "number", "min": 16, "max": 256, "value": asset.settings.resolution},
                ]
            }
        ]
    
    return templates.TemplateResponse("property_editor.html", {
        "request": request,
        "card_id": card_id,
        "card_name": card_name,
        "property_groups": property_groups
    })

from .websocket import broadcast_event

@router.post("/partials/property/{card_id}", response_class=HTMLResponse)
async def update_property(card_id: str, request: Request):
    form_data = await request.form()
    print(f"Updating property for {card_id}: {form_data}")
    
    asset = await load_asset(card_id)
    if asset:
        # Update name if provided
        if "name" in form_data:
            asset.name = form_data["name"]
        
        # Update settings if provided
        if "lod_count" in form_data:
            asset.settings.lod_count = int(form_data["lod_count"])
        if "resolution" in form_data:
            asset.settings.resolution = int(form_data["resolution"])
        
        # Save back to Librarian (this triggers background compilation)
        await save_asset(asset)
    
    # Simulate a compilation progress event for UI feedback
    await broadcast_event("compile:progress", {
        "asset_id": card_id,
        "progress": 15,
        "status": "Saving & Validating..."
    })
    
    # Re-render the property editor and progress bar
    editor_html = (await get_editor_partial(card_id, request)).body.decode()
    progress_html = templates.get_template("progress_bar.html").render({
        "asset_id": card_id,
        "progress": 15,
        "status": "Saving & Validating..."
    })
    
    return HTMLResponse(content=editor_html + progress_html)
