from datetime import datetime
from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional
from pathlib import Path

from src.librarian import (
    load_asset, save_asset, delete_asset, list_assets, search_assets, 
    update_asset_field, update_asset_rag, load_asset_doc,
    resolve_cache_path,
)
# index_asset imported lazily below to avoid loading torch/ROCm at startup
from generated.types import AssetMetadata
from .templates import templates
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi import Request, Body, BackgroundTasks, Form
from src.compiler import enqueue_compile, CompilePriority

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
async def find_assets(q: str = Query(..., min_length=1), limit: int = Query(50, ge=1, le=200)):
    return await search_assets(q, limit=limit)

# IMPORTANT: More specific routes must come BEFORE less specific ones
@router.get("/{asset_id}/binary")
async def get_asset_binary(asset_id: str):
    """Serve the compiled binary file for an asset.
    
    Cache-Control: no-store ensures browser always fetches fresh after recompile.
    """
    binary_path = resolve_cache_path(asset_id)

    if not binary_path.exists():
        raise HTTPException(status_code=404, detail="Binary file not found. Asset may not be compiled yet.")

    # Use Response wrapper to add cache-control headers
    from starlette.responses import Response
    import os
    
    with open(binary_path, "rb") as f:
        content = f.read()
    
    return Response(
        content=content,
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "Content-Disposition": f"attachment; filename={binary_path.name}",
        }
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
    
    # 2. Trigger RAG Indexing (Learning) - lazy import to avoid torch/ROCm at startup
    doc = await load_asset_doc(asset_id)
    if doc and "dna" in doc:
        from src.ai_pipeline import index_asset
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

    # If good rating, ensure it's learnt - lazy import to avoid torch/ROCm at startup
    if rating >= 4:
        doc = await load_asset_doc(asset_id)
        if doc and "dna" in doc:
            from src.ai_pipeline import index_asset
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
        items = await search_assets(q, limit=50)
    return templates.TemplateResponse("library_grid.html", {"request": request, "items": items, "library_type": "geometry"})

@router.get("/partials/browser", response_class=HTMLResponse)
async def get_browser_partial(request: Request):
    return templates.TemplateResponse("asset_browser.html", {"request": request})

def _get_texture_mode_from_doc(doc: dict | None) -> str:
    """Read texture_mode from asset doc settings; default 'procedural_triplanar'. Backward compat: splat_mode."""
    if not doc:
        return "procedural_triplanar"
    s = doc.get("settings", {})
    return s.get("texture_mode") or s.get("splat_mode", "procedural_triplanar")


@router.get("/partials/editor/{card_id}", response_class=HTMLResponse)
async def get_editor_partial(card_id: str, request: Request):
    asset = await load_asset(card_id)
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")
    doc = await load_asset_doc(card_id)
    texture_mode = _get_texture_mode_from_doc(doc)
    card_name = asset.name
    # Use shared logic to avoid duplication
    property_groups = _property_groups_from_doc(card_id, doc)
    
    # Pass JSON-serialized DNA for safely embedding in JS
    import json
    dna_json = json.dumps(doc.get("dna", {}))

    return templates.TemplateResponse("property_editor.html", {
        "request": request,
        "card_id": card_id,
        "card_name": card_name,
        "property_groups": property_groups,
        "has_dna": "dna" in (doc or {}),
        "dna_json": dna_json,
    })

from .websocket import broadcast_event

def _property_groups_from_doc(card_id: str, doc: dict) -> list:
    """Build property_groups from raw doc so we never rely on save_asset (which would wipe DNA)."""
    settings = doc.get("settings", {})
    category = doc.get("category", "Prop")
    category_display = category.value if hasattr(category, "value") else str(category)
    return [
        {
            "name": "General Settings",
            "properties": [
                {"name": "name", "label": "Asset Name", "type": "text", "value": doc.get("name", card_id)},
                {"name": "category", "label": "Category", "type": "text", "value": category_display, "readonly": True},
            ]
        },
        {
            "name": "Optimization",
            "properties": [
                {"name": "lod_count", "label": "LOD Count", "type": "number", "min": 0, "max": 5, "value": settings.get("lod_count", 3)},
                {"name": "resolution", "label": "Resolution", "type": "number", "min": 16, "max": 256, "value": settings.get("resolution", 128)},
                {
                    "name": "texture_mode",
                    "label": "Texture mode",
                    "type": "select",
                    "value": _get_texture_mode_from_doc(doc),
                    "options": [
                        {"value": "dense", "label": "Dense"},
                        {"value": "swatch", "label": "Swatch"},
                        {"value": "procedural_triplanar", "label": "Procedural (triplanar)"},
                    ],
                },
                {
                    "name": "triplanar_bake_mode",
                    "label": "Bake Mode (Triplanar)",
                    "type": "select",
                    "value": settings.get("triplanar_bake_mode", "gaussian"),
                    "options": [
                        {"value": "gaussian", "label": "Gaussian (Smooth)"},
                        {"value": "point", "label": "Point (Crisp/Voronoi)"},
                    ],
                },
            ]
        }
    ]


@router.post("/partials/property/{card_id}", response_class=HTMLResponse)
async def update_property(card_id: str, request: Request):
    form_data = await request.form()
    print(f"🔧 Updating property for {card_id}: {form_data}")
    doc = await load_asset_doc(card_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Asset not found")
    # Use update_asset_field only — never save_asset here, or we overwrite the doc and wipe DNA
    updates = {}
    if "name" in form_data:
        updates["name"] = form_data["name"]
    if "lod_count" in form_data:
        updates["settings.lod_count"] = int(form_data["lod_count"])
    if "resolution" in form_data:
        updates["settings.resolution"] = int(form_data["resolution"])
    if "texture_mode" in form_data and form_data["texture_mode"] in ("dense", "swatch", "procedural_triplanar"):
        updates["settings.texture_mode"] = form_data["texture_mode"]
    if "triplanar_bake_mode" in form_data and form_data["triplanar_bake_mode"] in ("gaussian", "point"):
        updates["settings.triplanar_bake_mode"] = form_data["triplanar_bake_mode"]
    if updates:
        updates["updated_at"] = datetime.utcnow()
        updates["version"] = doc.get("version", 0) + 1
        await update_asset_field(card_id, updates)
        await enqueue_compile(card_id)
    await broadcast_event("compile:progress", {"asset_id": card_id, "progress": 15, "status": "Saving & Validating..."})
    doc = await load_asset_doc(card_id)
    import json
    dna_json = json.dumps(doc.get("dna", {}))
    
    editor_html = templates.get_template("property_editor.html").render({
        "request": request,
        "card_id": card_id,
        "card_name": doc.get("name", card_id),
        "property_groups": _property_groups_from_doc(card_id, doc),
        "has_dna": "dna" in doc,
        "dna_json": dna_json,
    })
    progress_html = templates.get_template("progress_bar.html").render({
        "asset_id": card_id,
        "progress": 15,
        "status": "Saving & Validating..."
    })
    
    return HTMLResponse(content=editor_html + progress_html)

@router.post("/partials/update_dna/{card_id}")
async def update_asset_dna(card_id: str, dna: dict = Body(...)):
    """
    Update the DNA of an asset directly from the Tree Editor.
    """
    doc = await load_asset_doc(card_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Asset not found")

    print(f"🧬 Updating DNA for {card_id}, keys: {list(dna.keys())}")
    
    # Update DNA field
    # TODO: Validate DNA against schema here if possible
    await update_asset_field(card_id, {"dna": dna})
    
    # Trigger Compile
    # Future work: Return 'hotload' patch if supported
    job_id = await enqueue_compile(card_id, priority=CompilePriority.HIGH, force_recompile=True)
    
    return {"status": "ok", "job_id": job_id}

@router.post("/partials/recompile/{asset_id}", response_class=HTMLResponse)
async def recompile_asset(asset_id: str, request: Request):
    """
    Trigger a forced recompilation of the asset.
    Returns HTML snippet for the status indicator. If asset has no DNA, returns
    an error message in the same element instead of enqueueing.
    """
    doc = await load_asset_doc(asset_id)
    if not doc or "dna" not in doc:
        return HTMLResponse(
            content=f'<div id="status-{asset_id}" class="save-status error">No DNA — generate or import first</div>'
        )
    # Trigger compile with High priority and force_recompile=True
    job_id = await enqueue_compile(
        asset_id,
        priority=CompilePriority.HIGH,
        force_recompile=True
    )
    await broadcast_event("compile:started", {"asset_id": asset_id, "job_id": job_id})
    return HTMLResponse(
        content=f'<div id="status-{asset_id}" class="save-status">Recompiling... (Job: {job_id[:8]})</div>'
    )
