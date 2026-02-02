"""
World Editor API: current world and 8m×8m chunks (metadata only).
In-memory storage for Phase 2; no SDF/voxel data.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .templates import templates

router = APIRouter()

CHUNK_STATES = ("unprocessed", "analyzing", "baking", "finalized")
CHUNK_STATE_LABELS = {
    "unprocessed": "Unprocessed",
    "analyzing": "Analyzing",
    "baking": "Baking",
    "finalized": "Finalized",
}

# In-memory current world. None = no world; else { "id", "name", "chunks": [ { "id", "x", "z", "state" }, ... ] }
_current_world: Optional[dict[str, Any]] = None


class WorldCreate(BaseModel):
    """Request body for POST /api/world."""

    name: str = Field(default="Untitled World", min_length=1, max_length=256)
    grid_rows: int = Field(default=4, ge=1, le=16)
    grid_cols: int = Field(default=4, ge=1, le=16)


class ChunkUpdate(BaseModel):
    """Request body for PATCH /api/world/chunks/{chunk_id}."""

    state: str = Field(..., pattern="^(unprocessed|analyzing|baking|finalized)$")


def _make_chunks(rows: int, cols: int) -> list[dict[str, Any]]:
    """Build chunk list with ids A1, A2, ..., B1, ... and x,z in 8m units."""
    row_letters = "ABCDEFGHIJKLMNOP"[:rows]
    chunks = []
    for r, letter in enumerate(row_letters):
        for c in range(cols):
            chunk_id = f"{letter}{c + 1}"
            chunks.append({"id": chunk_id, "x": c, "z": r, "state": "unprocessed"})
    return chunks


@router.get("", response_class=JSONResponse)
async def get_world() -> dict[str, Any]:
    """Return current world or null. GET /api/world."""
    if _current_world is None:
        return {"world": None}
    return {"world": _current_world}


@router.post("", response_class=JSONResponse)
async def create_or_replace_world(body: WorldCreate) -> dict[str, Any]:
    """Create or replace current world; initialize chunk grid. POST /api/world."""
    global _current_world
    world_id = str(uuid.uuid4())
    chunks = _make_chunks(body.grid_rows, body.grid_cols)
    _current_world = {
        "id": world_id,
        "name": body.name,
        "chunks": chunks,
    }
    return {"world": _current_world}


@router.get("/partials/chunks", response_class=HTMLResponse)
async def get_chunks_partial(request: Request) -> HTMLResponse:
    """Return HTML partial for chunk grid (htmx swap). GET /api/world/partials/chunks."""
    chunks = _current_world["chunks"] if _current_world else []
    
    context = {
        "request": request,
        "chunks": chunks,
        "chunk_state_labels": CHUNK_STATE_LABELS,
    }
    return templates.TemplateResponse("world_chunks_partial.html", context)


@router.patch("/chunks/{chunk_id}", response_class=JSONResponse)
async def update_chunk_state(chunk_id: str, body: ChunkUpdate) -> dict[str, Any]:
    """Update one chunk's state. No real baking in Phase 2. PATCH /api/world/chunks/{chunk_id}."""
    global _current_world
    if _current_world is None:
        raise HTTPException(status_code=404, detail="No world")
    for ch in _current_world["chunks"]:
        if ch["id"] == chunk_id:
            ch["state"] = body.state
            return ch
    raise HTTPException(status_code=404, detail=f"Chunk {chunk_id} not found")
