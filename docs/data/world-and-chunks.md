# World and Chunks Data Model

**Purpose:** Metadata for the World Editor: one current world and its 8m×8m map bricks (chunks). No SDF/voxel data in this model—only identity, layout, and pipeline state.

**Related Docs:**
- [Forge World Editor](../tools/forge-world-editor.md) - Terrain tools, chunk map, geo-importer
- [LiDAR Pipeline](../workflows/lidar-pipeline.md) - Map bricks and halo blending
- [Compiler Pipeline](../workflows/compiler-pipeline.md) - World baking (future)

---

## World

A **World** is a single level or map document. The Architect exposes one "current" world at a time.

| Field   | Type   | Description                    |
|---------|--------|--------------------------------|
| `id`    | string | Unique world ID (e.g. UUID)    |
| `name`  | string | Display name                  |
| `chunks`| array  | List of [Chunk](#chunk) items |

---

## Chunk

A **Chunk** is one 8m×8m×8m map brick. Each has an id (e.g. "A1", "B2"), grid position, and pipeline state.

| Field   | Type   | Description                                      |
|---------|--------|--------------------------------------------------|
| `id`    | string | Chunk id (e.g. "A1", "B2")                       |
| `x`     | int    | Grid X in 8m units (world-space origin)          |
| `z`     | int    | Grid Z in 8m units (world-space origin)         |
| `state` | string | One of: `unprocessed`, `analyzing`, `baking`, `finalized` |
| `job_id`| string | Optional; bake/analyze job id for polling       |

**State semantics:**
- **unprocessed** - Raw data, not compiled
- **analyzing** - AI processing terrain/props
- **baking** - Generating splats
- **finalized** - Ready for runtime

No SDF, voxel, or binary payload in this document—only metadata. Chunk binaries (e.g. `.gve_bin` per chunk) are produced by the compiler and stored in cache; the API returns chunk list and state only.

---

## REST Shape (Architect API)

### GET /api/world

Returns the current world or empty.

**Response (200):**
```json
{
  "world": {
    "id": "uuid",
    "name": "My Level",
    "chunks": [
      { "id": "A1", "x": 0, "z": 0, "state": "unprocessed" },
      { "id": "A2", "x": 1, "z": 0, "state": "finalized" }
    ]
  }
}
```

If no world exists: `{ "world": null }` or `{ "world": { "id": null, "name": "", "chunks": [] } }`.

### POST /api/world

Create or replace the current world.

**Request body:**
```json
{
  "name": "My Level",
  "grid_rows": 4,
  "grid_cols": 4
}
```

Backend creates chunk records (ids A1–A4, B1–B4, …; x/z from grid; state `unprocessed`).

**Response (200):** Same shape as GET /api/world.

### GET /api/world/partials/chunks

Returns HTML partial for the chunk grid (htmx swap target). Same data as `world.chunks` rendered as grid.

### PATCH /api/world/chunks/{chunk_id}

Update one chunk’s state (e.g. for "Bake" or "Analyze" placeholders). No real baking in Phase 2—only state flip.

**Request body:**
```json
{ "state": "baking" }
```

**Response (200):** Updated chunk or 404 if not found.

---

**Version:** 1.0  
**Last Updated:** 2026-01-31
