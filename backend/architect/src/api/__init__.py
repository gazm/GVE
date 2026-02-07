from fastapi import FastAPI, APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
from .assets import router as assets_router
from .compile import router as compile_router
from .websocket import router as websocket_router
from .generate import router as generate_router
from .library import router as library_router
from .world import router as world_router

from contextlib import asynccontextmanager
from src.librarian import close_connections

@asynccontextmanager
async def lifespan(app: FastAPI):
    import asyncio
    from src.torch_preloader import preloader
    from .websocket import broadcast_event
    
    # Startup: Wire broadcast callback + begin torch preload (doesn't block)
    preloader.set_broadcast(broadcast_event, asyncio.get_running_loop())
    preloader.start_preload()
    
    # Database will lazy-initialize on first request
    yield
    
    # Shutdown: Clean up database connections
    await close_connections()

app = FastAPI(
    title="GVE Architect API",
    description="Orchestration layer for GVE Asset Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

# Forge UI static root: backend/architect/src/api -> repo root -> tools/forge-ui/static
_FORGE_STATIC = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "forge-ui", "static"))

# Cache root: backend/architect/cache (compiled .gve_bin files)
_CACHE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cache"))


@app.get("/")
async def root():
    index_path = os.path.join(_FORGE_STATIC, "index.html")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"Forge UI index not found: {index_path}")
    return FileResponse(index_path)


@app.get("/viewport")
async def viewport():
    viewport_path = os.path.join(_FORGE_STATIC, "viewport.html")
    if not os.path.exists(viewport_path):
        raise FileNotFoundError(f"Forge UI viewport not found: {viewport_path}")
    return FileResponse(viewport_path)


# Serve static assets (same path; independent of cwd)
app.mount("/static", StaticFiles(directory=_FORGE_STATIC), name="static")

# Serve compiled cache files (for loading .gve_bin directly)
if os.path.exists(_CACHE_ROOT):
    app.mount("/cache", StaticFiles(directory=_CACHE_ROOT), name="cache")

app.include_router(assets_router, prefix="/api/assets", tags=["assets"])
app.include_router(compile_router, prefix="/api/compile", tags=["compile"])
app.include_router(generate_router, prefix="/api/generate", tags=["generate"])
app.include_router(library_router, prefix="/api/library", tags=["library"])
app.include_router(world_router, prefix="/api/world", tags=["world"])
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])


@app.get("/api/status", tags=["status"])
async def get_system_status():
    """
    Get system status including torch preloader state.
    
    Used by frontend to show torch status indicator.
    """
    from src.torch_preloader import preloader
    return {
        "architect": "online",
        "torch": preloader.get_status(),
    }
