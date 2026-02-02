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
    # Startup: Database will lazy-initialize on first request
    yield
    # Shutdown: Clean up database connections
    await close_connections()

app = FastAPI(
    title="GVE Architect API",
    description="Orchestration layer for GVE Asset Pipeline",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    index_path = os.path.join(os.getcwd(), "..", "..", "tools", "forge-ui", "static", "index.html")
    if not os.path.exists(index_path):
        # Fallback for different working directory scenarios
        index_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "forge-ui", "static", "index.html"))
    return FileResponse(index_path)

@app.get("/viewport")
async def viewport():
    viewport_path = os.path.join(os.getcwd(), "..", "..", "tools", "forge-ui", "static", "viewport.html")
    if not os.path.exists(viewport_path):
        viewport_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "forge-ui", "static", "viewport.html"))
    return FileResponse(viewport_path)

# Serve static assets
static_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "tools", "forge-ui", "static"))
app.mount("/static", StaticFiles(directory=static_path), name="static")

app.include_router(assets_router, prefix="/api/assets", tags=["assets"])
app.include_router(compile_router, prefix="/api/compile", tags=["compile"])
app.include_router(generate_router, prefix="/api/generate", tags=["generate"])
app.include_router(library_router, prefix="/api/library", tags=["library"])
app.include_router(world_router, prefix="/api/world", tags=["world"])
app.include_router(websocket_router, prefix="/ws", tags=["websocket"])
