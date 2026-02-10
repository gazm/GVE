# backend/architect/src/paths.py
"""
Centralized path configuration for the GVE Architect project.

All paths are defined relative to the project root (GVE/) to avoid fragile
relative path calculations scattered throughout the codebase.
"""

import os
from pathlib import Path

# Find project root by looking for a marker file/directory
# We'll use the presence of 'backend' and 'tools' directories as markers
_current_file = Path(__file__).resolve()

# From backend/architect/src/paths.py, go up to find GVE root
# paths.py -> src -> architect -> backend -> GVE
_project_root = _current_file.parent.parent.parent.parent

# Validate we found the right directory
if not (_project_root / "backend").exists() or not (_project_root / "tools").exists():
    raise RuntimeError(
        f"Could not locate GVE project root. "
        f"Expected to find 'backend' and 'tools' directories in: {_project_root}"
    )

# Define all project paths
PROJECT_ROOT = _project_root
BACKEND_ROOT = PROJECT_ROOT / "backend"
ARCHITECT_ROOT = BACKEND_ROOT / "architect"
TOOLS_ROOT = PROJECT_ROOT / "tools"
FORGE_UI_ROOT = TOOLS_ROOT / "forge-ui"
FORGE_UI_STATIC = FORGE_UI_ROOT / "static"
FORGE_UI_TEMPLATES = FORGE_UI_ROOT / "templates"

# Cache and data directories
CACHE_ROOT = ARCHITECT_ROOT / "cache"
DATA_ROOT = ARCHITECT_ROOT / "data"

# Ensure cache directory exists
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


def get_template_dir() -> str:
    """Get the templates directory as a string for Jinja2."""
    return str(FORGE_UI_TEMPLATES)


def get_static_dir() -> str:
    """Get the static files directory as a string."""
    return str(FORGE_UI_STATIC)


def get_cache_dir() -> str:
    """Get the cache directory as a string."""
    return str(CACHE_ROOT)


# Print paths on import for debugging
if __name__ == "__main__" or os.getenv("DEBUG_PATHS"):
    print(f"[paths] PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"[paths] FORGE_UI_TEMPLATES: {FORGE_UI_TEMPLATES}")
    print(f"[paths] FORGE_UI_STATIC: {FORGE_UI_STATIC}")
    print(f"[paths] CACHE_ROOT: {CACHE_ROOT}")
