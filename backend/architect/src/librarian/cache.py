from enum import Enum
from pathlib import Path
from typing import Optional, Union
import os

from generated.types import AssetMetadata, AssetCategory

class CacheStatus(Enum):
    VALID = "valid"      # Binary matches DB
    STALE = "stale"      # Binary outdated
    MISSING = "missing"  # No binary exists

CACHE_ROOT = Path(os.getenv("GVE_CACHE_ROOT", "./cache"))

def resolve_cache_path(asset: Union[AssetMetadata, dict, str]) -> Path:
    """
    Calculate cache file path from metadata, raw document, or asset ID.
    Format: /cache/{category}/{name}_{short_id}.gve_bin
    """
    # Handle string asset ID
    if isinstance(asset, str):
        return CACHE_ROOT / "compiled" / f"{asset}.gve_bin"
    
    # Handle raw document dict
    if isinstance(asset, dict):
        asset_id = asset.get("id") or asset.get("_id", "unknown")
        name = asset.get("name", "asset")
        category = asset.get("category", "prop")
        if isinstance(category, str):
            category = category.lower()
        else:
            category = "prop"
        
        short_id = str(asset_id)[-6:]
        safe_name = "".join(c for c in name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_').lower()
        filename = f"{safe_name}_{short_id}.gve_bin"
        
        return CACHE_ROOT / category / filename
    
    # Handle AssetMetadata object
    short_id = str(asset.id)[-6:]
    
    # Sanitize name
    safe_name = "".join(c for c in asset.name if c.isalnum() or c in (' ', '_', '-')).strip().replace(' ', '_').lower()
    
    filename = f"{safe_name}_{short_id}.gve_bin"
    
    return CACHE_ROOT / asset.category.value.lower() / filename

def check_cache_validity(asset: AssetMetadata) -> CacheStatus:
    """
    Compare Asset version vs file on disk.
    Note: In a real impl, we'd read the header version from the binary.
    For now, we just check existence.
    """
    path = resolve_cache_path(asset)
    
    if not path.exists():
        return CacheStatus.MISSING
        
    # TODO: Open file and read header version
    # For Phase 2 foundation, we just check existence
    return CacheStatus.VALID
