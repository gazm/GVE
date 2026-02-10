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

    All branches resolve to the same canonical path keyed by full asset ID:
    ``cache/compiled/{asset_id}.gve_bin``

    This ensures the compiler write path and the API read path always match,
    regardless of whether the caller passes a string ID, a dict, or an
    AssetMetadata object.
    """
    if isinstance(asset, str):
        asset_id = asset
    elif isinstance(asset, dict):
        asset_id = str(asset.get("id") or asset.get("_id", "unknown"))
    else:
        asset_id = str(asset.id)

    return CACHE_ROOT / "compiled" / f"{asset_id}.gve_bin"

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
