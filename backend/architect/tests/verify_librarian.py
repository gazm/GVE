import sys
import os
import asyncio
from bson import ObjectId

# Add the project root to path
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)

from src.librarian.assets import save_asset, load_asset, search_assets, delete_asset
from generated.types import AssetMetadata, AssetCategory, AssetSettings

async def test_librarian():
    print("--- Librarian Verification (MongoDB Atlas) ---")
    
    # Create test asset
    asset_id = str(ObjectId())
    test_asset = AssetMetadata(
        id=asset_id,
        name="TestSphere-Librarian",
        category=AssetCategory.Primitive,
        settings=AssetSettings(lod_count=0, resolution=32),
        tags=["test", "unit"],
        version=0
    )
    
    try:
        # 1. Save
        print(f"Saving asset: {asset_id}...")
        saved_id = await save_asset(test_asset)
        print(f"Saved (expected {asset_id}): {saved_id}")
        
        # 2. Load
        print(f"Loading asset: {saved_id}...")
        loaded = await load_asset(saved_id)
        if loaded:
            print(f"Loaded successfully: {loaded.name}, Version: {loaded.version}")
            if loaded.name == "TestSphere-Librarian" and loaded.version > 0:
                print("SUCCESS: Field verification passed.")
            else:
                print(f"FAILURE: Field verification failed. Name: {loaded.name}, Version: {loaded.version}")
        else:
            print("FAILURE: Asset not found after save.")
            return

        # 3. Search
        print("Searching for 'TestSphere-Librarian'...")
        results = await search_assets("TestSphere-Librarian")
        if any(a.id == saved_id for a in results):
            print("SUCCESS: Asset found in search.")
        else:
            print("FAILURE: Asset not found in search.")

        # 4. Delete
        print(f"Deleting asset: {saved_id}...")
        await delete_asset(saved_id)
        
        # 5. Verify Deletion
        print("Verifying deletion...")
        final_load = await load_asset(saved_id)
        if final_load is None:
            print("SUCCESS: Deletion verified.")
        else:
            print("FAILURE: Asset still exists after deletion.")

    except Exception as e:
        print(f"ERROR during verification: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_librarian())
