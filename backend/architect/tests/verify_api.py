import sys
import os
import asyncio
import httpx
from bson import ObjectId

# Add project root to path
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)

from src.api import app
from generated.types import AssetMetadata

async def test_api_flow():
    print("--- API Implementation Verification (Async) ---")
    
    # Use AsyncClient with lifespan support
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Health Check
        print("Testing Root Health Check...")
        response = await client.get("/")
        assert response.status_code == 200
        print(f"SUCCESS: {response.json()}")

        # 2. Asset CRUD
        asset_id = str(ObjectId())
        test_asset = {
            "id": asset_id,
            "name": "API-Test-Sphere",
            "category": "Primitive",
            "settings": {"lod_count": 0, "resolution": 64},
            "tags": ["api", "test"],
            "version": 0
        }

        print(f"Testing Asset POST (Save): {asset_id}...")
        response = await client.post("/api/assets/", json=test_asset)
        assert response.status_code == 200
        print(f"SUCCESS: Saved ID {response.json()}")

        print(f"Testing Asset GET (Load): {asset_id}...")
        response = await client.get(f"/api/assets/{asset_id}")
        assert response.status_code == 200
        loaded = response.json()
        assert loaded["name"] == "API-Test-Sphere"
        print(f"SUCCESS: Loaded {loaded['name']}")

        print("Testing Asset Search...")
        response = await client.get("/api/assets/search?q=API-Test")
        assert response.status_code == 200
        results = response.json()
        assert any(a["id"] == asset_id for a in results)
        print("SUCCESS: Asset found in search results.")

        # 3. Compile Trigger
        print(f"Testing Compile Trigger: {asset_id}...")
        response = await client.post(f"/api/compile/{asset_id}", json={"priority": 1, "force_recompile": True})
        assert response.status_code == 200
        job_info = response.json()
        assert "job_id" in job_info
        print(f"SUCCESS: Job queued: {job_info['job_id']}")

        print(f"Testing Compile Status: {job_info['job_id']}...")
        response = await client.get(f"/api/compile/status/{job_info['job_id']}")
        assert response.status_code == 200
        status_info = response.json()
        assert status_info["status"] == "queued"
        print(f"SUCCESS: Status: {status_info['status']}")

        # 5. Cleanup
        print(f"Cleaning up: Deleting {asset_id}...")
        response = await client.delete(f"/api/assets/{asset_id}")
        assert response.status_code == 200
        print("SUCCESS: Cleanup complete.")

if __name__ == "__main__":
    try:
        asyncio.run(test_api_flow())
        print("\nALL API TESTS PASSED!")
    except Exception as e:
        print(f"\nAPI TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_api_flow()
        print("\nALL API TESTS PASSED!")
    except Exception as e:
        print(f"\nAPI TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
