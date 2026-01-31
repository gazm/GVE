import sys
import os
import asyncio
import httpx

# Add project root to path
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)

from src.api import app

async def verify_property_editor():
    print("--- Property Editor UI Verification ---")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Test Editor Partial
        print("Testing Property Editor Partial (/api/assets/partials/editor/1)...")
        response = await client.get("/api/assets/partials/editor/1")
        assert response.status_code == 200
        assert "Properties: AK Receiver" in response.text
        assert "Position X" in response.text
        assert "Roughness" in response.text
        assert "range" in response.text
        print("SUCCESS: Property editor partial returned correct mock data and controls.")

        # 2. Test Property Update (POST)
        print("Testing Property Update POST (/api/assets/partials/property/1)...")
        # Post a change to roughness
        data = {"roughness": "0.75"}
        response = await client.post("/api/assets/partials/property/1", data=data)
        assert response.status_code == 200
        # Since our mock just returns the editor again, we check if it's still valid
        assert "Properties: AK Receiver" in response.text
        print("SUCCESS: Property update endpoint accepted mock data and returned HTML.")

if __name__ == "__main__":
    try:
        asyncio.run(verify_property_editor())
        print("\nPROPERTY EDITOR VERIFICATION PASSED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
