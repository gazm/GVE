import sys
import os
import asyncio
import httpx

# Add project root to path
# CWD should be backend/architect
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)

from src.api import app

async def verify_forge_ui():
    print("--- Forge UI Integration Verification ---")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Test Root (index.html)
        print("Testing Root Route (index.html)...")
        response = await client.get("/")
        assert response.status_code == 200
        assert "GVE Forge" in response.text
        assert "forge-viewport" in response.text
        print("SUCCESS: Root HTML served correctly.")

        # 2. Test CSS
        print("Testing CSS Serving...")
        response = await client.get("/static/css/base.css")
        assert response.status_code == 200
        assert "--accent" in response.text
        # In some test environments content-type might vary, but status 200 + content check is solid
        print(f"SUCCESS: CSS served ({len(response.text)} bytes).")

        # 3. Test JS
        print("Testing Viewport JS Serving...")
        response = await client.get("/static/js/viewport.js")
        assert response.status_code == 200
        assert "initViewport" in response.text
        print(f"SUCCESS: JS served ({len(response.text)} bytes).")

if __name__ == "__main__":
    try:
        asyncio.run(verify_forge_ui())
        print("\nFORGE UI VERIFICATION PASSED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
