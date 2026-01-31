import sys
import os
import asyncio
import httpx

# Add project root to path
root = os.getcwd()
if root not in sys.path:
    sys.path.insert(0, root)

from src.api import app

async def verify_card_chain():
    print("--- Card Chain UI Verification ---")
    
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        # 1. Test Chain Partial
        print("Testing Card Chain Partial (/api/assets/partials/chain)...")
        response = await client.get("/api/assets/partials/chain")
        assert response.status_code == 200
        assert "card-chain" in response.text
        assert "AK Receiver" in response.text
        assert "Rusty Steel" in response.text
        assert "Modified Stock" in response.text
        assert "ai-generate-state" in response.text
        assert "cached-state" in response.text
        print("SUCCESS: Card chain partial returned correct mock data and classes.")

        # 2. Test Browser Partial
        print("Testing Asset Browser Partial (/api/assets/partials/browser)...")
        response = await client.get("/api/assets/partials/browser")
        assert response.status_code == 200
        assert "browser-modal" in response.text
        assert "Component Library" in response.text
        print("SUCCESS: Asset browser partial rendered correctly.")

if __name__ == "__main__":
    try:
        asyncio.run(verify_card_chain())
        print("\nCARD CHAIN VERIFICATION PASSED!")
    except Exception as e:
        print(f"\nVERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
