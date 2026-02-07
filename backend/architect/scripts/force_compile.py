import sys
import os
import asyncio
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.compiler.queue import enqueue_compile, CompilePriority
from src.librarian import load_asset_doc

async def main():
    parser = argparse.ArgumentParser(description="Force compile an asset from its DNA")
    parser.add_argument("asset_id", help="ID of the asset to compile")
    parser.add_argument("--priority", choices=["LOW", "NORMAL", "HIGH", "IMMEDIATE"], default="HIGH", help="Compilation priority")
    
    args = parser.parse_args()
    
    print(f"[*] Loading asset {args.asset_id}...")
    doc = await load_asset_doc(args.asset_id)
    if not doc:
        print(f"[!] Asset {args.asset_id} not found!")
        return

    if "dna" not in doc:
        print(f"[!] Asset {args.asset_id} has no DNA! Cannot compile.")
        return

    print(f"[*] Enqueuing compilation for {doc.get('name', 'Unknown Asset')} ({args.asset_id})...")
    
    priority_map = {
        "LOW": CompilePriority.LOW,
        "NORMAL": CompilePriority.NORMAL,
        "HIGH": CompilePriority.HIGH,
        "IMMEDIATE": CompilePriority.IMMEDIATE
    }
    
    job_id = await enqueue_compile(
        asset_id=args.asset_id,
        priority=priority_map[args.priority],
        force_recompile=True
    )
    
    print(f"[+] Compilation queued! Job ID: {job_id}")
    print(f"[*] You can check status at: /api/compile/status/{job_id}")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
