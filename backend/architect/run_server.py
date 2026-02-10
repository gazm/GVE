
import os
import argparse
import uvicorn

def main():
    parser = argparse.ArgumentParser(description="GVE Architect Server Runner")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    
    args = parser.parse_args()
    
    print(f"🚀 [run_server] Starting GVE Architect on http://{args.host}:{args.port}")
    if args.reload:
        print("🔄 [run_server] Auto-reload enabled")
        
    uvicorn.run(
        "src.api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()
