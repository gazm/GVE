# backend/architect/setup_vector_index.py
"""
Script to create MongoDB Atlas Vector Search index for RAG.

Run this once to set up the vector search index on your MongoDB Atlas cluster.
Requires: pymongo (or use MongoDB Atlas UI)
"""

import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi

# MongoDB connection (same as in librarian/assets.py)
MONGO_URI = "mongodb+srv://user:MRyoqoHiZ73yRkUk@gve.vurn2az.mongodb.net/?appName=GVE"
DB_NAME = "gve"
COLLECTION_NAME = "assets"
INDEX_NAME = "asset_vectors"

def create_vector_index():
    """Create the vector search index via MongoDB Atlas API."""
    
    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client[DB_NAME]
    
    # Vector search index definition
    index_definition = {
        "name": INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {
                    "path": "rag.embedding_fusion",
                    "numDimensions": 384,  # all-MiniLM-L6-v2 produces 384-dim vectors
                    "similarity": "cosine"
                }
            ]
        }
    }
    
    try:
        # Create index via Atlas Search API
        # Note: This requires Atlas Search API access
        # Alternative: Use MongoDB Atlas UI (see instructions below)
        
        print(f"📝 Creating vector search index '{INDEX_NAME}'...")
        print(f"   Collection: {COLLECTION_NAME}")
        print(f"   Field: rag.embedding_fusion")
        print(f"   Dimensions: 384")
        print(f"   Similarity: cosine")
        print()
        print("⚠️  Note: Vector search indexes must be created via MongoDB Atlas UI or Atlas Admin API.")
        print("   This script shows the required configuration.")
        print()
        print("📋 Index Definition (JSON):")
        import json
        print(json.dumps(index_definition, indent=2))
        print()
        print("✅ Copy the JSON above and create the index via:")
        print("   1. MongoDB Atlas UI → Search → Create Search Index")
        print("   2. Or use Atlas Admin API (requires API key)")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print()
        print("💡 Alternative: Create index via MongoDB Atlas UI:")
        print("   1. Go to https://cloud.mongodb.com")
        print("   2. Select your cluster")
        print("   3. Navigate to 'Search' tab")
        print("   4. Click 'Create Search Index'")
        print("   5. Choose 'JSON Editor'")
        print("   6. Paste the JSON definition shown above")
        print("   7. Click 'Next' and 'Create Search Index'")
    finally:
        client.close()


if __name__ == "__main__":
    create_vector_index()
