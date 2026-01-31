# backend/architect/test_vector_search.py
"""
Test script to verify vector search index is working.

Run this after creating the index to verify it's functional.
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.ai_pipeline.rag import create_embedding, semantic_search


async def test_vector_search():
    """Test the vector search functionality."""
    
    print("🧪 Testing Vector Search...")
    print()
    
    # Test 1: Create embedding
    print("1️⃣ Testing embedding generation...")
    test_text = "A weathered wooden barrel with iron bands"
    embedding = create_embedding(test_text)
    print(f"   ✅ Generated embedding: {len(embedding)} dimensions")
    print(f"   First 5 values: {embedding[:5]}")
    print()
    
    # Test 2: Semantic search
    print("2️⃣ Testing semantic search...")
    try:
        results = await semantic_search(test_text, limit=3)
        print(f"   ✅ Search completed: {len(results)} results")
        
        if results:
            print("   Top results:")
            for i, result in enumerate(results, 1):
                print(f"      {i}. {result.name} (score: {result.score:.3f})")
        else:
            print("   ℹ️  No results found (this is OK if database is empty)")
    except Exception as e:
        print(f"   ⚠️  Search failed: {e}")
        print("   💡 Make sure:")
        print("      - Vector search index 'asset_vectors' exists")
        print("      - Index has finished building (check Atlas UI)")
        print("      - MongoDB connection is working")
    
    print()
    print("✅ Test complete!")


if __name__ == "__main__":
    asyncio.run(test_vector_search())
