from datetime import datetime
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

from generated.types import AssetMetadata, AssetCategory
from src.compiler import enqueue_compile

MONGO_URI = "mongodb+srv://user:MRyoqoHiZ73yRkUk@gve.vurn2az.mongodb.net/?appName=GVE"
DB_NAME = "gve"

class AssetLibrarian:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def _ensure_db(self):
        """Lazy initialization of the database connection."""
        if self.db is None:
            self.client = AsyncIOMotorClient(MONGO_URI)
            self.db = self.client[DB_NAME]
        return self.db

    async def save_asset(self, asset: AssetMetadata) -> str:
        """
        Atomic save to MongoDB.
        """
        await self._ensure_db()
        # Convert Pydantic to dict (handling Enums/Dates for Mongo)
        doc = asset.model_dump(mode='json')
        doc["updated_at"] = datetime.utcnow()
        
        # Ensure ID is handled correctly (as _id)
        if "id" in doc:
            doc["_id"] = doc.pop("id")
            
        # Update version
        doc["version"] += 1
        
        # Upsert
        result = await self.db.assets.replace_one(
            {"_id": doc["_id"]},
            doc,
            upsert=True
        )
        
        asset_id = doc["_id"]
        
        # Trigger background compilation
        await enqueue_compile(asset_id)
        
        return asset_id

    async def load_asset(self, asset_id: str) -> Optional[AssetMetadata]:
        """Load asset from MongoDB."""
        await self._ensure_db()
        doc = await self.db.assets.find_one({"_id": asset_id})
        if not doc:
            return None
        
        # Normalize document to AssetMetadata schema
        doc["id"] = doc.pop("_id")
        
        # Handle old schema with nested meta fields
        if "meta" in doc:
            meta = doc.pop("meta")
            if "category" not in doc and "category" in meta:
                doc["category"] = meta["category"]
            if "tags" not in doc and "tags" in meta:
                doc["tags"] = meta["tags"]
        
        # Provide defaults for missing required fields
        if "category" not in doc:
            doc["category"] = "Prop"
        if "tags" not in doc:
            doc["tags"] = []
        if "settings" not in doc:
            doc["settings"] = {"lod_count": 3, "resolution": 128}
        if "version" not in doc:
            doc["version"] = 1
        
        # Normalize category to enum value (capitalize first letter)
        if isinstance(doc["category"], str):
            cat = doc["category"].lower()
            doc["category"] = {
                "prop": "Prop",
                "primitive": "Primitive", 
                "character": "Character",
                "environment": "Environment",
            }.get(cat, "Prop")
        
        return AssetMetadata(**doc)

    async def delete_asset(self, asset_id: str) -> None:
        """
        Delete asset and clean up cache files.
        """
        await self._ensure_db()
        await self.db.assets.delete_one({"_id": asset_id})
        # TODO: Garbage collect cache files via cache.py logic

    async def list_assets(self, limit: int = 50, skip: int = 0) -> List[AssetMetadata]:
        await self._ensure_db()
        # Filter out drafts by default
        cursor = self.db.assets.find({"is_draft": {"$ne": True}}).skip(skip).limit(limit)
        results = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            
            # Handle old schema with nested meta fields
            if "meta" in doc:
                meta = doc.pop("meta")
                if "category" not in doc and "category" in meta:
                    doc["category"] = meta["category"]
                if "tags" not in doc and "tags" in meta:
                    doc["tags"] = meta["tags"]
            
            # Provide defaults for missing required fields
            if "category" not in doc:
                doc["category"] = "Prop"
            if "tags" not in doc:
                doc["tags"] = []
            if "settings" not in doc:
                doc["settings"] = {"lod_count": 3, "resolution": 128}
            if "version" not in doc:
                doc["version"] = 1
            
            # Normalize category
            if isinstance(doc["category"], str):
                cat = doc["category"].lower()
                doc["category"] = {
                    "prop": "Prop",
                    "primitive": "Primitive",
                    "character": "Character",
                    "environment": "Environment",
                }.get(cat, "Prop")
            
            results.append(AssetMetadata(**doc))
        return results

    async def list_assets_by_tags(
        self,
        tags: list[str],
        limit: int = 50,
        skip: int = 0,
    ) -> list[AssetMetadata]:
        """List assets that have at least one of the given tags. Filtering done in MongoDB."""
        await self._ensure_db()
        cursor = (
            self.db.assets
            .find({"is_draft": {"$ne": True}, "tags": {"$in": tags}})
            .skip(skip)
            .limit(limit)
        )
        results: list[AssetMetadata] = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            if "meta" in doc:
                meta = doc.pop("meta")
                if "category" not in doc and "category" in meta:
                    doc["category"] = meta["category"]
                if "tags" not in doc and "tags" in meta:
                    doc["tags"] = meta["tags"]
            if "category" not in doc:
                doc["category"] = "Prop"
            if "tags" not in doc:
                doc["tags"] = []
            if "settings" not in doc:
                doc["settings"] = {"lod_count": 3, "resolution": 128}
            if "version" not in doc:
                doc["version"] = 1
            if isinstance(doc["category"], str):
                cat = doc["category"].lower()
                doc["category"] = {
                    "prop": "Prop", "primitive": "Primitive",
                    "character": "Character", "environment": "Environment",
                }.get(cat, "Prop")
            results.append(AssetMetadata(**doc))
        return results

    async def search_assets(self, query: str, limit: int = 50) -> list[AssetMetadata]:
        await self._ensure_db()
        # Simple regex search for prototype (excluding drafts)
        cursor = self.db.assets.find({
            "name": {"$regex": query, "$options": "i"},
            "is_draft": {"$ne": True}
        }).limit(limit)
        results = []
        async for doc in cursor:
            doc["id"] = doc.pop("_id")
            
            # Handle old schema with nested meta fields
            if "meta" in doc:
                meta = doc.pop("meta")
                if "category" not in doc and "category" in meta:
                    doc["category"] = meta["category"]
                if "tags" not in doc and "tags" in meta:
                    doc["tags"] = meta["tags"]
            
            # Provide defaults
            if "category" not in doc:
                doc["category"] = "Prop"
            if "tags" not in doc:
                doc["tags"] = []
            if "settings" not in doc:
                doc["settings"] = {"lod_count": 3, "resolution": 128}
            if "version" not in doc:
                doc["version"] = 1
            
            # Normalize category
            if isinstance(doc["category"], str):
                cat = doc["category"].lower()
                doc["category"] = {
                    "prop": "Prop",
                    "primitive": "Primitive",
                    "character": "Character",
                    "environment": "Environment",
                }.get(cat, "Prop")
            
            results.append(AssetMetadata(**doc))
        return results

    async def vector_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        pre_filter: Optional[dict] = None,
    ) -> List[dict]:
        """
        Semantic vector search using MongoDB Atlas Vector Search.
        
        Requires 'asset_vectors' index on rag.embedding_fusion field.
        Returns raw documents with similarity score.
        """
        await self._ensure_db()
        
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "asset_vectors",
                    "path": "rag.embedding_fusion",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "dna": 1,
                    "rag": 1,
                    "meta": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        
        # Add pre-filter if provided
        if pre_filter:
            pipeline[0]["$vectorSearch"]["filter"] = pre_filter
        
        results = []
        try:
            async for doc in self.db.assets.aggregate(pipeline):
                results.append(doc)
        except Exception as e:
            # Vector search may fail if index doesn't exist yet
            print(f"⚠️ Vector search failed: {e}")
        
        return results

    async def update_asset_rag(
        self,
        asset_id: str,
        embedding: List[float],
        semantic_desc: str,
    ) -> None:
        """
        Update RAG data for an asset (embedding + description).
        
        Called after asset save to index for semantic search.
        """
        await self._ensure_db()
        
        await self.db.assets.update_one(
            {"_id": asset_id},
            {
                "$set": {
                    "rag.embedding_fusion": embedding,
                    "rag.semantic_desc": semantic_desc,
                }
            },
        )

    async def load_asset_doc(self, asset_id: str) -> Optional[dict]:
        """
        Load raw asset document including DNA (for compiler pipeline).
        
        Unlike load_asset(), returns raw dict with DNA field preserved.
        Use this when you need access to DNA for compilation.
        """
        await self._ensure_db()
        doc = await self.db.assets.find_one({"_id": asset_id})
        if not doc:
            return None
        
        # Normalize _id to id for consistency
        doc["id"] = doc.pop("_id")
        return doc

    async def save_asset_doc(self, doc: dict) -> str:
        """
        Save a raw asset document (for AI pipeline).
        
        Unlike save_asset(), accepts raw dict with DNA and meta fields.
        """
        await self._ensure_db()
        
        # Generate ID if not present
        if "_id" not in doc:
            doc["_id"] = str(ObjectId())
        
        doc["created_at"] = datetime.utcnow()
        doc["updated_at"] = datetime.utcnow()
        doc["version"] = doc.get("version", 0) + 1
        
        # Upsert document
        await self.db.assets.replace_one(
            {"_id": doc["_id"]},
            doc,
            upsert=True,
        )
        
        # Trigger background compilation if DNA present
        if "dna" in doc:
            await enqueue_compile(doc["_id"])
        
        return doc["_id"]

    async def update_asset_field(self, asset_id: str, updates: dict) -> bool:
        """Update specific fields of an asset (e.g. is_draft, rating)."""
        await self._ensure_db()
        result = await self.db.assets.update_one(
            {"_id": asset_id},
            {"$set": updates}
        )
        return result.modified_count > 0

    # =========================================================================
    # Concept Image RAG (Learning Loop)
    # =========================================================================

    async def store_concept_rag(
        self,
        asset_id: str,
        prompt: str,
        embedding: List[float],
        concept_image: str,
        dna: Optional[dict] = None,
    ) -> None:
        """
        Store approved concept image for RAG retrieval.
        
        Creates entry in concepts collection for semantic search.
        Links concept to the asset that was successfully generated.
        """
        await self._ensure_db()
        
        doc = {
            "_id": str(ObjectId()),
            "asset_id": asset_id,
            "prompt": prompt,
            "embedding": embedding,
            "concept_image": concept_image,
            "dna": dna,
            "created_at": datetime.utcnow(),
            "approved": True,
        }
        
        await self.db.concepts.insert_one(doc)

    async def search_concepts(
        self,
        query_embedding: List[float],
        limit: int = 3,
    ) -> List[dict]:
        """
        Search for similar approved concepts using vector search.
        
        Returns concept documents with similarity scores.
        Requires 'concept_vectors' index on embedding field.
        """
        await self._ensure_db()
        
        # Try vector search first
        pipeline = [
            {
                "$vectorSearch": {
                    "index": "concept_vectors",
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": limit * 10,
                    "limit": limit,
                    "filter": {"approved": True},
                }
            },
            {
                "$project": {
                    "_id": 1,
                    "asset_id": 1,
                    "prompt": 1,
                    "concept_image": 1,
                    "dna": 1,
                    "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        
        results = []
        try:
            async for doc in self.db.concepts.aggregate(pipeline):
                results.append(doc)
        except Exception as e:
            # Fallback: return recent approved concepts if vector search fails
            print(f"⚠️ Concept vector search failed: {e}")
            try:
                cursor = self.db.concepts.find(
                    {"approved": True}
                ).sort("created_at", -1).limit(limit)
                
                async for doc in cursor:
                    doc["score"] = 0.5  # Default score for fallback
                    results.append(doc)
            except Exception as e2:
                print(f"⚠️ Concept fallback search also failed: {e2}")
        
        return results

    async def close_connections(self) -> None:
        """Close database connections. Call on application shutdown."""
        if self.client:
            self.client.close()
            self.client = None
            self.db = None

# Global instance for public API
_librarian = AssetLibrarian()

# Public API wrappers
load_asset = _librarian.load_asset
save_asset = _librarian.save_asset
delete_asset = _librarian.delete_asset
list_assets = _librarian.list_assets
list_assets_by_tags = _librarian.list_assets_by_tags
search_assets = _librarian.search_assets
close_connections = _librarian.close_connections
vector_search = _librarian.vector_search
update_asset_rag = _librarian.update_asset_rag
save_asset_doc = _librarian.save_asset_doc
load_asset_doc = _librarian.load_asset_doc
update_asset_field = _librarian.update_asset_field

# Concept RAG wrappers
store_concept_rag = _librarian.store_concept_rag
search_concepts = _librarian.search_concepts