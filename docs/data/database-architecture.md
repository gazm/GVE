# **GVE-1 Asset Database Configuration**

**Stack:** MongoDB (Source of Truth) \+ Local Cache (Runtime).

**Philosophy:** The database serves as the "Genotype" storage—holding the high-level recipes, metadata, and history. The file system serves as the "Phenotype" storage—holding the compiled, optimized binaries ready for the engine to ingest.

## **1\. The Schema (collections.assets)**

The assets collection is the primary registry for the entire project. It is designed to be **Document-Oriented**, allowing for flexible schemas that can evolve as the engine matures (e.g., adding new component types without migrating SQL tables).

### **1.1 The Document Model**

This structure represents a single asset. Unlike a standard game engine where an asset is a binary file on disk, here an asset is a JSON document.

{  
  "\_id": ObjectId("507f1f77bcf86cd799439011"),  
  "name": "Bunker\_Heavy\_01",  
  "version": 12, // Incrementing integer for cache invalidation  
  "created\_at": ISODate("2025-10-01T..."),  
  "updated\_at": ISODate("2025-10-05T..."),

  // 1\. The DNA (The Immutable Recipe)  
  // This defines the physical form. Changing this requires a Re-Bake.  
  "dna": {   
      "root\_node": {   
          "type": "operation",   
          "op": "union",   
          "children": \[...\]   
      },  
      "audio\_patch": {   
          "mode": "hybrid",   
          "synth\_params": { "ratio": 1.41 }   
      },  
      "baking\_profile": {  
          "iterations": 500,  
          "pruning\_aggressiveness": 0.05  
      }  
  },

  // 2\. The Entity (The Mutable Gameplay Data)  
  // This defines behavior. Changing this is instant (No Re-Bake).  
  "entity": {   
      "archetype": "static\_structure",  
      "traits": {   
          "health": { "max": 5000, "armor": "concrete\_heavy" },  
          "interaction": { "prompt": "Press E to Enter" },  
          "item": { "weight": 50000.0, "value": 0 }   
      }   
  },

  // 3\. Search & RAG Data (The Semantic Index)  
  // Generated automatically by the Librarian upon save.  
  "rag": {  
      // 512-float vector fusing Text Description \+ PointNet Shape  
      "embedding\_fusion": \[0.02, \-0.55, 0.12, ...\],   
      // Human-readable summary used for text-based embedding  
      "semantic\_desc": "Concrete structure, domed roof, machined bore holes."  
  },

  // 4\. Faceted Metadata (The UI Filters)  
  "meta": {   
      "category": "architecture",   
      "subcategory": "military",  
      "material\_primary": "ASTM\_C114",   
      "tags": \["bunker", "ww2", "concrete", "defensive"\],  
      "author": "AI\_Architect\_01"  
  },  
    
  // 5\. Version History (Safety Net)  
  // Stores deltas or full snapshots of previous versions.  
  "history": \[  
      { "v": 11, "timestamp": "...", "diff": "Changed wall thickness to 1.5m" }  
  \]  
}

### **1.2 Indexing Strategy**

To ensure the Forge UI remains snappy even with 100,000+ assets, we enforce specific indexes:

* **Compound Text Index:** name \+ meta.tags. Supports standard "Ctrl+F" search behavior.  
* **Vector Search Index (Atlas):** rag.embedding\_fusion. Enables "Find assets that look like this shape" queries.  
* **Faceted Indexes:** meta.category, meta.material\_primary. Allows for instant filtering (e.g., "Show me all *Steel* items in *Weapons*").

## **2\. Structured Cache (File System)**

While MongoDB holds the truth, the Rust engine requires raw binary data to run efficiently. We utilize a **"Type-First, Category-Second"** directory hierarchy in the local cache.

**Root Path:** /cache (or /game\_data in release builds)

### **2.1 The Hierarchy**

This structure allows developers to browse generated assets using standard OS file explorers for debugging, while keeping the root clean.

/cache  
  ├── /geometry           \# .gve\_bin (SDFs, Physics, Shell Meshes)  
  │     ├── /architecture  
  │     │     └── /military  
  │     │           └── bunker\_heavy\_01\_507f1f.gve\_bin  
  │     ├── /props  
  │     │     └── /tools  
  │     └── /vehicles  
  │  
  ├── /surfaces           \# .splats (Texture Data & Smart Masks)  
  │     ├── /architecture  
  │     │     └── /military  
  │     │           └── bunker\_heavy\_01\_507f1f.splats  
  │     └── ...  
  │  
  ├── /audio              \# .gve\_synth (DSP Chains & Synth Patches)  
  │     ├── /impacts  
  │     └── /ambience  
  │  
  └── /previews           \# .png (512x512 UI Thumbnails)  
        ├── /architecture  
        └── /props

### **2.2 Naming Convention**

* **Format:** {snake\_case\_name}\_{short\_oid}.{ext}  
* **Example:** plasma\_cutter\_a1b2c3.gve\_bin  
* **Reasoning:** Human readable for debugging, but the 6-char Object ID suffix prevents collisions if two users name an asset "Box".

## **3\. The Librarian Module**

The Librarian is no longer a passive file watcher. It is an active **Database Abstraction Layer (DAL)** residing within the Python Architect. It acts as the gatekeeper between the high-level UI and low-level storage.

### **3.1 Core Responsibilities**

* **CRUD Operations:** Wraps pymongo to handle saving, loading, and deleting assets. It ensures that when an asset is deleted, its corresponding binaries in /cache are also garbage collected.  
* **Path Resolution:** Dynamically calculates where a binary *should* be based on its metadata. This decouples the file system structure from the database ID, allowing us to reorganize folders without breaking links.  
  * *Logic:* path \= cache\_root \+ type \+ category \+ subcategory \+ name \+ id.  
* **Cache Invalidation:** Checks db\_version vs file\_version. If the DB record is newer (v12) than the binary on disk (v11), it triggers the Compiler to re-bake the asset immediately.

### **3.2 The Vectorization Pipeline**

When an asset is saved, the Librarian triggers an asynchronous job:

1. **Text Embedding:** Flattens the JSON structure into a string description and runs it through all-MiniLM-L6-v2.  
2. **Shape Embedding:** Feeds the RichSplat point cloud through **PointNet** to extract a geometric signature.  
3. **Fusion:** Combines these into the rag.embedding\_fusion vector and updates the MongoDB document.

### **3.3 API Endpoints**

* GET /api/assets/search: Performs an Aggregation Pipeline query on MongoDB.  
* POST /api/assets/save: Atomic write to MongoDB ![][image1] Triggers Background Compile ![][image1] Updates /cache.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABMAAAAXCAYAAADpwXTaAAAAiklEQVR4XmNgGAWjgGqAFYjZoJgyICsrqyMvL18Cwuhy5ABWOTm5DhCWkpKSRZckGUhLS6uBMNCV3YKCgvzo8iQBfIaBnO0MxCGkYmC41QLp00DaCWgOM9UNIxkAvagPwkBD+sTFxbnR5YkGMjIynEDXTABhimMTaIgxEJeDMLocOYB6OWAUUBcAAPXlJ9gwCBaWAAAAAElFTkSuQmCC>