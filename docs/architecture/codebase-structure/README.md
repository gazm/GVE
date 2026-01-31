# Codebase Structure

Documentation for GVE monorepo structure and AI-assisted development.

## Documents

| Document | Purpose |
|----------|---------|
| [Overview](./overview.md) | Monorepo layout, implementation phases, schema sync |
| [Module Breakdowns](./module-breakdowns.md) | File-by-file tables for Engine, Backend, Forge |
| [AI Context Guide](./ai-context-guide.md) | Templates and examples for `.agent/modules/` files |

## Quick Links

### By Layer
- **Engine (Rust)** → [Module Breakdowns](./module-breakdowns.md#engine-rust)
- **Backend (Python)** → [Module Breakdowns](./module-breakdowns.md#backend-python)  
- **Forge (Web)** → [Module Breakdowns](./module-breakdowns.md#forge-web)

### By Topic
- AI context files → [AI Context Guide](./ai-context-guide.md)
- Type generation → [Overview: Schema Sync](./overview.md#schema-sync-flow)
- Implementation order → [Overview: Phases](./overview.md#implementation-phases)

---

**Related Docs:**
- [System Overview](../overview.md) - High-level architecture
- [Engine API](../engine-api.md) - Tri-layer communication
