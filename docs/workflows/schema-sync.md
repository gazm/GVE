# Schema Sync Workflow

This workflow describes how to synchronize types from **Rust** (Source of Truth) to **Python** and **TypeScript**.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  engine/shared/src/types/*.rs                               │
│  (Source of Truth)                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ↓ typeshare                     ↓ schemars
   ┌─────────────┐                 ┌─────────────┐
   │ TypeScript  │                 │ JSON Schema │
   │ Interface   │                 │ schema.json │
   └─────────────┘                 └──────┬──────┘
                                          │ datamodel-code-generator
                                          ↓
                                   ┌─────────────┐
                                   │ Python      │
                                   │ Pydantic    │
                                   └─────────────┘
```

## Setup

### Prerequisites
- `cargo` (Rust)
- `uv` (Python)
- `typeshare-cli` (`cargo install typeshare-cli`)

### Python Dependencies
The backend requires `datamodel-code-generator`:
```bash
cd backend/architect
uv add --dev datamodel-code-generator
```

## Running the Sync

Run the automation script from the project root:

```powershell
./tools/generate_schemas.ps1
```

### Manual Steps (Under the hood)

1.  **Generate TypeScript**:
    ```bash
    typeshare . --lang=typescript --output-file tools/forge-ui/src/types.ts
    ```

2.  **Generate JSON Schema**:
    ```bash
    cd engine/shared
    cargo run --bin schema_gen > ../../schemas/schema.json
    ```

3.  **Generate Python**:
    ```bash
    cd backend/architect
    uv run datamodel-code-generator --input ../../schemas/schema.json --output generated/types.py
    ```
