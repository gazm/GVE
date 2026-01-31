$ErrorActionPreference = "Stop"

Write-Host "🚧 Starting Schema Sync..."

$ROOT = Resolve-Path "$PSScriptRoot/.."
$SHARED_DIR = "$ROOT/engine/shared"
$BACKEND_DIR = "$ROOT/backend/architect"
$SCHEMAS_DIR = "$ROOT/schemas"

# 1. Create schemas directory
if (!(Test-Path $SCHEMAS_DIR)) {
    New-Item -ItemType Directory -Path $SCHEMAS_DIR | Out-Null
}

# 2. Generate JSON Schema (Rust)
Write-Host "🦀 Generating JSON Schema from Rust..."
Push-Location $SHARED_DIR
# Pass output path directly to Rust binary to avoid shell encoding issues
$schemaPath = Join-Path $SCHEMAS_DIR "schema.json"
cargo run --quiet --bin schema_gen --features schema_gen -- --output "$schemaPath"
Pop-Location

# 3. Generate Python Models
Write-Host "🐍 Generating Python Pydantic models..."
Push-Location $BACKEND_DIR

# Use 'uv run python -m datamodel_code_generator' to be explicit
# Changes:
# 1. --input-file-type jsonschema
# 2. --formatters ruff-format
# 3. -W ignore::UserWarning (Suppresses 'uint32' format warning)
uv run python -W ignore::UserWarning -m datamodel_code_generator --input "$SCHEMAS_DIR/schema.json" --output generated/types.py --input-file-type jsonschema --output-model-type pydantic.BaseModel --formatters ruff-format

Pop-Location

Write-Host "✅ Schema Sync Complete!"
