# GVE WASM Build Script
# Builds the WASM engine and generates JS bindings for the web frontend

param(
    [switch]$Release,
    [switch]$Clean,
    [switch]$SkipCopy,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# Colors for output
function Write-Header { param($msg) Write-Host "`n=== $msg ===" -ForegroundColor Cyan }
function Write-Success { param($msg) Write-Host "[OK] $msg" -ForegroundColor Green }
function Write-Info { param($msg) Write-Host "[..] $msg" -ForegroundColor Blue }
function Write-Warn { param($msg) Write-Host "[!!] $msg" -ForegroundColor Yellow }
function Write-Err { param($msg) Write-Host "[XX] $msg" -ForegroundColor Red }

if ($Help) {
    Write-Host ""
    Write-Host "GVE WASM Build Script"
    Write-Host "====================="
    Write-Host ""
    Write-Host "Usage: .\build-wasm.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "    -Release    Build in release mode (optimized, smaller)"
    Write-Host "    -Clean      Clean build artifacts before building"
    Write-Host "    -SkipCopy   Do not copy pkg to forge-ui/static/wasm/"
    Write-Host "    -Help       Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "    .\build-wasm.ps1                    # Debug build"
    Write-Host "    .\build-wasm.ps1 -Release           # Release build"
    Write-Host "    .\build-wasm.ps1 -Release -Clean    # Clean release build"
    Write-Host ""
    exit 0
}

# Paths
$ProjectRoot = $PSScriptRoot
$WasmCrate = Join-Path $ProjectRoot "engine\wasm"
$RuntimeCrate = Join-Path $ProjectRoot "engine\runtime"
$ForgeUiWasm = Join-Path $ProjectRoot "tools\forge-ui\static\wasm\pkg"

Write-Header "GVE WASM Build"
Write-Info "Project root: $ProjectRoot"

# Check prerequisites
Write-Header "Checking Prerequisites"

# Check cargo - try common installation paths
$cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
if (-not $cargoCmd) {
    # Try adding common Rust paths
    $rustPaths = @(
        "$env:USERPROFILE\.cargo\bin",
        "$env:CARGO_HOME\bin",
        "C:\Users\$env:USERNAME\.cargo\bin"
    )
    foreach ($p in $rustPaths) {
        if (Test-Path "$p\cargo.exe") {
            $env:PATH = "$p;$env:PATH"
            Write-Info "Added $p to PATH"
            break
        }
    }
    $cargoCmd = Get-Command cargo -ErrorAction SilentlyContinue
}
if (-not $cargoCmd) {
    Write-Err "Cargo not found. Install Rust from https://rustup.rs"
    Write-Err "Searched: $env:USERPROFILE\.cargo\bin"
    exit 1
}
$cargoVersion = & cargo --version
Write-Success "Cargo: $cargoVersion"

# Check wasm32 target
$targets = & rustup target list --installed 2>&1
if ($targets -notcontains "wasm32-unknown-unknown") {
    Write-Warn "WASM target not installed. Installing..."
    & rustup target add wasm32-unknown-unknown
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to install wasm32-unknown-unknown target"
        exit 1
    }
}
Write-Success "WASM target: wasm32-unknown-unknown"

# Check wasm-bindgen (should be in same path as cargo now)
$wbCmd = Get-Command wasm-bindgen -ErrorAction SilentlyContinue
if (-not $wbCmd) {
    Write-Warn "wasm-bindgen not found. Installing..."
    & cargo install wasm-bindgen-cli
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to install wasm-bindgen-cli"
        exit 1
    }
    # Refresh command lookup after install
    $wbCmd = Get-Command wasm-bindgen -ErrorAction SilentlyContinue
    Write-Success "wasm-bindgen installed"
}
if ($wbCmd) {
    $wbVersion = & wasm-bindgen --version
    Write-Success "wasm-bindgen: $wbVersion"
}

# Clean if requested
if ($Clean) {
    Write-Header "Cleaning Build Artifacts"
    
    Push-Location $WasmCrate
    & cargo clean
    Pop-Location
    
    $pkgDir = Join-Path $WasmCrate "pkg"
    if (Test-Path $pkgDir) {
        Remove-Item -Recurse -Force $pkgDir
        Write-Info "Removed $pkgDir"
    }
    
    Write-Success "Clean complete"
}

# Build mode
$TargetDir = if ($Release) { "release" } else { "debug" }

# Build runtime first (to catch errors early)
Write-Header "Building Runtime Crate"
Push-Location $RuntimeCrate
if ($Release) {
    & cargo check --release
} else {
    & cargo check
}
if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Err "Runtime crate failed to compile"
    exit 1
}
Pop-Location
Write-Success "Runtime crate OK"

# Build WASM
Write-Header "Building WASM Crate"
Push-Location $WasmCrate

$modeFlag = if ($Release) { "--release" } else { "" }
Write-Info "Running: cargo build --target wasm32-unknown-unknown $modeFlag"
if ($Release) {
    & cargo build --target wasm32-unknown-unknown --release
} else {
    & cargo build --target wasm32-unknown-unknown
}

if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Err "WASM build failed"
    exit 1
}
Write-Success "WASM build complete"

# Generate JS bindings
Write-Header "Generating JS Bindings"

$wasmFile = Join-Path $WasmCrate "target\wasm32-unknown-unknown\$TargetDir\gve_wasm.wasm"
$pkgDir = Join-Path $WasmCrate "pkg"

if (-not (Test-Path $wasmFile)) {
    Pop-Location
    Write-Err "WASM file not found: $wasmFile"
    exit 1
}

# Create pkg directory
if (-not (Test-Path $pkgDir)) {
    New-Item -ItemType Directory -Path $pkgDir | Out-Null
}

Write-Info "Running: wasm-bindgen --target web --out-dir $pkgDir $wasmFile"
& wasm-bindgen --target web --out-dir $pkgDir $wasmFile

if ($LASTEXITCODE -ne 0) {
    Pop-Location
    Write-Err "wasm-bindgen failed"
    exit 1
}

Pop-Location
Write-Success "JS bindings generated in $pkgDir"

# Copy to forge-ui
if (-not $SkipCopy) {
    Write-Header "Deploying to Forge UI"
    
    if (-not (Test-Path $ForgeUiWasm)) {
        New-Item -ItemType Directory -Path $ForgeUiWasm -Force | Out-Null
    }
    
    Copy-Item -Path "$pkgDir\*" -Destination $ForgeUiWasm -Recurse -Force
    Write-Success "Copied to $ForgeUiWasm"
}

# Summary
Write-Header "Build Summary"

$wasmSize = (Get-Item $wasmFile).Length / 1KB
$jsFile = Join-Path $pkgDir "gve_wasm.js"
$jsSize = 0
if (Test-Path $jsFile) { 
    $jsSize = (Get-Item $jsFile).Length / 1KB 
}

$modeStr = "Debug"
if ($Release) { $modeStr = "Release" }

Write-Host ""
Write-Host "  Mode:      $modeStr"
Write-Host "  WASM:      $([math]::Round($wasmSize, 1)) KB"
Write-Host "  JS:        $([math]::Round($jsSize, 1)) KB"
Write-Host "  Output:    $pkgDir"
if (-not $SkipCopy) {
    Write-Host "  Deployed:  $ForgeUiWasm"
}
Write-Host ""

Write-Success "Build complete!"
Write-Host ""
Write-Info "To run the dev server:"
Write-Host "  cd tools\forge-ui"
Write-Host "  python serve.py"
Write-Host ""
