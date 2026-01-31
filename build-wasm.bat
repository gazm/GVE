@echo off
REM Quick WASM build script for GVE
REM Usage: build-wasm.bat [release]

setlocal

set "PROJECT_ROOT=%~dp0"
set "WASM_CRATE=%PROJECT_ROOT%engine\wasm"
set "FORGE_UI_WASM=%PROJECT_ROOT%tools\forge-ui\static\wasm\pkg"

if "%1"=="release" (
    set "BUILD_MODE=--release"
    set "TARGET_DIR=release"
    echo Building in RELEASE mode...
) else (
    set "BUILD_MODE="
    set "TARGET_DIR=debug"
    echo Building in DEBUG mode...
)

echo.
echo === Building WASM ===
cd /d "%WASM_CRATE%"
cargo build --target wasm32-unknown-unknown %BUILD_MODE%
if errorlevel 1 (
    echo ERROR: Cargo build failed
    exit /b 1
)

echo.
echo === Generating JS Bindings ===
if not exist pkg mkdir pkg
wasm-bindgen --target web --out-dir pkg "target\wasm32-unknown-unknown\%TARGET_DIR%\gve_wasm.wasm"
if errorlevel 1 (
    echo ERROR: wasm-bindgen failed
    exit /b 1
)

echo.
echo === Copying to Forge UI ===
if not exist "%FORGE_UI_WASM%" mkdir "%FORGE_UI_WASM%"
xcopy /Y /E pkg\* "%FORGE_UI_WASM%\" >nul

echo.
echo === Build Complete ===
echo Output: %WASM_CRATE%\pkg
echo Deployed: %FORGE_UI_WASM%
echo.
echo Run dev server with: cd tools\forge-ui ^& python serve.py

endlocal
