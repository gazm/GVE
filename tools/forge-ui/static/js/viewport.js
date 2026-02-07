/**
 * viewport.js
 * Handles the WASM viewport initialization and rendering loop.
 */

// Namespace import so missing exports (e.g. get_scene_snapshot in older WASM builds) don't break load
import init, * as gve_wasm from '../wasm/pkg/gve_wasm.js';
import { CameraController } from './viewport-camera.js';

const wasmGetSceneSnapshot = gve_wasm.get_scene_snapshot || (() => new Uint8Array(0));

const MSG_TYPE_LOAD_CHUNK = 0x30;
const MSG_TYPE_TRANSLATE_NODE = 0x31;
const MSG_TYPE_UPDATE_JOINT = 0x32;
const MSG_TYPE_UPDATE_CAMERA = 0x20;

let wasmEngineInitialized = false;
let wasmModuleLoaded = false;
let engineInitializing = false;  // Prevent duplicate initialization
window.viewportReady = false;
let cameraController = null;
let lastLoadedAsset = { id: 0n, url: null, data: null };

// Current scene bounds (updated on asset load)
let currentBounds = {
    min: [-1, -1, -1],
    max: [1, 1, 1],
    center: [0, 0, 0],
    radius: 1.732
};

/** Load WASM module once. Call on page load so WASM is ready when user opens Assets. */
export async function ensureWasmLoaded() {
    if (wasmModuleLoaded) return;
    await init();
    wasmModuleLoaded = true;
}

// Expose debug functions to window for console access
window.toggle_axes = () => {
    if (!wasmEngineInitialized) { console.warn("WASM not ready"); return; }
    gve_wasm.toggle_axes();
};

// Expose debug functions to window for console access
window.load_test_sdf = () => {
    if (!wasmEngineInitialized) {
        console.warn("WASM not ready");
        return;
    }
    gve_wasm.load_test_sdf();
};
window.clear_sdf = () => {
    if (!wasmEngineInitialized) return;
    gve_wasm.clear_sdf();
};

/**
 * Set the rendering view mode for the current asset.
 * @param {string} mode - 'shell' | 'sdf' | 'splat'
 */
window.set_view_mode = (mode) => {
    if (!wasmEngineInitialized) return;

    // If we have a loaded asset, use its ID. Otherwise 0 (global clear/reset might fail if ID required?)
    // The WASM function takes (mode, asset_id). If asset_id is wrong, it might just hide everything.
    const assetId = lastLoadedAsset.id;

    console.log(`👁️ Setting view mode to: ${mode} for asset ${assetId}`);

    switch (mode) {
        case 'shell':
        case 'mesh':
            gve_wasm.set_view_mode(0, assetId);
            break;
        case 'sdf':
            gve_wasm.set_view_mode(1, assetId);
            break;
        case 'splat':
            gve_wasm.set_view_mode(2, assetId);
            break;
        default:
            console.warn(`Unknown view mode: ${mode}`);
    }
};

window.snap_camera_to = (x, y, z, yaw, pitch) => {
    if (!wasmEngineInitialized) { console.warn("WASM not ready"); return; }
    if (cameraController) {
        cameraController.snap_to_position([x, y, z], yaw, pitch);
    } else {
        gve_wasm.snap_camera_to(x, y, z, yaw, pitch);
    }
};

/**
 * Get available view modes for the current or specified asset.
 * Returns { mesh: bool, sdf: bool, splat: bool, volume: bool }
 * WASM returns a packed bitmask u8: bit0=mesh, bit1=sdf, bit2=splat, bit3=volume
 */
window.get_asset_modes = (assetId = null) => {
    if (!wasmEngineInitialized) {
        console.warn("WASM not ready");
        return { mesh: false, sdf: false, splat: false, volume: false };
    }
    const id = assetId || lastLoadedAsset.id;
    const bits = gve_wasm.get_asset_modes(id);
    const modes = {
        mesh:   !!(bits & 0x01),
        sdf:    !!(bits & 0x02),
        splat:  !!(bits & 0x04),
        volume: !!(bits & 0x08),
    };
    console.log(`📊 Asset ${id} modes:`, modes);
    return modes;
};

/**
 * Fit camera to current scene bounds.
 * Positions camera to see the entire loaded asset.
 * Triggered by 'F' key.
 */
window.fit_to_bounds = (bounds = null) => {
    if (!wasmEngineInitialized) {
        console.warn("WASM not ready");
        return;
    }

    const b = bounds || currentBounds;

    // Calculate center and size
    const center = [
        (b.min[0] + b.max[0]) / 2,
        (b.min[1] + b.max[1]) / 2,
        (b.min[2] + b.max[2]) / 2
    ];

    const size = [
        b.max[0] - b.min[0],
        b.max[1] - b.min[1],
        b.max[2] - b.min[2]
    ];

    // Calculate bounding sphere radius
    const radius = Math.sqrt(size[0] * size[0] + size[1] * size[1] + size[2] * size[2]) / 2;

    // Position camera at 2.5x the bounding sphere radius, looking at center
    // Place camera on the -Z side looking towards +Z (front view)
    const cameraDistance = Math.max(radius * 2.5, 0.5);

    const camX = center[0];
    const camY = center[1];
    const camZ = center[2] - cameraDistance;  // Behind the object (looking +Z)

    // Yaw = PI/2 to look towards +Z
    const yaw = Math.PI / 2;
    const pitch = 0;

    // Use WASM snap_camera_to to position camera
    gve_wasm.snap_camera_to(camX, camY, camZ, yaw, pitch);

    console.log(`📷 Fit to bounds: center=[${center.map(v => v.toFixed(2))}], radius=${radius.toFixed(2)}, distance=${cameraDistance.toFixed(2)}`);

    return { center, radius, cameraDistance };
};

/**
 * Update current scene bounds from loaded asset.
 * Called automatically when loading .gve_bin files.
 */
function updateBoundsFromBinary(data) {
    // GVE Binary Header (84 bytes):
    // magic(4) + version(4) + flags(4) + sdf_bytecode_offset(8) + volume_offset(8) +
    // splat_offset(8) + shell_offset(8) + audio_offset(8) + metadata_offset(8) +
    // sdf_size(4) + volume_size(4) + splat_count(4) + vertex_count(4) + padding(8)

    if (data.byteLength < 84) {
        console.log(`📐 Binary too small: ${data.byteLength} < 84`);
        return null;
    }

    const view = new DataView(data);

    // Check magic
    const magic = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
    if (magic !== 'GVE1') {
        console.log(`📐 Invalid magic: ${magic}`);
        return null;
    }

    // Get shell mesh offset to estimate bounds from vertex data
    const shellOffset = Number(view.getBigUint64(36, true));
    const vertexCount = view.getUint32(72, true);

    console.log(`📐 Header: shellOffset=${shellOffset}, vertexCount=${vertexCount}, dataLen=${data.byteLength}`);

    if (shellOffset > 0 && vertexCount > 0) {
        // Shell mesh section has its own vertex_count and index_count at the start
        // Read the actual vertex count from shell section
        const shellVertCount = view.getUint32(shellOffset, true);
        const shellIdxCount = view.getUint32(shellOffset + 4, true);

        console.log(`📐 Shell section: verts=${shellVertCount}, idxs=${shellIdxCount}`);

        const vertStart = shellOffset + 8;  // Skip vertex_count, index_count headers
        const vertEnd = vertStart + shellVertCount * 24;

        if (vertEnd > data.byteLength) {
            console.log(`📐 Vertex data out of bounds: ${vertEnd} > ${data.byteLength}`);
            return null;
        }

        // Debug: print first few vertex bytes as hex and float
        const debugOffset = vertStart;
        const debugBytes = [];
        for (let b = 0; b < 24 && debugOffset + b < data.byteLength; b++) {
            debugBytes.push(view.getUint8(debugOffset + b).toString(16).padStart(2, '0'));
        }
        const v0x = view.getFloat32(debugOffset, true);
        const v0y = view.getFloat32(debugOffset + 4, true);
        const v0z = view.getFloat32(debugOffset + 8, true);
        console.log(`📐 First vertex raw hex: ${debugBytes.join(' ')}`);
        console.log(`📐 First vertex pos: [${v0x}, ${v0y}, ${v0z}]`);

        let minX = Infinity, minY = Infinity, minZ = Infinity;
        let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;

        // Sample every 100th vertex for speed (still accurate for bounds)
        const sampleCount = Math.min(shellVertCount, 500);
        const step = Math.max(1, Math.floor(shellVertCount / sampleCount));

        for (let i = 0; i < shellVertCount; i += step) {
            const offset = vertStart + i * 24;  // 24 bytes per vertex (pos + normal)
            if (offset + 12 > data.byteLength) break;

            const x = view.getFloat32(offset, true);
            const y = view.getFloat32(offset + 4, true);
            const z = view.getFloat32(offset + 8, true);

            // Skip invalid values
            if (!isFinite(x) || !isFinite(y) || !isFinite(z)) continue;
            if (Math.abs(x) > 1000 || Math.abs(y) > 1000 || Math.abs(z) > 1000) continue;

            minX = Math.min(minX, x); maxX = Math.max(maxX, x);
            minY = Math.min(minY, y); maxY = Math.max(maxY, y);
            minZ = Math.min(minZ, z); maxZ = Math.max(maxZ, z);
        }

        if (isFinite(minX) && (maxX - minX) > 0.001) {
            currentBounds = {
                min: [minX, minY, minZ],
                max: [maxX, maxY, maxZ],
                center: [(minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2],
                radius: Math.sqrt(
                    Math.pow(maxX - minX, 2) +
                    Math.pow(maxY - minY, 2) +
                    Math.pow(maxZ - minZ, 2)
                ) / 2
            };
            console.log(`📐 Bounds: [${minX.toFixed(3)}, ${minY.toFixed(3)}, ${minZ.toFixed(3)}] to [${maxX.toFixed(3)}, ${maxY.toFixed(3)}, ${maxZ.toFixed(3)}]`);
            return currentBounds;
        } else {
            console.log(`📐 Invalid bounds: minX=${minX}, maxX=${maxX}`);
        }
    } else {
        console.log(`📐 No shell data: shellOffset=${shellOffset}, vertexCount=${vertexCount}`);
    }

    return null;
}

window.clear_viewport = () => {
    if (!wasmEngineInitialized) { console.warn("WASM not ready"); return; }
    gve_wasm.clear_viewport();
};

/** Returns current scene as Uint8Array: u32 count, then per entry asset_id (u64 le), type (u8: 0=mesh, 1=sdf), active (u8). */
window.get_scene_snapshot = () => {
    if (!wasmEngineInitialized) return new Uint8Array(0);
    return wasmGetSceneSnapshot();
};

// Convert string asset ID to numeric (hash for WASM)
function assetIdToNumeric(id) {
    if (typeof id === 'number' || typeof id === 'bigint') {
        return BigInt(id);
    }
    // Hash string ID to a 64-bit number
    const str = String(id);
    let hash = 0n;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5n) - hash) + BigInt(str.charCodeAt(i));
        hash = hash & 0xFFFFFFFFFFFFFFFFn; // Keep to 64 bits
    }
    return hash;
}

// Load a .gve_bin file from URL and send to WASM renderer
// Options: { autoFit: true } to automatically fit camera to asset bounds
window.load_asset = async (url, assetId = 1, options = {}) => {
    if (!wasmEngineInitialized) {
        console.warn("⚠️ WASM not ready");
        return { success: false, error: "WASM not initialized" };
    }

    console.log(`📥 Loading asset from: ${url}`);
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }

        const data = await response.arrayBuffer();
        console.log(`📦 Loaded ${data.byteLength} bytes`);

        // Parse bounds from binary header
        const bounds = updateBoundsFromBinary(data);

        // Send as AssetReady message (type 0x01)
        const MSG_TYPE_ASSET_READY = 0x01;
        const numericId = assetIdToNumeric(assetId);
        sendMessage(MSG_TYPE_ASSET_READY, new Uint8Array(data), numericId, 1);

        console.log(`✅ Asset ${assetId} (id=${numericId}) sent to renderer`);

        // Update cache
        lastLoadedAsset = { id: numericId, url, data: new Uint8Array(data) };

        // Set view mode if explicitly requested via options.
        // Otherwise, trust the WASM engine's intelligent defaults set during load_geometry:
        // - If SDF exists → defaults to SDF mode
        // - Else if Splat exists → defaults to Splat mode  
        // - Else → defaults to Mesh mode
        // This ensures the best available representation is shown by default.
        if (options.viewMode) {
            window.set_view_mode(options.viewMode);
        }
        // Note: If viewMode is not specified, WASM engine already set the appropriate default

        // Auto-fit camera to see the asset (default: true for first load)
        if (options.autoFit !== false && bounds) {
            // Small delay to let WASM process the geometry
            setTimeout(() => window.fit_to_bounds(bounds), 150);
        }

        return { success: true, bytes: data.byteLength, assetId: numericId, bounds };
    } catch (err) {
        console.error("❌ Failed to load asset:", err);
        return { success: false, error: err.message };
    }
};

// Load the test sphere from static assets (CSG: sphere - box)
window.load_sphere = () => load_asset("/static/assets/test_sphere.gve_bin", 1);

// =============================================================================
// Primitive Library - Precompiled basic shapes
// =============================================================================
const PRIMITIVES_PATH = "/static/assets/primitives/";

// Load primitive by name: "sphere", "box", "cylinder", "capsule", "torus", "cone", "plane"
window.load_primitive = (name) => {
    const assetId = 100 + ["sphere", "box", "cylinder", "capsule", "torus", "cone", "plane"].indexOf(name);
    return load_asset(`${PRIMITIVES_PATH}${name}.gve_bin`, assetId);
};

// Convenience shortcuts for each primitive
window.primitives = {
    sphere: () => load_primitive("sphere"),
    box: () => load_primitive("box"),
    cylinder: () => load_primitive("cylinder"),
    capsule: () => load_primitive("capsule"),
    torus: () => load_primitive("torus"),
    cone: () => load_primitive("cone"),
    plane: () => load_primitive("plane"),
};

/**
 * Send binary message to WASM. Unifies raw usage.
 */
export function onMessage(type, callback) {
    // TODO: Implement WASM message callback registration
    console.warn("onMessage not implemented yet");
}

export function sendMessage(type, payload, assetId = 0n, version = 0) {
    if (!wasmEngineInitialized) {
        console.warn("WASM Engine not yet initialized. Skipping message.");
        return;
    }

    // Pack message into 18-byte header [Header (18 bytes)][Payload]
    // msg_type (u8) @ 0
    // asset_id (u64) @ 1
    // version  (u32) @ 9
    // payload_size (u32) @ 13
    // reserved (u8) @ 17
    const HEADER_SIZE = 18;
    const header = new ArrayBuffer(HEADER_SIZE);
    const view = new DataView(header);

    view.setUint8(0, type);
    view.setBigUint64(1, BigInt(assetId), true);
    view.setUint32(9, version, true);
    view.setUint32(13, payload.byteLength, true);
    view.setUint8(17, 0); // reserved

    const msg = new Uint8Array(HEADER_SIZE + payload.byteLength);
    msg.set(new Uint8Array(header), 0);
    msg.set(new Uint8Array(payload), HEADER_SIZE);

    gve_wasm.handle_message(msg);
}

function sendEngineCommand(type, payload) {
    sendMessage(type, payload, 0n, 1);
}

// Internal update helper for CameraController
function sendCameraUpdate(camera) {
    // Payload: [pos_x, pos_y, pos_z, yaw, pitch] (5 * 4 bytes)
    const payload = new Float32Array([
        camera.pos[0], camera.pos[1], camera.pos[2],
        camera.yaw, camera.pitch
    ]);
    sendMessage(MSG_TYPE_UPDATE_CAMERA, new Uint8Array(payload.buffer), 0n, 1);
}

window.load_chunk = (chunkId, x, z) => {
    const buffer = new ArrayBuffer(16);
    const view = new DataView(buffer);
    view.setBigUint64(0, assetIdToNumeric(chunkId), true);
    view.setInt32(8, x, true);
    view.setInt32(12, z, true);
    sendEngineCommand(MSG_TYPE_LOAD_CHUNK, new Uint8Array(buffer));
};

window.translate_node = (nodeId, dx, dy, dz) => {
    const buffer = new ArrayBuffer(20);
    const view = new DataView(buffer);
    view.setBigUint64(0, assetIdToNumeric(nodeId), true);
    view.setFloat32(8, dx, true);
    view.setFloat32(12, dy, true);
    view.setFloat32(16, dz, true);
    sendEngineCommand(MSG_TYPE_TRANSLATE_NODE, new Uint8Array(buffer));
};

window.update_joint = (jointId, qx, qy, qz, qw) => {
    const buffer = new ArrayBuffer(24);
    const view = new DataView(buffer);
    view.setBigUint64(0, assetIdToNumeric(jointId), true);
    view.setFloat32(8, qx, true);
    view.setFloat32(12, qy, true);
    view.setFloat32(16, qz, true);
    view.setFloat32(20, qw, true);
    sendEngineCommand(MSG_TYPE_UPDATE_JOINT, new Uint8Array(buffer));
};

export async function initViewport(canvasId) {
    // Prevent duplicate initialization (WASM engine is a singleton)
    if (wasmEngineInitialized) {
        console.log(`⚠️ WASM engine already initialized, skipping re-init for ${canvasId}`);
        return;
    }

    if (engineInitializing) {
        console.log(`⚠️ Engine initialization already in progress`);
        return;
    }

    engineInitializing = true;

    await ensureWasmLoaded();
    // WebSocket is connected by studio-navigation.js at load time — don't duplicate here
    const canvas = document.getElementById(canvasId);
    if (!canvas) {
        console.error(`Canvas with id ${canvasId} not found`);
        engineInitializing = false;
        return;
    }

    const container = canvas.parentElement;

    try {
        console.log("🚀 Initializing GVE Engine for", canvasId);
        await gve_wasm.init_engine(canvasId);

        wasmEngineInitialized = true;
        window.viewportReady = true;
        engineInitializing = false;
        console.log("✅ Viewport WASM initialized");

        // Sync initial size
        const syncSize = () => {
            const rect = container.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            canvas.width = rect.width * dpr;
            canvas.height = rect.height * dpr;
            gve_wasm.resize_viewport(canvas.width, canvas.height);
        };

        let resizePending = false;
        const resizeObserver = new ResizeObserver(() => {
            if (!resizePending) {
                resizePending = true;
                requestAnimationFrame(() => {
                    syncSize();
                    resizePending = false;
                });
            }
        });
        resizeObserver.observe(container);
        syncSize();

        const start = performance.now();
        let lastTime = start;

        function frame(time) {
            const dt = (time - lastTime) / 1000;
            lastTime = time;

            if (wasmEngineInitialized) {
                gve_wasm.render_frame(dt);
            }

            requestAnimationFrame(frame);
        }

        requestAnimationFrame(frame);

        // Initialize Camera Controller
        cameraController = new CameraController(canvas, sendCameraUpdate, gve_wasm.pick_view_cube);
        cameraController.start();

    } catch (err) {
        console.error("❌ Failed to initialize WASM viewport:", err);
        engineInitializing = false;
    }
}



