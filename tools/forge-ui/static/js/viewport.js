/**
 * viewport.js
 * Handles the WASM viewport initialization and rendering loop.
 */

// Namespace import so missing exports (e.g. get_scene_snapshot in older WASM builds) don't break load
import init, * as gve_wasm from '../wasm/pkg/gve_wasm.js';
import { connectWebSocket } from './events.js';

const wasmGetSceneSnapshot = gve_wasm.get_scene_snapshot || (() => new Uint8Array(0));

const MSG_TYPE_LOAD_CHUNK = 0x30;
const MSG_TYPE_TRANSLATE_NODE = 0x31;
const MSG_TYPE_UPDATE_JOINT = 0x32;



let wasmEngineInitialized = false;
let wasmModuleLoaded = false;
let engineInitializing = false;  // Prevent duplicate initialization
window.viewportReady = false;

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
    if (!wasmEngineInitialized) {
        console.warn("WASM not ready");
        return;
    }
    gve_wasm.clear_sdf();
};

window.snap_camera_to = (x, y, z, yaw, pitch) => {
    if (!wasmEngineInitialized) {
        console.warn("WASM not ready");
        return;
    }
    gve_wasm.snap_camera_to(x, y, z, yaw, pitch);
};

window.clear_viewport = () => {
    if (!wasmEngineInitialized) {
        console.warn("WASM not ready");
        return;
    }
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
window.load_asset = async (url, assetId = 1) => {
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

        // Send as AssetReady message (type 0x01)
        const MSG_TYPE_ASSET_READY = 0x01;
        const numericId = assetIdToNumeric(assetId);
        sendMessageRaw(MSG_TYPE_ASSET_READY, new Uint8Array(data), numericId, 1);

        console.log(`✅ Asset ${assetId} (id=${numericId}) sent to renderer`);
        return { success: true, bytes: data.byteLength, assetId: numericId };
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

// Internal: send raw binary message to WASM
function sendMessageRaw(type, payload, assetId, version) {
    const HEADER_SIZE = 18;
    const header = new ArrayBuffer(HEADER_SIZE);
    const view = new DataView(header);

    view.setUint8(0, type);
    view.setBigUint64(1, assetId, true);
    view.setUint32(9, version, true);
    view.setUint32(13, payload.byteLength, true);
    view.setUint8(17, 0);

    const msg = new Uint8Array(HEADER_SIZE + payload.byteLength);
    msg.set(new Uint8Array(header), 0);
    msg.set(payload, HEADER_SIZE);

    gve_wasm.handle_message(msg);
}

function sendEngineCommand(type, payload) {
    const zeroAsset = 0n;
    sendMessageRaw(type, payload, zeroAsset, 1);
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
    // Start WebSocket connection
    connectWebSocket();
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

        const resizeObserver = new ResizeObserver(() => {
            syncSize();
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

    } catch (err) {
        console.error("❌ Failed to initialize WASM viewport:", err);
        engineInitializing = false;
    }

    // =========================================================
    // Camera Controller
    // =========================================================
    const camera = {
        pos: [0.0, 0.0, 3.0], // x, y, z
        yaw: -Math.PI / 2,    // Look -Z
        pitch: 0.0,
        speed: 5.0,           // Units per second
        sensitivity: 0.002,
        keys: {}
    };

    const keys = {};

    let isOrbiting = false;
    let orbitMode = {
        radius: 3.0,
        theta: 0.0,
        phi: 0.0
    };

    window.addEventListener('keydown', (e) => keys[e.code] = true);
    window.addEventListener('keyup', (e) => keys[e.code] = false);

    // Prevent context menu on right click
    canvas.addEventListener('contextmenu', e => e.preventDefault());

    // Mouse Look & Orbit & View Cube
    let viewCubeDrag = false;
    let dragStartX = 0;
    let dragStartY = 0;

    canvas.addEventListener('mousedown', (e) => {
        if (e.button === 2) { // Right click (Fly Mode)
            if (document.pointerLockElement !== canvas) {
                canvas.requestPointerLock();
            }
        } else if (e.button === 0) { // Left click
            const rect = canvas.getBoundingClientRect();
            const x = (e.clientX - rect.left) * (canvas.width / rect.width);
            const y = (e.clientY - rect.top) * (canvas.height / rect.height);

            // Check View Cube hit
            const face = gve_wasm.pick_view_cube(x, y);
            if (face !== null) {
                viewCubeDrag = true;
                dragStartX = e.clientX;
                dragStartY = e.clientY;
                // Treat as orbit start
                isOrbiting = true;

                // Calculate orbit params
                const dx = camera.pos[0];
                const dy = camera.pos[1];
                const dz = camera.pos[2];
                orbitMode.radius = Math.sqrt(dx * dx + dy * dy + dz * dz);
                orbitMode.theta = Math.atan2(dx, dz);
                orbitMode.phi = Math.asin(dy / orbitMode.radius);
                return;
            }

            // Normal Scene Orbit
            isOrbiting = true;
            const dx = camera.pos[0];
            const dy = camera.pos[1];
            const dz = camera.pos[2];
            orbitMode.radius = Math.sqrt(dx * dx + dy * dy + dz * dz);
            orbitMode.theta = Math.atan2(dx, dz);
            orbitMode.phi = Math.asin(dy / orbitMode.radius);
        }
    });

    window.addEventListener('mouseup', (e) => {
        if (viewCubeDrag) {
            const dragDist = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
            if (dragDist < 5) {
                // Click (Snap)
                const rect = canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (canvas.height / rect.height);
                const face = gve_wasm.pick_view_cube(x, y);
                if (face !== null) {
                    snapCameraToFace(face);
                }
            }
            viewCubeDrag = false;
        }
        isOrbiting = false;
    });

    function snapCameraToFace(face) {
        // 0=Right, 1=Left, 2=Top, 3=Bottom, 4=Front, 5=Back
        const r = orbitMode.radius;
        let yaw = 0;
        let pitch = 0;

        switch (face) {
            case 0: // Right (+X)
                pos = [r, 0, 0]; break;
            case 1: // Left (-X)
                pos = [-r, 0, 0]; break;
            case 2: // Top (+Y)
                pos = [0, r, 0]; break;
            case 3: // Bottom (-Y)
                pos = [0, -r, 0]; break;
            case 4: // Front (+Z)
                pos = [0, 0, r]; break;
            case 5: // Back (-Z)
                pos = [0, 0, -r]; break;
        }

        camera.pos = pos;

        // Recalculate orbit params (Standard Math: x=r*cos(theta), z=r*sin(theta))
        // So theta = atan2(z, x)
        orbitMode.theta = Math.atan2(pos[2], pos[0]);
        orbitMode.phi = Math.asin(pos[1] / r);

        // Yaw needs to look AT origin.
        // Pos direction is (cos t, sin t). Look direction is opposite: (-cos t, -sin t).
        // (cos(yaw), sin(yaw)) = (-cos t, -sin t) = (cos(t+PI), sin(t+PI))
        // So Yaw = theta + PI.
        camera.yaw = orbitMode.theta + Math.PI;
        camera.pitch = -orbitMode.phi;

        sendCameraUpdate();
        console.log("Snapped to face:", face);
    }



    document.addEventListener('mousemove', (e) => {
        // Fly Mode (Pointer Lock)
        if (document.pointerLockElement === canvas) {
            camera.yaw -= e.movementX * camera.sensitivity;
            camera.pitch -= e.movementY * camera.sensitivity;

            // Clamp pitch
            const MAX_PITCH = Math.PI / 2 - 0.01;
            camera.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, camera.pitch));

            sendCameraUpdate();
        }
        // Orbit Mode (Left Drag)
        else if (isOrbiting) {
            // Update spherical angles
            orbitMode.theta -= e.movementX * camera.sensitivity;
            orbitMode.phi += e.movementY * camera.sensitivity;

            // Clamp phi (elevation)
            const MAX_PHI = Math.PI / 2 - 0.01;
            orbitMode.phi = Math.max(-MAX_PHI, Math.min(MAX_PHI, orbitMode.phi));

            // Convert back to Cartesian
            const r = orbitMode.radius;
            const theta = orbitMode.theta;
            const phi = orbitMode.phi;

            // Pipeline: x=cos(yaw), z=sin(yaw)
            // Orbit: x=cos(theta), z=sin(theta)
            camera.pos[0] = r * Math.cos(theta) * Math.cos(phi);
            camera.pos[1] = r * Math.sin(phi);
            camera.pos[2] = r * Math.sin(theta) * Math.cos(phi);

            // Look at origin: Yaw = theta + PI
            camera.yaw = theta + Math.PI;
            camera.pitch = -phi;

            sendCameraUpdate();
        }
    });

    document.addEventListener('pointerlockchange', () => {
        if (document.pointerLockElement !== canvas) {
            // Unlocked
        }
    });

    // Zoom (Wheel)
    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        // Move along view vector
        const zoomSpeed = 0.5;
        const dir = e.deltaY > 0 ? -1 : 1;

        const forwardX = Math.cos(camera.yaw) * Math.cos(camera.pitch);
        const forwardY = Math.sin(camera.pitch);
        const forwardZ = Math.sin(camera.yaw) * Math.cos(camera.pitch);

        camera.pos[0] += forwardX * zoomSpeed * dir;
        camera.pos[1] += forwardY * zoomSpeed * dir;
        camera.pos[2] += forwardZ * zoomSpeed * dir;

        sendCameraUpdate();
    }, { passive: false });

    function sendCameraUpdate() {
        if (!wasmEngineInitialized) return;

        // Packet: Type(1) + AssetID(8) + Ver(4) + Size(4) + Pad(1) + Payload(20)
        // Payload: [pos_x, pos_y, pos_z, yaw, pitch] (5 * 4 bytes)
        const payload = new Float32Array([
            camera.pos[0], camera.pos[1], camera.pos[2],
            camera.yaw, camera.pitch
        ]);

        const UPDATE_CAMERA = 0x20;
        sendMessageRaw(UPDATE_CAMERA, new Uint8Array(payload.buffer), 0n, 1);
    }

    // Hook into existing frame loop for smooth movement
    const originalRequestAnimationFrame = window.requestAnimationFrame;
    // We already have a frame loop inside initViewport, let's inject into it?
    // Actually, initViewport defines 'frame' and starts it.
    // I can't easily inject into the closure 'frame' function defined above 
    // without replacing the whole block.
    // BUT, I can run my own loop or just handle movement in a `setInterval` or 
    // add a hook if I redefine `render_frame` export? No.

    // Changing approach: I'll overwrite the 'frame' function in the text replacement
    // or just run a parallel loop for input processing.
    // Parallel loop is fine for input physics.

    let lastTime = performance.now();
    function inputLoop(time) {
        const dt = (time - lastTime) / 1000;
        lastTime = time;

        if (document.pointerLockElement === canvas) {
            let moved = false;
            const forwardX = Math.cos(camera.yaw); // Planar forward for walking
            const forwardZ = Math.sin(camera.yaw);
            const rightX = -forwardZ;
            const rightZ = forwardX;

            const moveSpeed = camera.speed * dt * (keys['ShiftLeft'] ? 2.0 : 1.0);

            if (keys['KeyW']) {
                camera.pos[0] += forwardX * moveSpeed;
                camera.pos[2] += forwardZ * moveSpeed;
                moved = true;
            }
            if (keys['KeyS']) {
                camera.pos[0] -= forwardX * moveSpeed;
                camera.pos[2] -= forwardZ * moveSpeed;
                moved = true;
            }
            if (keys['KeyA']) {
                camera.pos[0] -= rightX * moveSpeed;
                camera.pos[2] -= rightZ * moveSpeed;
                moved = true;
            }
            if (keys['KeyD']) {
                camera.pos[0] += rightX * moveSpeed;
                camera.pos[2] += rightZ * moveSpeed;
                moved = true;
            }
            if (keys['KeyE']) { // Up
                camera.pos[1] += moveSpeed;
                moved = true;
            }
            if (keys['KeyQ']) { // Down
                camera.pos[1] -= moveSpeed;
                moved = true;
            }

            if (moved) sendCameraUpdate();
        }

    requestAnimationFrame(inputLoop);

    function initViewcubeOverlay() {
        document.querySelectorAll('.viewcube-overlay').forEach((overlay) => {
            overlay.addEventListener('click', (event) => {
                const button = event.target.closest('.viewcube-face');
                if (!button) return;
                const yaw = parseFloat(button.dataset.yaw || 0);
                const pitch = parseFloat(button.dataset.pitch || 0);
                if (window.snap_camera_to) {
                    window.snap_camera_to(camera.pos[0], camera.pos[1], camera.pos[2], yaw, pitch);
                }
                setViewcubeActive(button.dataset.face);
            });
        });
    }

    function setViewcubeActive(face) {
        document.querySelectorAll('.viewcube-face').forEach((btn) => {
            btn.classList.toggle('active', btn.dataset.face === face);
        });
    }

    initViewcubeOverlay();
}
    requestAnimationFrame(inputLoop);

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


export function onMessage(type, callback) {
    console.log("Registering WASM listener for:", type);
    // Future: Handle messages FROM WASM to JS
}
