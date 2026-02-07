/**
 * viewport-camera.js
 * Handles camera movement, orbit, input events, and view cube interaction.
 */

import * as gve_wasm from '../wasm/pkg/gve_wasm.js';

export class CameraController {
    constructor(canvas, sendUpdateFn) {
        this.canvas = canvas;
        this.sendUpdateFn = sendUpdateFn;

        // Camera State
        this.camera = {
            pos: [0.0, 0.0, 3.0], // x, y, z
            yaw: -Math.PI / 2,    // Look -Z
            pitch: 0.0,
            speed: 5.0,           // Units per second
            sensitivity: 0.002,
        };

        // Input State
        this.keys = {};
        this.isOrbiting = false;

        // Orbit Mode State
        this.orbitMode = {
            radius: 3.0,
            center: [0, 0, 0],  // Orbit center
            theta: 0.0,
            phi: 0.0
        };

        // Expose to window for external access (fit_to_bounds, etc.)
        window._viewportCamera = this.camera;
        window._viewportOrbit = this.orbitMode;

        // Events
        this.boundInputLoop = this.inputLoop.bind(this);
        this.lastTime = performance.now();
        this.isRunning = false;

        // Store bound handlers for cleanup
        this._handlers = {};

        this._setupEventListeners();
        this._setupWindowGlobals();
    }

    start() {
        if (!this.isRunning) {
            this.isRunning = true;
            this.lastTime = performance.now();
            requestAnimationFrame(this.boundInputLoop);
        }
    }

    stop() {
        this.isRunning = false;
    }

    /** Remove all event listeners and stop the input loop. */
    destroy() {
        this.stop();
        const h = this._handlers;
        if (h.keydown) window.removeEventListener('keydown', h.keydown);
        if (h.keyup) window.removeEventListener('keyup', h.keyup);
        if (h.mouseup) window.removeEventListener('mouseup', h.mouseup);
        if (h.mousemove) document.removeEventListener('mousemove', h.mousemove);
        if (h.contextmenu) this.canvas.removeEventListener('contextmenu', h.contextmenu);
        if (h.mousedown) this.canvas.removeEventListener('mousedown', h.mousedown);
        if (h.wheel) this.canvas.removeEventListener('wheel', h.wheel);
        this._handlers = {};
    }

    _setupEventListeners() {
        // Keyboard
        this._handlers.keydown = (e) => {
            this.keys[e.code] = true;

            // 'F' key: Fit to bounds
            if (e.code === 'KeyF' && !e.ctrlKey && !e.altKey && !e.metaKey) {
                if (document.activeElement.tagName !== 'INPUT' &&
                    document.activeElement.tagName !== 'TEXTAREA') {
                    if (window.fit_to_bounds) window.fit_to_bounds();
                    e.preventDefault();
                }
            }

            // 'Home' key: Reset
            if (e.code === 'Home') this.resetCamera();
        };
        window.addEventListener('keydown', this._handlers.keydown);

        this._handlers.keyup = (e) => this.keys[e.code] = false;
        window.addEventListener('keyup', this._handlers.keyup);

        // Mouse Interact
        this._handlers.contextmenu = (e) => e.preventDefault();
        this.canvas.addEventListener('contextmenu', this._handlers.contextmenu);

        // Mouse Down (Orbit / Pick)
        let viewCubeDrag = false;
        let dragStartX = 0;
        let dragStartY = 0;

        this._handlers.mousedown = (e) => {
            if (e.button === 2) { // Right click (Fly Mode)
                if (document.pointerLockElement !== this.canvas) {
                    this.canvas.requestPointerLock();
                }
            } else if (e.button === 0) { // Left click
                const rect = this.canvas.getBoundingClientRect();
                const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
                const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);

                // Check View Cube
                const face = gve_wasm.pick_view_cube(x, y);
                if (face !== null) {
                    viewCubeDrag = true;
                    dragStartX = e.clientX;
                    dragStartY = e.clientY;
                    this.isOrbiting = true;
                    this._updateOrbitFromCamera();
                    return;
                }

                // Scene Orbit
                this.isOrbiting = true;
                this._updateOrbitFromCamera();
            }
        };
        this.canvas.addEventListener('mousedown', this._handlers.mousedown);

        // Mouse Up
        this._handlers.mouseup = (e) => {
            if (viewCubeDrag) {
                const dragDist = Math.hypot(e.clientX - dragStartX, e.clientY - dragStartY);
                if (dragDist < 5) { // Click
                    const rect = this.canvas.getBoundingClientRect();
                    const x = (e.clientX - rect.left) * (this.canvas.width / rect.width);
                    const y = (e.clientY - rect.top) * (this.canvas.height / rect.height);
                    const face = gve_wasm.pick_view_cube(x, y);
                    if (face !== null) this.snapCameraToFace(face);
                }
                viewCubeDrag = false;
            }
            this.isOrbiting = false;
        };
        window.addEventListener('mouseup', this._handlers.mouseup);

        // Mouse Move
        this._handlers.mousemove = (e) => {
            if (document.pointerLockElement === this.canvas) {
                // Fly Mode
                this.camera.yaw -= e.movementX * this.camera.sensitivity;
                this.camera.pitch -= e.movementY * this.camera.sensitivity;

                const MAX_PITCH = Math.PI / 2 - 0.01;
                this.camera.pitch = Math.max(-MAX_PITCH, Math.min(MAX_PITCH, this.camera.pitch));

                this.sendUpdateFn(this.camera);
            } else if (this.isOrbiting) {
                // Orbit Mode
                this.orbitMode.theta -= e.movementX * this.camera.sensitivity;
                this.orbitMode.phi += e.movementY * this.camera.sensitivity;

                const MAX_PHI = Math.PI / 2 - 0.01;
                this.orbitMode.phi = Math.max(-MAX_PHI, Math.min(MAX_PHI, this.orbitMode.phi));

                this._updateCameraFromOrbit();
                this.sendUpdateFn(this.camera);
            }
        };
        document.addEventListener('mousemove', this._handlers.mousemove);

        // Wheel Zoom
        this._handlers.wheel = (e) => {
            e.preventDefault();
            const zoomSpeed = 0.5;
            const dir = e.deltaY > 0 ? -1 : 1;

            const forwardX = Math.cos(this.camera.yaw) * Math.cos(this.camera.pitch);
            const forwardY = Math.sin(this.camera.pitch);
            const forwardZ = Math.sin(this.camera.yaw) * Math.cos(this.camera.pitch);

            this.camera.pos[0] += forwardX * zoomSpeed * dir;
            this.camera.pos[1] += forwardY * zoomSpeed * dir;
            this.camera.pos[2] += forwardZ * zoomSpeed * dir;

            this.sendUpdateFn(this.camera);
        };
        this.canvas.addEventListener('wheel', this._handlers.wheel, { passive: false });

        // Viewcube Overlay Click
        document.querySelectorAll('.viewcube-overlay').forEach((overlay) => {
            overlay.addEventListener('click', (event) => {
                const button = event.target.closest('.viewcube-face');
                if (!button) return;
                const yaw = parseFloat(button.dataset.yaw || 0);
                const pitch = parseFloat(button.dataset.pitch || 0);

                // Snap using global or logic
                if (window.snap_camera_to) {
                    window.snap_camera_to(this.camera.pos[0], this.camera.pos[1], this.camera.pos[2], yaw, pitch);
                }

                // Also update local state to match snap
                // Note: snap_camera_to sends updates to WASM, but we should sync local state too
                // For now, rely on WASM being source of truth or re-sync?
                // Actually, viewport.js snap_camera_to often updates _viewportCamera. 

                document.querySelectorAll('.viewcube-face').forEach(btn => {
                    btn.classList.toggle('active', btn.dataset.face === button.dataset.face);
                });
            });
        });
    }

    _updateOrbitFromCamera() {
        const dx = this.camera.pos[0] - this.orbitMode.center[0];
        const dy = this.camera.pos[1] - this.orbitMode.center[1];
        const dz = this.camera.pos[2] - this.orbitMode.center[2];
        this.orbitMode.radius = Math.sqrt(dx * dx + dy * dy + dz * dz);
        this.orbitMode.theta = Math.atan2(dx, dz);
        this.orbitMode.phi = Math.asin(dy / this.orbitMode.radius);
    }

    _updateCameraFromOrbit() {
        const r = this.orbitMode.radius;
        const theta = this.orbitMode.theta;
        const phi = this.orbitMode.phi;
        const cx = this.orbitMode.center[0];
        const cy = this.orbitMode.center[1];
        const cz = this.orbitMode.center[2];

        this.camera.pos[0] = cx + r * Math.cos(theta) * Math.cos(phi);
        this.camera.pos[1] = cy + r * Math.sin(phi);
        this.camera.pos[2] = cz + r * Math.sin(theta) * Math.cos(phi);

        this.camera.yaw = theta + Math.PI;
        this.camera.pitch = -phi;
    }

    inputLoop(time) {
        if (!this.isRunning) return;

        const dt = (time - this.lastTime) / 1000;
        this.lastTime = time;

        if (document.pointerLockElement === this.canvas) {
            this._processKeyboardMovement(dt);
        }

        requestAnimationFrame(this.boundInputLoop);
    }

    _processKeyboardMovement(dt) {
        let moved = false;
        const forwardX = Math.cos(this.camera.yaw);
        const forwardZ = Math.sin(this.camera.yaw);
        const rightX = -forwardZ;
        const rightZ = forwardX;

        const moveSpeed = this.camera.speed * dt * (this.keys['ShiftLeft'] ? 2.0 : 1.0);

        if (this.keys['KeyW']) { this.camera.pos[0] += forwardX * moveSpeed; this.camera.pos[2] += forwardZ * moveSpeed; moved = true; }
        if (this.keys['KeyS']) { this.camera.pos[0] -= forwardX * moveSpeed; this.camera.pos[2] -= forwardZ * moveSpeed; moved = true; }
        if (this.keys['KeyA']) { this.camera.pos[0] -= rightX * moveSpeed; this.camera.pos[2] -= rightZ * moveSpeed; moved = true; }
        if (this.keys['KeyD']) { this.camera.pos[0] += rightX * moveSpeed; this.camera.pos[2] += rightZ * moveSpeed; moved = true; }
        if (this.keys['KeyE']) { this.camera.pos[1] += moveSpeed; moved = true; }
        if (this.keys['KeyQ']) { this.camera.pos[1] -= moveSpeed; moved = true; }

        if (moved) this.sendUpdateFn(this.camera);
    }

    resetCamera() {
        this.camera.pos = [0, 0, 3];
        this.camera.yaw = -Math.PI / 2;
        this.camera.pitch = 0;
        this.orbitMode.radius = 3.0;
        this.orbitMode.center = [0, 0, 0];
        this.orbitMode.theta = 0;
        this.orbitMode.phi = 0;
        this.sendUpdateFn(this.camera);
    }

    snapCameraToFace(face) {
        const r = this.orbitMode.radius;
        let pos = [0, 0, 0];

        switch (face) {
            case 0: pos = [r, 0, 0]; break; // Right
            case 1: pos = [-r, 0, 0]; break; // Left
            case 2: pos = [0, r, 0]; break; // Top
            case 3: pos = [0, -r, 0]; break; // Bottom
            case 4: pos = [0, 0, r]; break; // Front
            case 5: pos = [0, 0, -r]; break; // Back
        }

        this.camera.pos = pos;
        this.orbitMode.theta = Math.atan2(pos[2], pos[0]);
        this.orbitMode.phi = Math.asin(pos[1] / r);
        this.camera.yaw = this.orbitMode.theta + Math.PI;
        this.camera.pitch = -this.orbitMode.phi;

        this.sendUpdateFn(this.camera);
        console.log("Snapped to face:", face);
    }

    _setupWindowGlobals() {
        // Expose helpers that were previously inline or globally available
        // Note: fit_to_bounds is defined in viewport.js and updates _viewportCamera
    }
}
