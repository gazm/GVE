/**
 * World Editor - Chunk map (from API), terrain tools, entity library.
 * Phase 2: Chunk grid from GET /api/world/partials/chunks; New World POST; Bake PATCH.
 */

const CHUNK_STATE_LABELS = {
    unprocessed: 'Unprocessed',
    analyzing: 'Analyzing',
    baking: 'Baking',
    finalized: 'Finalized',
};

const BRUSHES = ['Raise', 'Lower', 'Smooth', 'Flatten', 'Noise', 'Erode', 'Paint'];
const MATERIALS = [
    { id: 'grass', name: 'Grass', color: '#3a6b2e' },
    { id: 'dirt', name: 'Dirt', color: '#5c4033' },
    { id: 'rock', name: 'Rock', color: '#4a4a4a' },
    { id: 'sand', name: 'Sand', color: '#c4a574' },
    { id: 'snow', name: 'Snow', color: '#e8e8e8' },
];
const ENTITIES = [
    { id: 'tree', name: 'Tree', icon: '🌲' },
    { id: 'crate', name: 'Crate', icon: '📦' },
    { id: 'car', name: 'Car', icon: '🚗' },
    { id: 'lamp', name: 'Lamp', icon: '🪔' },
];

let selectedChunkIds = new Set();
let activeBrush = 'Raise';
let brushSize = 4.0;
let brushStrength = 0.5;
let activeMaterialId = 'grass';
let worldEditorInitialized = false;

function refreshChunkGrid() {
    const wrapper = document.getElementById('world-chunk-grid-wrapper');
    if (wrapper && typeof htmx !== 'undefined') {
        htmx.ajax('GET', '/api/world/partials/chunks', { target: '#world-chunk-grid-wrapper', swap: 'innerHTML' });
    }
}

function updateWorldStripSelection() {
    const el = document.getElementById('world-strip-selection');
    if (!el) return;
    if (selectedChunkIds.size === 0) {
        el.textContent = 'No chunks selected';
    } else {
        const list = Array.from(selectedChunkIds).sort().join(', ');
        el.innerHTML = `Selected: <span class="highlight">${list}</span>`;
    }
}

function applySelectionToChunkCells() {
    document.querySelectorAll('#world-panel-left .chunk-cell').forEach((el) => {
        const id = el.dataset.chunkId;
        el.classList.toggle('selected', selectedChunkIds.has(id));
    });
}

function setupChunkDelegation() {
    const panel = document.getElementById('world-panel-left');
    if (!panel) return;
    panel.addEventListener('click', (e) => {
        const cell = e.target.closest('.chunk-cell');
        if (!cell) return;
        e.preventDefault();
        const id = cell.dataset.chunkId;
        if (e.shiftKey) {
            selectedChunkIds.has(id) ? selectedChunkIds.delete(id) : selectedChunkIds.add(id);
        } else {
            selectedChunkIds.clear();
            selectedChunkIds.add(id);
        }
        applySelectionToChunkCells();
        updateWorldStripSelection();
    });
}

function renderBrushToolbar(container) {
    if (!container) return;
    container.innerHTML = BRUSHES.map(
        (name) =>
            `<button type="button" class="brush-btn ${activeBrush === name ? 'active' : ''}" data-brush="${name}">${name}</button>`
    ).join('');

    container.querySelectorAll('.brush-btn').forEach((el) => {
        el.addEventListener('click', () => {
            activeBrush = el.dataset.brush;
            container.querySelectorAll('.brush-btn').forEach((b) => b.classList.remove('active'));
            el.classList.add('active');
        });
    });
}

function renderMaterials(container) {
    if (!container) return;
    container.innerHTML = MATERIALS.map(
        (m) => `
        <button type="button" class="world-material-item ${activeMaterialId === m.id ? 'active' : ''}" data-material-id="${m.id}">
            <span class="mat-dot" style="background:${m.color}"></span>
            <span>${m.name}</span>
        </button>`
    ).join('');

    container.querySelectorAll('.world-material-item').forEach((el) => {
        el.addEventListener('click', () => {
            activeMaterialId = el.dataset.materialId;
            container.querySelectorAll('.world-material-item').forEach((b) => b.classList.remove('active'));
            el.classList.add('active');
        });
    });
}

function renderEntityGrid(container) {
    if (!container) return;
    container.innerHTML =
        ENTITIES.map(
            (e) => `
        <div class="world-entity-card" data-entity-id="${e.id}" data-entity-name="${e.name}" draggable="true">
            <span class="entity-icon">${e.icon}</span>
            <span>${e.name}</span>
        </div>`
        ).join('') + '<p class="world-entity-hint">Drag to viewport to place (coming soon)</p>';

    container.querySelectorAll('.world-entity-card').forEach((el) => {
        el.addEventListener('click', () => {
            if (typeof window.logOutput === 'function') {
                window.logOutput(`Place entity: ${el.dataset.entityName} (viewport placement coming soon)`, 'info');
            }
        });
        el.addEventListener('dragstart', (ev) => {
            ev.dataTransfer.setData('text/plain', el.dataset.entityId);
            ev.dataTransfer.effectAllowed = 'copy';
        });
    });
}

function initSliders() {
    const sizeInput = document.getElementById('world-brush-size');
    const strengthInput = document.getElementById('world-brush-strength');
    if (sizeInput) {
        sizeInput.value = brushSize;
        sizeInput.addEventListener('input', () => {
            brushSize = parseFloat(sizeInput.value);
            const label = document.getElementById('world-brush-size-value');
            if (label) label.textContent = `${brushSize.toFixed(1)}m`;
        });
        const sizeVal = document.getElementById('world-brush-size-value');
        if (sizeVal) sizeVal.textContent = `${brushSize.toFixed(1)}m`;
    }
    if (strengthInput) {
        strengthInput.value = brushStrength;
        strengthInput.addEventListener('input', () => {
            brushStrength = parseFloat(strengthInput.value);
            const label = document.getElementById('world-brush-strength-value');
            if (label) label.textContent = brushStrength.toFixed(2);
        });
        const strengthVal = document.getElementById('world-brush-strength-value');
        if (strengthVal) strengthVal.textContent = brushStrength.toFixed(2);
    }
}

async function createNewWorld() {
    try {
        const res = await fetch('/api/world', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: 'Untitled World', grid_rows: 4, grid_cols: 4 }),
        });
        if (!res.ok) throw new Error(await res.text());
        selectedChunkIds.clear();
        updateWorldStripSelection();
        refreshChunkGrid();
        if (typeof window.logOutput === 'function') {
            window.logOutput('World created', 'success');
        }
    } catch (err) {
        if (typeof window.logOutput === 'function') {
            window.logOutput(`New World failed: ${err.message}`, 'error');
        }
    }
}

async function bakeSelectedChunks() {
    if (selectedChunkIds.size === 0) {
        if (typeof window.logOutput === 'function') {
            window.logOutput('Select chunks first', 'warning');
        }
        return;
    }
    try {
        for (const id of selectedChunkIds) {
            const res = await fetch(`/api/world/chunks/${id}`, {
                method: 'PATCH',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ state: 'baking' }),
            });
            if (!res.ok) throw new Error(await res.text());
        }
        refreshChunkGrid();
        if (typeof window.logOutput === 'function') {
            window.logOutput(`Bake queued for ${selectedChunkIds.size} chunk(s)`, 'success');
        }
    } catch (err) {
        if (typeof window.logOutput === 'function') {
            window.logOutput(`Bake failed: ${err.message}`, 'error');
        }
    }
}

function initWorldEditor() {
    if (worldEditorInitialized) return;
    worldEditorInitialized = true;

    setupChunkDelegation();
    document.getElementById('world-panel-left')?.addEventListener('htmx:afterSwap', (ev) => {
        if (ev.detail?.target?.id === 'world-chunk-grid-wrapper') applySelectionToChunkCells();
    });
    renderBrushToolbar(document.getElementById('world-brush-toolbar'));
    renderMaterials(document.getElementById('world-materials-list'));
    renderEntityGrid(document.getElementById('world-entity-grid'));
    updateWorldStripSelection();
    initSliders();

    const newBtn = document.getElementById('world-new-btn');
    if (newBtn) newBtn.addEventListener('click', createNewWorld);

    const importBtn = document.getElementById('world-import-btn');
    if (importBtn) {
        importBtn.addEventListener('click', () => {
            if (typeof window.logOutput === 'function') {
                window.logOutput('Import Real-World Location (modal coming soon)', 'info');
            }
        });
    }

    const bakeBtn = document.getElementById('world-strip-bake');
    const artistBtn = document.getElementById('world-strip-artist');
    const translateBtn = document.getElementById('world-strip-translate');
    if (bakeBtn) bakeBtn.addEventListener('click', bakeSelectedChunks);
    if (artistBtn) {
        artistBtn.addEventListener('click', () => {
            if (typeof window.logOutput === 'function') {
                window.logOutput('Artist Pass (AI polish coming later)', 'info');
            }
        });
    }
    if (translateBtn) {
        translateBtn.addEventListener('click', () => {
            if (selectedChunkIds.size === 0) {
                if (typeof window.logOutput === 'function') {
                    window.logOutput('Select a chunk before translating', 'warning');
                }
                return;
            }
            const delta = 0.15;
            selectedChunkIds.forEach((id) => {
                window.translate_node(id, delta, 0.0, 0.0);
            });
            if (typeof window.logOutput === 'function') {
                window.logOutput('Issued translate command to selected chunk(s)', 'info');
            }
        });
    }
}

function onWorldPageShow() {
    initWorldEditor();
    refreshChunkGrid();
}

export { initWorldEditor, onWorldPageShow };
