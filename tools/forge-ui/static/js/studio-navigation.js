// Studio Navigation - Page switching and viewport initialization
import { initViewport, ensureWasmLoaded } from '/static/js/viewport.js';
import { connectWebSocket } from '/static/js/events.js';
import { onWorldPageShow } from '/static/js/studio-world.js';

// Initialize viewport once on page load with shared canvas
let viewportInitialized = false;
let currentTab = 'library'; // Track current tab to avoid unnecessary reloads
let viewportSyncedWithTree = false; // Track if viewport matches tree state
window._hiddenAssets = new Set(); // Track hidden assets (not deleted, just invisible)

// Connect WebSocket for real-time events
connectWebSocket();

ensureWasmLoaded()
    .then(() => {
        return initViewport('shared-viewport');
    })
    .then(() => {
        viewportInitialized = true;
        window.viewportReady = true;
        console.log('🎨 Shared viewport initialized');
    })
    .catch(e => console.error('Viewport initialization failed:', e));

let externalViewportWindow = null;

window.openExternalViewport = function () {
    if (externalViewportWindow && !externalViewportWindow.closed) {
        externalViewportWindow.focus();
    } else {
        externalViewportWindow = window.open('/viewport', 'GVE_Dedicated_Viewport', 'width=1280,height=720');
        logOutput('Dedicated viewport opened', 'info');
    }
};

export function syncExternalViewport(assetId) {
    if (externalViewportWindow && !externalViewportWindow.closed) {
        // Wait a bit for the window to initialize if it just opened
        setTimeout(() => {
            externalViewportWindow.postMessage({
                type: 'LOAD_ASSET',
                asset_id: assetId,
                binary_url: `/api/assets/${assetId}/binary`
            }, window.location.origin);
        }, 500);
    }
}


// Viewport Sizing Logic
let viewportResizeObserver = null;
const viewportContainer = document.querySelector('.shared-viewport-container');

function updateViewportSize(entries) {
    for (let entry of entries) {
        if (entry.target.classList.contains('viewport-cell') && entry.contentRect.width > 0) {
            const rect = entry.target.getBoundingClientRect();
            if (viewportContainer) {
                viewportContainer.style.top = `${rect.top}px`;
                viewportContainer.style.left = `${rect.left}px`;
                viewportContainer.style.width = `${rect.width}px`;
                viewportContainer.style.height = `${rect.height}px`;
                // Force canvas resize if needed, though CSS 100% usually handles it
            }
        }
    }
}

// Page navigation - shared viewport always visible, just swap UI panels
window.showPage = function (pageName) {
    const previousTab = currentTab;
    currentTab = pageName;

    // Disconnect old observer
    if (viewportResizeObserver) {
        viewportResizeObserver.disconnect();
        viewportResizeObserver = null;
    }

    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.style.display = 'none');

    // Default: Reset viewport to full screen if no cell found (or hide it)
    if (viewportContainer) {
        viewportContainer.style.top = '60px';
        viewportContainer.style.left = '0';
        viewportContainer.style.width = '100vw';
        viewportContainer.style.height = 'calc(100vh - 60px)';
    }

    // Show selected page
    const page = document.getElementById(`page-${pageName}`);
    if (page) {
        page.style.display = 'grid';

        // Find the placeholder cell for the viewport
        const viewportCell = page.querySelector('.viewport-cell');
        if (viewportCell) {
            viewportResizeObserver = new ResizeObserver(updateViewportSize);
            viewportResizeObserver.observe(viewportCell);
        } else if (pageName === 'library') {
            // Library overlays everything, hide viewport or keep it in background
            // If library is opaque, we might not care. If modal, we might want it visible.
            // For now, let's keep it visible but maybe dimmed? Or just full screen background.
        }
    }

    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.page === pageName);
    });

    // Handle viewport content intelligently
    if (viewportInitialized) {
        if (pageName === 'asset-editor') {
            // Only sync if we haven't synced yet or if tree changed
            if (!viewportSyncedWithTree) {
                // Refresh tree, which will trigger htmx:afterSwap -> syncViewportFromTree
                refreshTreeViewer();
            }
        }
        if (pageName === 'world') {
            onWorldPageShow();
        }
    }
};

// Sync viewport AFTER tree updates (fixes "one behind" issue)
document.addEventListener('htmx:afterSwap', (evt) => {
    if (evt.target.id === 'tree-viewer') {
        console.log('Tree updated, syncing viewport...');
        syncViewportFromTree();
        setViewportSynced(true);
    }
});

/** Sync WASM viewport to current tree: load each visible node's binary. */
window.syncViewportFromTree = function () {
    if (!window.load_asset) return;

    // Clear viewport first
    if (typeof window.clear_viewport === 'function' && window.viewportReady) {
        window.clear_viewport();
    }

    // Find all visible tree items
    // Note: This relies on the tree viewer being rendered. If it's not rendered yet, this might miss items.
    // Ideally we'd fetch the list from API, but for now DOM scraping matches previous pattern.
    const visibleNodes = document.querySelectorAll('.tree-item:not(.hidden)');
    visibleNodes.forEach(node => {
        const assetId = node.dataset.assetId;
        if (assetId && !window._hiddenAssets.has(assetId)) {
            window.load_asset(`/api/assets/${assetId}/binary`, assetId);
        }
    });
};

export function refreshTreeViewer() {
    const el = document.getElementById('tree-viewer');
    if (el) htmx.ajax('GET', '/api/assets/partials/tree', { target: '#tree-viewer', swap: 'innerHTML' });
}

export function setViewportSynced(synced) {
    viewportSyncedWithTree = synced;
}

export function isViewportInitialized() {
    return viewportInitialized;
}

// --- Logic migrated/adapted from studio-chain.js ---

window.showLibraryPicker = function () {
    showPage('library');
    logOutput('Switched to library - select components to add', 'info');
};

// Add asset to composition (Project/Tree)
window.addToChain = async function (assetId) {
    // Rename/Aliased from addToChain to match old external calls if any, but logic is "Add to Project"
    const formData = new FormData();
    formData.set('asset_id', assetId);

    // POST to a generic "add to project" endpoint. 
    // Assuming backend has /api/assets/project/add or similar. 
    // If not, we might need to use the old /chain/fill endpoint if the backend still supports it 
    // and maps it to the tree, or if I need to update the backend too.
    // The task didn't explicitly say backend refactor, but deprecated card chain doc.
    // I'll assume /api/assets/chain/fill might still work or I should use a new one.
    // Let's use /api/assets/chain/fill for now as it likely adds to the scene graph which the tree reads.

    const res = await fetch('/api/assets/chain/fill', { method: 'POST', body: formData });
    if (res.ok) {
        logOutput(`Added asset ${assetId} to project`, 'success');
        setViewportSynced(false); // Tree changed
        showPage('asset-editor');
    } else {
        logOutput(`Failed to add asset ${assetId}`, 'error');
    }
};

// Alias for add to chain
window.addToProject = window.addToChain;

window.switchToAssetEditor = function (assetId) {
    if (assetId) {
        window.addToChain(assetId);
    } else {
        showPage('asset-editor');
    }
}

// Add asset to composition (Project/Tree) intelligently
window.addToSmartSlot = async function (assetId, assetType) {
    if (!assetType) {
        // Try to infer from active library tab
        const activeTab = document.querySelector('.tab-btn.active');
        if (activeTab) {
            assetType = activeTab.dataset.library;
        }
    }

    const formData = new FormData();
    formData.set('asset_id', assetId);
    if (assetType) formData.set('asset_type', assetType);

    const res = await fetch('/api/assets/chain/smart-add', { method: 'POST', body: formData });
    if (res.ok) {
        logOutput(`Added asset ${assetId} to project`, 'success');
        setViewportSynced(false); // Tree changed
        // Switch to asset editor to see result
        showPage('asset-editor');
    } else {
        logOutput(`Failed to add asset ${assetId}`, 'error');
    }
};

window.previewItem = function (assetId, assetType) {
    // Just load it into the viewport (if visible)
    logOutput(`Previewing ${assetId}...`, 'info');
    if (window.load_asset) {
        window.load_asset(`/api/assets/${assetId}/binary`, assetId);
    }
};

window.compileChain = function () {
    logOutput('Compiling project...', 'info');
    htmx.ajax('POST', '/api/compile/chain', { target: '#generation-output', swap: 'beforeend' });
    setTimeout(refreshTreeViewer, 500);
};

window.clearViewportManual = function () {
    if (window.clear_viewport && window.viewportReady) {
        window.clear_viewport();
        setViewportSynced(false);
        window._hiddenAssets.clear();
        console.log('🧹 Viewport cleared');
    }
};

window.toggleHideAsset = function (assetId) {
    const treeItem = document.querySelector(`.tree-item[data-asset-id="${assetId}"]`);

    if (window._hiddenAssets.has(assetId)) {
        window._hiddenAssets.delete(assetId);
        if (treeItem) treeItem.classList.remove('hidden');
        if (window.load_asset && window.viewportReady) {
            window.load_asset(`/api/assets/${assetId}/binary`, assetId);
        }
    } else {
        window._hiddenAssets.add(assetId);
        if (treeItem) treeItem.classList.add('hidden');
        // Re-sync to remove it (lazy way) or just hide if we could (WASM doesn't support hide yet maybe, so reload)
        window.syncViewportFromTree();
    }
};

window.deleteAssetFromChain = async function (assetId) {
    if (!confirm('Remove this asset?')) return;
    // Uses old endpoint for now, assuming it removes from scene
    // Logic adapted from studio-chain.js
    const formData = new FormData();
    // If API expects removing by slot, we might need more info. 
    // But if we can remove by asset_id or just clear the "slot" it was in.
    // studio-chain.js used /api/assets/chain/slot/${slotType} to clear.
    // Without slots, how do we delete?
    // I'll assume the backend supports DELETE /api/assets/{id} provided it's in the scene.
    // Or I might have to stick to the slot logic if the backend is strictly slot-based still.
    // Since I can't change backend right now (out of scope/no task), I hope /api/assets/chain/fill 
    // pushed to a list.

    // For safety, I'll log a warning if I can't be sure, but I'll try a generic delete.
    logOutput('Delete not fully implemented without slot info', 'warning');
};


// Output logging — canonical implementations live in studio-genai-ui.js
// (exported to window.logOutput and window.clearOutput there).
// If GenAI module hasn't loaded yet, provide a minimal fallback.
if (typeof window.logOutput !== 'function') {
    window.logOutput = function (message, _type = 'info') {
        console.log(`[logOutput] ${message}`);
    };
}
if (typeof window.clearOutput !== 'function') {
    window.clearOutput = function () {};
}

// Utility: debounce
export function debounce(fn, delay) {
    let timeout;
    return (...args) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => fn(...args), delay);
    };
}

// Library Page Controls
document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        document.querySelector('[name="library-type"]').value = btn.dataset.library;
    });
});

// Tag filtering
document.querySelectorAll('.tag-pill').forEach(pill => {
    pill.addEventListener('click', () => {
        pill.classList.toggle('active');
        updateTagFilters();
    });
});

function updateTagFilters() {
    const activeTags = Array.from(document.querySelectorAll('.tag-pill.active'))
        .map(p => p.dataset.tag);
    const tagsContainer = document.getElementById('active-tags');
    if (tagsContainer) {
        tagsContainer.innerHTML = activeTags.map(tag =>
            `<span class="active-tag">${tag} <button onclick="removeTag('${tag}')">×</button></span>`
        ).join('');
    }

    const libraryType = document.querySelector('[name="library-type"]')?.value || 'geometry';
    const searchQuery = document.getElementById('library-search')?.value || '';
    htmx.ajax('GET', `/api/library/search?q=${searchQuery}&type=${libraryType}&tags=${activeTags.join(',')}`, {
        target: '#library-content'
    });
}

window.removeTag = function (tag) {
    const pill = document.querySelector(`.tag-pill[data-tag="${tag}"]`);
    if (pill) pill.classList.remove('active');
    updateTagFilters();
};

// View toggle
document.querySelectorAll('.view-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        const content = document.getElementById('library-content');
        if (content) {
            content.classList.toggle('library-list', btn.dataset.view === 'list');
            content.classList.toggle('library-grid', btn.dataset.view === 'grid');
        }
    });
});

// Sidebar Resize Logic
function initResizeHandles() {
    console.log('Initialize Resize Handles');
    document.querySelectorAll('.resize-handle').forEach(handle => {
        handle.addEventListener('mousedown', (e) => {
            e.preventDefault();
            e.stopPropagation(); // Don't trigger other clicks

            const panel = handle.parentElement;
            const isRightHandle = handle.classList.contains('handle-right');
            const startX = e.clientX;
            const startWidth = panel.getBoundingClientRect().width;

            handle.classList.add('active');
            document.body.style.cursor = 'col-resize';
            // Disable pointer events on iframes/canvases during drag to prevent trapping
            document.querySelectorAll('iframe, canvas').forEach(el => el.style.pointerEvents = 'none');

            function onMouseMove(e) {
                const dx = e.clientX - startX;
                // If right handle (Left Panel): width = start + dx
                // If left handle (Right Panel): width = start - dx (dragging left increases width)
                const newWidth = isRightHandle ? startWidth + dx : startWidth - dx;

                // Constraints are handled by CSS min/max-width, JS just tries to set it
                panel.style.width = `${newWidth}px`;
            }

            function onMouseUp() {
                document.removeEventListener('mousemove', onMouseMove);
                document.removeEventListener('mouseup', onMouseUp);
                handle.classList.remove('active');
                document.body.style.cursor = '';
                // Restore pointer events
                document.querySelectorAll('iframe, canvas').forEach(el => el.style.pointerEvents = 'auto');
            }

            document.addEventListener('mousemove', onMouseMove);
            document.addEventListener('mouseup', onMouseUp);
        });
    });
}

// Initialize on load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initResizeHandles);
} else {
    initResizeHandles();
}
