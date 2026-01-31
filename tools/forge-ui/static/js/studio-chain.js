// Studio Chain - Card chain and asset editor functionality
import { syncExternalViewport, refreshTreeViewer, setViewportSynced } from '/static/js/studio-navigation.js';

// Selection: keep chain strip and tree viewer in sync
window.selectCard = function (cardId) {
    document.querySelectorAll('.chain-scroll .card').forEach(c => c.classList.remove('selected'));
    document.querySelectorAll('.tree-item').forEach(el => el.classList.remove('selected'));
    const cardEl = document.getElementById('card-' + cardId);
    if (cardEl) cardEl.classList.add('selected');
    const treeEl = document.querySelector('.tree-item[data-card-id="' + cardId + '"]');
    if (treeEl) treeEl.classList.add('selected');
    updateChainCount();
};

function updateChainCount() {
    const countEl = document.getElementById('card-count');
    if (countEl) {
        const filled = document.querySelectorAll('.chain-scroll .slot-filled').length;
        countEl.textContent = filled;
    }
}

/** Sync WASM viewport to current chain: load each filled slot's binary so viewport reflects app state. */
function syncViewportFromChain() {
    const chainEl = document.getElementById('card-chain');
    if (!chainEl || !window.load_asset) return;
    const filledCards = chainEl.querySelectorAll('.slot-filled[id^="card-"]');
    if (typeof window.clear_viewport === 'function' && window.viewportReady) {
        window.clear_viewport();
    }
    filledCards.forEach(card => {
        const assetId = card.id.replace(/^card-/, '');
        // Only load if not hidden
        if (assetId && !window._hiddenAssets.has(assetId)) {
            window.load_asset('/api/assets/' + assetId + '/binary', assetId);
        }
    });
}

// Make syncViewportFromChain available globally
window.syncViewportFromChain = syncViewportFromChain;

// After chain partial is swapped: update count, sync viewport from chain, refresh tree
document.body.addEventListener('htmx:afterSwap', function (ev) {
    if (ev.detail.target.id === 'card-chain') {
        updateChainCount();
        syncViewportFromChain();
        setViewportSynced(true); // Mark as synced after chain update
        htmx.ajax('GET', '/api/assets/partials/tree', { target: '#tree-viewer', swap: 'innerHTML' });
    }
});

// Fill a chain slot (from library). Uses _chainTargetSlot if set, else POST to /chain/fill (first empty slot).
// optionalLabel: if provided, stored in window._sceneLabels for hierarchy display.
window.addToChain = async function (assetId, optionalLabel) {
    if (optionalLabel) {
        window._sceneLabels = window._sceneLabels || {};
        window._sceneLabels[String(assetId)] = optionalLabel;
    }
    const formData = new FormData();
    formData.set('asset_id', assetId);
    if (window._chainTargetSlot) {
        await fetch('/api/assets/chain/slot/' + window._chainTargetSlot, { method: 'POST', body: formData });
        window._chainTargetSlot = null;
    } else {
        const res = await fetch('/api/assets/chain/fill', { method: 'POST', body: formData });
        if (res.ok && res.headers.get('content-type')?.includes('text/html')) {
            const html = await res.text();
            const chainEl = document.getElementById('card-chain');
            if (chainEl) chainEl.innerHTML = html;
        }
    }
    setViewportSynced(false); // Chain changed, need to resync
    htmx.ajax('GET', '/api/assets/partials/chain', { target: '#card-chain', swap: 'innerHTML' });
    showPage('asset-editor');
};

/** Refresh hierarchy from app state (server). Tree is driven by chain; no WASM snapshot. */
window.refreshHierarchyFromWasm = function refreshHierarchyFromWasm() {
    refreshTreeViewer();
};

// Card chain functions
window.showLibraryPicker = function () {
    showPage('library');
    logOutput('Switched to library - select components to add', 'info');
};

window.switchToAssetEditor = function (assetId) {
    if (assetId) {
        // Store the asset ID for adding to chain
        window._pendingAssetId = assetId;
        showPage('asset-editor');
        // Add to chain after switching
        setTimeout(() => {
            if (window._pendingAssetId) {
                addToChain(window._pendingAssetId);
                window._pendingAssetId = null;
            }
        }, 100);
    } else {
        showPage('asset-editor');
    }
};

window.previewItem = function (id, type) {
    logOutput(`Loading preview for ${type} ${id}...`, 'info');

    // 1. Switch to asset editor view
    showPage('asset-editor');

    // 2. Trigger HTMX to load property editor
    // We use the htmx API to issue a swap request manually
    htmx.ajax('GET', `/api/assets/partials/editor/${id}`, {
        target: '#property-editor',
        swap: 'innerHTML'
    });

    // 3. Load binary into Viewport
    if (type === 'geometry' || type === 'recipes') {
        // Clear entire viewport (meshes + SDFs)
        if (typeof window.clear_viewport === 'function') window.clear_viewport();

        // Determine binary URL based on ID type (simulated here)
        // In real app, this might come from the card data or API
        const binaryUrl = `/api/assets/${id}/binary`;

        // Ensure viewport is ready
        if (window.viewportReady && window.load_asset) {
            window.load_asset(binaryUrl, id);
            syncExternalViewport(id);
        } else {
            // Retry once viewport inits
            setTimeout(() => {
                if (window.viewportReady && window.load_asset) {
                    window.load_asset(binaryUrl, id);
                    syncExternalViewport(id);
                }
            }, 1000);
        }
    }
};

window.compileChain = function () {
    logOutput('Compiling card chain...', 'info');
    // TODO: Trigger actual compilation
    htmx.ajax('POST', '/api/compile/chain', { target: '#generation-output', swap: 'beforeend' });
    // Refresh chain and tree when compile completes (e.g. via WebSocket); for now refresh tree after a short delay
    setTimeout(refreshTreeViewer, 500);
};

// Clear viewport manually (useful when starting fresh)
window.clearViewportManual = function () {
    if (window.clear_viewport && window.viewportReady) {
        window.clear_viewport();
        setViewportSynced(false);
        window._hiddenAssets.clear();
        console.log('🧹 Viewport cleared');
    }
};

// Toggle hide/show asset in viewport
window.toggleHideAsset = function (assetId) {
    const treeItem = document.querySelector(`.tree-item[data-asset-id="${assetId}"]`);
    
    if (window._hiddenAssets.has(assetId)) {
        // Show the asset
        window._hiddenAssets.delete(assetId);
        if (treeItem) treeItem.classList.remove('hidden');
        
        // Reload the asset in viewport
        if (window.load_asset && window.viewportReady) {
            window.load_asset(`/api/assets/${assetId}/binary`, assetId);
        }
        console.log(`👁 Showing asset: ${assetId}`);
    } else {
        // Hide the asset
        window._hiddenAssets.add(assetId);
        if (treeItem) treeItem.classList.add('hidden');
        
        console.log(`🚫 Hiding asset: ${assetId}`);
        
        // Resync viewport to apply hidden state (reload visible assets only)
        syncViewportFromChainNoDefault();
    }
};

/** Sync viewport without clearing first - just reload visible assets to avoid default mesh flash */
function syncViewportFromChainNoDefault() {
    const chainEl = document.getElementById('card-chain');
    if (!chainEl || !window.load_asset) return;
    
    // Get all assets that should be visible
    const filledCards = chainEl.querySelectorAll('.slot-filled[id^="card-"]');
    const visibleAssets = [];
    
    filledCards.forEach(card => {
        const assetId = card.id.replace(/^card-/, '');
        if (assetId && !window._hiddenAssets.has(assetId)) {
            visibleAssets.push(assetId);
        }
    });
    
    // Only clear and reload if we have visible assets, otherwise just clear
    if (visibleAssets.length > 0) {
        if (typeof window.clear_viewport === 'function' && window.viewportReady) {
            window.clear_viewport();
        }
        visibleAssets.forEach(assetId => {
            window.load_asset('/api/assets/' + assetId + '/binary', assetId);
        });
    } else {
        // No visible assets - just clear viewport
        if (typeof window.clear_viewport === 'function' && window.viewportReady) {
            window.clear_viewport();
        }
    }
}

// Delete asset from chain (removes from composition)
window.deleteAssetFromChain = async function (assetId, slotType) {
    if (!confirm('Remove this asset from the chain?')) return;
    
    try {
        // Clear the slot via API (POST with no asset_id clears the slot)
        const formData = new FormData();
        // Don't set asset_id - this clears the slot
        
        const response = await fetch(`/api/assets/chain/slot/${slotType}`, {
            method: 'POST',
            body: formData
        });
        
        if (response.ok) {
            // Refresh chain UI
            htmx.ajax('GET', '/api/assets/partials/chain', { 
                target: '#card-chain', 
                swap: 'innerHTML' 
            });
            
            // Remove from hidden set
            window._hiddenAssets.delete(assetId);
            
            // Mark as needing resync
            setViewportSynced(false);
            
            console.log(`🗑 Removed asset from chain: ${assetId} (${slotType})`);
        } else {
            console.error('Failed to remove asset from chain');
        }
    } catch (error) {
        console.error('Error removing asset:', error);
    }
};
