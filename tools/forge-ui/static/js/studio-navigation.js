// Studio Navigation - Page switching and viewport initialization
import { initViewport, ensureWasmLoaded } from '/static/js/viewport.js';
import { connectWebSocket } from '/static/js/events.js';

// Initialize viewport once on page load with shared canvas
let viewportInitialized = false;
let currentTab = 'library'; // Track current tab to avoid unnecessary reloads
let viewportSyncedWithChain = false; // Track if viewport matches chain state
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

// Page navigation - shared viewport always visible, just swap UI panels
window.showPage = function (pageName) {
    const previousTab = currentTab;
    currentTab = pageName;
    
    // Hide all pages
    document.querySelectorAll('.page').forEach(p => p.style.display = 'none');
    // Show selected page
    const page = document.getElementById(`page-${pageName}`);
    if (page) {
        page.style.display = 'grid';
    }
    // Update nav buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.page === pageName);
    });
    
    // Handle viewport content intelligently
    if (viewportInitialized) {
        if (pageName === 'asset-editor') {
            // Only sync if we haven't synced yet or if chain changed
            if (!viewportSyncedWithChain) {
                setTimeout(() => {
                    if (typeof syncViewportFromChain === 'function') {
                        syncViewportFromChain();
                        viewportSyncedWithChain = true;
                    }
                    refreshTreeViewer();
                }, 100);
            } else {
                // Just refresh the tree viewer, keep viewport as-is
                refreshTreeViewer();
            }
        }
        // GenAI and other tabs: keep viewport content unchanged
    }
};

export function refreshTreeViewer() {
    const el = document.getElementById('tree-viewer');
    if (el) htmx.ajax('GET', '/api/assets/partials/tree', { target: '#tree-viewer', swap: 'innerHTML' });
}

export function setViewportSynced(synced) {
    viewportSyncedWithChain = synced;
}

export function isViewportInitialized() {
    return viewportInitialized;
}

// Output logging
window.logOutput = function (message, type = 'info') {
    const output = document.getElementById('generation-output');
    const empty = output.querySelector('.output-empty');
    if (empty) empty.remove();

    const line = document.createElement('p');
    line.className = `output-line ${type}`;
    const timestamp = new Date().toLocaleTimeString();
    line.textContent = `[${timestamp}] ${message}`;
    output.appendChild(line);
    output.scrollTop = output.scrollHeight;
};

window.clearOutput = function () {
    const output = document.getElementById('generation-output');
    output.innerHTML = '<p class="output-empty">Generation output will appear here...</p>';
};

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
