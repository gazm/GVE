/**
 * events.js
 * Handles WebSocket connection and event dispatching for real-time updates.
 */

/**
 * Update the torch status indicator in the UI.
 * @param {Object} status - Torch status object with status, device, device_name, error
 */
export function updateTorchStatus(status) {
    const dot = document.querySelector('.torch-dot');
    const label = document.querySelector('.torch-label');
    
    if (!dot || !label) return;
    
    // Update dot class
    dot.className = 'dot torch-dot ' + status.status;
    
    // Update label text
    const labels = {
        cold: 'Torch Cold',
        loading: 'Torch Loading...',
        ready: `Torch Ready`,
        unavailable: 'Torch Unavailable'
    };
    
    // Show device name if ready
    if (status.status === 'ready' && status.device_name) {
        label.textContent = `Torch: ${status.device_name}`;
        label.title = `Device: ${status.device} (${status.device_name})`;
    } else if (status.status === 'unavailable' && status.error) {
        label.textContent = labels[status.status];
        label.title = `Error: ${status.error}`;
    } else {
        label.textContent = labels[status.status] || status.status;
        label.title = '';
    }
    
    console.log('🔥 Torch status:', status.status, status.device_name || '');
}

/**
 * Fetch initial system status from API.
 * Call this on page load to get torch status if preload already completed.
 */
export async function fetchInitialStatus() {
    try {
        const response = await fetch('/api/status');
        if (response.ok) {
            const data = await response.json();
            if (data.torch) {
                updateTorchStatus(data.torch);
            }
        }
    } catch (e) {
        console.warn('Failed to fetch initial status:', e);
    }
}

// Reconnection state for exponential backoff
let _reconnectDelay = 1000;
const _MAX_RECONNECT_DELAY = 30000;

export function connectWebSocket(url) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = url || `${protocol}//${window.location.host}/ws/events`;

    console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("✅ WebSocket connection established");
        document.body.classList.add('ws-connected');
        _reconnectDelay = 1000; // Reset backoff on successful connection
        
        // Fetch initial status when connected (in case torch already loaded)
        fetchInitialStatus();
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log("📨 Received event:", data);

            // Dispatch custom event for other components to listen to
            const customEvent = new CustomEvent(`gve:${data.type}`, { detail: data.payload });
            window.dispatchEvent(customEvent);

            // Handle torch status updates
            if (data.type === 'torch:status') {
                updateTorchStatus(data.payload);
            }
            // HTMX Integration: Trigger refresh for specific elements
            else if (data.type === 'compile:progress') {
                const progressBar = document.querySelector(`#progress-${data.payload.asset_id}`);
                if (progressBar) {
                    htmx.ajax('GET', `/api/assets/partials/progress/${data.payload.asset_id}`, {
                        target: `#progress-${data.payload.asset_id}`,
                        swap: 'outerHTML'
                    });
                }
            } else if (data.type === 'asset:updated') {
                const card = document.querySelector(`#card-${data.payload.asset_id}`);
                if (card) {
                    htmx.trigger(card, 'refresh');
                }
            }
        } catch (e) {
            console.error("❌ Error parsing WebSocket message:", e);
        }
    };

    ws.onclose = () => {
        document.body.classList.remove('ws-connected');
        console.log(`🔌 WebSocket closed. Reconnecting in ${(_reconnectDelay / 1000).toFixed(0)}s...`);
        setTimeout(() => connectWebSocket(url), _reconnectDelay);
        _reconnectDelay = Math.min(_reconnectDelay * 2, _MAX_RECONNECT_DELAY);
    };

    ws.onerror = (error) => {
        console.error("❌ WebSocket error:", error);
    };

    return ws;
}

// Global event bus helper
export function addEventListener(eventType, callback) {
    window.addEventListener(`gve:${eventType}`, (e) => callback(e.detail));
}
