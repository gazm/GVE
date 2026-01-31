/**
 * events.js
 * Handles WebSocket connection and event dispatching for real-time updates.
 */

export function connectWebSocket(url) {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = url || `${protocol}//${window.location.host}/ws/events`;

    console.log(`Connecting to WebSocket: ${wsUrl}`);
    const ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        console.log("WebSocket connection established");
        document.body.classList.add('ws-connected');
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log("Received event:", data);

            // Dispatch custom event for other components to listen to
            const customEvent = new CustomEvent(`gve:${data.type}`, { detail: data.payload });
            window.dispatchEvent(customEvent);

            // HTMX Integration: Trigger refresh for specific elements
            if (data.type === 'compile:progress') {
                const progressBar = document.querySelector(`#progress-${data.payload.asset_id}`);
                if (progressBar) {
                    // Trigger htmx to refresh the progress bar partial
                    htmx.ajax('GET', `/api/assets/partials/progress/${data.payload.asset_id}`, {
                        target: `#progress-${data.payload.asset_id}`,
                        swap: 'outerHTML'
                    });
                }
            } else if (data.type === 'asset:updated') {
                // Refresh the specific card in the chain
                const card = document.querySelector(`#card-${data.payload.asset_id}`);
                if (card) {
                    htmx.trigger(card, 'refresh');
                }
            }
        } catch (e) {
            console.error("Error parsing WebSocket message:", e);
        }
    };

    ws.onclose = () => {
        console.log("WebSocket connection closed. Retrying in 5s...");
        document.body.classList.remove('ws-connected');
        setTimeout(() => connectWebSocket(url), 5000);
    };

    ws.onerror = (error) => {
        console.error("WebSocket error:", error);
    };

    return ws;
}

// Global event bus helper
export function addEventListener(eventType, callback) {
    window.addEventListener(`gve:${eventType}`, (e) => callback(e.detail));
}
