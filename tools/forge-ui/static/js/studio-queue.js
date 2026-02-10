// tools/forge-ui/static/js/studio-queue.js
/**
 * Job Queue UI Controller
 * 
 * Manages the queue modal, real-time updates, and user interactions.
 */

// Queue modal state
let queueModal = null;
let queueRefreshInterval = null;

/**
 * Initialize queue UI
 */
export function initQueue() {
    queueModal = document.getElementById('queue-modal');

    if (!queueModal) {
        console.warn('Queue modal not found');
        return;
    }

    // Setup event listeners
    setupQueueListeners();

    console.log('✅ Queue UI initialized');
}

/**
 * Setup event listeners for queue interactions
 */
function setupQueueListeners() {
    // Open queue modal
    const queueButton = document.getElementById('queue-button');
    if (queueButton) {
        queueButton.addEventListener('click', openQueue);
    }

    // Close queue modal
    const closeButton = queueModal.querySelector('.queue-modal-close');
    if (closeButton) {
        closeButton.addEventListener('click', closeQueue);
    }

    // Close on overlay click
    queueModal.addEventListener('click', (e) => {
        if (e.target === queueModal) {
            closeQueue();
        }
    });

    // Close on Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && queueModal.classList.contains('active')) {
            closeQueue();
        }
    });

    // WebSocket event listeners for queue updates
    window.addEventListener('gve:queue:job_added', updateQueueUI);
    window.addEventListener('gve:queue:job_started', updateQueueUI);
    window.addEventListener('gve:queue:job_completed', updateQueueUI);
    window.addEventListener('gve:queue:job_failed', updateQueueUI);
    window.addEventListener('gve:queue:job_cancelled', updateQueueUI);

    // AI generation stage updates (real-time progress)
    window.addEventListener('gve:generate:stage_complete', (event) => {
        const data = event.detail;
        const streamEl = document.querySelector(`.queue-ai-stream[data-job-id="${data.job_id}"]`);
        if (streamEl && !streamEl.dataset.hasTokens) {
            // Only update stage if we're not showing token stream
            streamEl.textContent = `→ ${data.stage}`;
            streamEl.classList.add('stage-updated');
            setTimeout(() => streamEl.classList.remove('stage-updated'), 300);
        }
    });

    // AI token streaming (real-time text generation)
    window.addEventListener('gve:generate:token_stream', (event) => {
        const data = event.detail;
        const streamEl = document.querySelector(`.queue-ai-stream[data-job-id="${data.job_id}"]`);
        if (streamEl) {
            streamEl.textContent = data.text + '...';
            streamEl.dataset.hasTokens = 'true';
            streamEl.classList.add('text-streaming');
        }
    });
}

/**
 * Open queue modal
 */
function openQueue() {
    queueModal.classList.add('active');
    refreshQueueTable();

    // Start auto-refresh every 2 seconds
    if (queueRefreshInterval) {
        clearInterval(queueRefreshInterval);
    }
    queueRefreshInterval = setInterval(refreshQueueTable, 2000);

    // Update queue status indicator
    updateQueueStatus();
}

/**
 * Close queue modal
 */
function closeQueue() {
    queueModal.classList.remove('active');

    // Stop auto-refresh
    if (queueRefreshInterval) {
        clearInterval(queueRefreshInterval);
        queueRefreshInterval = null;
    }
}

/**
 * Refresh queue table via HTMX
 */
function refreshQueueTable() {
    const tableContainer = document.getElementById('queue-table-container');
    if (tableContainer) {
        htmx.ajax('GET', '/api/queue/partial', {
            target: '#queue-table-container',
            swap: 'innerHTML'
        });
    }
}

/**
 * Update queue UI in response to WebSocket events
 */
function updateQueueUI(event) {
    console.log('Queue update:', event.detail);

    // Refresh table if modal is open
    if (queueModal && queueModal.classList.contains('active')) {
        refreshQueueTable();
    }

    // Update status indicator
    updateQueueStatus();
}

/**
 * Update queue status indicator (dot color)
 */
async function updateQueueStatus() {
    try {
        const response = await fetch('/api/queue/summary');
        const summary = await response.json();

        const dot = document.querySelector('.queue-dot');
        if (!dot) return;

        // Remove all status classes
        dot.classList.remove('idle', 'active', 'queued', 'error');

        // Determine status
        if (summary.failed > 0) {
            dot.classList.add('error');
        } else if (summary.running > 0) {
            dot.classList.add('active');
        } else if (summary.queued > 0) {
            dot.classList.add('queued');
        } else {
            dot.classList.add('idle');
        }

        // Update count text
        const countEl = document.getElementById('queue-count');
        if (countEl) {
            const activeCount = summary.running + summary.queued;
            countEl.textContent = activeCount > 0 ? `(${activeCount})` : '';
        }
    } catch (error) {
        console.error('Failed to update queue status:', error);
    }
}

// Initialize on page load
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initQueue);
} else {
    initQueue();
}

// Auto-update queue status every 5 seconds
setInterval(updateQueueStatus, 5000);
updateQueueStatus(); // Initial update
