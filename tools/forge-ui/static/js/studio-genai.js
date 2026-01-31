// Studio GenAI - AI generation workflow
import { addEventListener } from '/static/js/events.js';
import { syncExternalViewport } from '/static/js/studio-navigation.js';
import { debounce } from '/static/js/studio-navigation.js';

// Type selector
document.querySelectorAll('.type-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.type-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateCostEstimate();
    });
});

// Style chips (multi-select)
document.querySelectorAll('.style-chips .chip').forEach(chip => {
    chip.addEventListener('click', () => {
        chip.classList.toggle('active');
        updateCostEstimate();
    });
});

// Viewport mode buttons
document.querySelectorAll('.mode-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        // TODO: Switch viewport rendering mode
        logOutput(`Viewport mode: ${btn.dataset.mode}`, 'info');
    });
});

// Prompt input - suggest materials
const promptInput = document.getElementById('ai-prompt');
if (promptInput) {
    promptInput.addEventListener('input', debounce(() => {
        const prompt = promptInput.value.trim();
        if (prompt.length > 2) {
            suggestMaterials(prompt);
            updateCostEstimate();
        }
    }, 500));
}

async function suggestMaterials(prompt) {
    const suggestions = document.getElementById('material-suggestions');
    
    try {
        // Call backend API for material suggestions
        const response = await fetch(`/api/generate/suggest/materials?prompt=${encodeURIComponent(prompt)}`);
        
        if (response.ok) {
            const data = await response.json();
            const materials = data.materials;
            
            if (materials.length > 0) {
                suggestions.innerHTML = materials.map(m =>
                    `<button class="chip" onclick="addMaterial('${m}')">${m}</button>`
                ).join('');
            } else {
                suggestions.innerHTML = '<span class="chip-placeholder">No suggestions for this prompt</span>';
            }
        } else {
            suggestions.innerHTML = '<span class="chip-placeholder">Failed to load suggestions</span>';
        }
    } catch (err) {
        console.error('Failed to fetch material suggestions:', err);
        suggestions.innerHTML = '<span class="chip-placeholder">Failed to load suggestions</span>';
    }
}

window.addMaterial = function (material) {
    logOutput(`Added material: ${material}`, 'info');
};

async function updateCostEstimate() {
    const activeType = document.querySelector('.type-btn.active')?.dataset.type || 'prop';
    const activeStyles = Array.from(document.querySelectorAll('.style-chips .chip.active'))
        .map(c => c.dataset.style || c.textContent.trim());
    const promptLength = document.getElementById('ai-prompt')?.value.length || 0;

    try {
        // Call backend API for cost estimation
        const response = await fetch('/api/generate/estimate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                category: activeType,
                styles: activeStyles,
                prompt_length: promptLength
            })
        });

        if (response.ok) {
            const data = await response.json();
            document.getElementById('cost-estimate').textContent = `$${data.cost_usd.toFixed(2)}`;
            document.querySelector('.estimate-time').textContent = `~${data.estimated_time_sec}s`;
        } else {
            // Fallback to default display on error
            document.getElementById('cost-estimate').textContent = '$--';
            document.querySelector('.estimate-time').textContent = '~--s';
        }
    } catch (err) {
        console.error('Failed to fetch cost estimate:', err);
        // Fallback to default display on error
        document.getElementById('cost-estimate').textContent = '$--';
        document.querySelector('.estimate-time').textContent = '~--s';
    }
}

// Generate asset via AI pipeline
window.generateAsset = async function () {
    const prompt = document.getElementById('ai-prompt')?.value?.trim();
    if (!prompt) {
        logOutput('Please enter a prompt first', 'warning');
        return;
    }

    const activeType = document.querySelector('.type-btn.active')?.dataset.type || 'prop';
    const activeStyles = Array.from(document.querySelectorAll('.style-chips .chip.active'))
        .map(c => c.dataset.style);

    // Build the full prompt with style context
    const styleStr = activeStyles.length > 0 ? ` (${activeStyles.join(', ')})` : '';
    const fullPrompt = `${activeType}: ${prompt}${styleStr}`;

    logOutput(`Starting generation: "${fullPrompt}"`, 'info');

    // Disable button while generating
    const btn = document.getElementById('btn-generate');
    btn.disabled = true;
    btn.innerHTML = '<span class="icon">⏳</span> Generating...';

    // Capitalize first letter for category (e.g. "weapon" -> "Weapon")
    const category = activeType.charAt(0).toUpperCase() + activeType.slice(1);

    try {
        // POST to generate endpoint
        const response = await fetch('/api/generate/', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: fullPrompt,
                category: category
            })
        });

        if (!response.ok) {
            const err = await response.text();
            throw new Error(`HTTP ${response.status}: ${err}`);
        }

        const { job_id, status } = await response.json();
        logOutput(`Job queued: ${job_id}`, 'info');

        // Listen for WebSocket completion event instead of polling
        listenForGenerationComplete(job_id);

    } catch (err) {
        logOutput(`Generation failed: ${err.message}`, 'error');
        btn.disabled = false;
        btn.innerHTML = '<span class="icon">⚡</span> Generate Asset';
    }
};

function listenForGenerationComplete(jobId) {
    const btn = document.getElementById('btn-generate');
    
    // Listen for generation progress events
    const progressHandler = (payload) => {
        if (payload.job_id === jobId) {
            logOutput(`⏳ ${payload.status}...`, 'info');
        }
    };
    addEventListener('generate:progress', progressHandler);
    
    // Listen for generation complete event
    const completeHandler = (payload) => {
        if (payload.job_id === jobId) {
            logOutput(`✅ Generation complete! Asset ID: ${payload.asset_id}`, 'success');
            if (payload.result) {
                logOutput(`   Time: ${payload.result.generation_time_sec?.toFixed(1)}s, Track: ${payload.result.track_used}`, 'info');
            }
            
            // Load asset in shared viewport
            if (payload.asset_id && window.load_asset && window.viewportReady) {
                if (window.clear_sdf) window.clear_sdf();
                window.load_asset(`/api/assets/${payload.asset_id}/binary`, payload.asset_id);
                syncExternalViewport(payload.asset_id);
            }

            // Show feedback UI with Add to Chain button
            showFeedbackUI(payload.asset_id);

            btn.disabled = false;
            btn.innerHTML = '<span class="icon">⚡</span> Generate Asset';
            
            // Clean up event listeners
            window.removeEventListener('gve:generate:complete', completeHandler);
            window.removeEventListener('gve:generate:progress', progressHandler);
            window.removeEventListener('gve:generate:failed', failedHandler);
        }
    };
    addEventListener('generate:complete', completeHandler);
    
    // Listen for generation failed event
    const failedHandler = (payload) => {
        if (payload.job_id === jobId) {
            logOutput(`❌ Generation failed: ${payload.error}`, 'error');
            btn.disabled = false;
            btn.innerHTML = '<span class="icon">⚡</span> Generate Asset';
            
            // Clean up event listeners
            window.removeEventListener('gve:generate:complete', completeHandler);
            window.removeEventListener('gve:generate:progress', progressHandler);
            window.removeEventListener('gve:generate:failed', failedHandler);
        }
    };
    addEventListener('generate:failed', failedHandler);
}

// Feedback & Save Handlers
function showFeedbackUI(assetId) {
    // Show the save/rate container
    const container = document.getElementById('feedback-container');
    if (container) {
        container.style.display = 'flex';
        container.dataset.assetId = assetId;
        // Reset state
        document.querySelectorAll('.star-btn').forEach(b => b.classList.remove('active'));
        const saveBtn = document.getElementById('btn-save-draft');
        if (saveBtn) {
            saveBtn.textContent = 'Save to Library';
            saveBtn.disabled = false;
        }
        // Show Add to Chain button
        const addBtn = document.getElementById('btn-add-to-chain');
        if (addBtn) {
            addBtn.style.display = 'block';
            addBtn.onclick = () => window.switchToAssetEditor(assetId);
        }
    }
}

window.rateAsset = async function (rating) {
    const container = document.getElementById('feedback-container');
    const assetId = container?.dataset.assetId;
    if (!assetId) return;

    // Visual update
    document.querySelectorAll('.star-btn').forEach((btn, idx) => {
        btn.classList.toggle('active', idx < rating);
    });

    try {
        await fetch(`/api/assets/${assetId}/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ rating: rating })
        });
        logOutput(`Rated ${rating} stars`, 'success');
    } catch (e) {
        logOutput('Failed to submit rating', 'error');
        console.error(e);
    }
};

window.saveAsset = async function () {
    const container = document.getElementById('feedback-container');
    const assetId = container?.dataset.assetId;
    const btn = document.getElementById('btn-save-draft');

    if (!assetId || !btn) return;

    btn.disabled = true;
    btn.textContent = 'Saving...';

    try {
        const res = await fetch(`/api/assets/${assetId}/save`, { method: 'POST' });
        if (res.ok) {
            logOutput('Asset saved to library and learnt!', 'success');
            btn.textContent = 'Saved ✓';
            // Refresh library view if open
            if (document.getElementById('page-library').style.display !== 'none') {
                document.querySelector('.tab-btn.active')?.click();
            }
        } else {
            throw new Error('Save failed');
        }
    } catch (e) {
        logOutput('Failed to save asset', 'error');
        btn.disabled = false;
        btn.textContent = 'Save to Library';
        console.error(e);
    }
};

// Initialize cost estimate on load
updateCostEstimate();
