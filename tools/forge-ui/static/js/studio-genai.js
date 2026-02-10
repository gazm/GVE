// Studio GenAI - AI generation workflow (Two-Phase: Concept -> 3D)
import { addEventListener } from '/static/js/events.js';
import { syncExternalViewport } from '/static/js/studio-navigation.js';
import { debounce } from '/static/js/studio-navigation.js';
import * as UI from '/static/js/studio-genai-ui.js';

// State for concept workflow
let currentConceptJobId = null;
let currentConceptImage = null;
let currentPrompt = null;

// State for 3D generation
let currentGenerationJobId = null;
let lastLoadedPreviewUrl = null;

// =============================================================================
// Initialization & Event Wiring
// =============================================================================

// Wire up workflow mode toggle
if (UI.ui.skipConceptCheckbox) {
    UI.ui.skipConceptCheckbox.addEventListener('change', () => {
        const icon = document.querySelector('#btn-generate .icon');
        const mode = UI.ui.skipConceptCheckbox.checked ? 'direct' : 'concept';
        UI.updateGenerateButtonState(false, mode);
    });
}

// Wire up concept preview buttons
document.getElementById('btn-approve-concept')?.addEventListener('click', () => {
    if (currentConceptJobId) approveConcept(currentConceptJobId);
});

document.getElementById('btn-regenerate-concept')?.addEventListener('click', () => {
    if (currentConceptJobId) UI.showRegenerateDialog();
});

document.getElementById('btn-cancel-concept')?.addEventListener('click', () => {
    if (currentConceptJobId) cancelConcept(currentConceptJobId);
});

// Wire up regenerate dialog buttons
document.getElementById('btn-submit-regenerate')?.addEventListener('click', () => {
    const feedback = document.getElementById('regenerate-feedback')?.value?.trim();
    if (feedback && currentConceptJobId) {
        regenerateConcept(currentConceptJobId, feedback);
        UI.hideRegenerateDialog();
    }
});

document.getElementById('btn-cancel-regenerate')?.addEventListener('click', UI.hideRegenerateDialog);

// Type selector
UI.ui.typeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        UI.ui.typeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateCostEstimate();
    });
});

// Style chips (multi-select)
UI.ui.styleChips.forEach(chip => {
    chip.addEventListener('click', () => {
        chip.classList.toggle('active');
        updateCostEstimate();
    });
});

// Viewport mode buttons
// Viewport mode buttons
UI.ui.modeBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        UI.ui.modeBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        const mode = btn.dataset.mode;
        UI.logOutput(`Viewport mode: ${mode}`, 'info');

        if (window.set_view_mode) {
            window.set_view_mode(mode);
        }
    });
});

// Prompt input - suggest materials
if (UI.ui.promptInput) {
    UI.ui.promptInput.addEventListener('input', debounce(() => {
        const prompt = UI.ui.promptInput.value.trim();
        if (prompt.length > 2) {
            suggestMaterials(prompt);
            updateCostEstimate();
        }
    }, 500));
}

// Quality Slider Logic
if (UI.ui.qualitySlider && UI.ui.qualityValue) {
    const qualityLevels = ['Draft', 'Standard', 'High', 'Ultra'];
    UI.ui.qualitySlider.addEventListener('input', () => {
        const idx = parseInt(UI.ui.qualitySlider.value);
        UI.ui.qualityValue.textContent = qualityLevels[idx];
        updateCostEstimate();
    });
}

// Global UI helper needed for inline onclick
window.addMaterial = function (material) {
    UI.logOutput(`Added material: ${material}`, 'info');
    // could append to prompt here if desired
};

// =============================================================================
// API calls
// =============================================================================

async function suggestMaterials(prompt) {
    try {
        const response = await fetch(`/api/generate/suggest/materials?prompt=${encodeURIComponent(prompt)}`);
        if (response.ok) {
            const data = await response.json();
            UI.renderMaterialSuggestions(data.materials || []);
        } else {
            UI.renderMaterialSuggestionsError();
        }
    } catch (err) {
        console.error('Failed to fetch material suggestions:', err);
        UI.renderMaterialSuggestionsError();
    }
}

async function updateCostEstimate() {
    const activeType = document.querySelector('.type-btn.active')?.dataset.type || 'prop';
    const activeStyles = Array.from(document.querySelectorAll('.style-chips .chip.active'))
        .map(c => c.dataset.style || c.textContent.trim());
    const promptLength = UI.ui.promptInput?.value.length || 0;

    try {
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
            UI.updateCostDisplay(data.cost_usd, data.estimated_time_sec);
        } else {
            UI.updateCostDisplayError();
        }
    } catch (err) {
        console.error('Failed to fetch cost estimate:', err);
        UI.updateCostDisplayError();
    }
}

// =============================================================================
// Concept Workflow (Phase 1)
// =============================================================================

// Generate asset - routes to concept-first or direct based on toggle
window.generateAsset = async function () {
    const skipConcept = UI.ui.skipConceptCheckbox?.checked;

    if (skipConcept) {
        return generateAssetDirect();
    }

    const prompt = UI.ui.promptInput?.value?.trim();
    if (!prompt) {
        UI.logOutput('⚠️ Please enter a prompt first', 'warning');
        return;
    }

    const activeType = document.querySelector('.type-btn.active')?.dataset.type || 'prop';
    const activeStyles = Array.from(document.querySelectorAll('.style-chips .chip.active'))
        .map(c => c.dataset.style);
    const style = activeStyles.length > 0 ? activeStyles[0] : 'realistic';
    const category = activeType.charAt(0).toUpperCase() + activeType.slice(1);

    currentPrompt = prompt;
    UI.logOutput(`🎨 Generating concept image for: "${prompt}"`, 'info');

    // Update UI state
    UI.updateGenerateButtonState(true, 'concept', '<span class="icon">🎨</span> <span>Generating Concept...</span>');
    UI.hideConceptPreview();

    try {
        const response = await fetch('/api/generate/concept', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                prompt: prompt,
                category: category,
                style: style,
                aspect_ratio: '1:1'
            })
        });

        if (!response.ok) {
            const err = await response.text();
            throw new Error(`HTTP ${response.status}: ${err}`);
        }

        const { job_id } = await response.json();
        currentConceptJobId = job_id;
        UI.logOutput(`📋 Concept job queued: ${job_id}`, 'info');

        pollConceptStatus(job_id);

    } catch (err) {
        UI.logOutput(`❌ Concept generation failed: ${err.message}`, 'error');
        resetGenerateButton();
    }
};

async function pollConceptStatus(jobId) {
    const maxAttempts = 60; // 2 minutes max
    let attempts = 0;

    const poll = async () => {
        attempts++;
        try {
            const response = await fetch(`/api/generate/concept/${jobId}`);
            const data = await response.json();

            if (data.status === 'ready') {
                UI.logOutput('✅ Concept image ready! Review and approve to generate 3D.', 'success');
                currentConceptImage = data.concept_image;
                UI.showConceptPreview(data.concept_image, data.prompt || currentPrompt);
                resetGenerateButton();
                return;
            } else if (data.status === 'failed') {
                throw new Error(data.error || 'Concept generation failed');
            } else if (data.status === 'generating') {
                UI.logOutput(`⏳ Generating concept... (${attempts * 2}s)`, 'info');
                if (attempts < maxAttempts) {
                    setTimeout(poll, 2000);
                } else {
                    throw new Error('Concept generation timed out');
                }
            } else if (attempts < maxAttempts) {
                setTimeout(poll, 2000);
            } else {
                throw new Error('Concept generation timed out');
            }
        } catch (err) {
            UI.logOutput(`❌ ${err.message}`, 'error');
            resetGenerateButton();
        }
    };

    poll();
}

function resetGenerateButton() {
    const skipConcept = UI.ui.skipConceptCheckbox?.checked;
    UI.updateGenerateButtonState(false, skipConcept ? 'direct' : 'concept');
}

// Approve concept and proceed to 3D generation
async function approveConcept(jobId) {
    UI.logOutput('✅ Concept approved! Starting 3D generation...', 'info');

    UI.updateGenerateButtonState(true, 'concept', '<span class="icon">⏳</span> <span>Generating 3D...</span>');
    UI.hideConceptPreview();

    try {
        const response = await fetch(`/api/generate/concept/${jobId}/approve`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        if (!response.ok) {
            const err = await response.text();
            throw new Error(`HTTP ${response.status}: ${err}`);
        }

        const { generation_job_id } = await response.json();
        UI.logOutput(`📋 3D generation job: ${generation_job_id}`, 'info');

        listenForGenerationComplete(generation_job_id);

    } catch (err) {
        UI.logOutput(`❌ Approval failed: ${err.message}`, 'error');
        resetGenerateButton();
    }
}

// Regenerate concept with feedback
async function regenerateConcept(jobId, feedback) {
    UI.logOutput(`🔄 Regenerating with feedback: "${feedback}"`, 'info');

    UI.updateGenerateButtonState(true, 'concept', '<span class="icon">🎨</span> <span>Regenerating...</span>');
    UI.hideConceptPreview();

    try {
        const response = await fetch(`/api/generate/concept/${jobId}/regenerate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                feedback: feedback,
                use_previous_as_reference: true
            })
        });

        if (!response.ok) {
            const err = await response.text();
            throw new Error(`HTTP ${response.status}: ${err}`);
        }

        const { job_id: newJobId } = await response.json();
        currentConceptJobId = newJobId;
        UI.logOutput(`📋 New concept job: ${newJobId}`, 'info');

        pollConceptStatus(newJobId);

    } catch (err) {
        UI.logOutput(`❌ Regeneration failed: ${err.message}`, 'error');
        resetGenerateButton();
    }
}

// Cancel concept and start fresh
async function cancelConcept(jobId) {
    try {
        await fetch(`/api/generate/concept/${jobId}`, { method: 'DELETE' });
    } catch (e) {
        console.error('🔴 Failed to cancel concept:', e);
    }

    UI.hideConceptPreview();
    currentConceptJobId = null;
    currentConceptImage = null;
    currentPrompt = null;

    UI.logOutput('🚫 Concept cancelled. Enter a new prompt to try again.', 'info');
}

// Export for window access
window.approveConcept = approveConcept;
window.showRegenerateDialog = UI.showRegenerateDialog;
window.cancelConcept = cancelConcept;


// =============================================================================
// Direct/3D Generation (Phase 2)
// =============================================================================

window.generateAssetDirect = async function () {
    const prompt = UI.ui.promptInput?.value?.trim();
    if (!prompt) {
        UI.logOutput('⚠️ Please enter a prompt first', 'warning');
        return;
    }

    const activeType = document.querySelector('.type-btn.active')?.dataset.type || 'prop';
    const activeStyles = Array.from(document.querySelectorAll('.style-chips .chip.active'))
        .map(c => c.dataset.style);

    // Get Quality Setting
    const qualityLevels = ['Draft', 'Standard', 'High', 'Ultra'];
    const qualityIdx = parseInt(UI.ui.qualitySlider?.value || "1");
    const qualityTag = `[Quality: ${qualityLevels[qualityIdx]}]`;

    const styleStr = activeStyles.length > 0 ? ` (${activeStyles.join(', ')})` : '';
    const fullPrompt = `${activeType}: ${prompt}${styleStr} ${qualityTag}`;

    UI.logOutput(`⚡ Starting direct 3D generation: "${prompt}"`, 'info');

    UI.updateGenerateButtonState(true, 'direct', '<span class="icon">⏳</span> <span>Generating 3D...</span>');

    const category = activeType.charAt(0).toUpperCase() + activeType.slice(1);

    try {
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

        const { job_id } = await response.json();
        UI.logOutput(`📋 Job queued: ${job_id}`, 'info');

        listenForGenerationComplete(job_id);

    } catch (err) {
        UI.logOutput(`❌ Generation failed: ${err.message}`, 'error');
        resetGenerateButton();
    }
};

function listenForGenerationComplete(jobId) {
    currentGenerationJobId = jobId;
    lastLoadedPreviewUrl = null;

    UI.showStageProgress();
    UI.resetStageProgress();

    // Progress handler
    const progressHandler = (payload) => {
        if (payload.job_id === jobId) {
            UI.logOutput(`⏳ ${payload.status}...`, 'info');
        }
    };
    addEventListener('generate:progress', progressHandler);

    // Stage complete handler
    const stageCompleteHandler = (payload) => {
        if (payload.job_id === jobId) {
            const stageName = payload.stage;
            const previewUrl = payload.preview_url;

            if (previewUrl === lastLoadedPreviewUrl) return;
            lastLoadedPreviewUrl = previewUrl;

            UI.logOutput(`📺 Stage ${stageName} complete, loading preview...`, 'info');
            UI.updateStageProgress(stageName, 'complete');

            if (previewUrl && window.load_asset && window.viewportReady) {
                if (window.clear_sdf) window.clear_sdf();
                window.load_asset(previewUrl, `preview_${stageName}`);
            }

            const stages = ['A1', 'A2', 'A3'];
            const nextIdx = stages.indexOf(stageName) + 1;
            if (nextIdx < stages.length) {
                UI.updateStageProgress(stages[nextIdx], 'active');
            }
        }
    };
    addEventListener('generate:stage_complete', stageCompleteHandler);

    // Complete handler
    const completeHandler = (payload) => {
        if (payload.job_id === jobId) {
            UI.logOutput(`✅ Generation complete! Asset ID: ${payload.asset_id}`, 'success');
            if (payload.result) {
                UI.logOutput(`   Time: ${payload.result.generation_time_sec?.toFixed(1)}s, Track: ${payload.result.track_used}`, 'info');
            }

            UI.updateStageProgress('A3', 'complete');

            if (payload.asset_id && window.load_asset && window.viewportReady) {
                if (window.clear_sdf) window.clear_sdf();
                window.load_asset(`/api/assets/${payload.asset_id}/binary`, payload.asset_id);
                syncExternalViewport(payload.asset_id);
            }

            if (payload.asset_id && window.load_asset && window.viewportReady) {
                if (window.clear_sdf) window.clear_sdf();
                window.load_asset(`/api/assets/${payload.asset_id}/binary`, payload.asset_id);
                syncExternalViewport(payload.asset_id);
            }

            resetGenerateButton();
            currentGenerationJobId = null;
            setTimeout(UI.hideStageProgress, 2000);

            // Cleanup
            window.removeEventListener('gve:generate:complete', completeHandler);
            window.removeEventListener('gve:generate:progress', progressHandler);
            window.removeEventListener('gve:generate:stage_complete', stageCompleteHandler);
            window.removeEventListener('gve:generate:failed', failedHandler);
        }
    };
    addEventListener('generate:complete', completeHandler);

    // Failed handler
    const failedHandler = (payload) => {
        if (payload.job_id === jobId) {
            UI.logOutput(`❌ Generation failed: ${payload.error}`, 'error');
            resetGenerateButton();
            UI.hideStageProgress();
            currentGenerationJobId = null;

            window.removeEventListener('gve:generate:complete', completeHandler);
            window.removeEventListener('gve:generate:progress', progressHandler);
            window.removeEventListener('gve:generate:stage_complete', stageCompleteHandler);
            window.removeEventListener('gve:generate:failed', failedHandler);
        }
    };
    addEventListener('generate:failed', failedHandler);

    UI.updateStageProgress('A1', 'active');
}

// =============================================================================
// Batch Actions: View & Save
// =============================================================================

/**
 * Load a generated asset into the viewport.
 * @param {string} assetId 
 */
window.viewAsset = function (assetId) {
    if (!assetId) return;

    UI.logOutput(`📺 Loading asset ${assetId} into viewport...`, 'info');

    if (window.load_asset && window.viewportReady) {
        if (window.clear_sdf) window.clear_sdf();
        window.load_asset(`/api/assets/${assetId}/binary`, assetId);
        syncExternalViewport(assetId);
    } else {
        UI.logOutput('⚠️ Viewport not ready', 'warning');
    }
};

/**
 * Promote an asset from draft to library.
 * @param {string} assetId 
 * @param {HTMLElement} btn - Optional button element to update state
 */
window.saveAsset = async function (assetId, btn = null) {
    // If no assetId provided, try to find it from the active document (legacy fallback)
    if (!assetId) {
        const container = document.getElementById('feedback-container');
        assetId = container?.dataset.assetId;
    }

    if (!assetId) return;

    // Use specific button if provided, otherwise fallback to global UI element
    const targetBtn = btn || UI.ui.btnSave;
    const originalText = targetBtn ? targetBtn.textContent : 'Save';

    if (targetBtn) {
        targetBtn.disabled = true;
        targetBtn.textContent = 'Saving...';
    }

    try {
        const res = await fetch(`/api/assets/${assetId}/save`, { method: 'POST' });
        if (res.ok) {
            UI.logOutput(`✅ Asset ${assetId} saved to library!`, 'success');
            if (targetBtn) {
                targetBtn.textContent = 'Saved ✓';
                targetBtn.classList.add('btn-success');
            }
            // Refresh library view if open
            if (document.getElementById('page-library').style.display !== 'none') {
                document.querySelector('.tab-btn.active')?.click();
            }
        } else {
            throw new Error('Save failed');
        }
    } catch (e) {
        UI.logOutput('❌ Failed to save asset', 'error');
        if (targetBtn) {
            targetBtn.disabled = false;
            targetBtn.textContent = originalText;
        }
        console.error(e);
    }
};

// Initialize cost estimate on load
updateCostEstimate();

