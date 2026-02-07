// Studio GenAI UI - DOM manipulation and UI states
// Separated from workflow logic

// =============================================================================
// DOM Elements
// =============================================================================

export const ui = {
    // Inputs
    promptInput: document.getElementById('ai-prompt'),
    skipConceptCheckbox: document.getElementById('skip-concept'),
    qualitySlider: document.getElementById('ai-quality'),
    qualityValue: document.getElementById('quality-value'),

    // Buttons
    btnGenerate: document.getElementById('btn-generate'),
    btnGenerateText: document.getElementById('btn-generate-text'),
    btnApprove: document.getElementById('btn-approve-concept'),
    btnRegenerate: document.getElementById('btn-regenerate-concept'),
    btnCancel: document.getElementById('btn-cancel-concept'),
    btnSave: document.getElementById('btn-save-draft'),
    btnAddChain: document.getElementById('btn-add-to-chain'),

    // Containers
    conceptPreview: document.getElementById('concept-preview'),
    conceptImage: document.getElementById('concept-image'),
    conceptPromptDisplay: document.getElementById('concept-prompt-display'),
    regenerateDialog: document.getElementById('regenerate-dialog'),
    stageProgress: document.getElementById('stage-progress'),
    feedbackContainer: document.getElementById('feedback-container'),
    costEstimate: document.getElementById('cost-estimate'),
    estimateTime: document.querySelector('.estimate-time'),
    outputLog: document.getElementById('generation-output'),
    materialSuggestions: document.getElementById('material-suggestions'),

    // Selectors
    typeBtns: document.querySelectorAll('.type-btn'),
    styleChips: document.querySelectorAll('.style-chips .chip'),
    modeBtns: document.querySelectorAll('.mode-btn'),
    starBtns: document.querySelectorAll('.star-btn'),
};

// =============================================================================
// UI State Updates
// =============================================================================

/**
 * Log a message to the output panel
 */
export function logOutput(message, type = 'info') {
    const timestamp = new Date().toLocaleTimeString();
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.innerHTML = `<span class="timestamp">[${timestamp}]</span> <span class="message">${message}</span>`;

    // Remove empty state if present
    const emptyState = ui.outputLog.querySelector('.output-empty');
    if (emptyState) emptyState.remove();

    ui.outputLog.appendChild(entry);
    ui.outputLog.scrollTop = ui.outputLog.scrollHeight;
}

/**
 * Clear the output log
 */
export function clearOutput() {
    ui.outputLog.innerHTML = '<p class="output-empty">Generation output will appear here...</p>';
}

/**
 * Update the generate button state and text
 */
export function updateGenerateButtonState(isGenerating, mode = 'concept', textOverride = null) {
    ui.btnGenerate.disabled = isGenerating;

    if (textOverride) {
        ui.btnGenerate.innerHTML = textOverride;
        return;
    }

    if (mode === 'direct') {
        ui.btnGenerate.innerHTML = '<span class="icon">⚡</span> <span id="btn-generate-text">Generate Asset</span>';
    } else {
        ui.btnGenerate.innerHTML = '<span class="icon">🎨</span> <span id="btn-generate-text">Generate Concept</span>';
    }
}

/**
 * Show concept image preview
 */
export function showConceptPreview(imageBase64, prompt) {
    if (ui.conceptImage) ui.conceptImage.src = `data:image/png;base64,${imageBase64}`;
    if (ui.conceptPromptDisplay) ui.conceptPromptDisplay.textContent = `"${prompt}"`;
    if (ui.conceptPreview) ui.conceptPreview.style.display = 'block';
}

/**
 * Hide concept preview
 */
export function hideConceptPreview() {
    if (ui.conceptPreview) ui.conceptPreview.style.display = 'none';
    if (ui.regenerateDialog) ui.regenerateDialog.style.display = 'none';
}

/**
 * Show regenerate dialog
 */
export function showRegenerateDialog() {
    if (ui.regenerateDialog) {
        ui.regenerateDialog.style.display = 'block';
        const feedbackInput = document.getElementById('regenerate-feedback');
        if (feedbackInput) {
            feedbackInput.value = '';
            feedbackInput.focus();
        }
    }
}

/**
 * Hide regenerate dialog
 */
export function hideRegenerateDialog() {
    if (ui.regenerateDialog) ui.regenerateDialog.style.display = 'none';
}

// =============================================================================
// Stage Progress UI
// =============================================================================

export function showStageProgress() {
    if (ui.stageProgress) ui.stageProgress.style.display = 'flex';
}

export function hideStageProgress() {
    if (ui.stageProgress) ui.stageProgress.style.display = 'none';
}

export function resetStageProgress() {
    document.querySelectorAll('.stage-item').forEach(item => {
        item.classList.remove('active', 'complete');
        item.classList.add('pending');
    });
}

export function updateStageProgress(stage, status) {
    const stageEl = document.querySelector(`.stage-item[data-stage="${stage}"]`);
    if (stageEl) {
        stageEl.classList.remove('pending', 'active', 'complete');
        stageEl.classList.add(status);
    }
}

// =============================================================================
// Feedback & Cost UI
// =============================================================================

export function showFeedbackUI(assetId) {
    if (ui.feedbackContainer) {
        ui.feedbackContainer.style.display = 'flex';
        ui.feedbackContainer.dataset.assetId = assetId;

        // Reset state
        ui.starBtns.forEach(b => b.classList.remove('active'));

        if (ui.btnSave) {
            ui.btnSave.textContent = 'Save to Library';
            ui.btnSave.disabled = false;
        }

        if (ui.btnAddChain) {
            ui.btnAddChain.style.display = 'block';
            // Note: onclick handler needs to be attached by controller
        }
    }
}

export function updateRatingVisuals(rating) {
    ui.starBtns.forEach((btn, idx) => {
        btn.classList.toggle('active', idx < rating);
    });
}

export function updateCostDisplay(cost_usd, time_sec) {
    if (ui.costEstimate) ui.costEstimate.textContent = `$${cost_usd.toFixed(2)}`;
    if (ui.estimateTime) ui.estimateTime.textContent = `~${time_sec}s`;
}

export function updateCostDisplayError() {
    if (ui.costEstimate) ui.costEstimate.textContent = '$--';
    if (ui.estimateTime) ui.estimateTime.textContent = '~--s';
}

export function renderMaterialSuggestions(materials) {
    if (ui.materialSuggestions) {
        if (materials.length > 0) {
            ui.materialSuggestions.innerHTML = materials.map(m =>
                `<button class="chip" onclick="window.addMaterial('${m}')">${m}</button>`
            ).join('');
        } else {
            ui.materialSuggestions.innerHTML = '<span class="chip-placeholder">No suggestions for this prompt</span>';
        }
    }
}

export function renderMaterialSuggestionsError() {
    if (ui.materialSuggestions) {
        ui.materialSuggestions.innerHTML = '<span class="chip-placeholder">Failed to load suggestions</span>';
    }
}

// Make logOutput globally available for other modules if needed (optional)
window.logOutput = logOutput;
window.clearOutput = clearOutput;
