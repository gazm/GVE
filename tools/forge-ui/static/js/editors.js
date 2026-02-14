/**
 * editors.js
 * Handles Property Editor interactivity: Accordion, DNA Tree, and Oklab Picker.
 */

// Global state for color picker
let activeColorCallback = null;
let activeColorSwatch = null;

// Oklab to RGB conversion for CSS display (Approximate)
// Source: https://bottosson.github.io/posts/oklab/
function oklab_to_linear_srgb(L, a, b) {
    let l_ = L + 0.3963377774 * a + 0.2158037573 * b;
    let m_ = L - 0.1055613458 * a - 0.0638541728 * b;
    let s_ = L - 0.0894841775 * a - 1.2914855480 * b;

    let l = l_ * l_ * l_;
    let m = m_ * m_ * m_;
    let s = s_ * s_ * s_;

    return {
        r: +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
        g: -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
        b: -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s,
    };
}

function linear_srgb_to_srgb(c) {
    if (c <= 0.0031308) {
        return 12.92 * c;
    } else {
        return 1.055 * Math.pow(c, 1.0 / 2.4) - 0.055;
    }
}

function oklabToCss(L, a, b) {
    const lin = oklab_to_linear_srgb(L, a, b);
    const r = Math.max(0, Math.min(255, Math.round(linear_srgb_to_srgb(lin.r) * 255)));
    const g = Math.max(0, Math.min(255, Math.round(linear_srgb_to_srgb(lin.g) * 255)));
    const b_ = Math.max(0, Math.min(255, Math.round(linear_srgb_to_srgb(lin.b) * 255)));
    return `rgb(${r}, ${g}, ${b_})`;
}

class OklabPicker {
    constructor() {
        this.el = document.createElement('div');
        this.el.className = 'oklab-picker';
        this.el.style.display = 'none';
        this.el.innerHTML = `
            <div class="picker-header">Oklab Color</div>
            <div class="picker-preview"></div>
            <div class="picker-controls">
                <div class="control-group">
                    <label>L</label>
                    <input type="range" class="slider-l" min="0" max="1" step="0.01">
                    <span class="val-l">0.00</span>
                </div>
                <div class="control-group">
                    <label>C</label>
                    <input type="range" class="slider-c" min="0" max="0.4" step="0.001"> 
                    <span class="val-c">0.00</span>
                </div>
                <div class="control-group">
                    <label>H</label>
                    <input type="range" class="slider-h" min="0" max="360" step="1">
                    <span class="val-h">0</span>
                </div>
            </div>
            <div class="picker-actions">
                <button class="picker-close">Close</button>
            </div>
        `;
        document.body.appendChild(this.el);

        this.preview = this.el.querySelector('.picker-preview');
        this.inputs = {
            l: this.el.querySelector('.slider-l'),
            c: this.el.querySelector('.slider-c'),
            h: this.el.querySelector('.slider-h'),
        };
        this.displays = {
            l: this.el.querySelector('.val-l'),
            c: this.el.querySelector('.val-c'),
            h: this.el.querySelector('.val-h'),
        };

        // Bind events
        Object.values(this.inputs).forEach(input => {
            input.addEventListener('input', () => this.updateState());
        });

        this.el.querySelector('.picker-close').addEventListener('click', () => this.hide());

        // Close on click outside
        document.addEventListener('mousedown', (e) => {
            if (this.el.style.display !== 'none' && !this.el.contains(e.target) && !activeColorSwatch?.contains(e.target)) {
                this.hide();
            }
        });
    }

    open(swatch, initialOklab, callback) {
        activeColorSwatch = swatch;
        activeColorCallback = callback;

        // Position
        const rect = swatch.getBoundingClientRect();
        this.el.style.top = `${rect.bottom + 8}px`;
        this.el.style.left = `${rect.left}px`;
        this.el.style.display = 'block';

        // Set initial state (Approximation: simplistic reverse not implemented, assuming initial is [L, a, b])
        // We need LCH from Lab to set sliders correctly.
        // L = L
        // C = sqrt(a^2 + b^2)
        // H = atan2(b, a)
        let [L, a, b] = initialOklab || [0.5, 0, 0];
        let C = Math.sqrt(a * a + b * b);
        let H = Math.atan2(b, a) * (180 / Math.PI);
        if (H < 0) H += 360;

        this.inputs.l.value = L;
        this.inputs.c.value = C;
        this.inputs.h.value = H;

        this.updateState();
    }

    hide() {
        this.el.style.display = 'none';
        activeColorSwatch = null;
        activeColorCallback = null;
    }

    updateState() {
        const L = parseFloat(this.inputs.l.value);
        const C = parseFloat(this.inputs.c.value);
        const H_deg = parseFloat(this.inputs.h.value);
        const H_rad = H_deg * (Math.PI / 180);

        // Convert LCH to Lab
        const a = C * Math.cos(H_rad);
        const b = C * Math.sin(H_rad);

        // Update displays
        this.displays.l.textContent = L.toFixed(2);
        this.displays.c.textContent = C.toFixed(3);
        this.displays.h.textContent = Math.round(H_deg);

        // Update preview
        const cssColor = oklabToCss(L, a, b);
        this.preview.style.backgroundColor = cssColor;

        // Callback
        if (activeColorCallback) {
            activeColorCallback([L, a, b], cssColor);
        }
    }
}

// Singleton picker instance
let picker = null;

function initEditors() {
    if (!picker) picker = new OklabPicker();

    // Initialize Accordions
    document.querySelectorAll('.accordion-header').forEach(header => {
        if (header.dataset.accordionBound === '1') return;
        header.dataset.accordionBound = '1';
        header.addEventListener('click', () => {
            const container = header.closest('.accordion-container');
            if (!container) return;

            const parentItem = header.parentElement;
            if (!parentItem) return;

            const willOpen = !parentItem.classList.contains('active');
            // Collapse all others
            container.querySelectorAll('.accordion-item').forEach(item => {
                if (item !== parentItem) {
                    item.classList.remove('active');
                }
            });
            // Toggle clicked (supports all-collapsed state)
            parentItem.classList.toggle('active', willOpen);
        });
    });

    // Initialize DNA Trees
    document.querySelectorAll('.dna-editor-root').forEach(root => {
        const rawJson = root.dataset.dna;
        if (rawJson) {
            try {
                const data = JSON.parse(rawJson);
                renderDNATreeInteractive(data, root.querySelector('.tree-content'));
            } catch (e) {
                console.error("Failed to parse DNA", e);
                root.querySelector('.tree-content').innerHTML = `<div class="error">Invalid DNA Data</div>`;
            }
        }
    });
}

function renderDNATree(data, container) {
    container.innerHTML = '';
    const tree = createNodes(data);
    container.appendChild(tree);
}

function createNodes(data) {
    const ul = document.createElement('div');
    ul.className = 'tree-list';

    if (typeof data !== 'object' || data === null) {
        // Primitive value
        return createValueInput(data);
    }

    Object.entries(data).forEach(([key, value]) => {
        const item = document.createElement('div');
        item.className = 'tree-node';

        const header = document.createElement('div');
        header.className = 'node-header';

        // Key Label
        const label = document.createElement('span');
        label.className = 'node-key';
        label.textContent = key;

        if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
            // Nested Object -> Collapsible
            item.classList.add('collapsible');
            item.classList.add('expanded'); // Default expanded

            const toggle = document.createElement('span');
            toggle.className = 'caret';
            toggle.textContent = '▶';
            toggle.onclick = (e) => {
                e.stopPropagation();
                item.classList.toggle('expanded');
            };

            header.appendChild(toggle);
            header.appendChild(label);
            item.appendChild(header);

            const children = createNodes(value);
            children.className += ' node-children';
            item.appendChild(children);
        } else if (Array.isArray(value)) {
            // Array - Check if it's a Color
            if ((key.includes('color') || key === 'c') && value.length >= 3 && typeof value[0] === 'number') {
                // Color Field
                header.appendChild(label);

                const swatch = document.createElement('div');
                swatch.className = 'color-swatch-container';
                const preview = document.createElement('div');
                preview.className = 'color-swatch';

                // Initial color
                const [L, a, b] = value;
                preview.style.backgroundColor = oklabToCss(L, a, b);

                // Hidden Input for value holding
                const input = document.createElement('input');
                input.type = 'hidden';
                input.className = 'dna-value';
                input.name = key; // Path handling needed for deep saving?
                input.value = JSON.stringify(value);
                input.dataset.type = 'color'; // Marker

                swatch.onclick = (e) => {
                    e.stopPropagation();
                    const currentVal = JSON.parse(input.value);
                    picker.open(swatch, currentVal, (newLab, css) => {
                        preview.style.backgroundColor = css;
                        input.value = JSON.stringify(newLab);
                        // Trigger change event if needed for auto-save
                    });
                };

                swatch.appendChild(preview);
                swatch.appendChild(input);
                header.appendChild(swatch);
                item.appendChild(header);

            } else {
                // Regular Array -> Treat as object for now or primitives list
                // Simplified: Render as generic object
                item.classList.add('collapsible');
                // ... same as object logic logic for now, or simplified
                header.appendChild(label);
                const valDisp = document.createElement('span');
                valDisp.className = 'array-preview';
                valDisp.textContent = `[${value.length}]`;
                header.appendChild(valDisp);
                item.appendChild(header);

                const children = createNodes(value); // recurses with index keys
                children.className += ' node-children';
                item.appendChild(children);
            }
        } else {
            // Leaf Value
            header.appendChild(label);
            const input = createValueInput(value);
            input.classList.add('dna-value');
            if (key === 'id' || key === 'type') input.readOnly = true;
            header.appendChild(input);
            item.appendChild(header);
        }

        ul.appendChild(item);
    });

    return ul;
}

function createValueInput(val) {
    const input = document.createElement('input');
    if (typeof val === 'number') {
        input.type = 'number';
        input.step = 'any';
        input.value = val;
    } else {
        input.type = 'text';
        input.value = val;
    }
    return input;
}

// Scrape DNA back from DOM
window.collectDNA = function (rootElement) {
    // This is a naive reconstruction. A better way would be modifying a data model connected to the DOM.
    // However, given the recursive DOM structure:

    function parseNode(node) {
        // If node has inputs directly, it's a leaf? No, our structure is .tree-list -> .tree-node
        // A tree-node can be an object (has .node-children) or leaf (has value input).

        // This is complex to scrape effectively without keeping a reference to the data.
        // For this MVP, we might need to rely on the fact that we edit the 'values' and we can just 
        // traverse the object structure we generated.

        // Alternative: Pass the original data object around and mutate it in place during edit? 
        // YES, that is much robust.
    }

    return null; // TODO: Implement extraction or data binding.
}

// Better Approach: Mutate data in place.
// Let's redefine createNodes to accept a 'context' function or bind inputs to the data object.

function getNodeAtPath(obj, path) {
    let cur = obj;
    for (const p of path.split('.')) {
        if (cur == null) return null;
        const idx = parseInt(p, 10);
        cur = isNaN(idx) ? cur[p] : cur[idx];
    }
    return cur;
}

function renderDNATreeInteractive(data, container, onChange) {
    container.innerHTML = '';
    const rootEl = container.closest('[id^="dna-editor-"]') || container.parentElement;
    const cardId = rootEl?.id?.replace('dna-editor-', '') || '';
    let selectedNodePath = null;

    function selectNode(nodeId, nodeInfo) {
        const center = nodeInfo?.center;
        const path = nodeInfo?.path ?? null;
        selectedNodePath = path;
        document.querySelectorAll('.node-header.selectable').forEach(h => h.classList.remove('selected'));
        const header = container.querySelector(`[data-node-id="${nodeId}"]`);
        if (header) header.classList.add('selected');
        if (window.gve_wasm?.set_selected_node_pos && center) {
            window.gve_wasm.set_selected_node_pos(center[0], center[1], center[2]);
        }
        const xyzSection = container.querySelector('.node-xyz-section');
        if (xyzSection) {
            xyzSection.style.display = header ? 'block' : 'none';
            const [xIn, yIn, zIn] = xyzSection.querySelectorAll('input[data-xyz]');
            if (center && xIn) {
                xIn.value = center[0].toFixed(4);
                yIn.value = center[1].toFixed(4);
                zIn.value = center[2].toFixed(4);
            }
        }
    }

    /** Update in-memory node position only. No save, no compile, no asset reload. */
    function syncPositionToDNA(pos) {
        if (!selectedNodePath || !data) return;
        const node = getNodeAtPath(data, selectedNodePath);
        if (!node) return;
        if (!node.transform) node.transform = {};
        node.transform.pos = [pos[0], pos[1], pos[2]];
        if (onChange) onChange();
    }

    function buildNode(obj, key, parentObj) {
        const item = document.createElement('div');
        item.className = 'tree-node';

        if (typeof obj === 'object' && obj !== null) {
            // Branch
            const header = document.createElement('div');
            header.className = 'node-header';

            const nodeId = obj.id;
            if (nodeId && typeof nodeId === 'string') {
                header.classList.add('selectable');
                header.dataset.nodeId = nodeId;
                header.title = `Select node (show gizmo at origin)`;
                header.onclick = (e) => {
                    if (e.target.classList.contains('caret')) return;
                    e.stopPropagation();
                    fetch(`/api/assets/${cardId}/dna/nodes`)
                        .then(r => r.json())
                        .then(({ nodes }) => {
                            const n = nodes?.find(x => x.id === nodeId);
                            selectNode(nodeId, n || null);
                        })
                        .catch(() => selectNode(nodeId, null));
                };
            }

            // Toggle
            const toggle = document.createElement('span');
            toggle.className = 'caret down';
            toggle.textContent = '▼';
            toggle.onclick = (e) => {
                e.stopPropagation();
                item.classList.toggle('collapsed');
                toggle.textContent = item.classList.contains('collapsed') ? '▶' : '▼';
            };

            const label = document.createElement('span');
            label.className = 'node-key';
            label.textContent = key !== undefined ? key : 'Root';

            header.appendChild(toggle);
            header.appendChild(label);

            // Vector/Color special handling
            if (Array.isArray(obj) && (key?.includes('color') || key === 'c') && obj.length >= 3 && typeof obj[0] === 'number') {
                // COLOR PICKER OVERRIDE
                const swatch = document.createElement('div');
                swatch.className = 'color-swatch-container';
                const preview = document.createElement('div');
                preview.className = 'color-swatch';
                const [L, a, b] = obj;
                preview.style.backgroundColor = oklabToCss(L, a, b);

                swatch.onclick = (e) => {
                    e.stopPropagation();
                    if (!picker) picker = new OklabPicker();
                    picker.open(swatch, obj, (newLab, css) => {
                        preview.style.backgroundColor = css;
                        // Mutation
                        obj[0] = newLab[0];
                        obj[1] = newLab[1];
                        obj[2] = newLab[2];
                        if (onChange) onChange();
                    });
                };
                swatch.appendChild(preview);
                header.appendChild(swatch);
                // Don't render children for color array
            } else {
                // Render Children
                const childrenContainer = document.createElement('div');
                childrenContainer.className = 'node-children';

                Object.keys(obj).forEach(k => {
                    childrenContainer.appendChild(buildNode(obj[k], k, obj));
                });

                item.appendChild(header);
                item.appendChild(childrenContainer);
                return item;
            }
            item.appendChild(header);
        } else {
            // Leaf
            const header = document.createElement('div');
            header.className = 'node-header leaf';

            const label = document.createElement('span');
            label.className = 'node-key';
            label.textContent = key;

            const input = document.createElement('input');
            input.className = 'dna-value';

            if (typeof obj === 'number') {
                input.type = 'number';
                input.step = 'any';
                input.value = obj;
                input.onchange = (e) => {
                    parentObj[key] = parseFloat(e.target.value);
                    if (onChange) onChange();
                };
            } else if (typeof obj === 'string') {
                input.type = 'text';
                input.value = obj;
                if (key === 'id' || key === 'type') {
                    // input.disabled = true; // Maybe allow editing?
                }
                input.onchange = (e) => {
                    parentObj[key] = e.target.value;
                    if (onChange) onChange();
                };
            }

            header.appendChild(label);
            header.appendChild(input);
            item.appendChild(header);
        }

        return item;
    }

    container.appendChild(buildNode(data, undefined, null));

    // XYZ position section (shown when node selected)
    const xyzSection = document.createElement('div');
    xyzSection.className = 'node-xyz-section';
    xyzSection.style.display = 'none';
    xyzSection.innerHTML = `
        <div class="node-xyz-header">Node Position</div>
        <div class="node-xyz-inputs">
            <label>X <input type="number" data-xyz="x" step="0.01"></label>
            <label>Y <input type="number" data-xyz="y" step="0.01"></label>
            <label>Z <input type="number" data-xyz="z" step="0.01"></label>
        </div>
    `;
    const [xIn, yIn, zIn] = xyzSection.querySelectorAll('input[data-xyz]');
    [xIn, yIn, zIn].forEach((input, i) => {
        input.addEventListener('change', () => {
            const pos = [parseFloat(xIn.value) || 0, parseFloat(yIn.value) || 0, parseFloat(zIn.value) || 0];
            if (window.gve_wasm?.set_selected_node_pos) {
                window.gve_wasm.set_selected_node_pos(pos[0], pos[1], pos[2]);
            }
            syncPositionToDNA(pos);
        });
    });
    container.appendChild(xyzSection);

    // Wire gizmo drag to update XYZ inputs and sync to DNA
    window._onGizmoDrag = (pos) => {
        const [xIn, yIn, zIn] = xyzSection.querySelectorAll('input[data-xyz]');
        if (pos && pos.length >= 3 && xIn) {
            xIn.value = pos[0].toFixed(4);
            yIn.value = pos[1].toFixed(4);
            zIn.value = pos[2].toFixed(4);
            syncPositionToDNA(pos);
        }
    };
    window._onGizmoDragEnd = () => {
        if (window.gve_wasm?.get_selected_node_pos) {
            const pos = window.gve_wasm.get_selected_node_pos();
            if (pos && pos.length >= 3) {
                syncPositionToDNA(pos);
            }
        }
    };
}

// Public API
window.ForgeEditors = {
    init: initEditors,
    renderDNA: renderDNATreeInteractive
};

// Initialize on first page load and after HTMX swaps.
document.addEventListener('DOMContentLoaded', initEditors);
document.body.addEventListener('htmx:afterSwap', initEditors);
