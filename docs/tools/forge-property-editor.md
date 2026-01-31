# Forge Editor: Property Table Editor

**Purpose:** Type-aware table UI for editing asset struct values with contextual input controls—sliders for ranges, dropdowns for enums, and textboxes for strings.

**Related Docs:**
- [Card-Chain Workflow](./forge-card-chain.md) - Asset assembly system
- [Component Libraries](./forge-libraries.md) - Reusable components
- [Engine API](../architecture/engine-api.md) - Rust struct definitions

---

## **Architecture**

```
┌─────────────────────────────────────────────────────────────┐
│  Property Table                                             │
│  (htmx + Server-Rendered HTML)                              │
├─────────────────────────────────────────────────────────────┤
│  Input Controls:                                            │
│  ├── Sliders: f32/f64 ranged values                         │
│  ├── Dropdowns: Enums, material specs                       │
│  ├── Textboxes: Strings, identifiers                        │
│  ├── Color Pickers: [u8; 3], Oklab                          │
│  ├── Vec3 Editors: Position, scale, rotation                │
│  └── Toggles: bool fields                                   │
└─────────────────────────────────────────────────────────────┘
         ↓ hx-post (on change)
┌─────────────────────────────────────────────────────────────┐
│  FastAPI Backend                                            │
│  - Validate against Pydantic schema                         │
│  - Return updated HTML partial                              │
│  - Trigger recompile if needed                              │
└─────────────────────────────────────────────────────────────┘
```

---

## **Type-to-Control Mapping**

| Rust Type | Control | Notes |
|-----------|---------|-------|
| `f32`, `f64` | **Slider** | Uses `#[schemars(range(min, max))]` for bounds |
| `f32` (normalized) | **Slider 0.0–1.0** | Roughness, metallic, opacity |
| `String` | **Textbox** | With optional validation regex |
| `enum` | **Dropdown** | Auto-populated from variants |
| `Option<T>` | **Checkbox + Control** | Enable/disable optional fields |
| `bool` | **Toggle** | On/off switch |
| `[u8; 3]` | **Color Picker** | sRGB color |
| `[f32; 3]` | **Vec3 Editor** | X/Y/Z inputs with drag |
| `MaterialSpecId` | **Material Dropdown** | Searchable material library |

---

## **UI Layout**

### Compact Table View

```
┌─────────────────────────────────────────────────────────────┐
│ Properties: MaterialSpec                     [Expand All]   │
├─────────────────────────────────────────────────────────────┤
│ ▸ Basic                                                     │
│ ├── spec_id          [ASTM_A36        ▼] dropdown           │
│ ├── display_name     [Structural Steel   ] textbox          │
│                                                             │
│ ▸ Physical Properties                                       │
│ ├── density_kg_m3    [====●=======] 7850    slider          │
│ ├── youngs_modulus   [======●=====] 200.0   slider (GPa)    │
│ ├── poissons_ratio   [===●========] 0.26    slider 0-0.5    │
│                                                             │
│ ▸ Audio Properties                                          │
│ ├── damping_coeff    [==●=========] 0.02    slider          │
│ ├── resonance_min_hz [====●=======] 200.0   slider          │
│ ├── resonance_max_hz [=======●====] 4000.0  slider          │
│                                                             │
│ ▸ Visual Properties                                         │
│ ├── base_color       [███] #8B7355         color picker     │
│ ├── metallic         [========●===] 0.9    slider 0-1       │
│ ├── roughness        [====●=======] 0.4    slider 0-1       │
│ └── color_mode       [Oklab       ▼]       dropdown         │
└─────────────────────────────────────────────────────────────┘
```

### Inline Editing

```
┌─────────────────────────────────────────────────────────────┐
│ asset.name      │ My Custom Weapon                          │
│─────────────────│───────────────────────────────────────────│
│ asset.category  │ [Weapon         ▼]                        │
│─────────────────│───────────────────────────────────────────│
│ material_spec   │ [ASTM_A36       ▼] 🔍                      │
│─────────────────│───────────────────────────────────────────│
│ roughness       │ [=======●=====] 0.65                      │
│─────────────────│───────────────────────────────────────────│
│ metallic        │ [=========●===] 0.85                      │
└─────────────────────────────────────────────────────────────┘
```

---

## **Control Components**

### Slider (Range Values)

```html
<!-- htmx slider with live preview -->
<div class="property-row">
  <label>roughness</label>
  <input type="range" 
         min="0" max="1" step="0.01" 
         value="0.4"
         name="roughness"
         hx-post="/api/asset/{id}/property"
         hx-trigger="change"
         hx-target="#preview-panel"
         hx-swap="outerHTML">
  <span class="value-display">0.4</span>
</div>
```

**Rust Schema Annotation:**
```rust
#[derive(JsonSchema)]
pub struct MaterialSpec {
    /// Surface roughness (0 = mirror, 1 = diffuse)
    #[schemars(range(min = 0.0, max = 1.0))]
    pub roughness: f32,
}
```

### Dropdown (Enums)

```html
<select name="category"
        hx-post="/api/asset/{id}/property"
        hx-trigger="change"
        hx-target="#card-chain">
  <option value="Architecture">Architecture</option>
  <option value="Weapon" selected>Weapon</option>
  <option value="Vehicle">Vehicle</option>
  <option value="Prop">Prop</option>
  <option value="Character">Character</option>
</select>
```

**Generated from Rust:**
```rust
#[typeshare]
pub enum AssetCategory {
    Architecture,
    Weapon,
    Vehicle,
    Prop,
    Character,
}
```

### Color Picker

```html
<div class="color-property">
  <input type="color" 
         value="#8B7355"
         name="base_color"
         hx-post="/api/asset/{id}/property"
         hx-trigger="change">
  <span class="hex-value">#8B7355</span>
  <span class="rgb-value">(139, 115, 85)</span>
</div>
```

### Vec3 Editor (Position/Scale)

```html
<div class="vec3-editor">
  <label>position</label>
  <div class="vec3-inputs">
    <span>X</span><input type="number" step="0.1" value="0.0" name="position.x">
    <span>Y</span><input type="number" step="0.1" value="1.5" name="position.y">
    <span>Z</span><input type="number" step="0.1" value="0.0" name="position.z">
  </div>
</div>
```

### Toggle (Boolean)

```html
<div class="toggle-property">
  <label>cast_shadows</label>
  <input type="checkbox" 
         name="cast_shadows" 
         checked
         hx-post="/api/asset/{id}/property"
         hx-trigger="change">
</div>
```

### Optional Fields

```html
<div class="optional-property">
  <input type="checkbox" id="enable-material-spec">
  <label for="enable-material-spec">material_spec</label>
  
  <!-- Shown only when checkbox enabled -->
  <select name="material_spec" disabled>
    <option value="">None</option>
    <option value="ASTM_A36">ASTM A36 Steel</option>
    <option value="WOOD_OAK">Oak Wood</option>
  </select>
</div>
```

---

## **Backend API**

### Property Update Endpoint

```python
@app.post("/api/asset/{asset_id}/property", response_class=HTMLResponse)
async def update_property(
    asset_id: str,
    request: Request,
):
    """Update a single property and return updated UI partial."""
    form = await request.form()
    property_name = list(form.keys())[0]
    value = form[property_name]
    
    # Load asset
    asset = await librarian.load_asset(ObjectId(asset_id))
    
    # Validate and update
    updated = update_nested_property(asset, property_name, value)
    
    # Save
    await librarian.save_asset(updated)
    
    # Return updated row HTML
    return templates.TemplateResponse(
        "partials/property_row.html",
        {"property": property_name, "value": value, "asset": updated}
    )
```

### Schema-Driven Form Generation

```python
def generate_property_form(schema: dict, values: dict) -> str:
    """Generate HTML form from JSON Schema + current values."""
    html = []
    
    for name, prop in schema["properties"].items():
        value = values.get(name)
        
        if "enum" in prop:
            # Dropdown
            html.append(render_dropdown(name, prop["enum"], value))
            
        elif prop.get("type") == "number":
            # Check for range
            if "minimum" in prop and "maximum" in prop:
                html.append(render_slider(
                    name, prop["minimum"], prop["maximum"], value
                ))
            else:
                html.append(render_number_input(name, value))
                
        elif prop.get("type") == "boolean":
            html.append(render_toggle(name, value))
            
        elif prop.get("type") == "string":
            html.append(render_textbox(name, value))
            
        elif prop.get("type") == "array" and prop.get("items", {}).get("type") == "number":
            # Vec3 or color
            if len(value) == 3 and all(isinstance(v, int) for v in value):
                html.append(render_color_picker(name, value))
            else:
                html.append(render_vec_editor(name, value))
    
    return "\n".join(html)
```

---

## **Live Preview Integration**

Property changes trigger viewport refresh via WASM bindings:

```javascript
// When slider changes
htmx.on("htmx:afterSwap", (e) => {
    if (e.detail.target.id === "property-table") {
        // Notify WASM viewport to refresh
        wasmViewport.requestPreviewUpdate();
    }
});

// WASM module receives compile progress callbacks
// (registered during WASM initialization)
wasmViewport.onCompileProgress((stage, progress) => {
    updateProgressBar(stage, progress);
});

wasmViewport.onCompileComplete((assetId) => {
    // WASM automatically loads the updated asset
    // and triggers re-render
});
```

**WASM Bindings (Rust):**
```rust
#[wasm_bindgen]
impl ForgeViewport {
    /// Called by backend when property changes are compiled
    pub fn notify_property_updated(&mut self, asset_id: &str, property: &str) {
        self.invalidate_asset(asset_id);
        self.request_redraw();
    }
    
    /// Called when compilation completes
    pub fn notify_compile_complete(&mut self, asset_id: &str, binary_path: &str) {
        self.hot_reload_asset(asset_id, binary_path);
    }
}
```

---

## **Keyboard Shortcuts**

| Shortcut | Action |
|----------|--------|
| `Tab` / `Shift+Tab` | Navigate between properties |
| `↑` / `↓` | Increment/decrement numeric values |
| `Enter` | Confirm edit, move to next |
| `Escape` | Revert to previous value |
| `Ctrl+Z` | Undo last change |
| `Ctrl+Shift+Z` | Redo |

---

## **Validation & Feedback**

### Inline Errors

```html
<div class="property-row error">
  <label>density_kg_m3</label>
  <input type="number" value="-500" class="invalid">
  <span class="error-message">Must be positive</span>
</div>
```

### Constraint Display

```html
<div class="property-row">
  <label>roughness</label>
  <input type="range" min="0" max="1">
  <span class="constraints">0.0 – 1.0</span>
</div>
```

---

## **Collapsible Sections**

Group related properties with htmx-powered expand/collapse:

```html
<details class="property-section" open>
  <summary hx-get="/api/asset/{id}/properties/physical"
           hx-trigger="toggle once"
           hx-target="#physical-props">
    ▸ Physical Properties
  </summary>
  <div id="physical-props">
    <!-- Loaded on first expand -->
  </div>
</details>
```

---

## **Implementation Checklist**

### Controls
- [ ] Slider component with live value display
- [ ] Dropdown from enum variants
- [ ] Color picker (sRGB + hex)
- [ ] Vec3 editor with drag handles
- [ ] Toggle switch for booleans
- [ ] Optional field enable/disable

### Backend
- [ ] Property update endpoint
- [ ] Schema-driven form generation
- [ ] Validation with Pydantic
- [ ] Partial HTML responses

### Integration
- [ ] WebSocket preview refresh
- [ ] Undo/redo stack
- [ ] Keyboard navigation
- [ ] Collapsible property groups

---

**Version:** 1.0  
**Last Updated:** January 26, 2026  
**Related:** [Card-Chain](./forge-card-chain.md) | [Engine API](../architecture/engine-api.md)
