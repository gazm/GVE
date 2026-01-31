# Forge Editor: Card-Chain Assembly Workflow

**Philosophy:** "Compose, Don't Generate." Assets are assembled from reusable component cards, where each card either references a library item or triggers AI generation for only that specific component.

**Related Docs:**
- [Forge Editor Overview](./forge-editor.md) - Main tool documentation
- [Component Libraries](./forge-libraries.md) - Geometry, materials, textures, audio
- [AI Pipeline](../workflows/ai-pipeline.md) - How AI generates components

---

## **The Card System**

### Visual Interface

```
┌─────────────────────────────────────────────────────────────┐
│ Asset Assembly: "AK-47"                     [Generate Asset]│
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   │
│  │ GEOMETRY     │ → │ MATERIAL     │ → │ AUDIO        │   │
│  │ CARD         │   │ CARD         │   │ CARD         │   │
│  └──────────────┘   └──────────────┘   └──────────────┘   │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ [1] GEOMETRY CARD                                  │    │
│  ├────────────────────────────────────────────────────┤    │
│  │ Component: Receiver                                │    │
│  │                                                    │    │
│  │ Source:                                            │    │
│  │  ● Library: "AK Receiver Pattern" ✓                │    │
│  │  ○ Generate New (AI)                               │    │
│  │                                                    │    │
│  │ [Preview: 3D wireframe]                            │    │
│  │ Tags: weapon, rifle, receiver                      │    │
│  │ Cost: $0 (cached)  Time: instant                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ [2] MATERIAL CARD                                  │    │
│  ├────────────────────────────────────────────────────┤    │
│  │ Target: Receiver (from Card 1)                     │    │
│  │                                                    │    │
│  │ Material Base:                                     │    │
│  │  ● Library: "ASTM_A36 (Steel)" ✓                   │    │
│  │  ○ Custom Specification                            │    │
│  │                                                    │    │
│  │ Texture:                                           │    │
│  │  ● Library: "Rusty Steel ★★★★" ✓                   │    │
│  │  ○ Generate New (AI)                               │    │
│  │  ○ Procedural                                      │    │
│  │                                                    │    │
│  │ Wear Modifiers:                                    │    │
│  │  edge_wear: [====·····] 0.7                        │    │
│  │  cavity_grime: [===······] 0.5                     │    │
│  │                                                    │    │
│  │ Cost: $0 (cached)  Time: instant                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  ┌────────────────────────────────────────────────────┐    │
│  │ [3] GEOMETRY CARD                                  │    │
│  ├────────────────────────────────────────────────────┤    │
│  │ Component: Stock                                   │    │
│  │                                                    │    │
│  │ Source:                                            │    │
│  │  ○ Library                                         │    │
│  │  ● Generate New (AI) ⚠                             │    │
│  │    Prompt: "Wooden rifle stock with grip texture" │    │
│  │                                                    │    │
│  │ Cost: $0.03  Time: ~3s                             │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
│  [+ Add Component Card]                                    │
│                                                             │
│  ────────────────────────────────────────────────────────  │
│  Total Cost: $0.03   Total Time: ~3s                       │
│  Cached Components: 3/4 (75%)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## **Key Concepts**

1. **Chain of Cards:** Each card represents one component (geometry, material, texture, audio)
2. **Library-First:** Default to library items (instant, free)
3. **Generate on Demand:** Only AI-generate missing components
4. **Live Cost Estimation:** See cost/time BEFORE generating
5. **Reusable:** Save entire card chains as "Recipes"

---

## **Smart Card Suggestions**

When user enters a prompt, the system analyzes it and suggests pre-filled cards:

```python
def suggest_cards(user_prompt: str) -> CardChain:
    # Analyze prompt
    weapon_type = classify_weapon(user_prompt)  # "rifle"
    weapon_model = extract_model(user_prompt)   # "AK-47"
    
    cards = []
    
    # Card 1: Main geometry
    if has_geometry_in_library(weapon_model):
        cards.append(GeometryCard(
            component="receiver",
            source="library",
            library_id="ak_receiver_pattern",
            cost=0,
            time=0
        ))
    else:
        cards.append(GeometryCard(
            component="receiver",
            source="ai-generate",
            prompt=f"{weapon_model} receiver",
            cost=0.04,
            time=5
        ))
    
    # Card 2: Material for receiver
    cards.append(MaterialCard(
        target="receiver",
        material="ASTM_A36",  # From material DB
        texture="rusty_steel_001",  # From texture library
        cost=0
    ))
    
    return CardChain(cards)
```

**UI Shows:**
```
┌─────────────────────────────────────────────┐
│ Suggested Card Chain for "AK-47":          │
├─────────────────────────────────────────────┤
│ ✓ Receiver (Library: ak_receiver_pattern)  │
│ ✓ Material (Rusty Steel ★★★★)              │
│ ⚠ Stock (Generate New - $0.03, 3s)         │
│ ✓ Material (Worn Oak ★★★★)                 │
│                                             │
│ [Accept All] [Customize]                   │
└─────────────────────────────────────────────┘
```

---

## **AI Generation Optimization**

**Only Generate What's Missing:**

```python
def generate_asset(card_chain: CardChain) -> Asset:
    components = {}
    
    for card in card_chain:
        if card.source == "library":
            # Instant retrieval
            components[card.id] = library.get(card.library_id)
            
        elif card.source == "ai-generate":
            # Build minimal prompt using existing context
            context = {
                "existing_components": list(components.keys()),
                "material_constraints": [c.material for c in card_chain if c.type == "material"],
                "style": infer_style_from_chain(card_chain)
            }
            
            # AI only generates THIS component
            prompt = build_contextual_prompt(
                card.prompt,
                context=context
            )
            
            components[card.id] = ai_pipeline.generate(
                prompt=prompt,
                constraints=context
            )
    
    # Assemble final asset
    return assemble_components(components)
```

**Example Comparison:**

**Without Card Context (wasteful):**
```
Prompt: "Generate AK-47"
→ AI generates EVERYTHING: receiver, stock, trigger, barrel, magazine
→ Cost: $0.12, Time: 15s
```

**With Card Context (efficient):**
```
Card 1 (Library): Receiver geometry ✓
Card 2 (Library): Material/texture ✓
Card 3 (AI): Stock only
  Prompt: "Wooden rifle stock compatible with AK receiver"
  Context: {"parent_geometry": "ak_receiver_pattern", "attachment_point": [0, -0.2, 0]}
→ AI generates ONLY stock
→ Cost: $0.03, Time: 3s
```

**75% cost reduction!**

---

## **Card Dependencies & Validation**

Cards can reference outputs from previous cards:

```
Card 1: Receiver Geometry (library)
  ↓ provides: attachment_points, material_zones
  
Card 2: Material for Receiver (library)
  ← requires: Card 1 output
  ↓ provides: surface_properties
  
Card 3: Stock Geometry (AI)
  ← requires: Card 1.attachment_points["stock_mount"]
  ↓ provides: stock_geometry
  
Card 4: Scope Rail (library)
  ← requires: Card 1.attachment_points["top_rail"]
```

**Validation Rules:**
```python
class CardValidator:
    def validate_chain(self, cards: list[Card]) -> ValidationResult:
        errors = []
        
        for i, card in enumerate(cards):
            # Check dependencies
            for dep in card.requires:
                if not any(c.provides(dep) for c in cards[:i]):
                    errors.append(
                        f"Card {i+1} requires '{dep}' but no prior card provides it"
                    )
            
            # Check material targets exist
            if card.type == "material":
                if card.target not in [c.id for c in cards if c.type == "geometry"]:
                    errors.append(
                        f"Material card targets '{card.target}' but no geometry card defines it"
                    )
        
        return ValidationResult(errors)
```

---

## **Recipe System**

Save card chains as reusable templates:

```json
{
  "recipe_name": "Standard AK-47 (Worn)",
  "tags": ["weapon", "rifle", "ak", "assault-rifle"],
  "cards": [
    {
      "type": "geometry",
      "component": "receiver",
      "source": "library",
      "library_id": "ak_receiver_pattern"
    },
    {
      "type": "material",
      "target": "receiver",
      "material_id": "ASTM_A36",
      "texture_id": "rusty_steel_001",
      "modifiers": {
        "edge_wear": 0.7,
        "cavity_grime": 0.5
      }
    }
  ],
  "metadata": {
    "total_cost": 0,
    "total_time_sec": 0,
    "created_by": "user_123",
    "usage_count": 47,
    "rating": 4.5
  }
}
```

---

## **Simple Workflow Example**

**User Input:**
- Asset Type: `prop`
- Text Prompt: `"chair"`
- Texture: `"oak"`

**System Generates:**

```
┌─────────────────────────────────────────────────────────────┐
│ Asset Assembly: "Chair"                     [Generate Asset]│
├─────────────────────────────────────────────────────────────┤
│ ┌────────────────────────────────────────────────────┐      │
│ │ [1] GEOMETRY CARD                                  │      │
│ │ Component: Chair Frame                             │      │
│ │ Source: Generate New (AI) ⚠                        │      │
│ │ Prompt: "Dining chair with 4 legs"                 │      │
│ │ Cost: $0.04  Time: ~4s                             │      │
│ └────────────────────────────────────────────────────┘      │
│                                                             │
│ ┌────────────────────────────────────────────────────┐      │
│ │ [2] MATERIAL CARD                                  │      │
│ │ Target: Chair Frame                                │      │
│ │ Material Base: WOOD_OAK ✓ (auto-selected)          │      │
│ │ Texture: "Natural Oak Wood ★★★★" ✓                 │      │
│ │ Cost: $0  Time: instant                            │      │
│ └────────────────────────────────────────────────────┘      │
│                                                             │
│ Total: $0.04, ~4s  |  Cached: 1/2 (50%)                    │
└─────────────────────────────────────────────────────────────┘
```

**After Generation:**
```
Generation Complete!
  
Save Components:
☑ Geometry: "Dining Chair Frame"
  Tags: furniture, chair, simple
☑ Recipe: "Simple Oak Chair" (2 cards)

[Save & Close]
```

**Next Time User Types "chair":**
```
Suggested: Use "Simple Oak Chair" recipe
Cost: $0, Time: instant ✓
```

---

## **Benefits Summary**

### Efficiency Gains

```
Traditional (monolithic prompt):
  "Generate AK-47"
  → Cost: $0.12
  → Time: 15s
  → Reusability: 0%

Card-Chain (modular):
  Receiver (library) + Material (library) + Stock (AI) + Material (library)
  → Cost: $0.03 (75% reduction)
  → Time: 3s (80% faster)
  → Reusability: 75%
  
Recipe (all library):
  4 cards, all from library
  → Cost: $0
  → Time: instant
  → Reusability: 100%
```

### UX Advantages

- ✅ **Granular Control:** Tweak individual components
- ✅ **Library Growth:** Every generated component can be saved for reuse
- ✅ **Cost Transparency:** See exactly what you're paying for
- ✅ **Iterative Refinement:** Regenerate only problematic parts
- ✅ **Recipes:** Share/reuse entire workflows
- ✅ **Learning:** System gets smarter as library grows

---

**Version:** 1.0  
**Last Updated:** January 25, 2026  
**Related:** [Component Libraries](./forge-libraries.md) | [Texture System](./texture-library-implementation.md)
