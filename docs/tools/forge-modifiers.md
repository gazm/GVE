# Forge Editor: Modifier Stack

**Purpose:** Domain-space modifiers that deform SDF primitives. Applied as a stack per-node in the DNA tree.

**Related Docs:**
- [Compiler Pipeline](../workflows/compiler-pipeline.md) - How modifiers are evaluated
- [DNA Specification](../data/data-specifications.md) - DNA format

---

## Overview

Modifiers transform the input coordinates before evaluating the base SDF. They "warp" space to create effects like twisting, bending, and tapering without changing the underlying primitive.

```
Input Point → [Modifier Stack] → Warped Point → [Base SDF] → Distance
```

---

## Available Modifiers

### 1. Twist

**Effect:** Rotates points around an axis, with rotation angle proportional to position along that axis.

```json
{
  "type": "twist",
  "axis": "y",
  "rate": 1.0
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `"x"`, `"y"`, or `"z"` | Axis to twist around |
| `rate` | `float` | Rotation rate in radians per unit |

**Example:** `rate: 3.14` = 180° twist per meter

### 2. Bend

**Effect:** Bends the object around an axis, creating a curved deformation.

```json
{
  "type": "bend",
  "axis": "x",
  "angle": 0.5
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `"x"`, `"y"`, or `"z"` | Axis to bend around |
| `angle` | `float` | Bend angle in radians |

### 3. Taper

**Effect:** Scales the cross-section along an axis, creating tapered shapes.

```json
{
  "type": "taper",
  "axis": "y",
  "scale_min": 0.5,
  "scale_max": 1.0
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `"x"`, `"y"`, or `"z"` | Axis along which to taper |
| `scale_min` | `float` | Scale at negative end (0.0-2.0) |
| `scale_max` | `float` | Scale at positive end (0.0-2.0) |

### 4. Mirror

**Effect:** Mirrors the SDF across a plane, creating symmetric shapes.

```json
{
  "type": "mirror",
  "axis": "x"
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `axis` | `"x"`, `"y"`, or `"z"` | Axis to mirror across |

### 5. Round

**Effect:** Rounds/bevels edges by offsetting the SDF.

```json
{
  "type": "round",
  "radius": 0.02
}
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `radius` | `float` | Rounding radius in meters |

---

## DNA Format

Modifiers are applied per-node in the `modifiers` array:

```json
{
  "id": "twisted_barrel",
  "type": "primitive",
  "shape": "cylinder",
  "params": {
    "radius": 0.05,
    "height": 0.5
  },
  "modifiers": [
    {"type": "twist", "axis": "y", "rate": 2.0},
    {"type": "round", "radius": 0.005}
  ]
}
```

**Evaluation Order:** Modifiers are applied in array order (first modifier runs first).

---

## SDF Math

### Twist (around Y axis)

```python
def twist_y(p: vec3, rate: float) -> vec3:
    angle = p.y * rate
    c, s = cos(angle), sin(angle)
    return vec3(c*p.x - s*p.z, p.y, s*p.x + c*p.z)
```

### Bend (around X axis)

```python
def bend_x(p: vec3, k: float) -> vec3:
    c, s = cos(k * p.y), sin(k * p.y)
    return vec3(p.x, c*p.y - s*p.z, s*p.y + c*p.z)
```

### Taper (along Y axis)

```python
def taper_y(p: vec3, scale_min: float, scale_max: float) -> vec3:
    t = (p.y + 1.0) / 2.0  # Normalize to 0-1
    scale = mix(scale_min, scale_max, t)
    return vec3(p.x / scale, p.y, p.z / scale)
```

### Mirror (across X)

```python
def mirror_x(p: vec3) -> vec3:
    return vec3(abs(p.x), p.y, p.z)
```

### Round

```python
def round_sdf(d: float, radius: float) -> float:
    return d - radius
```

---

## UI Integration

In the Forge Editor, modifiers appear in the property panel when a primitive is selected:

```
┌─────────────────────────────────────┐
│ MODIFIERS                     [+]   │
├─────────────────────────────────────┤
│ 1. Twist                       [×]  │
│    Axis: [Y ▼]  Rate: [2.0]         │
├─────────────────────────────────────┤
│ 2. Round                       [×]  │
│    Radius: [0.005]                  │
└─────────────────────────────────────┘
```

---

**Version:** 1.0  
**Last Updated:** January 28, 2026  
**Status:** ✅ Documented, Implementation Pending
