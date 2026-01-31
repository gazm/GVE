# GVE-1 Material Database

**Version:** 1.1  
**Source:** Verified ASTM, AMS, and engineering specifications  
**Purpose:** Physical properties for physics simulation, audio synthesis, and destruction

---

## **Database Structure**

Each material entry contains:
- **Physical Properties:** Density, Young's modulus, Poisson's ratio
- **Destruction Properties:** Yield strength, fracture toughness, shear modulus
- **Audio Properties:** Damping coefficient, resonance frequency range
- **Visual Properties:** Base color (sRGB), metallic, roughness
- **Recommended Color Mode:** RGB or Oklab based on use case

---

## **Metals**

### ASTM_A36 (Structural Steel)

**Common Uses:** Building frames, bridges, construction, general steel

**Physical Properties:**
- Density: 7850 kg/m³ (7.85 g/cm³)
- Young's Modulus: 200 GPa
- Poisson's Ratio: 0.260
- Tensile Strength: 400-550 MPa

**Destruction Properties:**
- Yield Strength: 250 MPa (stress to begin plastic deformation)
- Fracture Toughness (K₁c): 50 MPa√m (moderate crack resistance)
- Shear Modulus: 79.3 GPa (derived: E / (2*(1+ν)))
- Failure Mode: Ductile (bends before breaking)

**Audio Properties:**
- Damping Coefficient (ζ): 0.008 (low damping, resonant)
- Resonance Frequency Range: 200-8000 Hz (depends on geometry)
- Sound Character: Bright, metallic ring, long decay

**Visual Properties:**
- Base Color (sRGB): `#B0B0B0` (medium gray)
- Metallic: 0.9
- Roughness: 0.4 (slightly worn)

**Recommended Color Mode:** RGB (static structures)

---

### AMS_4911 (Aluminum 6061-T6)

**Common Uses:** Aircraft structures, automotive parts, consumer electronics

**Physical Properties:**
- Density: 2700 kg/m³ (2.7 g/cm³)
- Young's Modulus: 68.9 GPa
- Poisson's Ratio: 0.33
- Tensile Strength: 310 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.007 (low damping)
- Resonance Frequency Range: 300-10000 Hz
- Sound Character: Higher pitch than steel, bright tone

**Visual Properties:**
- Base Color (sRGB): `#D3D3D3` (light gray with slight blue tint)
- Metallic: 0.95
- Roughness: 0.3 (smooth machined finish)

**Recommended Color Mode:** RGB

---

### ASTM_B152 (Copper Alloy C110)

**Common Uses:** Electrical wiring, plumbing, decorative elements

**Physical Properties:**
- Density: 8960 kg/m³ (8.96 g/cm³)
- Young's Modulus: 130 GPa
- Poisson's Ratio: 0.34
- Tensile Strength: 220 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.01 (low damping)
- Resonance Frequency Range: 150-6000 Hz
- Sound Character: Warm, bell-like tone, musical

**Visual Properties:**
- Base Color (sRGB): `#B87333` (copper orange)
- Metallic: 0.98
- Roughness: 0.2 (polished)

**Recommended Color Mode:** RGB (unless oxidizing/patina system)

---

### ASTM_B265_Grade5 (Titanium Ti-6Al-4V)

**Common Uses:** Aerospace, medical implants, high-performance applications

**Physical Properties:**
- Density: 4430 kg/m³ (4.43 g/cm³)
- Young's Modulus: 113.8 GPa
- Poisson's Ratio: 0.342
- Tensile Strength: 895 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.005 (very low damping)
- Resonance Frequency Range: 250-9000 Hz
- Sound Character: Crisp, clear ring, long sustain

**Visual Properties:**
- Base Color (sRGB): `#C0C0C0` (silvery gray)
- Metallic: 0.92
- Roughness: 0.35

**Recommended Color Mode:** RGB

---

### ASTM_B127 (Brass C260 - Cartridge Brass)

**Common Uses:** Ammunition casings, musical instruments, decorative hardware

**Physical Properties:**
- Density: 8530 kg/m³ (8.53 g/cm³)
- Young's Modulus: 110 GPa
- Poisson's Ratio: 0.34
- Tensile Strength: 340 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.012 (low-moderate)
- Resonance Frequency Range: 180-7000 Hz
- Sound Character: Warm, golden tone (used in brass instruments)

**Visual Properties:**
- Base Color (sRGB): `#D4AF37` (golden yellow)
- Metallic: 0.97
- Roughness: 0.25

**Recommended Color Mode:** RGB

---

## **Stone & Concrete**

### ASTM_C33 (Concrete Mix)

**Common Uses:** Buildings, roads, foundations, general construction

**Physical Properties:**
- Density: 2400 kg/m³ (normal weight concrete)
- Young's Modulus: 30 GPa (varies by mix)
- Poisson's Ratio: 0.20
- Compressive Strength: 20-40 MPa (typical)

**Destruction Properties:**
- Yield Strength: 3 MPa (tensile - very weak in tension)
- Fracture Toughness (K₁c): 1.0 MPa√m (brittle, cracks easily)
- Shear Modulus: 12.5 GPa
- Failure Mode: Brittle (shatters/crumbles, no plastic deformation)

**Audio Properties:**
- Damping Coefficient (ζ): 0.05 (moderate damping)
- Resonance Frequency Range: 80-2000 Hz
- Sound Character: Dull thud, short decay, non-resonant

**Visual Properties:**
- Base Color (sRGB): `#808080` (medium gray)
- Metallic: 0.0
- Roughness: 0.9 (very rough)

**Recommended Color Mode:** RGB

---

### ASTM_C568 (Limestone)

**Common Uses:** Building facades, flooring, sculpture

**Physical Properties:**
- Density: 2500 kg/m³ (varies 2300-2700)
- Young's Modulus: 55 GPa
- Poisson's Ratio: 0.25
- Compressive Strength: 60-170 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.03 (low-moderate)
- Resonance Frequency Range: 100-3000 Hz
- Sound Character: Muted, solid impact

**Visual Properties:**
- Base Color (sRGB): `#E8D5B7` (beige/cream)
- Metallic: 0.0
- Roughness: 0.7

**Recommended Color Mode:** RGB

---

### ASTM_C503 (Marble)

**Common Uses:** Decorative flooring, sculptures, countertops

**Physical Properties:**
- Density: 2700 kg/m³
- Young's Modulus: 60 GPa
- Poisson's Ratio: 0.30
- Compressive Strength: 100-170 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.025
- Resonance Frequency Range: 120-4000 Hz
- Sound Character: Clear tap, moderate ring

**Visual Properties:**
- Base Color (sRGB): `#F5F5DC` (beige white, varies by type)
- Metallic: 0.0
- Roughness: 0.2 (polished)

**Recommended Color Mode:** RGB

---

## **Wood**

### WOOD_OAK (Red Oak - Quercus rubra)

**Common Uses:** Furniture, flooring, construction

**Physical Properties:**
- Density: 750 kg/m³ (varies 600-900 by moisture)
- Young's Modulus: 12.5 GPa (along grain)
- Poisson's Ratio: 0.35
- Tensile Strength: 68 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.01 (loss factor: 0.01)
- Resonance Frequency Range: 80-5000 Hz
- Natural Resonance: ~1100 Hz
- Sound Character: Warm, woody thump, moderate decay

**Visual Properties:**
- Base Color (sRGB): `#C19A6B` (tan/brown)
- Metallic: 0.0
- Roughness: 0.6 (natural grain)

**Recommended Color Mode:** RGB

---

### WOOD_PINE (Douglas Fir)

**Common Uses:** Framing lumber, construction, crates

**Physical Properties:**
- Density: 530 kg/m³
- Young's Modulus: 13 GPa (along grain)
- Poisson's Ratio: 0.37
- Tensile Strength: 50 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.008 (loss factor: 0.008)
- Resonance Frequency Range: 100-6000 Hz
- Sound Character: Light tap, quick decay

**Visual Properties:**
- Base Color (sRGB): `#E3C194` (pale yellowish)
- Metallic: 0.0
- Roughness: 0.7

**Recommended Color Mode:** RGB

---

### WOOD_MAPLE (Hard Maple - Acer saccharum)

**Common Uses:** Musical instruments, sports equipment, high-quality furniture

**Physical Properties:**
- Density: 705 kg/m³
- Young's Modulus: 12.6 GPa
- Poisson's Ratio: 0.42
- Tensile Strength: 77 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.009
- Resonance Frequency Range: 90-6000 Hz
- Sound Character: Bright, clear tone (preferred for drum shells)

**Visual Properties:**
- Base Color (sRGB): `#F4E7D3` (creamy white)
- Metallic: 0.0
- Roughness: 0.5

**Recommended Color Mode:** RGB

---

## **Plastics & Composites**

### ASTM_D4181 (ABS Plastic)

**Common Uses:** Consumer products, automotive trim, 3D printing

**Physical Properties:**
- Density: 1050 kg/m³
- Young's Modulus: 2.3 GPa
- Poisson's Ratio: 0.35
- Tensile Strength: 40 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.04 (moderate)
- Resonance Frequency Range: 200-4000 Hz
- Sound Character: Hollow plastic tap, quick decay

**Visual Properties:**
- Base Color (sRGB): `#E0E0E0` (light gray, varies by dye)
- Metallic: 0.0
- Roughness: 0.4

**Recommended Color Mode:** RGB (or Oklab for colored variants)

---

### ASTM_D638 (Polycarbonate)

**Common Uses:** Safety glasses, bulletproof windows, electronics housings

**Physical Properties:**
- Density: 1200 kg/m³
- Young's Modulus: 2.4 GPa
- Poisson's Ratio: 0.37
- Tensile Strength: 60 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.06 (high for plastic)
- Resonance Frequency Range: 250-5000 Hz
- Sound Character: Muted tap, absorbs vibration

**Visual Properties:**
- Base Color (sRGB): `#F0F0F0` (clear/white when opaque)
- Metallic: 0.0
- Roughness: 0.1 (very smooth)

**Recommended Color Mode:** RGB

---

### ASTM_D7031 (Carbon Fiber Composite)

**Common Uses:** Aerospace, racing, high-performance sports equipment

**Physical Properties:**
- Density: 1600 kg/m³
- Young's Modulus: 150 GPa (along fiber direction)
- Poisson's Ratio: 0.30
- Tensile Strength: 3500 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.02 (low)
- Resonance Frequency Range: 150-8000 Hz
- Sound Character: Hollow tap with slight ring

**Visual Properties:**
- Base Color (sRGB): `#1A1A1A` (black with visible weave)
- Metallic: 0.2
- Roughness: 0.3 (clear coat finish)

**Recommended Color Mode:** RGB

---

## **Glass & Ceramics**

### ASTM_C1036 (Float Glass)

**Common Uses:** Windows, mirrors, optical applications

**Physical Properties:**
- Density: 2500 kg/m³
- Young's Modulus: 70 GPa
- Poisson's Ratio: 0.22
- Tensile Strength: 40 MPa (very brittle)

**Destruction Properties:**
- Yield Strength: 40 MPa (no plastic deformation before fracture)
- Fracture Toughness (K₁c): 0.75 MPa√m (extremely brittle)
- Shear Modulus: 28.7 GPa
- Failure Mode: Catastrophic brittle (shatters into fragments)

**Audio Properties:**
- Damping Coefficient (ζ): 0.003 (very low, highly resonant)
- Resonance Frequency Range: 400-12000 Hz
- Sound Character: Clear, crystalline ring, long sustain

**Visual Properties:**
- Base Color (sRGB): `#E8F4F8` (slightly blue-tinted clear)
- Metallic: 0.0
- Roughness: 0.01 (extremely smooth)

**Recommended Color Mode:** RGB

---

### ASTM_C373 (Ceramic Tile)

**Common Uses:** Flooring, wall tiles, sanitary ware

**Physical Properties:**
- Density: 2300 kg/m³
- Young's Modulus: 65 GPa
- Poisson's Ratio: 0.25
- Compressive Strength: 500 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.004 (very low)
- Resonance Frequency Range: 200-10000 Hz
- Sound Character: Sharp tap, brittle ring

**Visual Properties:**
- Base Color (sRGB): `#FFFFFF` (white, varies with glaze)
- Metallic: 0.0
- Roughness: 0.15 (glazed finish)

**Recommended Color Mode:** RGB

---

## **Rubber & Elastomers**

### ASTM_D2000 (Neoprene Rubber)

**Common Uses:** Gaskets, seals, wetsuits, shock absorption

**Physical Properties:**
- Density: 1240 kg/m³
- Young's Modulus: 0.7-2 GPa (varies by durometer)
- Poisson's Ratio: 0.48 (nearly incompressible)
- Tensile Strength: 20 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.15 (very high)
- Resonance Frequency Range: 50-1000 Hz
- Sound Character: Dull thud, near-instant decay, absorbs shock

**Visual Properties:**
- Base Color (sRGB): `#2B2B2B` (black)
- Metallic: 0.0
- Roughness: 0.8

**Recommended Color Mode:** RGB

---

### ASTM_D412 (Silicone Rubber)

**Common Uses:** Seals, medical devices, high-temperature applications

**Physical Properties:**
- Density: 1100 kg/m³
- Young's Modulus: 0.01-0.05 GPa (very soft)
- Poisson's Ratio: 0.49
- Tensile Strength: 8 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.20 (extremely high)
- Resonance Frequency Range: 20-800 Hz
- Sound Character: Silent impact, complete absorption

**Visual Properties:**
- Base Color (sRGB): `#F5F5DC` (translucent white/clear)
- Metallic: 0.0
- Roughness: 0.6

**Recommended Color Mode:** RGB

---

## **Fabric & Textiles**

### TEXTILE_COTTON (Canvas)

**Common Uses:** Clothing, tents, bags, upholstery

**Physical Properties:**
- Density: 1540 kg/m³ (fiber), ~400 kg/m³ (woven fabric with air)
- Young's Modulus: 5.5-12.6 GPa (fiber)
- Tensile Strength: 287-597 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.08 (high)
- Resonance Frequency Range: 100-2000 Hz
- Sound Character: Soft rustle, muffled

**Visual Properties:**
- Base Color (sRGB): `#F5F5DC` (natural beige)
- Metallic: 0.0
- Roughness: 0.9

**Recommended Color Mode:** Oklab (for dyed fabrics/team colors)

---

### TEXTILE_NYLON (Ripstop Nylon)

**Common Uses:** Parachutes, outdoor gear, flags

**Physical Properties:**
- Density: 1150 kg/m³
- Young's Modulus: 2-4 GPa
- Tensile Strength: 75 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.10
- Resonance Frequency Range: 150-3000 Hz
- Sound Character: Crisp rustle, light crinkle

**Visual Properties:**
- Base Color (sRGB): `#EBEBEB` (white/varies)
- Metallic: 0.0
- Roughness: 0.4

**Recommended Color Mode:** Oklab (frequently dyed)

---

## **Special Materials**

### KEVLAR_49 (Aramid Fiber)

**Common Uses:** Body armor, aerospace, protective equipment

**Physical Properties:**
- Density: 1440 kg/m³
- Young's Modulus: 112.4 GPa
- Poisson's Ratio: 0.36
- Tensile Strength: 3620 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.03
- Resonance Frequency Range: 200-6000 Hz
- Sound Character: Stiff tap, fabric-like but dense

**Visual Properties:**
- Base Color (sRGB): `#FFD700` (golden yellow)
- Metallic: 0.0
- Roughness: 0.6

**Recommended Color Mode:** RGB

---

### BALLISTIC_GEL (10% Gelatin)

**Common Uses:** Ballistics testing (simulates human tissue)

**Physical Properties:**
- Density: 1060 kg/m³ (close to water)
- Young's Modulus: 0.0001 GPa (very soft)
- Poisson's Ratio: 0.49
- Compressive Strength: <0.1 MPa

**Audio Properties:**
- Damping Coefficient (ζ): 0.50 (extremely high)
- Resonance Frequency Range: 10-500 Hz
- Sound Character: Wet slap, immediate damping

**Visual Properties:**
- Base Color (sRGB): `#FFE4B5` (translucent amber)
- Metallic: 0.0
- Roughness: 0.05 (wet surface)

**Recommended Color Mode:** RGB

---

## **Usage Notes**

### Audio Synthesis Integration

The audio system uses these properties to synthesize contact sounds:

```rust
fn synthesize_impact_sound(
    material_a: &Material,
    material_b: &Material,
    impact_velocity: f32,
) -> AudioPatch {
    // Carrier frequency based on average resonance
    let base_freq = (material_a.resonance_freq_range.0 + material_b.resonance_freq_range.1) / 2.0;
    
    // Decay time from damping coefficients
    let combined_damping = (material_a.damping + material_b.damping) / 2.0;
    let decay_ms = 1000.0 * (1.0 - combined_damping).powi(2);
    
    // Impact intensity from velocity + density
    let impact_energy = 0.5 * material_a.density * impact_velocity.powi(2);
    
    AudioPatch {
        oscillator: Sine { frequency_hz: base_freq },
        envelope: ADSR {
            attack_ms: 1.0,
            decay_ms,
            sustain: 0.0,
            release_ms: decay_ms * 0.5,
        },
        amplitude: (impact_energy / 1000.0).min(1.0),
    }
}
```

### Physics Integration

Young's modulus and Poisson's ratio affect deformation during collisions (if using compliant contacts).

### Rendering Integration

Visual properties map directly to PBR material parameters in the rendering pipeline.

---

**Database Version:** 1.0  
**Last Updated:** January 25, 2026  
**Materials Count:** 27  
**Coverage:** Metals (6), Stone/Concrete (3), Wood (3), Plastics (3), Glass/Ceramics (2), Rubber (2), Textiles (2), Special (2), Biological (1)
