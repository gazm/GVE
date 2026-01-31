# Improvements & Optimizations

This directory contains technical proposals and optimization strategies for GVE-1.

## Documents

### [Splat Strategy](splat-strategy.md)
**Status:** Pending Approval  
**Impact:** High

Comprehensive analysis of using gaussian splats as primary surface representation across all asset types (weapons, characters, environments, foliage).

**Key Results:**
- 5-29× smaller file sizes vs traditional textures
- 2× faster GPU rendering
- JIT-compiled skinning: 3× faster than traditional (0.3ms vs 1ms)
- Voxel foliage freezing: 8× performance improvement

**Requires validation:** JIT performance on low-end hardware, voxel transition quality

---

## Contributing

When adding new improvement proposals:

1. Create detailed document with:
   - Problem statement
   - Proposed solution with algorithms
   - Performance analysis
   - Risk assessment
   - Implementation roadmap

2. Update this README with summary
3. Request technical review before implementation
