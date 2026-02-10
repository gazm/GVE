
import torch
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath("backend/architect"))

from src.compiler.splat_trainer import compile_splats

def sphere_sdf(p):
    return torch.norm(p, dim=1) - 0.5

def verify():
    print("Verifying splat compilation...")
    try:
        # Run with small count/iterations to just test the logic flow
        data = compile_splats(
            sdf_fn=sphere_sdf,
            bounds=([-1, -1, -1], [1, 1, 1]),
            target_count=100,
            iterations=10,
            device="cpu"
        )
        print("Verification successful! Binary data size:", len(data))
    except Exception as e:
        print(f"Verification FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify()
