
import sys
import os
sys.path.append(os.getcwd())

print("Attempting to import src.compiler.pipeline...")
try:
    from src.compiler.pipeline import VolumeBaker, MeshBaker, SplatBaker, TriplanarBaker
    print("Successfully imported bakers from pipeline")
    
    v = VolumeBaker()
    m = MeshBaker()
    s = SplatBaker()
    t = TriplanarBaker()
    
    print(f"Instantiated: {v.name()}, {m.name()}, {s.name()}, {t.name()}")
    
except ImportError as e:
    print(f"ImportError: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
