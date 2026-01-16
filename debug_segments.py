
import sys
import os

# Add current directory to path so we can import fpsim
sys.path.insert(0, os.getcwd())

try:
    print("Attempting to import fpsim.segments...")
    from fpsim.segments import segments_from_plddt
    print("Import successful.")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)

try:
    print("Testing segments_from_plddt execution...")
    # Dummy data
    seq_len = 100
    plddt = {i: 90.0 for i in range(1, 101)}
    fp_domains = [("GFP", 10, 20, 0.99)]
    
    segs = segments_from_plddt(seq_len, plddt, fp_domains)
    print("Execution successful.")
    print("Segments:", segs)
except Exception as e:
    print(f"Execution failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
