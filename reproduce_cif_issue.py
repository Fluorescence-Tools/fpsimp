
import sys
from pathlib import Path

# Add project root to path
sys.path.append('/Users/tpeulen/dev/fpsimp')

from fpsim.utils import extract_sequences_from_structure
from fpsim.segments import parse_plddt_from_pdb

def test_cif_parsing(cif_path):
    print(f"Testing CIF parsing for: {cif_path}")
    
    # 1. Extract sequences
    try:
        seqs = extract_sequences_from_structure(Path(cif_path))
        print(f"Extracted sequences for chains: {list(seqs.keys())}")
        for chain_id, seq in seqs.items():
            print(f"  Chain {chain_id}: {len(seq)} residues")
    except Exception as e:
        print(f"Failed to extract sequences: {e}")
        return

    # 2. Parse pLDDT for each chain
    for chain_id in seqs.keys():
        try:
            plddt = parse_plddt_from_pdb(Path(cif_path), chain_id)
            print(f"  Chain {chain_id}: pLDDT parsed? {'Yes' if plddt else 'No'} (count: {len(plddt)})")
            if plddt:
                print(f"    Sample: {list(plddt.items())[:5]}")
        except Exception as e:
            print(f"  Chain {chain_id}: Failed to parse pLDDT - {e}")

if __name__ == "__main__":
    cif_file = Path(__file__).parent / "examples" / "fold_mc4r_dimer_model_0.cif"
    test_cif_parsing(str(cif_file))
