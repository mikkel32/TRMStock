"""
8_check_data_alignment.py
Verify that the binary dataset strictly follows the [DOW, H, V, V, R] schema.
If this fails, the Training Metrics were meaningless (garbage in, garbage out).
"""
import os
import numpy as np

DATA_DIR = "data"

def check_file(filename, bins):
    print(f"🔎 Checking {filename}...")
    if not os.path.exists(filename):
        print("   ❌ File not found.")
        return

    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Calculate Offsets
    OFFSET_VOLAT = 31
    OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
    OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
    
    print(f"   Total Tokens: {len(data)}")
    print(f"   OFFSET_RET: {OFFSET_RET}")
    
    # Check 1: Length must be divisible by 5
    if len(data) % 5 != 0:
        print(f"   ❌ ERROR: Data length {len(data)} is NOT divisible by 5!")
        print("   This means incomplete sequences exist.")
    else:
        print("   ✅ Length is divisible by 5.")

    # Check 2: Sample Audit (Check first 1000 and random 1000)
    # We expect:
    # Index % 5 == 0 -> DOW (< 7)
    # Index % 5 == 1 -> HOUR (< 31)
    #Index % 5 == 4 -> RET (>= OFFSET_RET)
    
    errors = 0
    
    # Define a check function
    def audit_chunk(start_idx, length):
        chunk_errors = 0
        chunk = data[start_idx : start_idx + length]
        
        for i, token in enumerate(chunk):
            global_idx = start_idx + i
            pos = global_idx % 5
            
            if pos == 0: # DOW
                if token >= 7: chunk_errors += 1
            elif pos == 1: # HOUR
                if token >= 31: chunk_errors += 1
            elif pos == 4: # RET
                if token < OFFSET_RET: 
                    # Special case: It MIGHT be a padding token if we used padding? 
                    # But we don't use padding in 1_prepare.py for binary streams usually.
                    chunk_errors += 1
        return chunk_errors

    # Audit Start
    print("   Auditing Head...")
    e_head = audit_chunk(0, min(len(data), 5000))
    if e_head > 0: print(f"   ❌ HEAD ERRORS: {e_head} mismatches found in first 5000 tokens.")
    else: print("   ✅ Head looks perfect.")
    
    # Audit Random
    if len(data) > 10000:
        print("   Auditing Random Chunk...")
        rand_start = (len(data) // 10) * 5 # Align to 5
        e_rand = audit_chunk(rand_start, 5000)
        if e_rand > 0: print(f"   ❌ RANDOM ERRORS: {e_rand} mismatches found.")
        else: print("   ✅ Random chunk looks perfect.")
        
def main():
    bins_path = os.path.join(DATA_DIR, "bins.npy")
    if not os.path.exists(bins_path):
        print("Error: bins.npy not found")
        return
    bins = np.load(bins_path, allow_pickle=True).item()
    
    check_file(os.path.join(DATA_DIR, "train.bin"), bins)
    check_file(os.path.join(DATA_DIR, "val.bin"), bins)

if __name__ == "__main__":
    main()
