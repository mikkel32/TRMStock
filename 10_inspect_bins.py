"""
10_inspect_bins.py
Audit the bins.npy file to ensure log_ret bins are symmetric and correctly indexed.
Target: Verify range, count, and zero-index.
"""
import numpy as np
import os

DATA_DIR = "data"

def main():
    bins_path = os.path.join(DATA_DIR, "bins.npy")
    if not os.path.exists(bins_path):
        print("❌ bins.npy not found.")
        return

    bins = np.load(bins_path, allow_pickle=True).item()
    
    # 1. Inspect Log Returns
    log_ret = bins['log_ret']
    print(f"📊 Log Return Bins: {len(log_ret)} edges")
    print(f"   Min: {log_ret.min():.5f}")
    print(f"   Max: {log_ret.max():.5f}")
    print(f"   Mean: {log_ret.mean():.5f}")
    
    # Check Symmetry
    neg_count = (log_ret < 0).sum()
    pos_count = (log_ret > 0).sum()
    print(f"   Negative Bins: {neg_count}")
    print(f"   Positive Bins: {pos_count}")
    
    # Check Zero
    zero_idx = np.digitize([0.0], log_ret)[0]
    print(f"   Zero Value maps to Bin Index: {zero_idx}")
    print(f"   Value at {zero_idx}: {log_ret[min(zero_idx, len(log_ret)-1)]}")
    
    # 2. Re-verify Offsets
    OFFSET_VOLAT = 31
    OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
    OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
    
    print(f"Calculated OFFSET_RET: {OFFSET_RET}")
    print(f"Total Range: {OFFSET_RET} to {OFFSET_RET + len(log_ret)}")

if __name__ == "__main__":
    main()
