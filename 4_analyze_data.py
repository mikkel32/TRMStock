"""
4_analyze_data.py
Diagnose Class Imbalance / Zero Token Dominance.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"

def main():
    print("🔹 Analyzing Dataset Distribution...")
    
    # 1. Load Bins
    bins_path = os.path.join(DATA_DIR, "bins.npy")
    if not os.path.exists(bins_path):
        print("❌ bins.npy not found")
        return
    bins = np.load(bins_path, allow_pickle=True).item()
    
    # Calculate Offsets to find Return Tokens
    # [DOW, Hour, Volat, Vol, Ret]
    OFFSET_DOW = 0
    OFFSET_HOUR = 7
    OFFSET_VOLAT = 31
    OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
    OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
    
    log_ret_bins = bins['log_ret']
    n_ret_bins = len(log_ret_bins)
    
    print(f"Return Token Range: {OFFSET_RET} to {OFFSET_RET + n_ret_bins}")
    
    # Find Zero Token
    # We look for the bin closest to 0.0
    zero_idx = np.abs(np.array(log_ret_bins)).argmin()
    ZERO_TOKEN_ID = OFFSET_RET + zero_idx
    print(f"Zero Token ID: {ZERO_TOKEN_ID} (Value: {log_ret_bins[zero_idx]})")
    
    # 2. Scan Train Data
    filename = os.path.join(DATA_DIR, 'train.bin')
    if not os.path.exists(filename):
        print("❌ train.bin not found")
        return
        
    print("   Scanning train.bin (this may take a moment)...")
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Extract only Return Tokens (Every 5th token, offset 4)
    # Indices: 4, 9, 14...
    n_tokens = len(data)
    n_steps = n_tokens // 5
    
    # We can slice memmap? Yes.
    ret_tokens = data[4::5]
    
    print(f"   Total Steps: {len(ret_tokens)}")
    
    # 3. Statistics
    # Check for Zero Dominance
    n_zeros = np.sum(ret_tokens == ZERO_TOKEN_ID)
    pct_zeros = (n_zeros / len(ret_tokens)) * 100
    
    print(f"   Zero Tokens: {n_zeros}")
    print(f"   Zero Dominance: {pct_zeros:.2f}%")
    
    # Histogram of Return Token usages
    # Map back to 0-indexed bin IDs
    ret_bin_ids = ret_tokens.astype(np.int32) - OFFSET_RET
    
    # Filter out valid range (just in case)
    valid_mask = (ret_bin_ids >= 0) & (ret_bin_ids < n_ret_bins)
    ret_bin_ids = ret_bin_ids[valid_mask]
    
    counts = np.bincount(ret_bin_ids, minlength=n_ret_bins)
    
    # Plot Distribution
    plt.figure(figsize=(12, 6))
    plt.bar(range(n_ret_bins), counts, color='blue', alpha=0.7)
    plt.axvline(zero_idx, color='red', linestyle='--', label='Zero Bin')
    plt.title(f"Return Token Distribution (Zero Dominance: {pct_zeros:.2f}%)")
    plt.xlabel("Bin Index")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("token_distribution.png")
    print("📉 Saved token_distribution.png")

if __name__ == "__main__":
    main()
