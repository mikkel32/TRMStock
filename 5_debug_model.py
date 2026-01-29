"""
5_debug_model.py
Deep Probe of Model, Data, and Bins.
"""
import os
import torch
import numpy as np
from src.model import TRM, TRMConfig

DATA_DIR = "data"
CHECKPOINT_PATH = "checkpoints/final.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Must match 2_train.py
config = TRMConfig(
    vocab_size=8192, 
    dim=192,         
    n_layers=3,      
    n_recurrence=2,  
    n_heads=6,       
    n_kv_heads=2,    
    max_seq_len=2048,
    dropout=0.1
)

def main():
    print("🔹 Debugging Model & Data...")
    
    # 1. Inspect Bins
    bins_path = os.path.join(DATA_DIR, "bins.npy")
    bins = np.load(bins_path, allow_pickle=True).item()
    log_ret_bins = bins['log_ret']
    
    print("\n--- Bin Analysis ---")
    print(f"Num Return Bins: {len(log_ret_bins)}")
    print(f"Min Val: {log_ret_bins.min()}")
    print(f"Max Val: {log_ret_bins.max()}")
    print(f"Mean Val: {log_ret_bins.mean()}")
    print(f"Unique Vals: {len(np.unique(log_ret_bins))}")
    print(f"Sample First 10: {log_ret_bins[:10]}")
    
    # Offsets
    OFFSET_DOW = 0
    OFFSET_HOUR = 7
    OFFSET_VOLAT = 31
    OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
    OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
    
    # 2. Inspect Validation Data
    print("\n--- Validation Data Analysis ---")
    filename = os.path.join(DATA_DIR, 'val.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    print(f"Val Data Size: {len(data)} tokens")
    
    # Top 1000 tokens Analysis
    sample_tokens = data[:1000]
    # Extract Returns (Every 5th, offset 4)
    ret_tokens = sample_tokens[4::5]
    
    # Decode them
    decoded_vals = []
    for t in ret_tokens:
        idx = t - OFFSET_RET
        if 0 <= idx < len(log_ret_bins):
            decoded_vals.append(log_ret_bins[idx])
        else:
            decoded_vals.append(np.nan)
            
    decoded_vals = np.array(decoded_vals)
    print(f"Sample Decoded Returns (First 20): {decoded_vals[:20]}")
    print(f"Sample Std Dev: {np.std(decoded_vals)}")
    print(f"Is Sample All Zeros? {np.allclose(decoded_vals, 0)}")

    # 3. Model Probe
    print("\n--- Model Probe ---")
    
    # Find Checkpoint
    abs_checkpoints_dir = os.path.abspath("checkpoints")
    ckpt_path = os.path.join(abs_checkpoints_dir, "final.pt")
    if not os.path.exists(ckpt_path):
        # Fallback
        files = sorted([f for f in os.listdir(abs_checkpoints_dir) if f.endswith(".pt")])
        if files: ckpt_path = os.path.join(abs_checkpoints_dir, files[-1])
        
    print(f"Loading: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model = TRM(config).to(DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Prepare Input (First 50 steps from val)
    ctx_len = 50 * 5
    inp = torch.from_numpy(data[:ctx_len].astype(np.int64)).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits, _ = model(inp)
        # Last step logits
        last_logits = logits[0, -1, :]
        probs = torch.nn.functional.softmax(last_logits, dim=-1)
        
        # Get Top 10
        topk_probs, topk_indices = torch.topk(probs, 10)
        
        print("\nTOP 10 PREDICTIONS (Next Token):")
        print(f"{'TokenID':<10} | {'Prob':<10} | {'Type':<10} | {'Value'}")
        print("-" * 50)
        
        for p, idx in zip(topk_probs, topk_indices):
            idx = idx.item()
            val = "N/A"
            type_str = "?"
            
            if OFFSET_DOW <= idx < OFFSET_HOUR:
                type_str = "DOW"
                val = idx - OFFSET_DOW
            elif OFFSET_HOUR <= idx < OFFSET_VOLAT:
                type_str = "HOUR"
                val = idx - OFFSET_HOUR
            elif OFFSET_VOLAT <= idx < OFFSET_VOL:
                type_str = "VOLAT"
            elif OFFSET_VOL <= idx < OFFSET_RET:
                type_str = "VOL"
            elif OFFSET_RET <= idx < OFFSET_RET + len(log_ret_bins):
                type_str = "RET"
                val = log_ret_bins[idx - OFFSET_RET]
            
            print(f"{idx:<10} | {p:.4f}     | {type_str:<10} | {val}")

if __name__ == "__main__":
    main()
