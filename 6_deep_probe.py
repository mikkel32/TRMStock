"""
6_deep_probe.py
Analyze the Model's "Brain Activity" (Logits) to diagnose Mode Collapse.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from src.model import TRM, TRMConfig

DATA_DIR = "data"
CHECKPOINT_PATH = "checkpoints/ckpt_350.pt" # Force valid ckpt
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    print("🔹 Deep Probe: Logit Analysis")
    
    # 1. Load Bins & Model
    bins = np.load(os.path.join(DATA_DIR, "bins.npy"), allow_pickle=True).item()
    log_ret_bins = bins['log_ret']
    
    OFFSET_VOLAT = 31
    OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
    OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
    
    print(f"Loading {CHECKPOINT_PATH}...")
    if not os.path.exists(CHECKPOINT_PATH):
        print("❌ Checkpoint not found. Listing dir:")
        print(os.listdir("checkpoints") if os.path.exists("checkpoints") else "No checkpoints dir")
        return

    model = TRM(config).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    # 2. Load Data (Random Sample)
    filename = os.path.join(DATA_DIR, 'val.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    # Take 50 steps (250 tokens) + 4 tokens (DOW, H, V, V) of the 51st step
    # So we predict the 51st Return.
    # Total input length: 254
    ctx_len = 250 + 4
    ctx_tokens = torch.from_numpy(data[:ctx_len].astype(np.int64)).unsqueeze(0).to(DEVICE)
    
    print(f"Input Shape: {ctx_tokens.shape} (Should be ...4)")
    print(f"Last Input Token: {ctx_tokens[0, -1]}")
    print(f"OFFSET_RET: {OFFSET_RET}")
    
    if ctx_tokens[0, -1] >= OFFSET_RET:
        print("⚠️ WARNING: Last token looks like a Return token? Check offsets.")
    
    # 3. Forward Pass
    with torch.no_grad():
        logits, _ = model(ctx_tokens)
        final_logits = logits[0, -1, :] # (Vocab,)
        
    probs = F.softmax(final_logits, dim=-1).cpu().numpy()
    
    # 4. Analyze Predictions
    # Focus on the Return Token Range
    ret_start = OFFSET_RET
    ret_end = OFFSET_RET + len(log_ret_bins)
    
    ret_probs = probs[ret_start:ret_end]
    ret_logits = final_logits[ret_start:ret_end].cpu().numpy()
    
    # Check if mass is here
    total_mass = np.sum(ret_probs)
    print(f"\n--- Prediction Diagnosis (Target: Return Token) ---")
    print(f"Total Probability Mass on Returns: {total_mass:.4f}")
    
    if total_mass < 0.1:
        print("🚨 MASS MISSING: Model is NOT predicting a Return token.")
        # Find where the mass is
        top_overall = np.argsort(probs)[-10:][::-1]
        print("Top 10 Global Predictions:")
        for idx in top_overall:
            print(f"   Token {idx}: {probs[idx]:.4f}")
        return
    max_prob_idx = np.argmax(ret_probs)
    max_prob = ret_probs[max_prob_idx]
    predicted_bin_val = log_ret_bins[max_prob_idx]
    
    print("\n--- Prediction Diagnosis ---")
    print(f"Total Probability Mass on Returns: {np.sum(ret_probs):.4f}")
    print(f"Peak Probability: {max_prob:.4f}")
    print(f"Predicted Value: {predicted_bin_val:.6f}")
    print(f"Entropy: {-np.sum(ret_probs * np.log(ret_probs + 1e-9)):.4f}")
    
    # Top 5
    top5_inds = np.argsort(ret_probs)[-5:][::-1]
    print("\nTop 5 Return Candidates:")
    for i in top5_inds:
        print(f"   Bin {i} (Val {log_ret_bins[i]:.5f}): {ret_probs[i]*100:.2f}%")

    # 5. Plot Probability Distribution
    plt.figure(figsize=(10, 6))
    plt.plot(log_ret_bins, ret_probs, color='blue', alpha=0.7)
    plt.axvline(predicted_bin_val, color='red', linestyle='--', label=f'Model Pick ({predicted_bin_val:.5f})')
    plt.title(f"Model Probability Distribution (Next Return)\nPeak Prob: {max_prob*100:.1f}%")
    plt.xlabel("Log Return Value")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.savefig("logit_probe.png")
    print("📉 Saved logit_probe.png")
    
    if max_prob > 0.5:
        print("🚨 MODE COLLAPSE DETECTED: Model is obsessively confident in one value.")
    else:
        print("✅ Distribution looks healthy (dispersed).")

if __name__ == "__main__":
    main()
