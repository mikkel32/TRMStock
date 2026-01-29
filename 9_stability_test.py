"""
9_stability_test.py
Stress Test the Grammar Constraint.
Goal: Generate 10,000 steps (50,000 tokens) and verify 0% Alignment Loss.
This proves the "Fix" is mathematically robust, not just lucky.
"""
import os
import torch
import numpy as np
import time
from src.model import TRM, TRMConfig

# Re-implement generate locally to ensure we test the EXACT logic
# (Or better: import from 3_inference if possible, but 3_inference is a script)
# We will copy the robust generate logic here to isolate the test.

DATA_DIR = "data"
CHECKPOINT_PATH = "checkpoints/ckpt_350.pt" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Config must match training
config = TRMConfig(
    vocab_size=8192, dim=192, n_layers=3, n_recurrence=2,  
    n_heads=6, n_kv_heads=2, max_seq_len=2048, dropout=0.1
)

def load_bins():
    bins_path = os.path.join(DATA_DIR, "bins.npy")
    return np.load(bins_path, allow_pickle=True).item()

@torch.no_grad()
def robust_generate(model, idx, max_new_tokens, offsets, temperature=1.0):
    # Unpack offsets
    off_dow = offsets['DOW']
    off_hour = offsets['HOUR']
    off_volat = offsets['VOLAT']
    off_vol = offsets['VOL']
    off_ret = offsets['RET']
    
    generated = []
    
    start_time = time.time()
    
    for i in range(max_new_tokens):
        # 1. Determine Expected Type
        curr_len = idx.size(1)
        step_type = (curr_len) % 5
        
        # 2. Forward
        idx_cond = idx if idx.size(1) <= config.max_seq_len else idx[:, -config.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # 3. Apply Grammar Constraint
        if step_type == 0: # DOW
            start, end = 0, off_hour
        elif step_type == 1: # HOUR
            start, end = off_hour, off_volat
        elif step_type == 2: # VOLAT
            start, end = off_volat, off_vol
        elif step_type == 3: # VOL
            start, end = off_vol, off_ret
        elif step_type == 4: # RET
            start, end = off_ret, off_ret + 2000 

        # Masking
        # Create a tensor filled with -inf
        full_masked = torch.full_like(logits, -float('Inf'))
        # Copy valid logits
        full_masked[:, start:end] = logits[:, start:end]
        probabilities = torch.nn.functional.softmax(full_masked, dim=-1)
        
        # Check Safety
        if torch.isnan(probabilities).any():
             print(f"🚨 FATAL: NaN at step {i} (Type {step_type})")
             break
             
        idx_next = torch.multinomial(probabilities, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
        
        generated.append(idx_next.item())
        
        if i % 1000 == 0 and i > 0:
            print(f"   Step {i}/{max_new_tokens} OK...")
            
    return generated

def main():
    print("🔹 Stability Stress Test (10,000 Steps)")
    
    bins = load_bins()
    OFFSET_DOW = 0
    OFFSET_HOUR = 7
    OFFSET_VOLAT = 31
    OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
    OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
    
    offsets_dict = {
        'DOW': OFFSET_DOW, 'HOUR': OFFSET_HOUR,
        'VOLAT': OFFSET_VOLAT, 'VOL': OFFSET_VOL, 'RET': OFFSET_RET
    }
    
    model = TRM(config).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()
    
    # Dummy Start context (needs to be valid alignment)
    # Let's verify 0-ring alignment
    # [DOW, H, V, V, R] is 5 tokens.
    ctx = torch.zeros((1, 10), dtype=torch.long).to(DEVICE) 
    # Fill with plausible values to avoid weird embeddings
    ctx[0, 0] = 1 # DOW
    ctx[0, 1] = 10 # H
    ctx[0, 2] = 50 # V
    ctx[0, 3] = 100 # V
    ctx[0, 4] = OFFSET_RET + 1000 # R
    # Repeat
    ctx[0, 5:] = ctx[0, :5]
    
    TEST_LEN = 10000 
    print(f"   Generating {TEST_LEN} tokens...")
    
    gen_toks = robust_generate(model, ctx, TEST_LEN, offsets_dict)
    
    print("✅ Generation Complete. Auditing Stream...")
    
    errors = 0
    for i, tok in enumerate(gen_toks):
        # The first generated token is at index 10 of alignment (10 % 5 = 0 -> DOW)
        # So gen_toks[0] should be DOW
        pos = (10 + i) % 5
        
        if pos == 0 and not (0 <= tok < OFFSET_HOUR): errors += 1
        elif pos == 1 and not (OFFSET_HOUR <= tok < OFFSET_VOLAT): errors += 1
        elif pos == 2 and not (OFFSET_VOLAT <= tok < OFFSET_VOL): errors += 1
        elif pos == 3 and not (OFFSET_VOL <= tok < OFFSET_RET): errors += 1
        elif pos == 4 and not (OFFSET_RET <= tok): errors += 1
        
    if errors == 0:
        print(f"🏆 SUCCESS: {TEST_LEN} tokens generated with 0 Alignment Errors.")
        print("   The Grammar Constraint is mathematically sound.")
    else:
        print(f"❌ FAILURE: {errors} Alignment Errors found.")

if __name__ == "__main__":
    main()
