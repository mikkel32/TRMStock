"""
3_inference.py
Step 3: Hallucination Test (Generative Inference).

This script performs auto-regressive generation ("Dreaming") to verify
if the model has learned market physics or just mean-reversion.

It takes 50 steps of real context, and asks the model to dream the next 100 steps.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.model import TRM, TRMConfig

# --- Configuration ---
DATA_DIR = "data"
CHECKPOINT_PATH = "checkpoints/final.pt" 
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Must match 2_train.py config exactly
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

OFFSET_DOW = 0
OFFSET_HOUR = 7
OFFSET_VOLAT = 31

def load_bins():
    bins_path = os.path.join(DATA_DIR, "bins.npy")
    if not os.path.exists(bins_path):
        raise FileNotFoundError("bins.npy not found")
    return np.load(bins_path, allow_pickle=True).item()

def get_stable_sample(bins):
    # Load val data
    filename = os.path.join(DATA_DIR, 'val.bin')
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    
    log_ret_bins = bins['log_ret']
    
    # We need:
    # 1. 50 steps of context (250 tokens)
    # 2. 100 steps of future (500 tokens)
    # Total 750 tokens.
    SEQ_LEN = 150 * 5
    CONTEXT_STEPS = 50
    OFFSET_VOLAT = 31
    OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
    OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
    
    # Helper to decode a slice
    def decode_returns(slice_tokens):
        # Extract returns (every 5th, indices 4, 9...)
        ret_toks = slice_tokens[4::5]
        inds = ret_toks.astype(np.int32) - OFFSET_RET
        # Clip to valid range
        inds = np.clip(inds, 0, len(log_ret_bins)-1)
        return log_ret_bins[inds]
    
    # Random search for a "Normal" period
    print("🔍 Searching for a stable context (no crashes)...")
    attempts = 0
    while attempts < 100:
        torch.manual_seed(42 + attempts)
        start_idx = torch.randint(0, len(data) - SEQ_LEN * 2, (1,)).item()
        
        # Ensure start_idx is aligned to 5 tokens (Model expects strict [DOW, H, V, V, R] sequence)
        # Actually dataset is continuous, but let's try to align to block of 5 just in case 
        # (Though DataEngine writes in blocks of 5, so 0 is aligned)
        start_idx = (start_idx // 5) * 5
        
        chunk = data[start_idx : start_idx + SEQ_LEN]
        
        # Analyze Context (First 50 steps)
        ctx_chunk = chunk[:CONTEXT_STEPS*5]
        rets = decode_returns(ctx_chunk)
        
        cum_ret = np.sum(rets)
        price_impact = np.exp(cum_ret)
        
        # Criteria: Price shouldn't drop below 80 or go above 120 (start 100)
        # i.e. cum_ret between log(0.8) and log(1.2) -> -0.22 to +0.18
        if -0.25 < cum_ret < 0.25:
            # Also check for "Dead" data (all zeros)
            if np.allclose(rets, 0):
                attempts += 1
                continue
                
            print(f"✅ Found stable context at idx {start_idx} (CumRet: {cum_ret:.3f})")
            print(f"   Context Rets First 5: {rets[:5]}")
            return torch.from_numpy(chunk.astype(np.int64)).to(DEVICE)
            
        attempts += 1
        
    print("⚠️ Could not find stable context. Using random.")
    start_idx = torch.randint(0, len(data) - SEQ_LEN, (1,)).item()
    chunk = data[start_idx : start_idx + SEQ_LEN]
    return torch.from_numpy(chunk.astype(np.int64)).to(DEVICE)

@torch.no_grad()
def generate(model, idx, max_new_tokens, offsets, temperature=1.0, top_k=None, suppress_zero=False, zero_token_id=None):
    # Unpack offsets for Grammar Constraint
    # offsets dict has keys: DOW, HOUR, VOLAT, VOL, RET
    off_dow = offsets['DOW']
    off_hour = offsets['HOUR']
    off_volat = offsets['VOLAT']
    off_vol = offsets['VOL']
    off_ret = offsets['RET']
    max_vocab = config.vocab_size

    for _ in range(max_new_tokens):
        # 1. Determine Expected Type
        curr_len = idx.size(1)
        step_type = (curr_len) % 5
        
        # 0=DOW, 1=HOUR, 2=VOLAT, 3=VOL, 4=RET
        
        # 2. Forward
        idx_cond = idx if idx.size(1) <= config.max_seq_len else idx[:, -config.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # 3. Apply Grammar Constraint (Masking)
        mask = torch.ones_like(logits).bool() # True = Keep, False = Mask (Wait, conventionally mask is True to hide?)
        # Let's set logits to -inf
        
        # Define allowed range [start, end)
        if step_type == 0: # DOW
            start, end = 0, off_hour
        elif step_type == 1: # HOUR
            start, end = off_hour, off_volat
        elif step_type == 2: # VOLAT
            start, end = off_volat, off_vol
        elif step_type == 3: # VOL
            start, end = off_vol, off_ret
        elif step_type == 4: # RET
            start, end = off_ret, off_ret + 2000 # Approx max bins
            
        # Create mask: Only indices in [start, end) are allowed
        # Efficient way:
        # We can just set everything outside to -inf
        # Or clone a -inf tensor and fill in the window
        
        valid_logits = logits[:, start:end]
        full_masked = torch.full_like(logits, -float('Inf'))
        full_masked[:, start:end] = valid_logits
        logits = full_masked

        # 4. Zero Suppression
        if suppress_zero and zero_token_id is not None:
             logits[:, zero_token_id] = -float('Inf')
        
        # 5. Top K
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Safety: Check for NaN (if constraint masked everything)
        if torch.isnan(probs).any():
             print(f"⚠️ NaN probs at StepType {step_type}. Fallback to Uniform.")
             probs = torch.zeros_like(logits)
             probs[:, start:end] = 1.0 / (end - start)
             
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def main():
    print("🔹 Hallucination Test: Generative Inference (Constrained)")
    print(f"   CWD: {os.getcwd()}")
    
    # 1. Load Model
    model = TRM(config).to(DEVICE)
    
    # Robust Path Check
    abs_checkpoints_dir = os.path.abspath("checkpoints")
    # ckpt_path = os.path.join(abs_checkpoints_dir, "final.pt")
    ckpt_path = os.path.join(abs_checkpoints_dir, "ckpt_350.pt")
    
    if not os.path.exists(ckpt_path):
        print(f"❌ Checkpoint not found at: {ckpt_path}")
        if os.path.exists(abs_checkpoints_dir):
            files = [f for f in os.listdir(abs_checkpoints_dir) if f.endswith(".pt")]
            if files:
                # Sort by modification time (pick latest)
                # files.sort(key=lambda x: os.path.getmtime(os.path.join(abs_checkpoints_dir, x)))
                # Fix: Sort numerically if possible, or trust getmtime if filesystem is reliable
                # But let's just pick the one with highest number in name
                
                # regex to extract number
                import re
                def extract_iter(name):
                    m = re.search(r'ckpt_(\d+).pt', name)
                    return int(m.group(1)) if m else -1
                
                files.sort(key=extract_iter)
                latest = files[-1]
                ckpt_path = os.path.join(abs_checkpoints_dir, latest)
                print(f"⚠️  Falling back to latest (Numeric Sort): {ckpt_path}")
            else:
                print(f"   No .pt files in {abs_checkpoints_dir}")
                return
        else:
            print(f"   '{abs_checkpoints_dir}' folder missing.")
            return

    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    
    # 2. Setup Data
    bins = load_bins()
    log_ret_bins = bins['log_ret']
    
    OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
    OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
    
    # 3. Get Sample
    TOKENS_PER_STEP = 5
    CONTEXT_STEPS = 50
    FUTURE_STEPS = 100
    
    full_sample = get_stable_sample(bins)
    full_sample = full_sample.unsqueeze(0) 
    
    # Split
    cutoff = CONTEXT_STEPS * TOKENS_PER_STEP
    context_tokens = full_sample[:, :cutoff]
    true_future_tokens = full_sample[:, cutoff : cutoff + (FUTURE_STEPS * TOKENS_PER_STEP)]
    
    print(f"Context Tokens: {context_tokens.shape}")
    
    # Helper
    def extract_returns(toks):
        ret_inds = np.arange(4, len(toks), 5)
        ret_toks = toks[ret_inds]
        inds = ret_toks - OFFSET_RET
        inds = np.clip(inds, 0, len(log_ret_bins)-1)
        return log_ret_bins[inds]

    # --- Prepare Real Data First ---
    ctx_np = context_tokens[0].cpu().numpy()
    true_np = true_future_tokens[0].cpu().numpy()
    
    rets_ctx = extract_returns(ctx_np)
    rets_true = extract_returns(true_np)
    
    price = 100.0
    prices_ctx = [price]
    for r in rets_ctx:
        price = price * np.exp(r)
        prices_ctx.append(price)
        
    prices_true = [prices_ctx[-1]]
    price_t = prices_ctx[-1]
    for r in rets_true:
        price_t = price_t * np.exp(r)
        prices_true.append(price_t)

    # Calculate Zero Token ID (for suppression)
    zero_idx = np.abs(np.array(log_ret_bins)).argmin()
    ZERO_TOKEN_ID = OFFSET_RET + zero_idx
    print(f"Zero Token ID: {ZERO_TOKEN_ID} (Value: {log_ret_bins[zero_idx]})")
    
    # Grammar Constraint Offsets
    offsets_dict = {
        'DOW': OFFSET_DOW,
        'HOUR': OFFSET_HOUR,
        'VOLAT': OFFSET_VOLAT,
        'VOL': OFFSET_VOL,
        'RET': OFFSET_RET
    }

    # 4. Generate - THE SPECTRUM OF DREAMING
    feature_configs = [
        {"name": "Conservative (0.9)", "temp": 0.9, "color": "orange", "style": "--", "suppress": False},
        {"name": "Balanced (1.1)", "temp": 1.1, "color": "red", "style": "-.", "suppress": False},
        {"name": "Volatile (1.3)", "temp": 1.3, "color": "purple", "style": ":", "suppress": False},
        {"name": "Forced Motion (Zero Suppressed)", "temp": 1.0, "color": "green", "style": "-", "suppress": True}
    ]
    
    dreams = []

    print("🧠 Dreaming futures (Spectrum Test)...")
    
    for cfg in feature_configs:
        print(f"   > Dreaming {cfg['name']}...")
        suppress = cfg.get('suppress', False)
        
        dream_tokens_full = generate(
            model, 
            context_tokens, 
            max_new_tokens=FUTURE_STEPS*TOKENS_PER_STEP, 
            offsets=offsets_dict,
            temperature=cfg['temp'], 
            top_k=100,
            suppress_zero=suppress,
            zero_token_id=ZERO_TOKEN_ID
        )
        dream_future_tokens = dream_tokens_full[:, cutoff:]
        
        dream_np = dream_future_tokens[0].cpu().numpy()
        rets_dream = extract_returns(dream_np)
        
        # Calculate Price Path
        prices_dream = [prices_ctx[-1]]
        price_d = prices_ctx[-1]
        for r in rets_dream:
            price_d = price_d * np.exp(r)
            prices_dream.append(price_d)
            
        dreams.append(prices_dream)
        
    # 5. Plot
    steps_ctx = np.arange(len(prices_ctx))
    steps_fut = np.arange(len(prices_ctx)-1, len(prices_ctx)-1 + len(prices_true))
    
    plt.figure(figsize=(12, 6))
    
    plt.plot(steps_ctx, prices_ctx, color='black', label='Context (Real)', linewidth=2)
    plt.plot(steps_fut, prices_true, color='blue', label='Future (Real)', linewidth=2, alpha=0.4)
    
    for i, cfg in enumerate(feature_configs):
        plt.plot(steps_fut, dreams[i], color=cfg['color'], label=f"Dream: {cfg['name']}", linewidth=1.5, linestyle=cfg['style'])
    
    plt.title("Generative Inference: Mode Collapse Test (Spectrum + Forced Motion)")
    plt.xlabel("Minutes")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    out_path = "hallucination_test.png"
    plt.savefig(out_path)
    print(f"📉 Saved chart to {out_path}")
    print("Green Line = What the model thinks happens IF price moves.")

if __name__ == "__main__":
    main()
