"""
2_train.py
Step 2: Model Training.

This script loads the tokenized data from 'data/' and trains the
Recursive Transformer (TRM).

Usage:
    python 2_train.py
"""

import os
import time
import math
import argparse
import numpy as np
import torch
from src.model import TRM, TRMConfig # Import from src/

# --- Configuration ---
DATA_DIR = "data"
OUT_DIR = "checkpoints"

# System
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8 # Fit VRAM

# Model Config (V4) - Optimized for 27.9M Tokens (Chinchilla Verified)
# Target Params: ~1.4M (D/20)
# Calculation: 12 * 3(layers) * 192^2 = ~1.33M Params
# Ratio: ~0.95x (Near Perfect 1.0)
config = TRMConfig(
    vocab_size=8192, 
    dim=192,         # Reduced to 192
    n_layers=3,      # Reduced to 3
    n_recurrence=2,  # Reduced recurrence slightly
    n_heads=6,       # 192 / 32 = 6 heads
    n_kv_heads=2,    
    max_seq_len=2048,
    dropout=0.1
)

# Training Config
BLOCK_SIZE = config.max_seq_len
GRADIENT_ACCUMULATION_STEPS = 8
# The Math: 262,008,610 Tokens
# Batch 8 * Block 2048 * GradAcc 8 = 131,072 Tokens/Step
# 262,008,610 / 131,072 = 1998.9 Steps
# Round up to 2000 for full coverage
TRAIN_EPOCHS = 1
MAX_ITERS = 2000 # Strictly 1 Epoch (262M Tokens)
LEARNING_RATE = 3e-4 
WARMUP_ITERS = 100
LR_DECAY_ITERS = MAX_ITERS
MIN_LR = 6e-5
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 50   
LOG_INTERVAL = 5
EVAL_ITERS = 20

def get_batch(split):
    filename = os.path.join(DATA_DIR, f'{split}.bin')
    if not os.path.exists(filename):
        print(f"❌ File not found: {filename}")
        print("💡 Run 'python 1_prepare.py' first!")
        exit(1)
        
    data = np.memmap(filename, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    
    if DEVICE == 'cuda':
        x, y = x.pin_memory().to(DEVICE, non_blocking=True), y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)
    return x, y

def get_lr(it):
    if it < WARMUP_ITERS: return LEARNING_RATE * it / WARMUP_ITERS
    if it > LR_DECAY_ITERS: return MIN_LR
    decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LEARNING_RATE - MIN_LR)

def estimate_tokens():
    train_path = os.path.join(DATA_DIR, 'train.bin')
    val_path = os.path.join(DATA_DIR, 'val.bin')
    
    total_bytes = 0
    if os.path.exists(train_path): total_bytes += os.path.getsize(train_path)
    if os.path.exists(val_path): total_bytes += os.path.getsize(val_path)
    
    # uint16 = 2 bytes per token
    total_tokens = total_bytes // 2
    return total_tokens

@torch.no_grad()
def generate_dream(model, idx, max_new_tokens, offsets, temperature=1.0, top_k=None, suppress_zero=False, zero_token_id=None):
    # Unpack offsets for Grammar Constraint
    off_dow = offsets['DOW']
    off_hour = offsets['HOUR']
    off_volat = offsets['VOLAT']
    off_vol = offsets['VOL']
    off_ret = offsets['RET']
    
    for step_i in range(max_new_tokens):
        # 1. Determine Expected Type
        curr_len = idx.size(1)
        step_type = (curr_len) % 5
        
        # 2. Forward
        idx_cond = idx if idx.size(1) <= config.max_seq_len else idx[:, -config.max_seq_len:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature
        
        # 3. Apply Grammar Constraint
        start, end = 0, 0
        if step_type == 0: start, end = 0, off_hour
        elif step_type == 1: start, end = off_hour, off_volat
        elif step_type == 2: start, end = off_volat, off_vol
        elif step_type == 3: start, end = off_vol, off_ret
        elif step_type == 4: start, end = off_ret, off_ret + 2000 
            
        full_masked = torch.full_like(logits, -float('Inf'))
        full_masked[:, start:end] = logits[:, start:end]
        logits = full_masked

        # 4. Zero Suppression (Forced Motion)
        if suppress_zero and zero_token_id is not None:
             logits[:, zero_token_id] = -float('Inf')
        
        # 5. Top K
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
            
        probs = torch.nn.functional.softmax(logits, dim=-1)
        if torch.isnan(probs).any():
             probs = torch.zeros_like(logits)
             probs[:, start:end] = 1.0 / (end - start)
             
        # DEBUG: Inspect Return Generation (Step Type 4)
        if step_type == 4:
            # Check max prob
            max_prob, max_idx = torch.max(probs, dim=-1)
            # Local index relative to start
            local_idx = max_idx.item() - start
            if step_i < 20: # Only first few to avoid spam
                print(f"🐛 Step {step_i} (RET): Range=[{start}:{end}]. MaxProb={max_prob.item():.4f} at GlobIdx={max_idx.item()} (Loc={local_idx})")
                
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx

def get_stable_sample(bins, data, block_size, attempts=100):
    # Find a context that ends with [DOW, H, V, V] (len % 5 == 4)
    # And has valid values to prevent initial crash
    valid_ctx = None
    
    for _ in range(attempts):
        start_idx = np.random.randint(0, len(data) - block_size - 100)
        # Align to block start? No, we just need a sequence.
        # But we WANT the sequence to END at a boundary where we predict RET next.
        # Current TRM predicts next token. 
        # If we input [A, B, C, D], it predicts E.
        # We want input to be [..., DOW, H, V, V] so next is RET.
        # Input length % 5 should be 4.
        
        # Check alignment
        # data[i] is the token at i.
        # If we take chunk data[i : i+L]
        # We want len(chunk) % 5 == 4.
        
        L = 254 # Ends at 253. 254 % 5 = 4. Perfect.
        chunk = data[start_idx : start_idx + L]
        
        # Check values to ensure not padding/garbage
        if np.all(chunk > 0):
             return torch.from_numpy(chunk.astype(np.int64)).unsqueeze(0).to(DEVICE)
             
    # Fallback
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    args = parser.parse_args()
    
    global BATCH_SIZE
    BATCH_SIZE = args.batch

    print("🔹 ExperimentLM V4: Training")
    print(f"   Device: {DEVICE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # --- Chinchilla Analysis ---
    total_tokens = estimate_tokens()
    optimal_params = total_tokens / 20.0
    print(f"📊 Dataset: {total_tokens/1e6:.1f}M Tokens")
    print(f"🎯 Optimal Params (Chinchilla): {optimal_params/1e6:.1f}M")
    
    torch.manual_seed(1337)
    model = TRM(config).to(DEVICE)
    current_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model Parameters: {current_params/1e6:.2f}M")
    
    ratio = current_params / optimal_params if optimal_params > 0 else 0
    print(f"⚖️  Ratio (Model/Optimal): {ratio:.2f}x (1.0 is perfect)")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.95))
    
    print("🚀 Starting Loop...")

    # --- Load Zero Token ID ---
    bins_path = os.path.join(DATA_DIR, "bins.npy")
    if os.path.exists(bins_path):
        bins = np.load(bins_path, allow_pickle=True).item()
        
        # Recalculate Offsets (Must match dataset.py)
        # Schema: [DOW, Hour, Volat(t-1), Vol(t-1), Ret(t)]
        OFFSET_DOW = 0
        OFFSET_HOUR = 7
        OFFSET_VOLAT = 31
        OFFSET_VOL = OFFSET_VOLAT + len(bins['volat_lag']) + 1
        OFFSET_RET = OFFSET_VOL + len(bins['vol_lag']) + 1
        
        # Find ID for 0.0 return
        # np.digitize returns 1-based index for bins, so we subtract 1 or keep logic consistent with dataset.py
        # dataset.py uses: inds = np.digitize(values, bins) -> returns index i such that bins[i-1] <= x < bins[i]
        # We need to find which bin 0.0 falls into.
        zero_bin_idx = np.digitize([0.0], bins['log_ret'])[0]
        ZERO_TOKEN_ID = int(OFFSET_RET + zero_bin_idx)
        print(f"🎯 Zero Token ID: {ZERO_TOKEN_ID} (Offset: {OFFSET_RET}, Bin: {zero_bin_idx})")
    else:
        ZERO_TOKEN_ID = None
        print("⚠️  bins.npy not found. Cannot calculate Actionable Accuracy.")

    X, Y = get_batch('train') 
    t0 = time.time()
    
    for iter_num in range(MAX_ITERS):
        lr = get_lr(iter_num)
        for pg in optimizer.param_groups: pg['lr'] = lr
            
        for _ in range(GRADIENT_ACCUMULATION_STEPS):
            _, loss = model(X, Y)
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            loss.backward()
            X, Y = get_batch('train') 
            
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()
        
        if iter_num % LOG_INTERVAL == 0:
            loss_f = loss.item() * GRADIENT_ACCUMULATION_STEPS
            dt = time.time() - t0
            print(f"iter {iter_num}: loss {loss_f:.4f}, lr {lr:.2e}, time {dt*1000:.2f}ms")
            t0 = time.time()
            
        if iter_num > 0 and iter_num % EVAL_INTERVAL == 0:
            print("📉 Evaluating & Eye Test...")
            model.eval()
            losses = torch.zeros(EVAL_ITERS)
            accuracies = torch.zeros(EVAL_ITERS)
            return_accuracies = torch.zeros(EVAL_ITERS)
            actionable_accuracies = torch.zeros(EVAL_ITERS)
            
            with torch.no_grad():
                for k in range(EVAL_ITERS):
                    Xval, Yval = get_batch('val')
                    logits, loss = model(Xval, Yval)
                    losses[k] = loss.item()
                    
                    # Calculate Accuracy
                    # logits: (B, T, V)
                    predictions = torch.argmax(logits, dim=-1) # (B, T)
                    acc = (predictions == Yval).float().mean()
                    accuracies[k] = acc.item()

                    # Return Token Accuracy (Every 5th token is Ret(t))
                    # Tokens: [DOW, Hour, Volat, Vol, Ret]
                    T = Yval.shape[1]
                    indices = torch.arange(T, device=DEVICE)
                    is_return_token = (indices % 5) == 4
                    
                    # predictions: (B, T), Yval: (B, T)
                    # Select only columns where is_return_token is True
                    pred_ret = predictions[:, is_return_token]
                    targ_ret = Yval[:, is_return_token]
                    
                    acc_ret = (pred_ret == targ_ret).float().mean()
                    return_accuracies[k] = acc_ret.item()

                    # Actionable Accuracy (Non-Zero Targets)
                    if ZERO_TOKEN_ID is not None:
                        is_moving = (targ_ret != ZERO_TOKEN_ID)
                        if is_moving.sum() > 0:
                            acc_actionable = (pred_ret[is_moving] == targ_ret[is_moving]).float().mean()
                            actionable_accuracies[k] = acc_actionable.item()
                        else:
                            actionable_accuracies[k] = float('nan')
            
            # --- Live Hallucination: The Dream of the Model ---
            print("🧠 Dreaming a future (Green Line)...")
            try:
                if 'bins' in locals():
                    import matplotlib.pyplot as plt
                    
                    # 1. Get Stable Context
                    # We need the full val_data to sample from
                    # We can't easily access it here as get_batch only returns a batch.
                    # But we can mmap it quickly just for this sample.
                    val_path = os.path.join(DATA_DIR, 'val.bin')
                    val_mmap = np.memmap(val_path, dtype=np.uint16, mode='r')
                    
                    ctx = get_stable_sample(bins, val_mmap, config.max_seq_len)
                    
                    if ctx is not None:
                        # 2. Generate Dream (Green Line - Zero Suppressed)
                        # Offsets must be packed
                        off_dict = {
                            'DOW': OFFSET_DOW, 'HOUR': OFFSET_HOUR,
                            'VOLAT': OFFSET_VOLAT, 'VOL': OFFSET_VOL, 'RET': OFFSET_RET
                        }
                        
                        FUTURE_STEPS = 50
                        dream_toks = generate_dream(
                            model, ctx, 
                            max_new_tokens=FUTURE_STEPS*5, 
                            offsets=off_dict,
                            temperature=1.0, 
                            suppress_zero=True, 
                            zero_token_id=ZERO_TOKEN_ID
                        )
                        
                        # 3. Decode & Plot
                        # We need to extract Returns from the sequence
                        # New tokens start at ctx.shape[1]
                        generated_part = dream_toks[0, ctx.shape[1]:].cpu().numpy()
                        
                        # Extract RET tokens
                        # We want indices where (ctx_len + i) % 5 == 4
                        # i = (4 - ctx_len) % 5
                        ctx_len = ctx.shape[1]
                        start_offset = (4 - ctx_len) % 5
                        ret_tokens = generated_part[start_offset::5]
                        
                        # Map to values
                        log_ret_bins = bins['log_ret']
                        def get_val(tok):
                            idx = np.clip(tok - OFFSET_RET, 0, len(log_ret_bins)-1)
                            return log_ret_bins[idx]
                        
                        dream_rets = [get_val(t) for t in ret_tokens]
                        
                        # Compute Price Path
                        # Get last known price from context? We don't have it. Assume 100.
                        price_dream = [100.0]
                        curr_p = 100.0
                        for r in dream_rets:
                            curr_p *= np.exp(r)
                            price_dream.append(curr_p)
                            
                        # Plot
                        plt.figure(figsize=(10, 5))
                        plt.plot(price_dream, color='green', label='Dream (Forced Motion)')
                        plt.title(f"Live Hallucination (Iter {iter_num})")
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        
                        plot_path = os.path.join(OUT_DIR, f"hallucination_step_{iter_num}.png")
                        plt.savefig(plot_path)
                        plt.close()
                        print(f"📸 Saved hallucination: {plot_path}")
                    else:
                        print("⚠️ Could not find stable context for dream.")
            except Exception as e:
                print(f"❌ Hallucination Failed: {e}")
                import traceback
                traceback.print_exc()

            val_loss = losses.mean()
            val_acc = accuracies.mean()
            val_ret_acc = return_accuracies.mean()
            val_act_acc = np.nanmean(actionable_accuracies.numpy()) # Handle NaNs
            
            print(f"🔍 VAL LOSS: {val_loss:.4f} | ACCURACY: {val_acc*100:.2f}% | RETURN ACCURACY: {val_ret_acc*100:.2f}% | ACTIONABLE: {val_act_acc*100:.2f}%")
            
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"ckpt_{iter_num}.pt"))
            model.train()

    print("🏁 Training Complete.")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "final.pt"))

if __name__ == "__main__":
    main()
