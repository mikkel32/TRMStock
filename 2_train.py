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
BATCH_SIZE = 8 # Prevent OOM

# Model Config (V4) - Optimized for ~650M Tokens (Chinchilla)
# Params: ~34M (Perfect match for 650M tokens)
config = TRMConfig(
    vocab_size=8192, 
    dim=768,         # 384 -> 768
    n_layers=4,      
    n_recurrence=3,  
    n_heads=12,      # 6 -> 12
    n_kv_heads=4,    # 2 -> 4
    max_seq_len=2048,
    dropout=0.0
)

# Training Config
BLOCK_SIZE = config.max_seq_len
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-3 
MAX_ITERS = 4960 # ~650M Tokens / 131,072 tokens per step
WARMUP_ITERS = 200
LR_DECAY_ITERS = 4960
MIN_LR = 1e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 50   # More frequent "Eye Test"
LOG_INTERVAL = 10
EVAL_ITERS = 50

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
                    
            val_loss = losses.mean()
            val_acc = accuracies.mean()
            val_ret_acc = return_accuracies.mean()
            print(f"🔍 VAL LOSS: {val_loss:.4f} | ACCURACY: {val_acc*100:.2f}% | RETURN ACCURACY: {val_ret_acc*100:.2f}%")
            
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"ckpt_{iter_num}.pt"))
            model.train()

    print("🏁 Training Complete.")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "final.pt"))

if __name__ == "__main__":
    main()
