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
import numpy as np
import torch
from src.model import TRM, TRMConfig # Import from src/

# --- Configuration ---
DATA_DIR = "data"
OUT_DIR = "checkpoints"

# System
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8 # Prevent OOM

# Model Config (V4)
config = TRMConfig(
    vocab_size=8192, 
    dim=384,
    n_layers=4,      
    n_recurrence=3,  
    n_heads=6,
    n_kv_heads=2,    
    max_seq_len=2048,
    dropout=0.0
)

# Training Config
BLOCK_SIZE = config.max_seq_len
GRADIENT_ACCUMULATION_STEPS = 8
LEARNING_RATE = 1e-3 
MAX_ITERS = 10000
WARMUP_ITERS = 200
LR_DECAY_ITERS = 10000
MIN_LR = 1e-4
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0
EVAL_INTERVAL = 500
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

def main():
    print("🔹 ExperimentLM V4: Training")
    print(f"   Device: {DEVICE}")
    print(f"   Batch Size: {BATCH_SIZE}")
    
    os.makedirs(OUT_DIR, exist_ok=True)
    
    torch.manual_seed(1337)
    model = TRM(config).to(DEVICE)
    print(f"🧠 Model Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
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
            print("📉 Evaluating...")
            model.eval()
            losses = torch.zeros(EVAL_ITERS)
            with torch.no_grad():
                for k in range(EVAL_ITERS):
                    Xval, Yval = get_batch('val')
                    _, loss = model(Xval, Yval)
                    losses[k] = loss.item()
            print(f"🔍 VAL LOSS: {losses.mean():.4f}")
            torch.save(model.state_dict(), os.path.join(OUT_DIR, f"ckpt_{iter_num}.pt"))
            model.train()

    print("🏁 Training Complete.")
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "final.pt"))

if __name__ == "__main__":
    main()
