"""
model.py
The TRM (Recursive Transformer) Architecture.

Key Innovations:
- Recurrence: Physical layers are reused 'n_recurrence' times to deepen the network without parameter bloat.
- Efficiency: SwiGLU, RMSNorm, RoPE, GQA.
- Compactness: Vocab size optimized for 20M parameter scale.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class TRMConfig:
    vocab_size: int = 8192
    dim: int = 512
    n_layers: int = 2        # Physical layers
    n_recurrence: int = 6    # How many times to loop the layers (Total depth = n_layers * n_recurrence)
    n_heads: int = 8
    n_kv_heads: int = 2      # GQA: 8 query heads, 2 KV heads (4:1 ratio)
    head_dim: int = 64
    multiple_of: int = 32    # For SwiGLU hidden dim
    dropout: float = 0.0
    max_seq_len: int = 16384 # V2: Support 365 * 8 bars * 5 tokens ~= 14,600
    norm_eps: float = 1e-5

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        var = torch.mean(x ** 2, dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(var + self.eps)
        return self.weight * x_normed

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    # xq: (B, T, H, D) -> complex
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    
    # Broadcast freqs
    freqs_cis = freqs_cis[:xq.shape[1]].view(1, xq.shape[1], 1, -1)
    
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class Attention(nn.Module):
    def __init__(self, args: TRMConfig):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.head_dim = args.head_dim
        self.scale = args.head_dim ** -0.5
        
        self.wq = nn.Linear(args.dim, args.n_heads * args.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, args.n_kv_heads * args.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * args.head_dim, args.dim, bias=False)

    def forward(self, x, freqs_cis, mask=None):
        B, T, C = x.shape
        
        xq = self.wq(x).view(B, T, self.n_heads, self.head_dim)
        xk = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim)
        
        # RoPE
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        # GQA Replication
        if self.n_kv_heads != self.n_heads:
            n_rep = self.n_heads // self.n_kv_heads
            xk = xk.repeat_interleave(n_rep, dim=2)
            xv = xv.repeat_interleave(n_rep, dim=2)
            
        # Transpose for Attention: (B, H, T, D)
        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)
        
        # Flash Attention
        output = F.scaled_dot_product_attention(xq, xk, xv, is_causal=True)
        
        # (B, H, T, D) -> (B, T, H, D) -> (B, T, C)
        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.wo(output)

class FeedForward(nn.Module):
    def __init__(self, args: TRMConfig):
        super().__init__()
        hidden_dim = 4 * args.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # Round to multiple of
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of - 1) // args.multiple_of)
        
        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)

    def forward(self, x):
        # SwiGLU: w2(F.silu(w1(x)) * w3(x))
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class RecursiveBlock(nn.Module):
    def __init__(self, args: TRMConfig):
        super().__init__()
        self.attention = Attention(args)
        self.feed_forward = FeedForward(args)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cis):
        h = x + self.attention(self.attention_norm(x), freqs_cis)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class TRM(nn.Module):
    def __init__(self, config: TRMConfig):
        super().__init__()
        self.config = config
        
        self.token_emb = nn.Embedding(config.vocab_size, config.dim)
        
        # Physical Layers
        self.layers = nn.ModuleList([
            RecursiveBlock(config) for _ in range(config.n_layers)
        ])
        
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Weight Tying
        self.output.weight = self.token_emb.weight
        
        # Precompute RoPE frequencies
        freqs_cis = precompute_freqs_cis(config.head_dim, config.max_seq_len * 2) # *2 for safety
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        x = self.token_emb(idx)
        
        # RoPE Frequencies
        freqs_cis = self.freqs_cis[:T]
        
        # Recursive Forward Pass
        for _ in range(self.config.n_recurrence):
            for layer in self.layers:
                x = layer(x, freqs_cis)
                
        x = self.norm(x)
        logits = self.output(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next 
        return idx
