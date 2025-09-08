# MultiHeadAttention, FeedForward, GELU, LayerNorm
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self,
        d_in,
        d_out,
        context_length,
        dropout,
        num_heads,
        qkv_bias=False,
    ):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.Wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.Wv = nn.Linear(d_in, d_out, bias=qkv_bias)

        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        # Causal mask: lower triangular ones; we'll mask where entries are 0
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context_length, context_length)).view(1, 1, context_length, context_length)
        )
        
    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.Wk(x) # (b, num_tokens, d_out)
        queries = self.Wq(x)
        values = self.Wv(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim) # (b, num_tokens, num_heads, head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2) # (b, num_heads, num_tokens, head_dim)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        
        attn_scores = queries @ keys.transpose(2, 3) # (b, num_heads, num_tokens, num_tokens)

        # Apply causal mask: allow looking at current and past tokens only
        mask = self.mask[:, :, :num_tokens, :num_tokens]
        attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = nn.functional.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) # (b, num_heads, num_tokens, num_tokens)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2) # (b, num_heads, num_tokens, head_dim)
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out) # (b, num_tokens, d_out)
        context_vec = self.out_proj(context_vec) # (b, num_tokens, d_out)

        return context_vec

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5 # prevent division by 0
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean)/torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]), # 768
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
        
    def forward(self, x):
        return self.layers(x)