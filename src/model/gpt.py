import torch
import torch.nn as nn

from src.model.transformer import TransformerBlock
from src.model.layer import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
        
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)

        logits = self.out_head(x)
        
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Autoregressively generate tokens given initial `input_ids`.

        Args:
            input_ids: LongTensor of shape (batch, seq_len)
            max_new_tokens: number of tokens to sample
            temperature: softmax temperature
            top_k: if set, keep only top_k logits before sampling
        """
        self.eval()
        batch_size = input_ids.size(0)
        device = input_ids.device
        context_length = self.pos_emb.num_embeddings

        generated = input_ids
        for _ in range(max_new_tokens):
            # Crop context to the last context_length tokens
            input_cond = generated[:, -context_length:]
            logits = self.forward(input_cond)
            logits = logits[:, -1, :]
            if temperature != 1.0:
                logits = logits / max(1e-8, temperature)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
        return generated

    