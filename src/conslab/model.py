"""A tiny decoder-only transformer that learns column addition from scratch.

Deliberately nanoGPT-shaped: learned token + position embeddings, a stack of
causal self-attention + MLP blocks, and a linear head.  At ~2 layers / 64 dims
it is a few tens of thousands of parameters, which is exactly why the whole
self-consistency study runs in CPU seconds and every curve is reproducible.

The model continues the prompt ``a.. + b.. =`` with the column-addition chain;
``generate_cot`` decodes either greedily (the k=1 baseline) or by temperature
sampling, which is what draws the *diverse* reasoning chains majority voting -
and the adaptive stopping rule - operate on.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NVOCAB, PAD


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        return (x + a) + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 48, n_head: int = 4, n_layer: int = 2,
                 max_len: int = 24):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(NVOCAB, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NVOCAB)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(ids) + self.pos(pos)
        mask = torch.triu(torch.full((t, t), float("-inf"), device=ids.device),
                          diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def sft_loss(self, prompt: torch.Tensor, cot: torch.Tensor) -> torch.Tensor:
        """Teacher-forced CE over the CoT span only (prompt positions unmasked)."""
        x = torch.cat([prompt, cot[:, :-1]], dim=1)
        logits = self.forward(x)
        p = prompt.shape[1]
        preds = logits[:, p - 1: p - 1 + cot.shape[1], :]
        return F.cross_entropy(preds.reshape(-1, NVOCAB), cot.reshape(-1))

    @torch.no_grad()
    def sample_cot(self, prompt: torch.Tensor, n_tokens: int,
                   temperature: float) -> torch.Tensor:
        """Sample one continuation per row of ``prompt`` (B, P) -> (B, n_tokens)."""
        self.eval()
        x = prompt.clone()
        for _ in range(n_tokens):
            logits = self.forward(x)[:, -1, :] / max(temperature, 1e-5)
            tok = torch.multinomial(F.softmax(logits, dim=-1), 1)
            x = torch.cat([x, tok], dim=1)
        return x[:, prompt.shape[1]:]

    @torch.no_grad()
    def greedy_cot(self, prompt: torch.Tensor, n_tokens: int) -> torch.Tensor:
        self.eval()
        x = prompt.clone()
        for _ in range(n_tokens):
            tok = self.forward(x)[:, -1, :].argmax(dim=-1, keepdim=True)
            x = torch.cat([x, tok], dim=1)
        return x[:, prompt.shape[1]:]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
