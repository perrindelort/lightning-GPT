import torch.nn as nn
import torch

from layers import Head
from einops import rearrange
from typing import Optional


class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self-attention in parallel
    """

    def __init__(
        self,
        head_size: int,
        n_embeddings: int,
        block_size: int,
        n_heads: int,
        dropout: Optional[float] = 0.0,
    ):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embeddings, block_size, dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = rearrange([h(x) for h in self.heads], "nh b t c -> b t (nh c)")
        # out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
