import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Optional


class Head(nn.Module):
    """
    One head of self-attention
    """

    def __init__(
        self, head_size: int, n_embeddings: int, block_size: int, dropout: Optional[float] = 0.0
    ):
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C)
        q = self.query(x)
        # Compute attention score "affinities"
        w = q @ rearrange(k, "b t c -> b c t") * C**-0.5  # (B, T, C) @ (B, C, T) -> (B, T, T)
        w = w.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # (B, T, T) lower triangular "exponentially-wise"
        w = F.softmax(w, dim=-1)  # (B, T, T)
        w = self.dropout(w)
        # Perform the weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = w @ v  # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
