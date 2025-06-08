"""Defines a head layer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class Head(nn.Module):
    """A single head of causal self-attention.

    This module computes attention over input embeddings using learned
    key, query, and value projections. It applies a causal (lower-triangular)
    mask to prevent attending to future positions, and performs a weighted
    aggregation based on attention scores.
    """

    def __init__(
        self,
        head_size: int,
        n_embeddings: int,
        block_size: int,
        dropout: float = 0.0,
    ):
        """Initialize the self-attention head.

        Args:
            head_size (int): Dimensionality of each attention head.
            n_embeddings (int): Dimensionality of input embeddings.
            block_size (int): Maximum sequence length (for causal mask).
            dropout (float, optional): Dropout rate applied to attention weights. Defaults to 0.0.

        """  # noqa: E501
        super().__init__()
        self.key = nn.Linear(n_embeddings, head_size, bias=False)
        self.query = nn.Linear(n_embeddings, head_size, bias=False)
        self.value = nn.Linear(n_embeddings, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute attention output for a batch of sequences.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embeddings)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_len, head_size)

        """
        B, T, C = x.shape
        k = self.key(x)  # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores
        w = q @ rearrange(k, "b t c -> b c t") * k.shape[-1] ** -0.5  # (B, T, T)
        w = w.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # Causal mask
        w = F.softmax(w, dim=-1)  # Attention weights
        w = self.dropout(w)

        # Aggregate values
        v = self.value(x)  # (B, T, head_size)
        out = w @ v  # Weighted sum: (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out
