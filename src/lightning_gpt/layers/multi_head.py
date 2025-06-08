"""Defines the MultiHeadAttention layer."""
import torch
import torch.nn as nn
from einops import rearrange

from lightning_gpt.layers import Head


class MultiHeadAttention(nn.Module):
    """Implements multi-head self-attention as used in transformer architectures.

    This module applies several parallel self-attention "heads" to the input and then
    projects the concatenated outputs back into the embedding space. Each head operates
    independently and captures different aspects of the relationships between tokens.

    Args:
        head_size (int): Dimensionality of each attention head.
        n_embeddings (int): Total embedding size of the model; usually `head_size * n_heads`.
        block_size (int): Maximum sequence length the model will be trained on.
        n_heads (int): Number of parallel attention heads.
        dropout (float, optional): Dropout probability applied after the output projection. Default is 0.0.

    """  # noqa: E501

    def __init__(
        self,
        head_size: int,
        n_embeddings: int,
        block_size: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        """Initializes the multi head attention."""
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(head_size, n_embeddings, block_size, dropout) for _ in range(n_heads)]
        )
        self.proj = nn.Linear(n_embeddings, n_embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass of multi-head self-attention.

        Each head independently computes self-attention over the input `x`, and their
        outputs are concatenated and projected back to the embedding space.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, n_embeddings).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, n_embeddings),
                          representing the transformed input after multi-head attention.

        """  # noqa: E501
        out = rearrange([h(x) for h in self.heads], "nh b t c -> b t (nh c)")
        out = self.dropout(self.proj(out))
        return out
