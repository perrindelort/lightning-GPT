"""Defines a block layer."""
import torch.nn as nn
from torch import Tensor

from lightning_gpt.layers import FeedForward, MultiHeadAttention


class Block(nn.Module):
    """Transformer block : communication followed by computation."""

    def __init__(
        self,
        n_embeddings: int,
        block_size: int,
        n_heads: int,
        dropout: float = 0.0,
    ):
        """Initialize the block instance.

        Args:
            n_embeddings (int): number of embeddings.
            block_size (int): block_size
            n_heads (int): number of heads.
            dropout (float, optional): dropout. Defaults to 0.0.

        """
        super().__init__()
        head_size = n_embeddings // n_heads
        self.sa = MultiHeadAttention(
            head_size,
            n_embeddings,
            block_size,
            n_heads,
            dropout
        )
        self.ffwd = FeedForward(n_embeddings, dropout)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self, x: Tensor) -> Tensor:
        """Apply a Transformer block consisting of self-attention and feed-forward layers.

        This includes:
        - Layer normalization before each sub-layer (pre-norm)
        - Residual connections after each sub-layer

        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, n_embeddings)

        Returns:
            Tensor: Output tensor of the same shape as input (batch_size, sequence_length, n_embeddings)

        """  # noqa: E501
        x = x + self.sa(self.ln1(x))  # Residual connection
        x = x + self.ffwd(self.ln2(x))  # Residual connection
        return x
