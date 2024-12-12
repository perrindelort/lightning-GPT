import torch.nn as nn

from typing import Optional

from layers import MultiHeadAttention, FeedForward


class Block(nn.Module):
    """
    Transformer block : communication followed by computation
    """

    def __init__(
        self, n_embeddings: int, block_size: int, n_heads: int, dropout: Optional[float] = 0.0
    ):
        super().__init__()
        head_size = n_embeddings // n_heads
        self.sa = MultiHeadAttention(head_size, n_embeddings, block_size, n_heads, dropout)
        self.ffwd = FeedForward(n_embeddings, dropout)
        self.ln1 = nn.LayerNorm(n_embeddings)
        self.ln2 = nn.LayerNorm(n_embeddings)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # Residual connection
        x = x + self.ffwd(self.ln2(x))  # Residual connection
        return x
