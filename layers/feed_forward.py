import torch.nn as nn

from typing import Optional


class FeedForward(nn.Module):
    """
    Simple linear layer followed by a non-linearity
    """

    def __init__(self, n_embeddings: int, dropout: Optional[float] = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embeddings, 4 * n_embeddings),  # 4 comes from "Attention is all you need"
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),  # 4 comes from "Attention is all you need"
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
