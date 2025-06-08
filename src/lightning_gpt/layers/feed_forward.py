"""Defines a feed-forward layer."""
import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """A two-layer feed-forward neural network with ReLU activation and dropout.

    This module is typically used in transformer architectures after the self-attention layer.
    It expands the embedding dimension by a factor of 4, applies a non-linearity, then projects back.
    """  # noqa: E501

    def __init__(self, n_embeddings: int, dropout: float = 0.0):
        """Initialize the FeedForward module.

        Args:
            n_embeddings (int): The input and output embedding dimension.
            dropout (float, optional): Dropout rate applied after the second linear layer. Defaults to 0.0.

        """  # noqa: E501
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                n_embeddings,
                4 * n_embeddings, # 4 comes from "Attention is All You Need"
                ),
            nn.ReLU(),
            nn.Linear(4 * n_embeddings, n_embeddings),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the feed-forward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, n_embeddings)

        Returns:
            torch.Tensor: Output tensor of the same shape as input (batch_size, seq_len, n_embeddings)

        """  # noqa: E501
        return self.net(x)
