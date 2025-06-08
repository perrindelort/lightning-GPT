"""Defines the GPT architecture from Attention is all you need using custom made layers."""  # noqa: E501
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from lightning import LightningModule
from lightning.pytorch.cli import (
    LRSchedulerCallable,
    OptimizerCallable,
    ReduceLROnPlateau,
)
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from lightning_gpt.layers import Block


class GPT(LightningModule):
    """A simplified GPT-style Transformer model implemented using PyTorch Lightning.

    This model includes token/position embeddings, a stack of transformer blocks,
    a language modeling head, and training utilities for autoregressive generation.

    Attributes:
        block_size (int): Maximum number of tokens the model can handle in a sequence.
        token_embedding_table (nn.Embedding): Embedding layer that maps token IDs to vectors.
        position_embeddings_table (nn.Embedding): Embedding layer that encodes positional information.
        blocks (nn.Sequential): A stack of transformer blocks performing attention and feedforward computation.
        ln_f (nn.LayerNorm): Final layer normalization layer applied after transformer blocks.
        lm_head (nn.Linear): Output projection layer that maps hidden states to vocabulary logits.
        optimizer (OptimizerCallable): Function to initialize the optimizer.
        scheduler (LRSchedulerCallable): Function to initialize the learning rate scheduler.

    """  # noqa: E501

    def __init__(
        self,
        block_size: int,
        n_embeddings: int,
        n_heads: int,
        dropout: float,
        n_layers: int,
        vocab_size: int,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ):
        """Initialize the GPT model.

        Args:
            block_size (int): Maximum sequence length.
            n_embeddings (int): Dimensionality of token embeddings.
            n_heads (int): Number of attention heads.
            dropout (float): Dropout rate used throughout the model.
            n_layers (int): Number of transformer blocks.
            vocab_size (int): Size of the token vocabulary.
            optimizer (OptimizerCallable): Function that returns an optimizer.
            scheduler (LRSchedulerCallable): Function that returns a learning rate scheduler.

        """  # noqa: E501
        super().__init__()
        self.save_hyperparameters()
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.position_embeddings_table = nn.Embedding(block_size, n_embeddings)

        self.blocks = nn.Sequential(
            *[
                Block(
                    n_embeddings=n_embeddings,
                    block_size=block_size,
                    n_heads=n_heads,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocab_size)

        self.optimizer = optimizer
        self.scheduler = scheduler

    def configure_model(self) -> None:
        """Apply custom weight initialization across the model."""
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights following normal distribution."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            idx (torch.Tensor): Tensor of token indices with shape (batch_size, seq_len)

        Returns:
            torch.Tensor: Logits of shape (batch_size, seq_len, vocab_size)

        """
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)
        pos_embeddings = self.position_embeddings_table(
            torch.arange(T, device=self.device)
            )
        x = token_embeddings + pos_embeddings
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def _calculate_loss(
        self,
        batch: tuple[torch.Tensor, torch.Tensor] | torch.Tensor,
        mode: str = "train",
    ) -> dict[str, torch.Tensor | None]:
        """Compute cross-entropy loss and logits from a batch.

        Args:
            batch (tuple or Tensor): Tuple of (input_ids, targets) or just input_ids.
            mode (str): Indicates the current mode ('train', 'val', 'predict').

        Returns:
            dict: Dictionary with keys 'loss' and 'logits'.

        """
        if isinstance(batch, tuple) and len(batch) == 2:
            x, targets = batch
        else:
            x = batch
            targets = None

        logits = self(x)

        if targets is None or mode == "predict":
            loss = None
        else:
            logits = rearrange(logits, "b t c -> (b t) c")
            targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits, targets)
            self.log(f"{mode}_loss", loss, prog_bar=True)

        return {"loss": loss, "logits": logits}

    def training_step(
        self,
        batch: torch.Tensor,
        batch_idx: int = 0
        ) -> dict[str, torch.Tensor | None]:
        """Training step for a single batch."""
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int = 0) -> None:
        """Validation step for a single batch."""
        self._calculate_loss(batch, mode="val")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Set up optimizer and scheduler."""
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        if isinstance(
            scheduler,
            (ReduceLROnPlateau, torch.optim.lr_scheduler.ReduceLROnPlateau),
            ):
            return { # type: ignore[return-value]
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_fit_start(self) -> None:
        """Generate and log example output before training begins."""
        self.eval()
        self._generate_and_log_example(epoch=-1)
        self.train()

    def on_validation_end(self) -> None:
        """Generate and log example output after each validation phase."""
        if not self.trainer.sanity_checking:
            self._generate_and_log_example()

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Autoregressively generate tokens from a starting index.

        Args:
            idx (torch.Tensor): Initial token indices (batch_size, seq_len).
            max_new_tokens (int): Maximum number of new tokens to generate.

        Returns:
            torch.Tensor: Tensor of shape (batch_size, seq_len + max_new_tokens)

        """
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self.predict_step(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def _generate_and_log_example(
            self,
            max_new_tokens: int = 1_000,
            epoch: Optional[int] = None,
        ) -> None:
        """Generate and log sample text from the model.

        Args:
            max_new_tokens (int): Number of tokens to generate.
            epoch (Optional[int]): Epoch number to tag the log with.

        """
        epoch = self.trainer.current_epoch if epoch is None else epoch
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        logger = getattr(self.trainer, "logger", None)
        if (
            logger is not None
            and hasattr(self.trainer, "datamodule")
            and hasattr(logger, "_experiment")
        ):
            generated_text = self.trainer.datamodule.decode(
                self.generate(context, max_new_tokens)[0].tolist()
            )
            logger._experiment.add_text("Generated sample", generated_text, epoch)

            f_path = Path(f"{logger.log_dir}/generated_text.txt").resolve()
            with open(f_path, "a") as f:
                f.write(f"Epoch {epoch}:{generated_text}\n")

    def infinite_generation(
            self,
            decode_fn: Callable[[list[int]], str],
            temperature: float = 1.0,
        ):
        """Infinite text generator yielding decoded tokens one at a time.

        Args:
            decode_fn (Callable): Function to convert token IDs into string/text.
            temperature (float): Sampling temperature to control randomness.

        Yields:
            str: Decoded token or text fragment.

        """
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        while True:
            context_cond = context[:, -self.block_size:]
            logits = self(context_cond)[:, -1, :]
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
            decoded_text = decode_fn(next_token[0].tolist())
            yield decoded_text
