"""Defines the BigramLanguageModel."""
from typing import Any, Callable, Dict, Generator, Optional, Tuple, Union

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


class BigramLanguageModel(LightningModule):
    """Bigram language model implemented with PyTorch Lightning.

    Args:
        block_size (int): Maximum context size for the model.
        n_embeddings (int): Dimension of the token embeddings.
        vocab_size (int): Vocabulary size.
        optimizer (OptimizerCallable): Callable returning an optimizer instance.
        scheduler (LRSchedulerCallable): Callable returning a learning rate scheduler.

    Attributes:
        token_embedding_table (nn.Embedding): Embedding layer for tokens.
        lm_head (nn.Linear): Linear layer projecting embeddings to vocab logits.
        block_size (int): Maximum sequence length the model can handle.
        optimizer (OptimizerCallable): Optimizer constructor.
        scheduler (LRSchedulerCallable): Scheduler constructor.

    """

    def __init__(
        self,
        block_size: int,
        n_embeddings: int,
        vocab_size: int,
        optimizer: OptimizerCallable,
        scheduler: LRSchedulerCallable,
    ) -> None:
        """Initialized the BigramLanguageModel."""
        super().__init__()
        self.save_hyperparameters()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embeddings)
        self.lm_head = nn.Linear(n_embeddings, vocab_size)
        self.block_size = block_size
        self.optimizer = optimizer
        self.scheduler = scheduler

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits tensor of shape (batch_size, sequence_length, vocab_size).

        """  # noqa: E501
        token_embeddings = self.token_embedding_table(x)  # (B, T, C)
        logits = self.lm_head(token_embeddings)  # (B, T, vocab_size)
        return logits

    def _calculate_loss(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        mode: str = "train",
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Calculate cross-entropy loss for a batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor] or torch.Tensor): Input batch,
                either (inputs, targets) tuple or inputs only.
            mode (str, optional): Mode name for logging ('train', 'val', 'predict').
                Defaults to "train".

        Returns:
            Dict[str, Optional[torch.Tensor]]: Dictionary containing:
                - "loss": Loss tensor or None if no targets.
                - "logits": Logits tensor.

        """
        if len(batch) == 2:
            x, targets = batch
        else:
            x = batch
            targets = None
        logits = self(x)  # (B, T, C)

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
        batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int = 0,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Training step for a single batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor] or torch.Tensor): Training batch.
            batch_idx (int, optional): Index of the batch. Defaults to 0.

        Returns:
            Dict[str, Optional[torch.Tensor]]: Output of _calculate_loss with loss and logits.

        """  # noqa: E501
        return self._calculate_loss(batch, mode="train")

    def validation_step(
        self,
        batch: Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
        batch_idx: int = 0,
    ) -> None:
        """Validation step for a single batch.

        Args:
            batch (Tuple[torch.Tensor, torch.Tensor] or torch.Tensor): Validation batch.
            batch_idx (int, optional): Index of the batch. Defaults to 0.

        """
        self._calculate_loss(batch, mode="val")

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Configure optimizer and learning rate scheduler.

        Returns:
            OptimizerLRScheduler: Dictionary containing optimizer and scheduler config.

        """
        optimizer = self.optimizer(self.parameters())
        scheduler = self.scheduler(optimizer)
        if isinstance(
            scheduler,
            (ReduceLROnPlateau, torch.optim.lr_scheduler.ReduceLROnPlateau),
        ):
            return {  # type: ignore[return-value]
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val_loss",
            }
        else:
            return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_fit_start(self) -> None:
        """Called at the start of fitting to generate and log an example."""
        self.eval()
        self._generate_and_log_example(epoch=-1)
        self.train()

    def on_validation_end(self) -> None:
        """Called at the end of validation to generate and log an example, skipping during sanity check."""  # noqa: E501
        if not self.trainer.sanity_checking:
            self._generate_and_log_example()

    def generate(self, x: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate tokens autoregressively given an initial context.

        Args:
            x (torch.Tensor): Initial input tensor of shape (batch_size, context_length).
            max_new_tokens (int): Number of new tokens to generate.

        Returns:
            torch.Tensor: Generated sequence including initial context and new tokens.

        """  # noqa: E501
        for _ in range(max_new_tokens):
            logits = self.predict_step(x)
            logits = logits[:, -1, :]  # Get logits for the last token
            probs = F.softmax(logits, dim=-1)
            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x

    def _generate_and_log_example(
        self, max_new_tokens: int = 1_000, epoch: Optional[int] = None
    ) -> None:
        """Generate text example and log it to the experiment logger.

        Args:
            max_new_tokens (int, optional): Number of tokens to generate. Defaults to 1000.
            epoch (Optional[int], optional): Epoch index for logging. Defaults to None.

        """  # noqa: E501
        epoch = self.trainer.current_epoch if epoch is None else epoch
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        logger = getattr(self.trainer, "logger", None)
        if  (
            logger is not None
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.logger, "_experiment")
        ):
            generated_text = self.trainer.datamodule.decode(
                self.generate(context, max_new_tokens=max_new_tokens)[0].tolist()
            )
            logger._experiment.add_text(
                "Generated sample", generated_text, epoch
            )

    def infinite_generation(
        self,
        decode_fn: Callable[[list[int]], Any],
        temperature: float = 1.0,
    ) -> Generator[Any, None, None]:
        """Generator for smooth character-by-character text generation.

        Args:
            decode_fn (Callable[[list[int]], Any]): Function to decode tokens to text.
            temperature (float, optional): Sampling temperature controlling randomness. Defaults to 1.0.

        Yields:
            Any: Decoded generated text incrementally.

        """  # noqa: E501
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)

        while True:
            context_cond = context[:, -self.block_size :]
            logits = self(context_cond)[:, -1, :]  # Get logits for the last position
            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            context = torch.cat((context, next_token), dim=1)
            decoded_text = decode_fn(next_token[0].tolist())
            yield decoded_text
