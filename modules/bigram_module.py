import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from argparse import Namespace
from einops import rearrange
from typing import Optional


class BigramLanguageModel(LightningModule):
    def __init__(self, config: Namespace, vocab_size: int):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.token_embedding_table = nn.Embedding(vocab_size, self.config.n_embeddings)
        self.lm_head = nn.Linear(self.config.n_embeddings, vocab_size)

    def forward(self, x):
        token_embeddings = self.token_embedding_table(x)  # (B, T, C)
        logits = self.lm_head(token_embeddings)  # (B, T, C)
        return logits

    def _calculate_loss(self, batch, mode="train"):
        if len(batch) == 2:
            x, targets = batch
        else:
            x = batch
            targets = None
        logits = self(x)  # (B, T, C) for einops ;)

        if targets is None or mode == "predict":
            loss = None
        else:
            logits = rearrange(logits, "b t c -> (b t) c")
            targets = rearrange(targets, "b t -> (b t)")
            loss = F.cross_entropy(logits, targets)

            self.log(f"{mode}_loss", loss, prog_bar=True)

        return {"loss": loss, "logits": logits}

    def training_step(self, batch, batch_idx=0):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx=0):
        self._calculate_loss(batch, mode="val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer

    def on_fit_start(self):
        # Log an example without any training
        self.eval()
        self._generate_and_log_example(epoch=-1)
        self.train()

    def on_validation_end(self):
        if not self.trainer.sanity_checking:
            self._generate_and_log_example()

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self.predict_step(x)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x

    def _generate_and_log_example(self, max_new_tokens: int = 1_000, epoch: Optional[int] = None):
        epoch = self.trainer.current_epoch if epoch is None else epoch
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        generated_text = self.trainer.datamodule.decode(self.generate(context, max_new_tokens=max_new_tokens)[0].tolist())
        self.trainer.logger._experiment.add_text("Generated sample", generated_text, epoch)

    def infinite_generation(
        self,
        decode_fn,
        temperature: float = 1.0,
    ):
        """
        Smooth character-by-character text generation using a generator.

        Args:
            temperature: Sampling temperature for controlling randomness.
        Yields:
            Generated characters or text incrementally.
        """
        context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        block_size = self.config.block_size

        while True:
            context_cond = context[:, -block_size:]
            logits = self(context_cond)[:, -1, :]  # Get logits for the last position

            probs = F.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # Sample the next token

            context = torch.cat((context, next_token), dim=1)
            decoded_text = decode_fn(next_token[0].tolist())

            yield decoded_text
