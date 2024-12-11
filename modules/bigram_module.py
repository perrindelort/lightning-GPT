import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from argparse import Namespace
from einops import rearrange


class BigramLanguageModel(LightningModule):
    def __init__(self, config: Namespace, vocab_size: int):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        logits = self.token_embedding_table(x)
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

    def predict_step(self, batch, batch_idx=0):
        return self._calculate_loss(batch, mode="predict")["logits"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer

    def generate(self, x, max_new_tokens):
        for _ in range(max_new_tokens):
            logits = self.predict_step(x)

            logits = logits[:, -1, :]

            probs = F.softmax(logits, dim=-1)

            x_next = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, x_next), dim=1)
        return x
