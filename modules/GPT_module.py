import torch
import torch.nn as nn
import torch.nn.functional as F

from lightning import LightningModule
from argparse import Namespace
from einops import rearrange

from layers import Block


class GPT(LightningModule):
    def __init__(self, config: Namespace, vocab_size: int):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(vocab_size, config.n_embeddings)
        self.position_embeddings_table = nn.Embedding(config.block_size, config.n_embeddings)
        self.blocks = nn.Sequential(
            *[
                Block(config.n_embeddings, config.block_size, config.n_heads, config.dropout)
                for _ in range(config.n_layers)
            ],
        )
        self.ln_f = nn.LayerNorm(config.n_embeddings)
        self.lm_head = nn.Linear(self.config.n_embeddings, vocab_size)

        self.last_val_loss = float("inf")
        self.last_generated_sample = float("-inf")

    def configure_model(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx)  # (B, T, C)
        pos_embeddings = self.position_embeddings_table(
            torch.arange(T, device=self.device)
        )  # (T, C))
        x = token_embeddings + pos_embeddings  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(token_embeddings)  # (B, T, C)
        return logits

    def _calculate_loss(self, batch, mode="train"):
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

    def training_step(self, batch, batch_idx=0):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx=0):
        self._calculate_loss(batch, mode="val")

    def predict_step(self, batch, batch_idx=0):
        return self._calculate_loss(batch, mode="predict")["logits"]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.lr)
        return optimizer

    def on_validation_end(self):
        current_val_loss = self.trainer.callback_metrics["val_loss"]
        if (
            current_val_loss < self.last_val_loss
            and self.trainer.current_epoch - self.last_generated_sample >= 5
        ):
            context = torch.zeros((1, 1), dtype=torch.long, device=self.device)
            generated_text = self.trainer.datamodule.decode(
                self.generate(context, max_new_tokens=500)[0].tolist()
            )
            self.trainer.logger._experiment.add_text(
                "Generated sample", generated_text, self.trainer.current_epoch
            )

        self.last_generated_sample = self.trainer.current_epoch
        self.last_val_loss = current_val_loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]

            logits = self.predict_step(idx_cond)

            logits = logits[:, -1, :]  # (B, C)

            probs = F.softmax(logits, dim=-1)  # (B, C)

            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx
