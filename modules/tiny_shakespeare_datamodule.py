import requests
import torch

from pathlib import Path

from lightning import LightningDataModule
from argparse import Namespace
from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get a sequence of length block_size starting from index idx
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class TinyShakeSpeare(LightningDataModule):
    def __init__(self, config: Namespace):
        super().__init__()
        self.data_dir = Path(__file__).parent.parent / "data"
        if not self.data_dir.exists():
            self.data_dir.mkdir()
        self.config = config

    def prepare_data(self):
        file = self.data_dir / "tiny_shakespeare.txt"
        if not file.exists():
            try:
                response = requests.get(
                    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                )
                response.raise_for_status()  # Ensure the request was successful
                with open(file, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"File saved to {self.data_dir}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading the file: {e}")

    def setup(self, stage: str):
        with open(self.data_dir / "tiny_shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for i, ch in enumerate(chars)}

        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: "".join([itos[i] for i in l])
        setattr(self, "encode", encode)
        setattr(self, "decode", decode)

        if stage == "fit":
            data = torch.tensor(encode(text), dtype=torch.long)
            n = int(0.9 * len(data))
            train_data = data[:n]
            val_data = data[n:]

            self.train_dataset = TextDataset(
                train_data, block_size=self.config.block_size
            )
            self.val_dataset = TextDataset(val_data, block_size=self.config.block_size)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.config.batch_size)
