import requests
import os
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
                response = requests.get("https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
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

        self._set_encode(chars)
        self._set_decode(chars)

        if stage == "fit":
            data = torch.tensor(self.encode(text), dtype=torch.long)
            n = int(0.9 * len(data))
            train_data = data[:n]
            val_data = data[n:]

            self.train_dataset = TextDataset(
                train_data,
                block_size=self.config.block_size,
            )
            self.val_dataset = TextDataset(
                val_data,
                block_size=self.config.block_size,
            )

    def _set_encode(self, chars):
        stoi = {ch: i for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        setattr(self, "encode", encode)

    def _set_decode(self, chars):
        itos = {i: ch for i, ch in enumerate(chars)}
        decode = lambda l: "".join([itos[i] for i in l])
        setattr(self, "decode", decode)

    def encode(self):
        """
        Defined in '_set_encode' used in 'setup' method because of the prepare_data / setup hooks optimization regarding ranks and devices if the training were to be parallelized.
        """
        pass

    def decode(self, txt):
        """
        Defined in '_set_decode' used in 'setup' method because of the prepare_data / setup hooks optimization regarding ranks and devices if the training were to be parallelized.
        """
        pass

    def get_num_workers(self):
        """
        Retrieve the total number of cores, leaving it 2 free

        Returns:
            int: Number of cores that can work for the DataLoader
        """
        num_cpus = os.cpu_count()
        if num_cpus is None:
            return 0
        return max(1, num_cpus - 2)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.get_num_workers(),
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.get_num_workers(),
            persistent_workers=True,
            pin_memory=True,
        )
