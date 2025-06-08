"""Defines the Data Module containing Shakespeare's body of work."""
import os
from pathlib import Path

import requests
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class TextDataset(Dataset):
    """A PyTorch Dataset that creates input-target pairs from a tensor of text data.

    Each item is a tuple (x, y) where:
        - x is a block of `block_size` consecutive tokens.
        - y is the same block shifted by one character into the future.

    Args:
        data (torch.Tensor): The tokenized input data as a 1D tensor.
        block_size (int): The length of each input sequence.

    """

    def __init__(self, data: torch.Tensor, block_size: int) -> None:
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns a single training example (x, y).

        Args:
            idx (int): Index to start the sequence.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A pair where `x` is the input sequence
                                               and `y` is the target sequence (shifted by one).

        """  # noqa: E501
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


class TinyShakeSpeare(LightningDataModule):
    """PyTorch LightningDataModule for training on the Tiny Shakespeare dataset.

    Handles downloading, preprocessing, dataset creation, and dataloader construction.

    Args:
        data_dir (str | Path): Directory to save/download the dataset.
        block_size (int): Length of each input sequence.
        batch_size (int): Batch size for training and validation.

    """

    def __init__(self, data_dir: str | Path, block_size: int, batch_size: int) -> None:
        super().__init__()
        self.data_dir = data_dir if isinstance(data_dir, Path) else Path(data_dir)
        if not self.data_dir.exists():
            self.data_dir.mkdir()
        self.block_size = block_size
        self.batch_size = batch_size

        self.prepare_data()
        self.setup()

    def prepare_data(self) -> None:
        """Downloads the Tiny Shakespeare dataset if it's not already present."""
        file = self.data_dir / "tiny_shakespeare.txt"
        if not file.exists():
            try:
                response = requests.get(
                    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
                )
                response.raise_for_status()
                with open(file, "w", encoding="utf-8") as f:
                    f.write(response.text)
                print(f"File saved to {self.data_dir}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading the file: {e}")

    def setup(self, stage: str = "fit") -> None:
        """Prepares training and validation datasets and initializes vocabulary encoding/decoding.

        Args:
            stage (str): The stage of training (e.g., 'fit', 'test'). Defaults to 'fit'.

        """  # noqa: E501
        with open(self.data_dir / "tiny_shakespeare.txt", "r", encoding="utf-8") as f:
            text = f.read()
        chars = sorted(set(text))
        self.vocab_size = len(chars)

        self._set_encode(chars)
        self._set_decode(chars)

        if stage == "fit":
            data = torch.tensor(self.encode(text), dtype=torch.long)
            n = int(0.9 * len(data))
            train_data = data[:n]
            val_data = data[n:]

            self.train_dataset = TextDataset(train_data, block_size=self.block_size)
            self.val_dataset = TextDataset(val_data, block_size=self.block_size)

    def _set_encode(self, chars) -> None:
        """Defines the character-to-index encoding function and stores it as `self.encode`.

        Args:
            chars (list[str]): List of unique characters in the dataset.

        """  # noqa: E501
        stoi = {ch: i for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s]
        setattr(self, "encode", encode)

    def _set_decode(self, chars) -> None:
        """Defines the index-to-character decoding function and stores it as `self.decode`.

        Args:
            chars (list[str]): List of unique characters in the dataset.

        """  # noqa: E501
        itos = {i: ch for i, ch in enumerate(chars)}
        decode = lambda l: "".join([itos[i] for i in l])
        setattr(self, "decode", decode)

    def encode(self, txt):
        """Placeholder for encode function, defined in `_set_encode`.

        Args:
            txt (str): Input text to be encoded.

        Returns:
            list[int]: Encoded list of token indices.

        """
        pass

    def decode(self, txt):
        """Placeholder for decode function, defined in `_set_decode`.

        Args:
            txt (list[int]): List of token indices.

        Returns:
            str: Decoded text string.

        """
        pass

    def get_num_workers(self) -> int:
        """Calculate the number of worker processes to use for data loading.

        Returns:
            int: Number of usable CPU cores, leaving 2 free for system processes.

        """
        num_cpus = os.cpu_count()
        if num_cpus is None:
            return 0
        return max(1, num_cpus - 2)

    def train_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the training dataset.

        Returns:
            DataLoader: DataLoader for training.

        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Returns the DataLoader for the validation dataset.

        Returns:
            DataLoader: DataLoader for validation.

        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2,
            persistent_workers=True,
            pin_memory=True,
        )
