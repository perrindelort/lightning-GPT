import yaml

from argparse import ArgumentParser

from pathlib import Path


class ConfigParser(ArgumentParser):
    def __init__(self, config_file=Path(__file__).parent.parent / "config.yaml", *args, **kwargs):
        super().__init__(*args, **kwargs)

        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.add_argument("--float32_matmul_precision", type=str, default=config.get("float32_matmul_precision"))

        self.add_argument("--batch_size", type=int, default=config.get("batch_size"))
        self.add_argument("--block_size", type=int, default=config.get("block_size"))
        self.add_argument("--dropout", type=int, default=config.get("dropout"))
        self.add_argument("--eval_interval", type=int, default=config.get("eval_interval"))
        self.add_argument("--head_size", type=int, default=config.get("head_size"))
        self.add_argument("--lr", type=float, default=config.get("lr"))
        self.add_argument("--max_epochs", type=int, default=config.get("max_epochs"))
        self.add_argument("--max_steps", type=int, default=config.get("max_steps"))
        self.add_argument("--n_embeddings", type=int, default=config.get("n_embeddings"))
        self.add_argument("--n_heads", type=int, default=config.get("n_heads"))
        self.add_argument("--n_layers", type=int, default=config.get("n_layers"))
