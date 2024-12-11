import yaml

from argparse import ArgumentParser

from pathlib import Path


class ConfigParser(ArgumentParser):
    def __init__(
        self, config_file=Path(__file__).parent.parent / "config.yaml", *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        self.add_argument("--batch_size", type=int, default=config.get("batch_size"))
        self.add_argument("--block_size", type=int, default=config.get("block_size"))
        self.add_argument("--dropout", type=int, default=config.get("dropout"))
        self.add_argument(
            "--eval_interval", type=int, default=config.get("eval_interval")
        )
        self.add_argument("--eval_iters", type=int, default=config.get("eval_iters"))
        self.add_argument("--lr", type=float, default=config.get("lr"))
        self.add_argument("--max_epochs", type=int, default=config.get("max_epochs"))
        self.add_argument("--max_steps", type=int, default=config.get("max_steps"))
        self.add_argument("--n_embd", type=int, default=config.get("n_embd"))
        self.add_argument("--n_head", type=int, default=config.get("n_head"))
        self.add_argument("--n_layer", type=int, default=config.get("n_layer"))
