"""Launches either fit or predict from given model."""

import os
import sys

import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

torch.set_float32_matmul_precision("medium")


class LightningGPTCLI(LightningCLI):
    """Custom LightningCLI for argument linking.

    Args:
        LightningCLI (LightningCLI): LightningCLI.

    """

    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        """Adds and links argument.

        Args:
            parser (LightningArgumentParser): The parser used by the LightningCLI.

        """
        parser.add_argument("--ckpt")
        parser.link_arguments("data.init_args.block_size", "model.init_args.block_size")
        parser.link_arguments(
            "data.vocab_size",
            "model.init_args.vocab_size",
            apply_on="instantiate",
        )


def run_fit() -> None:
    """Uses a LightningCLI to run the fit command of the trainer."""
    LightningGPTCLI(auto_configure_optimizers=False)


def run_predict(args: list[str]) -> None:
    """Infinitely generates text."""
    cli = LightningGPTCLI(
        auto_configure_optimizers=False,
        args=args,
        run=False,
    )
    cli.model = cli.model.__class__.load_from_checkpoint(
        cli.config["ckpt"],
        vocab_size=cli.datamodule.vocab_size
    )
    generator = cli.model.infinite_generation(
        decode_fn=cli.datamodule.decode,
        temperature=1.0,
    )
    os.system("cls" if os.name == "nt" else "clear")
    print("Press any key to stop generation.\n")
    try:
        for char in generator:
            # Print each character without newline and flush the output
            print(char, end="", flush=True)
    except KeyboardInterrupt:
        print("\nGeneration interrupted by user.")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "fit"
    if mode == "fit":
        run_fit()
    elif mode == "predict":
        args = sys.argv[2:] if len(sys.argv) > 2 else []
        run_predict(args)
    else:
        raise ValueError(f"Unknown {mode=}")
