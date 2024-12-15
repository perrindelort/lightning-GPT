import torch

from pathlib import Path
from lightning import Trainer, seed_everything

from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary, RichProgressBar

from modules import TinyShakeSpeare, BigramLanguageModel, GPT
from utils import ConfigParser

if __name__ == "__main__":
    parser = ConfigParser()
    config = parser.parse_args()

    # TODO : add flag in order to detect if GPU has tensor cores
    torch.set_float32_matmul_precision(config.float32_matmul_precision)
    if config.seed:
        seed_everything(config.seed)

    data_module = TinyShakeSpeare(config=config)
    data_module.prepare_data()
    data_module.setup(stage=None)

    model_class = GPT if config.model == "GPT" else BigramLanguageModel
    model = model_class(config=config, vocab_size=data_module.vocab_size)
    model_name = model._get_name()

    # model = torch.compile(model)  # Python < 3.11

    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        results_dir.mkdir()

    logs_dir = results_dir / model_name

    logger = TensorBoardLogger(
        save_dir=logs_dir,
        name="",
        log_graph=True,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            verbose=True,
        ),
        # LearningRateMonitor,
        ModelCheckpoint(
            dirpath=logger.log_dir,  # Contains the eventual sub_directories created
            filename="{epoch}-{val_loss:.3f}",
            monitor="val_loss",
            verbose=True,
        ),
        RichModelSummary(),
        RichProgressBar(),
    ]

    trainer = Trainer(
        logger=logger,
        callbacks=callbacks,
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        limit_train_batches=config.eval_interval,
        # limit_val_batches=config.eval_interval,
        inference_mode=True,
    )

    if config.load_model == "":
        trainer.fit(model, data_module)
    else:
        model_to_load = Path(__file__).parent / "checkpoints" / config.load_model
        model = model_class.load_from_checkpoint(model_to_load)
        model.eval()
        generator = model.infinite_generation(decode_fn=data_module.decode, temperature=1.0)
        print("Press any key to stop generation.\n")
        try:
            for char in generator:
                print(char, end="", flush=True)  # Print each character without newline and flush the output
        except KeyboardInterrupt:
            print("\nGeneration interrupted by user.")
