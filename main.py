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
    torch.set_float32_matmul_precision(config.float32_matmul_precision)
    seed_everything(1337)

    data_module = TinyShakeSpeare(config=config)
    data_module.prepare_data()
    data_module.setup(stage=None)
    # model = BigramLanguageModel(config=config, vocab_size=data_module.vocab_size)
    model = GPT(config=config, vocab_size=data_module.vocab_size)
    # model = torch.compile(model) # Python < 3.12

    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        results_dir.mkdir()

    logs_dir = results_dir / model._get_name()

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
    )

    trainer.fit(model, data_module)
