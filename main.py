import torch

from pathlib import Path
from lightning import Trainer, seed_everything

from modules import TinyShakeSpeare, BigramLanguageModel, GPT
from utils import ConfigParser

if __name__ == "__main__":
    parser = ConfigParser()
    config = parser.parse_args()

    seed_everything(42)

    data_module = TinyShakeSpeare(config=config)
    data_module.prepare_data()
    data_module.setup(stage=None)
    # model = BigramLanguageModel(config=config, vocab_size=data_module.vocab_size)
    model = GPT(config=config, vocab_size=data_module.vocab_size)
    # model = torch.compile(model)

    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        results_dir.mkdir()
    trainer = Trainer(
        default_root_dir=results_dir / model._get_name(),
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
        limit_train_batches=config.eval_interval,
        limit_val_batches=config.eval_interval,
    )

    trainer.fit(model, data_module)
