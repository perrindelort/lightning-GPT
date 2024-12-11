import torch

from pathlib import Path
from lightning import Trainer, seed_everything

from modules import TinyShakeSpeare, BigramLanguageModel
from utils import ConfigParser

if __name__ == "__main__":
    parser = ConfigParser()
    config = parser.parse_args()

    seed_everything(42)

    data_module = TinyShakeSpeare(config=config)
    data_module.prepare_data()
    data_module.setup(stage=None)
    model = BigramLanguageModel(config=config, vocab_size=data_module.vocab_size)

    results_dir = Path(__file__).parent / "results"
    if not results_dir.exists():
        results_dir.mkdir()
    trainer = Trainer(
        default_root_dir=results_dir / model._get_name(),
        max_epochs=config.max_epochs,
        max_steps=config.max_steps,
    )
    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    trainer.fit(model, data_module)

    context = torch.zeros((1, 1), dtype=torch.long)
    print(data_module.decode(model.generate(context, max_new_tokens=500)[0].tolist()))
