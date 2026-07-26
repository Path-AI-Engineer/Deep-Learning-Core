from __future__ import annotations

from pytorch_tabular.contracts import ExperimentConfig
from pytorch_tabular.data import load_task_data
from pytorch_tabular.models import ClassificationMLP
from pytorch_tabular.training import fit
from pytorch_tabular.utils import seed_everything

data = load_task_data("classification")
loaders = data.loaders(24)
for optimizer in ("sgd", "adam"):
    seed_everything(42)
    model = ClassificationMLP(13, 3)
    result = fit(
        model,
        loaders.train,
        loaders.validation,
        ExperimentConfig(
            task="classification",
            optimizer=optimizer,
            learning_rate=0.02 if optimizer == "sgd" else 0.001,
            epochs=80,
        ),
    )
    print({"optimizer": optimizer, "validation_loss": result.best_validation_loss})
