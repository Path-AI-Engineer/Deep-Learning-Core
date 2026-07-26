from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from pytorch_tabular.contracts import FeatureSpec, TaskName
from pytorch_tabular.utils import seed_everything


@dataclass(frozen=True)
class TaskLoaders:
    train: DataLoader
    validation: DataLoader
    test: DataLoader


@dataclass
class PreparedTaskData:
    task: TaskName
    dataset_name: str
    feature_names: list[str]
    feature_specs: list[FeatureSpec]
    class_names: list[str]
    target_unit: str | None
    scaler: StandardScaler
    x_train: np.ndarray
    x_validation: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_validation: np.ndarray
    y_test: np.ndarray
    seed: int
    source: str = "scikit-learn"

    def loaders(self, batch_size: int) -> TaskLoaders:
        generator = torch.Generator().manual_seed(self.seed)
        target_dtype = torch.float32 if self.task == "regression" else torch.long

        def dataset(features: np.ndarray, target: np.ndarray) -> TensorDataset:
            return TensorDataset(
                torch.as_tensor(features, dtype=torch.float32),
                torch.as_tensor(target, dtype=target_dtype),
            )

        return TaskLoaders(
            train=DataLoader(
                dataset(self.x_train, self.y_train),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            ),
            validation=DataLoader(
                dataset(self.x_validation, self.y_validation),
                batch_size=batch_size,
                shuffle=False,
            ),
            test=DataLoader(
                dataset(self.x_test, self.y_test),
                batch_size=batch_size,
                shuffle=False,
            ),
        )

    def split_summary(self) -> dict[str, int]:
        return {
            "train": len(self.x_train),
            "validation": len(self.x_validation),
            "test": len(self.x_test),
        }


def _specs(names: list[str], values: np.ndarray) -> list[FeatureSpec]:
    return [
        FeatureSpec(
            name=name,
            display_name=name.replace("_", " ").title(),
            description=f"Observed numeric feature: {name}.",
            minimum=float(np.min(values[:, index])),
            maximum=float(np.max(values[:, index])),
            example=float(np.median(values[:, index])),
        )
        for index, name in enumerate(names)
    ]


def _load_california_housing() -> tuple[np.ndarray, np.ndarray, list[str], str]:
    project_root = Path(__file__).resolve().parents[3]
    local_csv = project_root / "data" / "raw" / "california_housing.csv"
    reference_csv = (
        project_root / "data" / "samples" / "california_housing_reference_sample.csv"
    )
    if local_csv.is_file():
        frame = pd.read_csv(local_csv)
        source = "local full California Housing CSV"
    elif reference_csv.is_file():
        frame = pd.read_csv(reference_csv)
        source = "bundled official-source California Housing reference sample"
    else:
        raw = fetch_california_housing(
            data_home=project_root / ".runtime" / "scikit-learn-data"
        )
        return (
            np.asarray(raw.data, dtype=np.float32),
            np.asarray(raw.target, dtype=np.float32),
            list(raw.feature_names),
            "scikit-learn fetch_california_housing",
        )

    expected = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "MedHouseVal",
    ]
    if frame.columns.tolist() != expected:
        raise ValueError(
            "California Housing CSV must use the official eight features followed "
            "by MedHouseVal."
        )
    return (
        frame[expected[:-1]].to_numpy(dtype=np.float32),
        frame["MedHouseVal"].to_numpy(dtype=np.float32),
        expected[:-1],
        source,
    )


def load_task_data(task: TaskName, seed: int = 42) -> PreparedTaskData:
    seed_everything(seed)
    if task == "regression":
        features, target, names, source = _load_california_housing()
        dataset_name = "California Housing"
        classes: list[str] = []
        unit = "USD 100,000"
        stratify = None
    else:
        raw = load_wine()
        features = np.asarray(raw.data, dtype=np.float32)
        target = np.asarray(raw.target, dtype=np.int64)
        names = list(raw.feature_names)
        dataset_name = "Wine"
        classes = [f"class_{index}" for index in range(len(raw.target_names))]
        unit = None
        stratify = target
        source = "scikit-learn load_wine"

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=seed,
        stratify=stratify,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=0.50,
        random_state=seed,
        stratify=y_holdout if task == "classification" else None,
    )
    scaler = StandardScaler().fit(x_train)
    return PreparedTaskData(
        task=task,
        dataset_name=dataset_name,
        feature_names=names,
        feature_specs=_specs(names, features),
        class_names=classes,
        target_unit=unit,
        scaler=scaler,
        x_train=scaler.transform(x_train).astype(np.float32),
        x_validation=scaler.transform(x_validation).astype(np.float32),
        x_test=scaler.transform(x_test).astype(np.float32),
        y_train=y_train,
        y_validation=y_validation,
        y_test=y_test,
        seed=seed,
        source=source,
    )
