from __future__ import annotations

from torchvision import transforms

from cnn_foundations.contracts.config import AugmentationConfig, DatasetConfig


def build_inference_transform(dataset: DatasetConfig) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((28, 28), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize((dataset.mean,), (dataset.std,)),
        ]
    )


def build_split_transforms(
    dataset: DatasetConfig,
    augmentation: AugmentationConfig,
) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[object] = []
    if augmentation.random_crop_padding:
        train_steps.append(
            transforms.RandomCrop(28, padding=augmentation.random_crop_padding)
        )
    if augmentation.horizontal_flip:
        train_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((dataset.mean,), (dataset.std,)),
        ]
    )
    deterministic = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((dataset.mean,), (dataset.std,)),
        ]
    )
    return transforms.Compose(train_steps), deterministic

