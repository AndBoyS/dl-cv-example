from typing import Any

from torchvision import transforms  # type: ignore[import-untyped]
import torchvision
import torch
from torch.utils.data import random_split, Subset

from src.scripts.params import DATASET_DIR, DataConfig


def get_data() -> tuple[Subset, Subset, Subset, dict[str, Any]]:  # type: ignore[type-arg]
    IMAGE_SIZE = (128, 128)
    dataset_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(IMAGE_SIZE),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(str(DATASET_DIR), transform=dataset_transform)
    labels = dataset.classes

    torch.random.manual_seed(DataConfig.seed)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [DataConfig.train_split, DataConfig.val_split, DataConfig.test_split]
    )

    dataset_info = {
        "num_channels": 3,
        "labels": labels,
    }

    return train_dataset, val_dataset, test_dataset, dataset_info
