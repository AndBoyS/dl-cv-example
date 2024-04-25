import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dvclive import Live  # type: ignore[attr-defined]

from src import modeling
from src.scripts.params import MODEL_PATH, TrainConfig
from src.scripts.data import get_data


def main() -> None:
    train_dataset, val_dataset, test_dataset, dataset_info = get_data()

    train_loader = DataLoader(train_dataset, batch_size=TrainConfig.batch_size)
    val_loader = DataLoader(val_dataset, batch_size=TrainConfig.batch_size)

    criterion = nn.CrossEntropyLoss()
    model = modeling.Model(
        num_channels=dataset_info["num_channels"],
        num_classes=len(dataset_info["labels"]),
        criterion=criterion,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=TrainConfig.lr)

    with Live() as logger:
        logger.log_params(vars(TrainConfig()))

        trainer = modeling.Trainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            model_path=MODEL_PATH,
        )

        trainer.train(TrainConfig.num_epochs)


if __name__ == "__main__":
    main()
