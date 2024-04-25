import torch
from torch.utils.data import DataLoader
from dvclive import Live  # type: ignore[attr-defined]

from src import modeling
from src.scripts.params import MODEL_PATH
from src.scripts.data import get_data


def main() -> None:
    _, _, test_dataset, _ = get_data()
    test_loader = DataLoader(test_dataset, batch_size=1)

    with Live() as logger:
        model: modeling.Model = torch.load(MODEL_PATH)
        acc = modeling.evaluate_loader(test_loader, model)["acc"]
        logger.log_metric("test_acc", acc)


if __name__ == "__main__":
    main()
