from src import utils


REPO_DIR = utils.get_repo_root()
DATASET_DIR = REPO_DIR / "data/cats_dogs/train"
CHECKPOINT_DIR = REPO_DIR / "models"
CHECKPOINT_DIR.mkdir(exist_ok=True)
MODEL_PATH = CHECKPOINT_DIR / "model.pth"


class DataConfig:
    seed = 69
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1


class TrainConfig:
    batch_size = 1024
    lr = 1e-3
    num_epochs = 2
