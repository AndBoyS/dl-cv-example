from pathlib import Path
from typing import Protocol, Self

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.notebook import tqdm


class LocalModelType(Protocol):
    device: torch.device

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

    def compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor: ...

    def predict(self, x: torch.Tensor) -> torch.Tensor: ...

    def train(self, mode: bool = True) -> Self: ...

    def eval(self) -> Self: ...

    def to(self, device: torch.device) -> Self: ...


class Model(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        criterion: torch.nn.modules.loss._Loss,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(num_channels, 12, 3),
            nn.GELU(),
            nn.BatchNorm2d(12),
            nn.MaxPool2d(4),
            nn.Conv2d(12, 20, 3),
            nn.GELU(),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2),
            nn.Conv2d(20, 40, 3),
            nn.GELU(),
            nn.BatchNorm2d(40),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1440, 500),
            nn.GELU(),
            nn.Linear(500, num_classes),
        )
        self.device = device
        self.criterion = criterion

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[reportRedeclaration]
        x = self.body(x)
        x: torch.Tensor = self.fc(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        pred = logits.argmax(dim=1)
        return pred

    def compute_loss(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.forward(x)
        loss: torch.Tensor = self.criterion(input=logits, target=y)
        return loss


class Trainer:
    def __init__(
        self,
        model: LocalModelType,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,  # type: ignore[type-arg]
        val_loader: DataLoader | None = None,  # type: ignore[type-arg]
        writer: SummaryWriter | None = None,
        model_path: str | Path | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.model = model

        self.optimizer = optimizer

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = writer
        self.model_path = model_path

        self.device = device
        self.model = self.model.to(self.device)

    def train(self, num_epochs: int) -> None:
        batch_step = 0
        best_val_acc = 0.0

        for epoch in tqdm(range(num_epochs), desc="Training:"):
            epoch_losses: list[float] = []

            self.model.train()
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                self.optimizer.zero_grad()
                loss = self.model.compute_loss(batch)
                loss.backward()
                self.optimizer.step()

                epoch_losses.append(loss.item())
                if self.writer is not None:
                    self.writer.add_scalar("train_loss_batch", loss, batch_step)

                batch_step += 1

            if self.writer is None:
                continue

            epoch_loss = np.mean(epoch_losses)
            self.writer.add_scalar("train_loss", epoch_loss, epoch)

            self.model.eval()
            train_acc = evaluate_loader(self.train_loader, self.model)["acc"]
            self.writer.add_scalar("train_acc", train_acc, epoch)

            if self.val_loader is None:
                continue

            val_acc = evaluate_loader(self.val_loader, self.model)["acc"]
            self.writer.add_scalar("val_acc", val_acc, epoch)

            if val_acc > best_val_acc and self.model_path is not None:
                best_val_acc = val_acc
                torch.save(self.model, self.model_path)


@torch.no_grad()
def evaluate_loader(
    loader: DataLoader,  # type: ignore[type-arg]
    model: LocalModelType,
) -> dict[str, float]:
    model.eval()

    pred_batches: list[torch.Tensor] = []
    target_batches: list[torch.Tensor] = []

    for x, y in tqdm(loader, desc="Computing metrics"):
        x = x.to(model.device)
        pred = model.predict(x)

        pred_batches.append(pred.cpu())
        target_batches.append(y)

    target = torch.concat(target_batches)
    pred = torch.concat(pred_batches)

    acc = (pred == target).float().mean().item()

    return {
        "acc": acc,
    }
