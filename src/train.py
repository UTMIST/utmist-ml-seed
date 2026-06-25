import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from src.data import build_dataloaders, build_dataset
from src.models import build_model
from src.utils import get_device, save_checkpoint, save_git_hash, save_metrics, set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)
    log.info(f"Using device: {device}")

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_git_hash(output_dir)

    if cfg.model.name == "yolo_v8":
        _train_yolo(cfg, output_dir)
        return

    writer = None
    if cfg.logging.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))
        log.info(f"TensorBoard logging to {output_dir / 'tensorboard'}")

    model = build_model(cfg.model).to(device)
    train_ds, val_ds = build_dataset(cfg.dataset)
    train_loader, val_loader = build_dataloaders(train_ds, val_ds, cfg.training.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    all_metrics = []

    for epoch in range(1, cfg.training.epochs + 1):
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion, device, cfg)
        val_loss, val_acc = _validate(model, val_loader, criterion, device)

        log.info(
            f"Epoch {epoch}/{cfg.training.epochs} — "
            f"train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, "
            f"val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}"
        )

        all_metrics.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if writer:
            writer.add_scalars("loss", {"train": train_loss, "val": val_loss}, epoch)
            writer.add_scalars("accuracy", {"train": train_acc, "val": val_acc}, epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if cfg.logging.save_checkpoint and val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, output_dir / "best_model.pt")
            log.info(f"Saved best model (val_acc={val_acc:.4f})")

    if writer:
        writer.close()

    save_metrics(all_metrics, output_dir / "metrics.json")
    log.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")


def _train_epoch(model, loader, optimizer, criterion, device, cfg):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)

        if (batch_idx + 1) % cfg.logging.log_interval == 0:
            pbar.set_postfix(loss=total_loss / total, acc=correct / total)

    return total_loss / total, correct / total


@torch.no_grad()
def _validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        correct += outputs.argmax(dim=1).eq(targets).sum().item()
        total += inputs.size(0)

    return total_loss / total, correct / total


def _train_yolo(cfg, output_dir):
    from src.models import _build_yolo

    model = _build_yolo(cfg.model)
    model.train(
        epochs=cfg.training.epochs,
        imgsz=cfg.model.img_size,
        batch=cfg.training.batch_size,
        project=str(output_dir),
        name="yolo_run",
    )


if __name__ == "__main__":
    main()
