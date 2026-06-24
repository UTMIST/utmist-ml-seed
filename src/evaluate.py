import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig

from src.data import build_dataloaders, build_dataset
from src.models import build_model
from src.utils import get_device, load_checkpoint, save_metrics, set_seed

log = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    set_seed(cfg.training.seed)
    device = get_device(cfg.training.device)

    model = build_model(cfg.model).to(device)

    checkpoint_path = Path(cfg.get("checkpoint", "best_model.pt"))
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, model)
        log.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        log.warning(f"No checkpoint found at {checkpoint_path}, evaluating untrained model")

    _, val_ds = build_dataset(cfg.dataset)
    _, val_loader = build_dataloaders(val_ds, val_ds, cfg.training.batch_size)

    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = _evaluate(model, val_loader, criterion, device)

    log.info(f"Evaluation — loss: {val_loss:.4f}, accuracy: {val_acc:.4f}")

    output_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    save_metrics([{"val_loss": val_loss, "val_acc": val_acc}], output_dir / "eval_metrics.json")


@torch.no_grad()
def _evaluate(model, loader, criterion, device):
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


if __name__ == "__main__":
    main()
