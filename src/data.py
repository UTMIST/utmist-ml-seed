import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split


def build_dataset(cfg):
    if cfg.name == "cifar10":
        return _build_cifar10(cfg)
    if cfg.name == "custom":
        raise NotImplementedError(
            "Custom dataset not implemented. "
            "Add your dataset logic in src/data.py:_build_custom() "
            "and update build_dataset() to dispatch to it."
        )
    raise ValueError(f"Unknown dataset: {cfg.name}")


def build_dataloaders(train_ds, val_ds, batch_size: int):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader


def _build_cifar10(cfg):
    train_transform, val_transform = _cifar10_transforms(cfg.augment)

    train_full = torchvision.datasets.CIFAR10(
        root=cfg.path, train=True, download=True, transform=train_transform
    )
    val_ds = torchvision.datasets.CIFAR10(
        root=cfg.path, train=False, download=True, transform=val_transform
    )

    train_ds, _ = random_split(train_full, [len(train_full), 0])
    return train_ds, val_ds


def _cifar10_transforms(augment: bool):
    normalize = T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])

    if augment:
        train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            normalize,
        ])
    else:
        train_transform = T.Compose([T.ToTensor(), normalize])

    val_transform = T.Compose([T.ToTensor(), normalize])
    return train_transform, val_transform
