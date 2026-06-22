# data/

Raw and processed datasets. This directory is gitignored — do not commit data files.

## Convention

```
data/
├── cifar-10-batches-py/   # auto-downloaded by torchvision
├── custom/                # your custom dataset
│   ├── train/
│   ├── val/
│   └── test/
└── raw/                   # unprocessed source data
```

## Where to store data

- **Small datasets** (CIFAR-10, MNIST): auto-downloaded by torchvision into this folder
- **Medium datasets**: download manually, place here, document the source in your experiment config
- **Large datasets** (ImageNet, COCO): store on a shared drive or cluster filesystem, symlink into this folder
- **Google Drive** (Colab): mount and symlink, or set `dataset.path` in your config to the Drive path
