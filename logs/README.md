# logs/

Training logs, TensorBoard event files, and W&B local logs.

## Structure

```
logs/
├── tensorboard/    # TensorBoard event files (if using TensorBoard)
└── wandb/          # W&B local sync directory (if using W&B)
```

## Viewing TensorBoard logs

```bash
tensorboard --logdir logs/tensorboard/
```

This directory is gitignored.
