# outputs/

Hydra auto-generated run outputs. Each training run creates a timestamped subdirectory.

## Structure

```
outputs/
└── 2026-06-21/
    └── 14-30-00/
        ├── .hydra/          # Hydra internals (config snapshot, overrides)
        ├── config.yaml      # Resolved config for this run
        ├── metrics.json     # Per-epoch train/val loss and accuracy
        ├── best_model.pt    # Best checkpoint by validation accuracy
        └── git_hash.txt     # Git commit hash at time of run
```

This directory is gitignored. For results you want to keep long-term, copy them to `results/`.
