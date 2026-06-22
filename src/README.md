# src/

Core source code. Flat by design — no deep directory nesting.

## Files

- `train.py` — Training entry point. Uses `@hydra.main` to load config and run training. Handles both standard PyTorch training loops (ResNet) and ultralytics delegation (YOLO).
- `evaluate.py` — Standalone evaluation from a saved checkpoint.
- `models.py` — Model registry. `build_model(cfg)` dispatches to per-model builder functions.
- `data.py` — Dataset and dataloader construction. `build_dataset(cfg)` dispatches to per-dataset builders.
- `utils.py` — Shared utilities: seed management, device detection, git hash saving, checkpoint I/O.

## How to add code

- New model: add `_build_<name>(cfg)` in `models.py`, register in `build_model()`
- New dataset: add `_build_<name>(cfg)` in `data.py`, register in `build_dataset()`
- New utility: add to `utils.py`
- Only create new files if a component genuinely needs its own module
