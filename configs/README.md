# configs/

Hydra configuration files. Everything is composed from `config.yaml`.

## Structure

- `config.yaml` — Base config and Hydra entry point. Sets defaults for model, dataset, and training.
- `model/` — One YAML per model architecture. Selected via `model=<name>` on the CLI.
- `dataset/` — One YAML per dataset. Selected via `dataset=<name>` on the CLI.
- `experiment/` — Override presets that change training params. Applied via `+experiment=<name>`.

## How composition works

```
config.yaml (base defaults)
  ├── model/resnet50.yaml (or yolo_v8, transformer)
  ├── dataset/cifar10.yaml (or custom)
  └── experiment/long_run.yaml (optional override)
  └── CLI args (highest priority)
```

## Adding a new config

Copy an existing YAML in the relevant folder and modify it. The `name` field must match what `build_model()` or `build_dataset()` dispatches on in `src/`.
