# tests/

Smoke tests that verify the repo's core components work without requiring GPU or large data downloads.

## Running

```bash
pytest tests/ -v
```

## What's tested

- Config loading via Hydra compose API
- Model/experiment config overrides
- ResNet50 forward pass (random input, CPU)
- Seed reproducibility
- Device detection

## Adding tests

- Keep tests fast (no training loops, no data downloads)
- Use `OmegaConf.create()` for inline configs
- Use small random tensors for forward pass tests
