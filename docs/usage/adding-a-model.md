# Adding a Model

1. `configs/model/my_model.yaml` — add a YAML with a `name:` field and any hyperparams
2. `src/models.py` — add `_build_my_model(cfg)` and register it in `build_model()`
3. Run: `python src/train.py model=my_model`
