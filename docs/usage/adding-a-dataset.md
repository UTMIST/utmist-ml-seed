# Adding a Dataset

1. `configs/dataset/my_dataset.yaml` — add a YAML with a `name:` field and `path:`
2. `src/data.py` — add `_build_my_dataset(cfg)` returning `(train_ds, val_ds)` and register it in `build_dataset()`
3. Run: `python src/train.py dataset=my_dataset`

See also: [data formats](../data/formats.md) for dataset formats and the Parquet workflow.
