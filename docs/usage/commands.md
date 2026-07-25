# Commands

```bash
# Train (default: ResNet50 on CIFAR-10)
python src/train.py

# Switch model or override any param from CLI
python src/train.py model=yolo_v8
python src/train.py training.epochs=20 training.lr=0.0005

# Apply an experiment preset
python src/train.py +experiment=long_run

# Evaluate a checkpoint
python src/evaluate.py checkpoint=outputs/<date>/<time>/best_model.pt

# Monitor with TensorBoard (in a second terminal)
tensorboard --logdir outputs/

# Lint / test
ruff check src/ tests/
pytest tests/ -v
```

Every run saves a full config snapshot, metrics, and best checkpoint to `outputs/<date>/<time>/` automatically.

See also: [project structure](project-structure.md), [configuration](configuration.md).
