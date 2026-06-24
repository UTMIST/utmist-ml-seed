# Gradio Demo

A minimal web interface for running inference on trained models.

## Setup

Install the demo dependency:

```bash
pip install -e ".[demo]"
```

## Usage

1. Train a model first:
   ```bash
   python src/train.py
   ```

2. Copy the best checkpoint:
   ```bash
   cp outputs/<date>/<time>/best_model.pt checkpoints/best_model.pt
   ```

3. Launch the demo:
   ```bash
   python app/app.py
   ```

4. Open the URL printed in your terminal (usually `http://127.0.0.1:7860`).

## Customizing

`app.py` is a starting point. Common modifications:

- **Different model**: change the `OmegaConf.create(...)` config in `_load_model()`
- **Different classes**: update `CIFAR10_CLASSES` to match your dataset
- **Different input type**: swap `gr.Image` for `gr.Textbox`, `gr.Audio`, etc.
- **Multiple models**: add tabs with `gr.TabbedInterface`

See the [Gradio docs](https://www.gradio.app/docs) for more components and layouts.
