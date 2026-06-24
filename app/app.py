import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import torch
import torchvision.transforms as T
from PIL import Image

from src.models import build_model
from src.utils import get_device, load_checkpoint

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

CHECKPOINT_PATH = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pt"

device = get_device("auto")
model = None


def _load_model(checkpoint_path: Path) -> torch.nn.Module:
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"name": "resnet50", "pretrained": False, "num_classes": 10})
    m = build_model(cfg).to(device)
    if checkpoint_path.exists():
        load_checkpoint(checkpoint_path, m)
    return m


def _get_model() -> torch.nn.Module:
    global model
    if model is None:
        model = _load_model(CHECKPOINT_PATH)
        model.eval()
    return model


transform = T.Compose([
    T.Resize((32, 32)),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
])


@torch.no_grad()
def predict(image: Image.Image) -> dict[str, float]:
    if image is None:
        return {}
    img = transform(image.convert("RGB")).unsqueeze(0).to(device)
    logits = _get_model()(img)
    probs = torch.softmax(logits, dim=1)[0]
    return {CIFAR10_CLASSES[i]: probs[i].item() for i in range(len(CIFAR10_CLASSES))}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Label(num_top_classes=5, label="Predictions"),
    title="CIFAR-10 Classifier",
    description="Upload an image to classify it into one of 10 CIFAR-10 categories.",
    examples=[],
)

if __name__ == "__main__":
    demo.launch()
