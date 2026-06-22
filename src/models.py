import torch.nn as nn
import torchvision.models as tv_models


def build_model(cfg) -> nn.Module:
    if cfg.name == "resnet50":
        return _build_resnet50(cfg)
    if cfg.name == "yolo_v8":
        return _build_yolo(cfg)
    if cfg.name == "transformer":
        raise NotImplementedError(
            "Transformer not implemented. "
            "Add your model in src/models.py:_build_transformer() "
            "and update build_model() to dispatch to it."
        )
    raise ValueError(f"Unknown model: {cfg.name}")


def _build_resnet50(cfg) -> nn.Module:
    weights = tv_models.ResNet50_Weights.DEFAULT if cfg.pretrained else None
    model = tv_models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    return model


def _build_yolo(cfg):
    from ultralytics import YOLO

    model_variant = "yolov8n.pt" if cfg.pretrained else "yolov8n.yaml"
    return YOLO(model_variant)
