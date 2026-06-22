import os

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")


def test_config_loads():
    with initialize_config_dir(config_dir=os.path.abspath(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="config")
        assert cfg.training.batch_size == 32
        assert cfg.model.name == "resnet50"
        assert cfg.dataset.name == "cifar10"


def test_config_model_override():
    with initialize_config_dir(config_dir=os.path.abspath(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["model=yolo_v8"])
        assert cfg.model.name == "yolo_v8"


def test_config_experiment_override():
    with initialize_config_dir(config_dir=os.path.abspath(CONFIGS_DIR), version_base=None):
        cfg = compose(config_name="config", overrides=["+experiment=long_run"])
        assert cfg.training.epochs == 100
        assert cfg.training.lr == 0.0005


def test_resnet_forward_pass():
    from src.models import build_model

    cfg = OmegaConf.create({"name": "resnet50", "pretrained": False, "num_classes": 10})
    model = build_model(cfg)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 10)


def test_seed_reproducibility():
    from src.utils import set_seed

    set_seed(42)
    a = torch.randn(5)
    set_seed(42)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_device_detection():
    from src.utils import get_device

    device = get_device("cpu")
    assert device == torch.device("cpu")

    device = get_device("auto")
    assert isinstance(device, torch.device)
