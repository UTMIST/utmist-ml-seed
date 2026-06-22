# checkpoints/

Saved model weights and training checkpoints. This directory is gitignored — model files are too large for git.

## Where to store and share models

**Do not use git or git LFS for model weights.** Use one of these instead:

### Option 1: HuggingFace Hub (recommended)

Free, ML-native, version-controlled model hosting.

```python
# Upload
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/best_model.pt",
    path_in_repo="best_model.pt",
    repo_id="UTMIST/my-project",
    repo_type="model",
)

# Download
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id="UTMIST/my-project", filename="best_model.pt")
```

### Option 2: Google Drive

Simple for small teams. Upload manually or via Colab's Drive mount.

### Option 3: Weights & Biases Artifacts

If you're already using W&B for experiment tracking.

```python
import wandb
run = wandb.init()
artifact = wandb.Artifact("model", type="model")
artifact.add_file("checkpoints/best_model.pt")
run.log_artifact(artifact)
```

## Convention

```
checkpoints/
├── best_model.pt          # best by val accuracy
├── latest_model.pt        # most recent epoch
└── epoch_50.pt            # specific epoch snapshot
```
