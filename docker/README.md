# docker/

Optional Docker support for shared GPU servers and reproducible environments. Not required for local development.

## Files

- `Dockerfile` — PyTorch CUDA runtime image with all dependencies.
- `docker-compose.yml` — GPU-enabled compose config with volume mount.

## Usage

```bash
# Default training
docker compose -f docker/docker-compose.yml run train

# With config overrides
docker compose -f docker/docker-compose.yml run train python src/train.py model=yolo_v8 training.epochs=20
```

## When to use

- Shared GPU servers with CUDA version mismatches
- Reproducing exact training environments
- Serving models (extend the Dockerfile for inference)

For local development, use conda or venv instead.
