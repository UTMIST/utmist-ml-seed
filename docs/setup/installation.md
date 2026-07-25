# Setup

## Prerequisites

- **Miniconda** — [docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)
- **Git LFS** — `brew install git-lfs` / `sudo apt install git-lfs`, then `git lfs install`

## Install

```bash
git clone <your-repo-url>
cd <your-repo>

# Conda (recommended)
conda env create -f environment.yml
conda activate utmist-ml

# or venv
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Installing new packages

```bash
# Conda — install then export so teammates get it
pip install <package>
conda env export --from-history > environment.yml

# Sync from someone else's updated environment.yml
conda env update -f environment.yml --prune
```

Add permanent packages to `pyproject.toml` under the appropriate group (`ml`, `cv`, `dev`) and install with `pip install -e ".[ml]"`.

## Next

- [Commands](../usage/commands.md) — run training, evaluation, and TensorBoard
