# Contributing

## Getting started

1. Follow the [Prerequisites](README.md#prerequisites) and [Setup](README.md#setup) sections in the README
2. Create a new branch for your work:
   ```bash
   git checkout -b your-name/short-description
   ```

## Workflow

1. **Pick a story** — grab an unassigned issue from the [Issues](../../issues) tab
2. **Create a branch** — use `your-name/short-description` (e.g., `javier/add-vit-model`)
3. **Do the work** — commit as you go, keep commits small
4. **Open a PR** — link the story, fill out the acceptance criteria checklist
5. **Get a review** — at least one approval before merging

## Branch naming

```
your-name/short-description
```

Examples: `alice/add-data-augmentation`, `bob/fix-cifar-loader`, `carol/update-readme`

## Commit messages

Keep them short and descriptive. No strict format — just make it clear what changed.

```
add ViT model config and builder
fix learning rate scheduler bug
update README with setup instructions
```

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting. Check before pushing:

```bash
ruff check src/ tests/
```

## Running tests

```bash
pytest tests/ -v
```

## Adding models or datasets

See the README sections on [Adding a new model](README.md#adding-a-new-model) and [Adding a new dataset](README.md#adding-a-new-dataset).

## Questions?

Ask in the UTMIST Slack or open an issue — no question is too small.
