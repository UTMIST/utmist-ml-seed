# Contributing

## Workflow

1. Pick an issue from the [Issues](../../../issues) tab — assign it to yourself
2. Create a branch off `main` following the naming convention below
3. Do the work — commit as you go
4. Open a PR — link the issue, fill out the PR template
5. Get one approval, then **squash and merge**

---

## Branch naming

```
feature/<issue-id>-<short-description>
```

- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation only
- `chore/` for maintenance (deps, config, cleanup)

The `<issue-id>` is the GitHub issue number. The description uses hyphens, lowercase, no spaces.

**Examples:**

```
feature/42-add-vit-model
fix/17-cifar-loader-crash
docs/31-update-experiment-guide
chore/8-bump-torch-version
```

---

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add ViT model config and builder
fix: correct learning rate scheduler step timing
docs: add MLflow setup instructions
chore: update torch to 2.3
test: add forward pass test for ResNet
refactor: simplify build_model dispatch
```

Format: `<type>: <short description>` — lowercase, no period, present tense.

Since we squash on merge, commit messages within your branch don't need to be perfect. The PR title becomes the squash commit message on `main`, so **make the PR title a clean conventional commit message**.

---

## PR title

Match the branch prefix and make it a conventional commit message:

```
feat: add ViT model config and builder (#42)
fix: correct learning rate scheduler step timing (#17)
docs: update experiment tracking guide (#31)
```

The issue number in parentheses is optional but helpful for traceability.

---

## Squash and merge

We use **squash and merge** — not regular merge or rebase merge.

This means:
- All your branch commits get squashed into one commit on `main`
- `main` history stays clean — one commit per PR
- You don't need to rebase or clean up commits before merging
- The PR itself preserves the full commit history

To set this as the default in GitHub: repo **Settings → General → Pull Requests** → enable "Allow squash merging", disable the others.

---

## Code style

```bash
ruff check src/ tests/     # lint
pytest tests/ -v           # tests
```

Both run in CI on every PR. Fix lint errors before requesting review.

---

## Questions?

Open an issue or ask in the UTMIST Slack.
