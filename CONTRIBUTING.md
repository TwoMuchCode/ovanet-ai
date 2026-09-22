# Contributing

## Branches
- `main` is always runnable.
- Work on short-lived branches: `madhumita/<topic>` or `sahina/<topic>`.
- Open a PR into `main`; the other member reviews (~5 min: does it run, do tests pass, does it touch only its own folders).
- Merge at least once per day. No long-lived branches.

## Commits
Small, frequent, prefixed: `feat(vision): ...`, `fix(...)`, `docs`, `test`, `exp`.

## Ownership (no overlap = no conflicts)
- Madhumita: `ovanet/data/`, `ovanet/preprocessing.py`, `ovanet/radiomics.py`, `ovanet/vision/`, `ovanet/graph.py`, `notebooks/`, `experiments/vision/`
- Sahina: `ovanet/contracts.py`, `ovanet/mocks.py`, `ovanet/gnn.py`, `ovanet/fusion/`, `ovanet/evaluation/`, `ovanet/longitudinal.py`, `ovanet/report.py`, `ovanet/pipeline.py`, `app/`, `docs/`, `experiments/results/`

Never commit raw data, checkpoints, `.env`, or huge notebook outputs.
Shared files (`requirements.txt`, `docs/CONTRACTS.md`, `docs/DECISIONS.md`) change only via a small PR, announced to the partner.

## Hand-off ritual (end of every day)
TEST → COMMIT → PUSH → PR/merge → update your own `docs/status/<name>.md` → message partner: "Day N: done ___, need ___, blocked on ___".