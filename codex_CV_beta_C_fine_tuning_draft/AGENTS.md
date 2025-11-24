# Extra Human-made comments
## Comments from Prof. Richard Roettger:
- To avoid the combinatorial explosion problem of tuning $\beta$ and the capacity constant $C$, I would need to do a coarse halving grid search is a faster version of a standard grid search that uses a successive halving strategy to tune hyperparameters.   

# Repository Guidelines

## Project Structure & Module Organization
- `DisentangledBetaVAE.py` hosts the PyTorch beta-VAE model, CLI, and cross-validation driver.
- `disentanbledBetaVaeUtil.py` (name preserved for compatibility) contains scaling, masking, and coverage utilities.
- `cv_trainer_params.json` stores beta/C ranges, data paths, scheduler hints; copy it before editing.
- Long-running jobs emit `beta_analysis.csv`, `lock.txt`, and `trained_models/`; keep large artifacts out of git.

## Build, Test, and Development Commands
- Set up an isolated env: `python -m venv .venv && source .venv/bin/activate` followed by `python -m pip install torch pandas scikit-learn`.
- Run one fold: `python DisentangledBetaVAE.py 1 --config cv_trainer_params.json`; the index selects a beta/C/fold tuple.
- Use `tail -f beta_analysis.csv` to monitor metrics; inspect checkpoints in `trained_models/` after runs.
- For quick debugging, clone the config, shrink `k_folds` or `epoch_granularity`, and pass the new path to the CLI.

## Coding Style & Naming Conventions
- Stick to 4-space indents, type hints, and concise docstrings matching `evaluate_model` and `train_one_fold`.
- Group shared helpers inside `disentanbledBetaVaeUtil.py`; split out new classes when they outgrow a single file.
- Name files and functions snake_case, classes CamelCase; reuse existing config key casing when expanding JSON.
- Surface new CLI flags through `argparse` and document defaults in the tracked config file.

## Testing Guidelines
- Add `pytest` suites under `tests/`, naming modules `test_<feature>.py` and seeding torch/numpy for determinism.
- Build tiny CSV fixtures to exercise scaling/imputation flows and assert metrics such as `prop_80` or `multi_mae`.
- Run `pytest -q` before PRs and note expected stochastic variance when sharing coverage numbers.

## Commit & Pull Request Guidelines
- Write imperative commits scoped to the touched area (e.g., `Refine imputation variance estimator`).
- PR descriptions should outline behaviour shifts, list config changes, and paste key `beta_analysis.csv` rows.
- Link relevant issues, flag new dependencies, and call out artifacts that must stay outside version control.

## Data & Configuration Tips
- Replace placeholder paths in `cv_trainer_params.json` with secure locations; never commit raw datasets.
- Version alternative configs with suffixes like `cv_params_local.json` instead of editing the shared default.
- After crashes, delete stray `lock.txt` and prune old checkpoints once archived elsewhere.
- Hyperparam search now supports successive halving over beta/C ranges: set `beta_range`, `C_range`, grid point counts, and `halving_epoch_budgets` in `cv_trainer_params.json`. Omit the ranges to fall back to the legacy fixed grids.
