# Extra Human-made comments
## Comments from Prof. Richard Roettger:
- To avoid the combinatorial explosion problem of tuning $\beta$ and the capacity constant $C$, I would need to do a coarse halving grid search is a faster version of a standard grid search that uses a successive halving strategy to tune hyperparameters.   

# Repository Guidelines

## Updated Hyperparameter Search Flow
- Entry point: `python DisentangledBetaVAE.py --config cv_trainer_params.json` runs a 4-corner halving search over $(\beta, C)$ ranges, parallelizing candidates across GPUs/CPUs.
- Each halving iteration evaluates the four current corners via `run_candidate_cv` (full $K$-fold CV with early stopping) and shrinks the search box toward the winning corner until both spans are within `accuracy * original_span`.
- Epoch budgets come from `halving_epoch_budgets` (min = chunk, max = cap); fallback to `epoch_chunk`/`max_epochs` when budgets are absent. Convergence uses `convergence_tolerance`/`convergence_patience` with an optional `min_epochs_before_convergence`.
- Metrics are logged to `beta_analysis.csv` for every fold/epoch chunk; the best $(\beta, C)$ is retrained once on fold 0 and checkpointed under `model_outdir`.

## Project Structure & Module Organization
- `DisentangledBetaVAE.py` hosts the PyTorch beta-VAE model, CLI, and cross-validation driver.
- `disentangledBetaVaeUtil.py` (name preserved for compatibility) contains scaling, masking, and coverage utilities.
- `cv_trainer_params.json` stores beta/C ranges, data paths, scheduler hints; copy it before editing.
- Long-running jobs emit `beta_analysis.csv`, `lock.txt`, and `trained_models/`; keep large artifacts out of git.

## Build, Test, and Development Commands
- Set up an isolated env: `python -m venv .venv && source .venv/bin/activate` followed by `python -m pip install torch pandas scikit-learn`.
- Run the halving search: `python DisentangledBetaVAE.py --config cv_trainer_params.json`.
- Monitor progress with `tail -f beta_analysis.csv`; inspect checkpoints in `trained_models/` after runs.
- For quick debugging, clone the config, shrink `k_folds`, `halving_epoch_budgets`, or narrow `beta_range`/`C_range`, then point `--config` to the clone.

## Coding Style & Naming Conventions
- Stick to 4-space indents, type hints, and concise docstrings matching `evaluate_model` and `train_one_fold`.
- Group shared helpers inside `disentangledBetaVaeUtil.py`; split out new classes when they outgrow a single file.
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
- Hyperparam search is now 2D corner-halving only: set `beta_range`, `C_range`, and `accuracy` to control shrinkage; `halving_metric` picks the scorer (fallback to `multi_mae` or `average_variance`). Budgets drive epoch chunks/caps, and convergence is gated by tolerance/patience with a minimum epoch guard.
