# Extra Human-made comments
## Comments from Prof. Richard Roettger:
- To avoid the combinatorial explosion problem of tuning $\beta$ and the capacity constant $C$, I would need to do a coarse halving grid search is a faster version of a standard grid search that uses a successive halving strategy to tune hyperparameters.   

# Repository Guidelines

## Updated Hyperparameter Search Flow
- Entry point: `python -m VAEQL_plus.step2_beta_C_tuning.fine_tuner` runs a 4-corner halving search over $(\beta, C)$ ranges, parallelizing candidates across GPUs/CPUs. The default CV-config path is resolved through `VAEQL_plus.conf.config.DisentangledBetaVaeTuningConfig`.
- Each halving iteration evaluates the four current corners via `run_candidate_cv` (full $K$-fold CV with early stopping) and shrinks the search box toward the winning corner until both spans are within `granularity * original_span`.
- Epoch budgets come from `halving_epoch_budgets=[start_val, max_val, step_size]`: the first validation happens after `start_val` epochs, later validations advance by `step_size`, and training stops at `max_val`. Fallback remains `epoch_chunk`/`max_epochs` when budgets are absent. Convergence uses `convergence_tolerance` (finite and at least `1e-4`; default `1e-3`) and `convergence_patience`.
- Metrics are logged to `beta_analysis.csv` for every fold/epoch chunk; the selected $(\beta, C)$ and CV score are written to `best_beta_C_summary.json`. Step 2 does not refit or export a model checkpoint; downstream Step 3 trains its own model using the selected parameters.

## Project Structure & Module Organization
- `torch_nn.py` hosts the PyTorch $\beta$-VAE neural-network module.
- `lightning_mod.py` hosts the Lightning training wrapper and $\beta$/C tuning helpers.
- `utils.py` hosts shared masking helpers and coverage metrics.
- The shared $\beta$/C tuning defaults and default CV-config path are exposed through `VAEQL_plus.conf.config.DisentangledBetaVaeTuningConfig`; keep edits consistent with that contract.
- Long-running jobs emit `beta_analysis.csv`, `lock.txt`, and the best-parameter summary; keep generated artifacts out of git.

## Build, Test, and Development Commands
- Set up an isolated env: `python -m venv .venv && source .venv/bin/activate` followed by `python -m pip install torch pandas scikit-learn`.
- Run the halving search: `python -m VAEQL_plus.step2_beta_C_tuning.fine_tuner`, or pass `--cv_config` explicitly when overriding the config resolved by `DisentangledBetaVaeTuningConfig`.
- Monitor progress with `tail -f beta_analysis.csv`; inspect `best_beta_C_summary.json` for the selected parameters and score.
- For quick debugging, clone the config, keep `k_folds` at 5, shorten `halving_epoch_budgets`, or narrow `beta_range`/`C_range`, then point `--config` to the clone.

## Coding Style & Naming Conventions
- Stick to 4-space indents, type hints, and concise docstrings matching `evaluate_model` and `train_one_fold`.
- Keep neural-network code in `torch_nn.py`, Lightning/training orchestration in `lightning_mod.py`, and reusable evaluation/masking helpers in `utils.py`.
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
- Replace placeholder paths in the CV config resolved by `DisentangledBetaVaeTuningConfig` with secure locations; never commit raw datasets.
- Version alternative configs with suffixes like `cv_params_local.json` instead of editing the shared default.
- After crashes, delete stray `lock.txt` and prune old checkpoints once archived elsewhere.
- Hyperparameter search uses 2D corner-halving: set `beta_range`, `C_range`, and `granularity` to control shrinkage. Candidate selection uses the weighted composite validation metric on type-3 cells; optional `halving_metric_weights` override the defaults in `BetaGausMixedDVAETrainer`. `halving_epoch_budgets=[start_val, max_val, step_size]` controls the initial evaluation point, hard epoch cap, and re-evaluation cadence; convergence is gated by tolerance and patience.
