# VAEQL-plus Imputation

Research code for mixed-type tabular data imputation with a disentangled beta-VAE
and Q-learning-oriented reconstruction pipeline.

This repository is an active research prototype. I intend to develop it toward a
future medical-related CS/ML conference publication.

## Overview

`VAEQL_plus` explores imputation for heterogeneous tabular data with:

- type-aware preprocessing for continuous, positive-continuous, count, binary,
  categorical, and ordinal features;
- a Gaussian-mixture disentangled beta-VAE core;
- grouped reconstruction losses for expanded categorical and ordinal features;
- monotone cumulative ordinal reconstruction logits;
- research notes and baseline papers for comparison against VAE, GAN, and
  reinforcement-learning imputation methods.

## Repository Layout

```text
VAEQL_plus/                 main package
VAEQL_plus/beta_DVAE/       beta-VAE model, training wrapper, tuning notes
VAEQL_plus/step1_preprocessing/
                             feature preprocessing utilities
VAEQL_plus/step2_beta_C_tuning/
                             beta/C tuning entry points
VAEQL_plus/tests/           smoke tests and import tests
baseline_papers/            reference papers
AWS_dev_notes.md            infrastructure and data-governance notes
```

## Setup

Create the Conda environment:

```bash
conda env create -f ml_env_gpu.yml
conda activate ml_env_gpu
```

Run the smoke tests:

```bash
pytest
pytest VAEQL_plus/tests/nn_smoke_tests.py -q
pytest VAEQL_plus/tests/import_batch_mods_test.py -q
```

## Notes

- The code is research software, not a clinical decision-support tool.
- Some datasets and generated artifacts are archived for development history;
  review data-use constraints before making any derived data public.
- The current method should be described as a hybrid type-aware reconstruction
  objective, not as a fully likelihood-native HI-VAE implementation.

## License

See `LICENSE`.
