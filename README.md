# VAEQL-plus Imputation

Research code for mixed-type tabular data imputation with a disentangled $\beta$-VAE
and Q-learning-oriented reconstruction pipeline.

This repository is an active research prototype. I intend to develop it toward a
future medical-related CS/ML conference publication.

## Overview

`VAEQL_plus` explores imputation for heterogeneous tabular data with:

- type-aware preprocessing for continuous, positive-continuous, count, binary,
  categorical, and ordinal features;
- a Gaussian-mixture disentangled $\beta$-VAE core;
- grouped reconstruction losses for expanded categorical and ordinal features;
- monotone cumulative ordinal reconstruction logits;
- a preliminary two-phase AWS Batch workflow: Ministral 8B annotation on GPU,
  encrypted S3 structured-dataset handoff, and GPU DRL training with the final
  model and evaluation statistics saved to S3;
- research notes and baseline papers for comparison against VAE, GAN, and
  reinforcement-learning imputation methods.

## Repository Layout

```text
VAEQL_plus/                         main package
VAEQL_plus/beta_DVAE/               $\beta$-VAE model, training wrapper, and notes
VAEQL_plus/conf/                    model parameters, feature types, and Nextflow profiles
VAEQL_plus/conf/nextflow_conf.nf    preliminary Step 0-to-Step 3 workflow
VAEQL_plus/step0_SLM_metadata_profiling/
                                     Ministral 8B feature-type profiling scaffold
VAEQL_plus/step1_preprocessing/     feature preprocessing utilities and specification
VAEQL_plus/step2_beta_C_tuning/     $\beta$/C tuning entry points and pseudo-algorithm
VAEQL_plus/util/                    AWS Batch/S3 interfaces and shared utilities
VAEQL_plus/tests/                   AWS, Step 0, import, and neural-network tests
baseline_papers/                    reference papers
Codex_drafts/                       development prompts and research drafts
drafts_private/                     ignored private research plans
AWS_dev_notes.md                    infrastructure and data-governance notes
VAEQL_plus/research_todos.md        active research roadmap
pytest.ini                          test discovery and pytest configuration
ml_env_gpu.yml                      Conda environment specification
```

The current Nextflow scaffold calls Step 3 through a configured command
adapter; a dedicated `VAEQL_plus/step3_*` package has not yet been added.

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
```

## Notes

- The code is research software, not a clinical decision-support tool.
- Some datasets and generated artifacts are archived for development history;
  review data-use constraints before making any derived data public.
- The current method should be described as a hybrid type-aware reconstruction
  objective, not as a fully likelihood-native HI-VAE implementation.
- Reference Link for pipeline integration of AWS Batch, AWS S3 and Nextflow: <a>https://aws.amazon.com/blogs/hpc/optimize-nextflow-workflows-on-aws-batch-with-mountpoint-for-amazon-s3/</a>
- Reference Link for setting up AWS Batch HPC environment: <a>https://docs.aws.amazon.com/batch/latest/userguide/get-set-up-for-aws-batch.html</a>
- Installation guide of Nextflow at an on-premise server: <a>https://docs.seqera.io/nextflow/install</a>

## License

See `LICENSE`.
