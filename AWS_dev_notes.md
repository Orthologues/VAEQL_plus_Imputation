# AWS Dev Notes for VAEQL_plus

Minimal cloud notes for making `VAEQL_plus` reproducible on PDS-style clinical
tabular datasets without turning the project into a full platform.

## Current Goal

Use AWS only for the parts that benefit from shared, auditable storage and
repeatable runs:

- private storage of raw and processed clinical tables;
- versioned feature metadata before preprocessing;
- beta-DVAE tuning/training outputs;
- compact evaluation reports for later manuscript work.

The modeling focus stays in the repository: hybrid type-aware reconstruction,
processed-space Q-learning actions, selected inverse transforms for clinical
interpretability, and beta/C tuning.

## Minimal S3 Layout

Use one private S3 bucket per project or environment. PDS datasets stored in S3
must never be public. Avoid adding separate buckets until access boundaries
actually require them.

```text
s3://<bucket>/<dataset_id>/
  raw/                         original input files, read-only after upload
  metadata/                    dataset manifests and feature metadata
  processed/                   preprocessed tables used by VAEQL_plus
  runs/<run_id>/config/         exact config and feature dictionary for the run
  runs/<run_id>/tuning/         beta/C search outputs such as beta_analysis.csv
  runs/<run_id>/models/         checkpoints and selected model artifacts
  runs/<run_id>/evaluation/     transformed-space metrics and small raw-scale tables
```

Minimum controls:

- block all public access at bucket/account level;
- require server-side encryption with AWS KMS (`SSE-KMS`) for every object;
- use a customer-managed KMS key for PDS raw and processed datasets;
- restrict KMS decrypt permission to the specific preprocessing/training roles
  that need it;
- keep raw data read-only after ingestion;
- do not upload PHI/PII into prompts, logs, or public artifacts;
- keep local/public repo files free of raw datasets and generated clinical data.

## Dataset Manifest

Each dataset version should have a small manifest under `metadata/`.

```json
{
  "dataset_id": "pds_example",
  "source": "PDS",
  "version": "2026-07-13",
  "raw_object": "raw/source_file.csv",
  "feature_metadata": "metadata/feature_metadata.json",
  "notes": "No raw data should be committed to git."
}
```

This is enough for traceability without building a large lineage system.

## Feature Metadata Adapter

Keep dataset-specific coding evidence separate from the compact model-facing
`FeatureTypeDict`.

Example adapter record:

```json
{
  "source_feature": "ECOGPS",
  "canonical_feature": "ecog",
  "model_type": "ordinal",
  "num_levels": 6,
  "raw_to_canonical": {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5
  },
  "missing_values": ["", "NA", "9", "99"],
  "evidence": "Dataset dictionary",
  "confidence": 0.99
}
```

Compilation target:

```json
{
  "ord_feats": {
    "ecog": 6
  }
}
```

Recommended rule:

- deterministic rules and dataset dictionaries first;
- mandatory Bedrock-hosted SLM assistance for metadata standardization,
  feature-type identification, and cohort demographics summaries;
- human review for low-confidence mappings;
- save both adapter metadata and compiled `FeatureTypeDict` with the run config.

## Mandatory Bedrock SLM Metadata Tasks

Use Amazon Bedrock as the mandatory default runtime for SLM metadata calls. The
SLM is a bounded metadata assistant, not a modeling component. It must help
standardize messy clinical metadata before preprocessing and reporting.

Runtime rule:

- Bedrock: required for metadata standardization, feature-type identification,
  and cohort demographic summaries.
- SageMaker: reserved for beta-DVAE training/tuning, or for a future self-hosted
  or fine-tuned SLM if Bedrock models are insufficient.

Mandatory SLM-assisted tasks:

1. Standardize source metadata into the adapter schema:
   - source feature name;
   - canonical feature name;
   - raw-to-canonical coding;
   - missing-value codes;
   - evidence and confidence.
2. Identify feature types for `FeatureTypeDict`:
   - continuous;
   - positive continuous;
   - count;
   - binary;
   - categorical;
   - ordinal.
3. Summarize cohort demographics from aggregate statistics only:
   - cohort size;
   - age/BMI summaries where available;
   - sex/gender distribution where allowed;
   - ECOG or other baseline-status distributions;
   - missingness overview.

SLM output should remain metadata, for example:

```json
{
  "source_feature": "ECOGPS",
  "canonical_feature": "ecog",
  "suggested_model_type": "ordinal",
  "num_levels": 6,
  "missing_values": ["", "NA", "9", "99"],
  "confidence": 0.99,
  "evidence": "Dataset dictionary says ECOG performance status ranges 0-5",
  "needs_human_review": false
}
```

The code must validate SLM output before compiling it into `FeatureTypeDict`.
Do not send raw PHI/PII rows to the SLM; use data dictionaries, column summaries,
value counts, missingness summaries, and aggregate cohort statistics.

## Training And Tuning Artifacts

For beta-DVAE runs, save:

- the exact config used for preprocessing and beta/C tuning;
- the compiled `FeatureTypeDict`;
- `beta_analysis.csv`;
- selected checkpoint(s);
- random seeds and package/environment notes.

This matches the current tuning notes: beta/C search is the important expensive
step, so the AWS path should make it reproducible rather than more complex.

## Evaluation Outputs

Primary metrics should stay in the transformed modeling space:

| Family | Metric |
| --- | --- |
| continuous | transformed MAE/RMSE |
| positive continuous | transformed MAE/RMSE |
| count | log1p-space MAE/RMSE |
| binary | AUROC, F1 |
| categorical | macro-F1 |
| ordinal | ordinal MAE / weighted kappa |

For manuscript interpretability, add a small secondary table only for selected
clinically important variables, such as age, BMI, ECOG, tumor burden, baseline
labs, selected blood counts, and selected treatment-history counts.

This keeps inverse transformation scoped: use it for demonstration and selected
raw-scale checks, not as the default metric path for every feature.

## Immediate Backlog

1. Define the S3 prefix convention above in config.
2. Add a small dataset manifest writer/reader.
3. Add adapter metadata validation before compiling `FeatureTypeDict`.
4. Save run config, compiled feature dictionary, and `beta_analysis.csv` together.
5. Add the two-tier evaluation report:
   - primary transformed-space metrics;
   - optional selected-variable raw-scale table.

## Non-Goals For Now

- Multi-account AWS architecture.
- Full Bedrock labeling platform.
- Automatic clinical interpretation.
- Cohort segmentation and cluster summarization.
- Raw-scale metrics for every feature.
- Fully likelihood-native HI-VAE implementation.
