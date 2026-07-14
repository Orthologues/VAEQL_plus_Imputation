# AWS Dev Notes for `VAEQL_plus`

Use AWS for private PDS dataset storage, metadata preparation, beta-DVAE
training, and reproducible evaluation outputs. Keep the modeling code and
research decisions in this repository.

## Data Storage and Security

- Store raw data, processed data, metadata, model artifacts, and evaluation
  outputs in a private S3 bucket.
- PDS datasets stored in S3 shall not be public unless explicitly required by
  reviewer.
- Block public access by default.
- Encrypt all S3 objects with AWS KMS (`SSE-KMS`). Use a customer-managed KMS
  key for raw and processed PDS data.
- Restrict S3 access and KMS decryption to the roles used for preprocessing,
  training, and evaluation.
- Keep raw data read-only after ingestion.
- Do not send PHI/PII to unapproved model endpoints, application logs, or
  public artifacts.

### PHI/PII Handling for Bedrock

PHI (Protected Health Information) includes diagnoses, laboratory results,
medications, and other health data linked to a person. PII (Personally
Identifiable Information) includes names, patient IDs, dates of birth,
addresses, and contact details; `PHII` is a common typo for `PII`.

Patient-level PHI/PII may be sent to the approved Bedrock Mistral path when it
is necessary for a metadata task and the data-governance requirements for the
deployment have been approved. In that case:

- send the minimum fields and rows needed for the task;
- prefer pseudonymous IDs and remove direct identifiers whenever they are not
  needed;
- use only the approved AWS account, region, IAM role, network path, and model
  configuration;
- keep prompts and responses out of application logs, notebooks, public
  artifacts, and error messages;
- encrypt stored inputs, outputs, and audit records with the project KMS key;
- retain only the approved records for the documented retention period; and
- require human review for uncertain mappings or clinically sensitive output.

Use data dictionaries, column summaries, value counts, missingness summaries,
and aggregate cohort statistics instead of patient-level rows whenever they are
sufficient for the task.

A minimal S3 structure is sufficient:

```text
s3://<bucket>/<dataset_id>/
  raw/                         original input files, read-only after upload
  metadata/                    dataset manifests and feature metadata
  processed/                   preprocessed tables used by VAEQL_plus
  runs/<run_id>/config/        exact config and feature dictionary for the run
  runs/<run_id>/tuning/        beta/C search outputs such as beta_analysis.csv
  runs/<run_id>/models/        checkpoints and selected model artifacts
  runs/<run_id>/logs/          training and evaluation logs through convergence or the maximum episode count
  runs/<run_id>/evaluation/    transformed-space metrics and small raw-scale tables
```

## Mandatory Bedrock Mistral SLM Metadata Tasks

Use Amazon Bedrock as the mandatory runtime for Mistral SLM metadata calls. The
SLM is a bounded metadata assistant, not a modeling component. It standardizes
clinical metadata before preprocessing and reporting.

Runtime rule:

- Bedrock with Mistral: required for metadata standardization, feature-type
  identification, and cohort demographic summaries.
- SageMaker: reserved for beta-DVAE training and beta/C tuning, or for a future
  self-hosted SLM if Bedrock models are insufficient.

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

Example SLM metadata output:

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

Keep dataset-specific evidence in metadata containing the source feature,
canonical feature, model type, coding rules, missing-value codes, evidence, and
confidence. Human review remains required for uncertain mappings.

Validate SLM output before compiling it into `FeatureTypeDict`. Use metadata
and aggregate summaries by default; if raw PHI/PII rows are necessary, apply
the controls in the PHI/PII handling section and document why they were needed.

## Run Outputs

Each run should retain:

- the dataset and metadata versions;
- preprocessing and beta/C tuning configuration;
- the compiled `FeatureTypeDict`;
- `beta_analysis.csv` and selected model checkpoints;
- random seeds and environment notes;
- transformed-space evaluation metrics;
- selected raw-scale metrics for clinically important features.

## Next Tasks

1. Define the S3 paths in project configuration.
2. Validate Mistral metadata output before compiling `FeatureTypeDict`.
3. Save each run's configuration, feature dictionary, tuning output, model, and
   evaluation results together.
