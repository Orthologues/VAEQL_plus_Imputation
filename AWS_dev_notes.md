# AWS Dev Notes for `VAEQL_plus`

Use AWS Batch for `a.` private SLM-empowered PDS metadata preparation, `b.` beta/C tuning prior to training, `c.` beta-DVAE training, and `d.` reproducible evaluation jobs. Use private S3 for dataset and
run storage. Keep the modeling code and research decisions in this repository.

## Data Storage and Security

- Store raw data, processed data, metadata, model artifacts, and evaluation
  outputs in a private S3 bucket.
- PDS datasets stored in S3 shall not be public unless explicitly required by
  the reviewer.
- Block public access by default.
- Encrypt all S3 objects with native S3 server-side encryption (`SSE-S3`,
  `AES256`).
- Restrict S3 access to the roles used for preprocessing, training, and
  evaluation.
- Keep raw data read-only after ingestion.
- Do not send PHI/PII to unapproved model endpoints, application logs, or
  public artifacts.

### PHI/PII Handling for AWS Batch

PHI (Protected Health Information) includes diagnoses, laboratory results,
medications, and other health data linked to a person. PII (Personally
Identifiable Information) includes names, patient IDs, dates of birth,
addresses, and contact details; `PHII` is a common typo for `PII`.

Patient-level PHI/PII may be mounted or streamed into an approved AWS Batch
container when it is necessary for a metadata task and the data-governance
requirements for the deployment have been approved. In that case:

- send the minimum fields and rows needed for the task;
- prefer pseudonymous IDs and remove direct identifiers whenever they are not
  needed;
- use only the approved AWS account, region, IAM role, network path, and model
  configuration;
- keep prompts and responses out of application logs, notebooks, public
  artifacts, and error messages;
- store inputs, outputs, and audit records only in the encrypted S3 locations;
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

## AWS Batch Runtime and SLM Metadata Tasks

Use AWS Batch as the mandatory runtime for metadata standardization,
feature-type identification, cohort demographic summaries, beta-DVAE training,
beta/C tuning, and evaluation. Each job runs an approved, versioned container.
The SLM is a bounded metadata assistant, not a modeling component; it
standardizes clinical metadata before preprocessing and reporting.

Candidate SLMs for the Batch container are:

- Llama 3.1 8B;
- Ministral 8B; and
- Ministral 14B.

AWS Batch is the preferred runtime because it is a lower-cost, academically
appropriate job-orchestration choice for this research prototype. Batch does
not provide model weights itself: the selected SLM must be packaged in an
approved container or accessed through an approved endpoint from the job.

The boilerplate interfaces are [AWS_Batch_interface.py](VAEQL_plus/util/AWS_Batch_interface.py)
and [AWS_S3_interface.py](VAEQL_plus/util/AWS_S3_interface.py). They provide
the shared API boundary for `stepX_X` jobs and encrypted S3 run artifacts. The
intended step modules are:

- `step1_preprocessing`: metadata adapter validation and type-aware
  preprocessing;
- `step2_beta_C_tuning`: beta/C cross-validation and model selection; and
- future `stepX_X` modules: training, Q-learning, and evaluation stages using
  the same Batch and S3 interfaces.

Pass dataset, run, and S3 identifiers through job environment variables or the
versioned run configuration. Do not put private data in Batch job definitions,
command lines, logs, or public artifacts.

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

Validate SLM output inside the Batch preprocessing job before compiling it into
`FeatureTypeDict`. Use metadata and aggregate summaries by default; if raw
PHI/PII rows are necessary, apply the controls in the PHI/PII handling section
and document why they were needed.

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

1. Define the S3 paths, Batch job queue, and job definitions in project
   configuration.
2. Package and version the selected SLM container; validate its metadata output
   before compiling `FeatureTypeDict`.
3. Save each run's configuration, feature dictionary, tuning output, model, and
   evaluation results together.
