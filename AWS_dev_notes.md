# AWS Dev Notes for `VAEQL_plus`

Use AWS Batch for `a.` private SLM-empowered PDS metadata preparation, `b.` $\beta$/C tuning prior to training, `c.` $\beta$-DVAE training, and `d.` reproducible evaluation jobs. Use private S3 for
structured datasets and run storage. Keep the modeling code and research
decisions in this repository.

## Preliminary Two-Phase Workflow

The preliminary deployment workflow separates clinical-data annotation from
DRL training. AWS Batch schedules both jobs onto approved GPU compute, while
S3 is the versioned handoff between them:

```text
[ Phase 1: Annotation Job ] --> GPU instance --> Ministral 8B
                                  --> structured dataset --> encrypted S3
                                                               |
                                                               v
[ Phase 2: DRL Training Job ] --> GPU instance --> pulls structured dataset
                                  --> trains DRL model --> final model in S3
```

Phase 1 runs Ministral 8B as a bounded annotation and metadata-standardization
job. It validates source-feature mappings, compiles the model-facing
`FeatureTypeDict`, and writes a versioned structured dataset plus its metadata
manifest to encrypted S3. The annotation job is not the DRL model and must not
silently alter the source data.

Phase 2 starts only from the versioned Phase 1 S3 output. It pulls the
structured dataset, applies the documented preprocessing contract, conducts halving grid search to select the $\beta$ and $C$ hyperparameters and thus trains the DRL model (including its $\beta$-DVAE and Q-learning steps) on a GPU instance, and saves the final model, run configuration (including the applied hyperparameters), and evaluation statistics to S3.

The Phase 1 to Phase 2 contract must include the dataset metadata, compiled `FeatureTypeDict`, preprocessing parameters, random seeds,
model configuration, and source-to-canonical feature mappings. A failed or
uncertain annotation job must block Phase 2 until human review resolves it.

## Data Storage and Security

- Store raw data, processed data, metadata, model artifacts, and evaluation
  outputs in a private S3 bucket.
- PDS datasets stored in S3 shall not be public unless explicitly required by
  the reviewer.
- Block public access by default.
- Encrypt all S3 objects with server-side encryption using customer-provided
  keys (`SSE-C`, `AES256`) and a base64-encoded 256-bit AES key. Require the
  same key for each S3 read/write request and never commit or log it.
- Enable SSE-C for the bucket before uploading objects by setting
  `BlockedEncryptionTypes` to `NONE`.
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
  runs/<run_id>/tuning/        hyperparameter search outputs such as beta_analysis.csv
  runs/<run_id>/models/        checkpoints and selected model artifacts
  runs/<run_id>/logs/          training and evaluation logs through convergence or the maximum episode count
  runs/<run_id>/evaluation/    transformed-space metrics and small raw-scale tables
```

## AWS Batch Runtime and SLM Metadata Tasks

Use AWS Batch as the mandatory runtime for metadata standardization,
feature-type identification, cohort demographic summaries, $\beta$-DVAE training,
$\beta$/C tuning, and evaluation. Each job runs an approved, versioned container.
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
- `step2_beta_C_tuning`: $\beta$/C cross-validation and model selection; and
- `step3_drl_training`: DRL-agent, $\beta$-DVAE, and Q-learning training from the
  versioned Phase 1 S3 dataset; and
- future `stepX_X` modules: evaluation and reporting stages using the same
  Batch and S3 interfaces.

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
- the Phase 1 structured-dataset version and annotation manifest;
- preprocessing and $\beta$/C tuning configuration;
- the compiled `FeatureTypeDict`;
- `beta_analysis.csv` and selected model checkpoints;
- the Phase 2 DRL-agent configuration and final model artifact;
- random seeds and environment notes;
- transformed-space evaluation metrics;
- selected raw-scale metrics for clinically important features.

## `Nextflow` Integration with AWS S3 Mount and AWS Batch

Nextflow is the orchestration layer for the modular imputation workflow. It
should connect independent `stepX_X` modules, submit each process to AWS Batch, and publish versioned outputs including evaluation logs and GPU computing time. The scientific
and data-processing logic remains in Python modules under `VAEQL_plus`; a
Nextflow process should call a stable module entry point at `__init__.py` files under each `VAEQL_plus.stepX_X` module.

Each process must declare an explicit input and output contract. Use the NF Dataflow Channels
for artifacts, e.g., models and logs produced by an upstream process, and pass the dataset version,
configuration URI pegged to a series of run IDs, run ID, and review status explicitly between steps. Do not depend on an undeclared shared local directory or on files left behind by an
earlier process. A process should write its outputs to its isolated and idempotent work
directory identifiable by a configuration URI, publish only idempotent and versioned artifacts to the configured S3 run path for each run ID.

`VAEQL_plus/conf/nextflow_conf.nf` currently provides a preliminary beta/C
tuning workflow wrapper. Future modular workflows should extend this pattern
with separate processes for Step 0 metadata profiling, Step 1 preprocessing,
Step 2 beta/C tuning, Step 3 DRL training, and evaluation. The AWS deployment
configuration must select the AWS Batch executor, approved container images,
job queues, CPU/GPU resources, memory, timeout, retry, and IAM settings without
putting private data or encryption keys in the workflow file.

Mountpoint for Amazon S3 is an optional optimization for large, read-heavy,
static reference data. It is not a replacement for the repository's
`AWS_S3_Interface`, manifest validation, or versioned artifact handoff. When a
Mountpoint path is used as a static reference, pass that path as a Nextflow
`val` input; use ordinary `path` inputs for files that Nextflow must stage and
track as process artifacts.

The PDS and run objects in this project use SSE-C. Mountpoint access must not be
assumed to support the project's Base64-encoded SSE-C customer key, because its
documented authentication path is IAM-based and the key-injection contract has
not been validated here. Until a dedicated AWS integration test proves
otherwise, read and write SSE-C-protected objects through
`AWS_S3_Interface`, with `VAEQL_S3_SSE_CUSTOMER_KEY_B64` supplied only through
the approved Batch runtime environment. Never place the key in a Nextflow
parameter, command line, log, or published artifact.

See the AWS guidance on [Nextflow with AWS Batch and Mountpoint for Amazon
S3](https://aws.amazon.com/blogs/hpc/optimize-nextflow-workflows-on-aws-batch-with-mountpoint-for-amazon-s3/),
[Mountpoint for Amazon S3](https://docs.aws.amazon.com/AmazonS3/latest/userguide/mountpoint.html),
and [S3 SSE-C](https://docs.aws.amazon.com/AmazonS3/latest/userguide/specifying-s3-c-encryption.html).

## Next Tasks

1. Define the Phase 1 and Phase 2 S3 paths, Batch job queue, GPU compute
   environment, and job definitions in project configuration.
2. Package and version the Ministral 8B annotation container; validate its
   structured output before compiling `FeatureTypeDict`.
3. Implement the Phase 1-to-Phase 2 dataset manifest and review gate.
4. Save each run's configuration, feature dictionary, tuning output, DRL model,
   and evaluation results together.
