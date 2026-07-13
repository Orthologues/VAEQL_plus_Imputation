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
- Do not send PHI/PII to model prompts, logs, or public artifacts.

A minimal S3 structure is sufficient:

```text
s3://<bucket>/<dataset_id>/
  raw/
  metadata/
  processed/
  runs/<run_id>/
```

## Mandatory Bedrock Mistral SLM Agent

A Mistral SLM agent hosted through Amazon Bedrock is mandatory for:

- metadata standardization;
- feature-type identification;
- cohort demographics summaries from aggregate statistics.

The agent should use dataset dictionaries, column descriptions, value counts,
missingness summaries, and other non-identifying aggregate information. Its
output must be validated before it is used to build `FeatureTypeDict`.

Keep dataset-specific evidence in a small metadata record containing the source
feature, canonical feature, model type, coding rules, missing-value codes,
evidence, and confidence. Human review remains required for uncertain mappings.

Use SageMaker for beta-DVAE training and beta/C tuning. It is not the default
runtime for the mandatory Mistral metadata agent.

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
