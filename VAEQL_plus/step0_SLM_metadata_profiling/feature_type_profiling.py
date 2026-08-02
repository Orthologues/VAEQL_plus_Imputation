#########################################################
# Author: Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-08-02
# Description: AWS Batch boilerplate for Ministral feature-type profiling.
# Development: Mainly written with GPT-5.6 Luna-XHigh on Codex, with Jiawei Zhao's human
# review and revisions. Human review of the generated profile and this boilerplate is required.
#########################################################

"""Step 0 AWS Batch entry point for preliminary feature-type profiling.

The SLM runs inside an approved AWS Batch GPU container. This module only
coordinates bounded schema profiling and writes a pending-human-review manifest;
it is not a clinical metadata authority or a production annotation system.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from VAEQL_plus.util.AWS_Batch_interface import (
    SSE_CUSTOMER_KEY_ENV_VAR,
    AWS_Batch_Interface,
)
from VAEQL_plus.util.AWS_S3_interface import AWS_S3_Interface


DEFAULT_MODEL_NAME = "mistralai/Ministral-8B-Instruct-2410"
JOB_MODULE = "VAEQL_plus.step0_SLM_metadata_profiling.feature_type_profiling"
ALLOWED_MODEL_TYPES = {
    "continuous",
    "positive_continuous",
    "count",
    "binary",
    "categorical",
    "ordinal",
}
_TEXT_METADATA_SUFFIXES = {".csv", ".json", ".md", ".tex", ".txt", ".yaml", ".yml"}


def summarize_raw_data_with_pandas(path: str | Path, *, max_rows: int = 256) -> dict[str, Any]:
    """Build a sanitized Pandas metadata summary without sending raw rows to the SLM."""
    if max_rows < 1:
        raise ValueError("max_rows must be greater than zero")
    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    try:
        import pandas as pd
    except ModuleNotFoundError as exc:  # pragma: no cover - Batch image dependency
        raise RuntimeError("pandas is required for the Step 0 metadata summary") from exc

    if input_path.suffix.lower() == ".json":
        frame = pd.read_json(input_path)
    else:
        frame = pd.read_csv(input_path, nrows=max_rows)
    frame = frame.head(max_rows)

    columns: list[dict[str, Any]] = []
    for name in frame.columns:
        series = frame[name]
        non_missing = int(series.notna().sum())
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_non_missing = int(numeric.notna().sum())
        column: dict[str, Any] = {
            "source_feature": str(name),
            "pandas_dtype": str(series.dtype),
            "sampled_non_missing": non_missing,
            "sampled_missing": int(series.isna().sum()),
            "sampled_unique": int(series.nunique(dropna=True)),
            "numeric_parse_rate": round(numeric_non_missing / non_missing, 8)
            if non_missing
            else 0.0,
        }
        if numeric_non_missing:
            column["numeric_min"] = float(numeric.min())
            column["numeric_max"] = float(numeric.max())
        columns.append(column)

    return {
        "source_file": input_path.name,
        "summary_library": "pandas",
        "sampled_rows": int(len(frame)),
        "features": columns,
    }


def summarize_tabular_file(path: str | Path, *, max_rows: int = 256) -> dict[str, Any]:
    """Backward-compatible alias for the Pandas metadata summary."""
    return summarize_raw_data_with_pandas(path, max_rows=max_rows)


def build_metadata_bundle(
    metadata_paths: Sequence[str | Path],
    *,
    source_uris: Sequence[str] = (),
    max_chars_per_file: int = 20_000,
) -> list[dict[str, Any]]:
    """Read approved non-raw metadata files for the SLM prompt.

    Text metadata is included as UTF-8 content. Binary metadata is represented
    by its filename and size because it cannot be safely placed in a text-only
    prompt by this boilerplate.
    """
    if max_chars_per_file < 1:
        raise ValueError("max_chars_per_file must be greater than zero")
    if source_uris and len(source_uris) != len(metadata_paths):
        raise ValueError("source_uris must align one-to-one with metadata_paths")

    bundle: list[dict[str, Any]] = []
    for index, raw_path in enumerate(metadata_paths):
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        item: dict[str, Any] = {
            "source_uri": source_uris[index] if source_uris else None,
            "file_name": path.name,
            "size_bytes": path.stat().st_size,
        }
        if path.suffix.lower() in _TEXT_METADATA_SUFFIXES:
            content = path.read_text(encoding="utf-8", errors="replace")
            item["content"] = content[:max_chars_per_file]
            item["content_truncated"] = len(content) > max_chars_per_file
        else:
            item["content"] = None
            item["content_note"] = "Binary metadata content is not included in this text prompt."
        bundle.append(item)
    return bundle


def build_profiling_prompt(
    schema_summary: Mapping[str, Any],
    metadata_bundle: Sequence[Mapping[str, Any]] = (),
) -> str:
    """Build a constrained JSON-only prompt for the Batch-local SLM."""
    return (
        "Profile the tabular schema and approved non-raw metadata below for a "
        "research preprocessing pipeline. "
        "Return JSON only with one object per source feature under `features`. "
        "Use exactly one model_type from continuous, positive_continuous, count, "
        "binary, categorical, or ordinal. Do not invent clinical facts. "
        "Set confidence between 0 and 1 and explain evidence briefly. "
        "Human review is mandatory. Raw row-level data is excluded from this prompt.\n\n"
        + json.dumps(
            {
                "pandas_metadata_summary": schema_summary,
                "approved_non_raw_metadata": list(metadata_bundle),
            },
            ensure_ascii=True,
            sort_keys=True,
        )
    )


def parse_profile_response(
    response_text: str,
    source_features: Sequence[str],
) -> list[dict[str, Any]]:
    """Parse and validate the SLM JSON response before it reaches preprocessing."""
    start = response_text.find("{")
    end = response_text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("SLM response did not contain a JSON object")
    try:
        payload = json.loads(response_text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("SLM response did not contain valid JSON") from exc

    raw_features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(raw_features, list):
        raise ValueError("SLM response must contain a `features` list")

    expected = list(source_features)
    expected_set = set(expected)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_features:
        if not isinstance(item, dict):
            raise ValueError("Each profiled feature must be a JSON object")
        source_feature = item.get("source_feature")
        model_type = item.get("model_type")
        if source_feature not in expected_set:
            raise ValueError(f"Unknown source feature from SLM: {source_feature!r}")
        if source_feature in seen:
            raise ValueError(f"Duplicate source feature from SLM: {source_feature!r}")
        if model_type not in ALLOWED_MODEL_TYPES:
            raise ValueError(f"Unsupported model_type from SLM: {model_type!r}")
        confidence = item.get("confidence")
        if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
            raise ValueError(f"Invalid confidence for {source_feature!r}")
        seen.add(source_feature)
        normalized.append(
            {
                "source_feature": source_feature,
                "canonical_feature": item.get("canonical_feature", source_feature),
                "model_type": model_type,
                "num_levels": item.get("num_levels"),
                "raw_to_canonical": item.get("raw_to_canonical", {}),
                "missing_value_codes": item.get("missing_value_codes", []),
                "evidence": item.get("evidence", ""),
                "confidence": float(confidence),
            }
        )

    missing = expected_set - seen
    if missing:
        raise ValueError(f"SLM response omitted source features: {sorted(missing)!r}")
    return normalized


def build_profile_manifest(
    *,
    source_s3_uri: str,
    model_name: str,
    schema_summary: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    metadata_s3_uris: Sequence[str] = (),
) -> dict[str, Any]:
    """Build a versioned manifest that cannot be consumed without human review."""
    return {
        "schema_version": "0.1",
        "source_s3_uri": source_s3_uri,
        "model_name": model_name,
        "profile_status": "pending_human_review",
        "human_review_required": True,
        "metadata_s3_uris": list(metadata_s3_uris),
        "schema_summary": dict(schema_summary),
        "features": [dict(feature) for feature in features],
    }


class MinistralFeatureTypeProfiler:
    """Minimal local-in-container adapter for Ministral feature-type profiling."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = 1024,
    ) -> None:
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be greater than zero")
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens

    def profile(
        self,
        schema_summary: Mapping[str, Any],
        metadata_bundle: Sequence[Mapping[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Run Ministral inside the AWS Batch container and validate its JSON output."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ModuleNotFoundError as exc:  # pragma: no cover - Batch image dependency
            raise RuntimeError(
                "torch and transformers are required in the AWS Batch SLM image"
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            torch_dtype="auto",
        )
        prompt = build_profiling_prompt(schema_summary, metadata_bundle)
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
        else:  # pragma: no cover - model-tokenizer compatibility fallback
            inputs = tokenizer(prompt, return_tensors="pt")["input_ids"]
        inputs = inputs.to(model.device)
        with torch.inference_mode():
            outputs = model.generate(inputs, max_new_tokens=self.max_new_tokens)
        generated = tokenizer.decode(
            outputs[0, inputs.shape[-1] :],
            skip_special_tokens=True,
        )
        source_features = [item["source_feature"] for item in schema_summary["features"]]
        return parse_profile_response(generated, source_features)


def run_feature_type_profiling(
    input_s3_uri: str,
    output_s3_uri: str,
    *,
    metadata_s3_uris: Sequence[str] = (),
    model_name: str = DEFAULT_MODEL_NAME,
    region_name: str | None = None,
    max_rows: int = 256,
) -> str:
    """Run Step 0 in AWS Batch and write its SSE-C-protected review manifest.

    The SSE-C customer key protects S3 object reads and writes; AWS IAM
    credentials still authorize the S3 and Batch API calls.
    """
    sse_customer_key_b64 = os.environ.get(SSE_CUSTOMER_KEY_ENV_VAR)
    if not sse_customer_key_b64:
        raise RuntimeError(
            f"{SSE_CUSTOMER_KEY_ENV_VAR} must be set in the AWS Batch container environment"
        )
    s3 = AWS_S3_Interface(
        sse_customer_key_b64=sse_customer_key_b64,
        region_name=region_name,
    )
    _, input_key = s3.parse_s3_uri(input_s3_uri)
    output_bucket, output_key = s3.parse_s3_uri(output_s3_uri)
    with tempfile.TemporaryDirectory(prefix="vaeql_step0_") as temp_dir:
        input_suffix = Path(input_key).suffix.lower() or ".csv"
        input_path = s3.download_file(
            input_s3_uri,
            Path(temp_dir) / f"pds_trial_input{input_suffix}",
        )
        metadata_paths: list[Path] = []
        for index, metadata_uri in enumerate(metadata_s3_uris):
            if metadata_uri == input_s3_uri:
                raise ValueError("metadata_s3_uris must exclude the raw input URI")
            _, metadata_key = s3.parse_s3_uri(metadata_uri)
            metadata_suffix = Path(metadata_key).suffix.lower() or ".metadata"
            metadata_paths.append(
                s3.download_file(
                    metadata_uri,
                    Path(temp_dir) / f"metadata_{index}{metadata_suffix}",
                )
            )
        schema_summary = summarize_raw_data_with_pandas(input_path, max_rows=max_rows)
        metadata_bundle = build_metadata_bundle(
            metadata_paths,
            source_uris=metadata_s3_uris,
        )
        features = MinistralFeatureTypeProfiler(model_name=model_name).profile(
            schema_summary,
            metadata_bundle,
        )
        manifest = build_profile_manifest(
            source_s3_uri=input_s3_uri,
            model_name=model_name,
            schema_summary=schema_summary,
            features=features,
            metadata_s3_uris=metadata_s3_uris,
        )
        return s3.write_json_metadata(manifest, output_bucket, output_key)


def submit_feature_type_profiling_job(
    batch: AWS_Batch_Interface,
    *,
    job_name: str,
    job_queue: str,
    job_definition: str,
    input_s3_uri: str,
    output_s3_uri: str,
    metadata_s3_uris: Sequence[str] = (),
    model_name: str = DEFAULT_MODEL_NAME,
    sse_customer_key_b64: str | None = None,
    retry_attempts: int | None = 1,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Submit Step 0 to an AWS Batch GPU job definition.

    The supplied job definition must request the GPU resources required by the
    selected SLM image; this helper only submits the job.
    """
    AWS_S3_Interface.parse_s3_uri(input_s3_uri)
    AWS_S3_Interface.parse_s3_uri(output_s3_uri)
    arguments = [
        "--input-uri",
        input_s3_uri,
        "--output-uri",
        output_s3_uri,
    ]
    for metadata_uri in metadata_s3_uris:
        arguments.extend(("--metadata-uri", metadata_uri))
    arguments.extend(("--model-name", model_name))
    return batch.submit_training_step(
        job_name=job_name,
        job_queue=job_queue,
        job_definition=job_definition,
        step_module=JOB_MODULE,
        arguments=arguments,
        sse_customer_key_b64=sse_customer_key_b64,
        retry_attempts=retry_attempts,
        timeout_seconds=timeout_seconds,
    )


def main() -> None:
    """Run the Batch-container entry point for Step 0."""
    parser = argparse.ArgumentParser(description="Profile PDS feature types with Ministral on AWS Batch")
    parser.add_argument("--input-uri", required=True, help="SSE-C-protected input S3 URI")
    parser.add_argument("--output-uri", required=True, help="SSE-C-protected output manifest S3 URI")
    parser.add_argument(
        "--metadata-uri",
        action="append",
        default=[],
        help="Approved non-raw metadata S3 URI; repeat for multiple files",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--region-name", default=None)
    parser.add_argument("--max-rows", type=int, default=256)
    args = parser.parse_args()
    run_feature_type_profiling(
        args.input_uri,
        args.output_uri,
        metadata_s3_uris=args.metadata_uri,
        model_name=args.model_name,
        region_name=args.region_name,
        max_rows=args.max_rows,
    )


if __name__ == "__main__":
    main()
