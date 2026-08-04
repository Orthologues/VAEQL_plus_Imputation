#########################################################
# Author: Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-08-02
# Description: Boilerplate tests for AWS Batch Step 0 feature profiling.
# Development: Mainly written with GPT-5.6 Luna-XHigh on Codex, with Jiawei Zhao's human
# review and revisions. Human review of the profiling output remains necessary.
#########################################################

from __future__ import annotations

import base64
import json

import pytest

from VAEQL_plus.step0_SLM_metadata_profiling.feature_type_profiling import (
    DEFAULT_MODEL_NAME,
    build_profile_manifest,
    parse_profile_response,
    run_feature_type_profiling,
    summarize_tabular_file,
    submit_feature_type_profiling_job,
)
from VAEQL_plus.util.AWS_Batch_interface import SSE_CUSTOMER_KEY_ENV_VAR, AWS_Batch_Interface


class FakeBatchClient:
    def __init__(self) -> None:
        self.submitted: dict = {}

    def submit_job(self, **request):
        self.submitted = request
        return {"jobId": "step0-job"}


def test_summarize_csv_without_raw_value_examples(tmp_path) -> None:
    source = tmp_path / "trial.csv"
    source.write_text("age,ecog\n61,1\n62,2\n", encoding="utf-8")

    summary = summarize_tabular_file(source)

    assert summary["sampled_rows"] == 2
    assert summary["features"][0]["source_feature"] == "age"
    assert "sampled_values" not in json.dumps(summary)


def test_parse_profile_response_requires_all_source_features() -> None:
    response = json.dumps(
        {
            "features": [
                {
                    "source_feature": "age",
                    "canonical_feature": "age",
                    "model_type": "continuous",
                    "confidence": 0.9,
                }
            ]
        }
    )

    with pytest.raises(ValueError, match="omitted source features"):
        parse_profile_response(response, ["age", "ecog"])


def test_profile_manifest_requires_human_review() -> None:
    manifest = build_profile_manifest(
        source_s3_uri="s3://pds/raw/trial.csv",
        model_name=DEFAULT_MODEL_NAME,
        schema_summary={"features": []},
        features=[],
    )

    assert manifest["human_review_required"] is True
    assert manifest["profile_status"] == "pending_human_review"


def test_submit_step0_job_forwards_s3_uris_and_key() -> None:
    client = FakeBatchClient()
    batch = AWS_Batch_Interface(client=client)

    response = submit_feature_type_profiling_job(
        batch,
        job_name="trial-step0",
        job_queue="gpu-queue",
        job_definition="ministral-8b-profile",
        input_s3_uri="s3://pds/raw/trial.csv",
        output_s3_uri="s3://pds/metadata/trial.json",
        metadata_s3_uris=("s3://pds/metadata/dictionary.md",),
        sse_customer_key_b64="encoded-key",
    )

    assert response == {"jobId": "step0-job"}
    assert client.submitted["containerOverrides"]["command"] == [
        "python",
        "-m",
        "VAEQL_plus.step0_SLM_metadata_profiling.feature_type_profiling",
        "--input-uri",
        "s3://pds/raw/trial.csv",
        "--output-uri",
        "s3://pds/metadata/trial.json",
        "--metadata-uri",
        "s3://pds/metadata/dictionary.md",
        "--model-name",
        DEFAULT_MODEL_NAME,
    ]
    assert client.submitted["containerOverrides"]["environment"] == [
        {"name": SSE_CUSTOMER_KEY_ENV_VAR, "value": "encoded-key"}
    ]


def test_step0_reads_a_mountpoint_path_and_writes_a_local_manifest_copy(tmp_path, monkeypatch) -> None:
    from VAEQL_plus.step0_SLM_metadata_profiling import feature_type_profiling as module

    source = tmp_path / "trial.csv"
    source.write_text("age\n61\n62\n", encoding="utf-8")
    local_manifest = tmp_path / "phase1" / "manifest.json"
    writes: list[dict] = []

    class FakeS3:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

        @staticmethod
        def parse_s3_uri(uri: str) -> tuple[str, str]:
            assert uri == "s3://pds/metadata/trial.json"
            return "pds", "metadata/trial.json"

        def write_json_metadata(self, payload, bucket: str, key: str) -> str:
            writes.append(payload)
            return f"s3://{bucket}/{key}"

    class FakeProfiler:
        def __init__(self, **kwargs) -> None:
            assert kwargs["model_name"] == DEFAULT_MODEL_NAME

        def profile(self, schema_summary, metadata_bundle):
            assert schema_summary["source_file"] == "trial.csv"
            assert metadata_bundle == []
            return [
                {
                    "source_feature": "age",
                    "canonical_feature": "age",
                    "model_type": "continuous",
                    "confidence": 0.9,
                }
            ]

    monkeypatch.setattr(module, "AWS_S3_Interface", FakeS3)
    monkeypatch.setattr(module, "MinistralFeatureTypeProfiler", FakeProfiler)
    monkeypatch.setenv(SSE_CUSTOMER_KEY_ENV_VAR, base64.b64encode(bytes(range(32))).decode("ascii"))

    output_uri = run_feature_type_profiling(
        None,
        "s3://pds/metadata/trial.json",
        input_path=source,
        manifest_path=local_manifest,
    )

    assert output_uri == "s3://pds/metadata/trial.json"
    assert local_manifest.is_file()
    assert json.loads(local_manifest.read_text(encoding="utf-8"))["source_reference"] == "mountpoint://trial.csv"
    assert writes[0]["source_s3_uri"] is None
