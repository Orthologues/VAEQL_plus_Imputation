#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-08-01
# Description: AWS Batch and S3 interface unit tests.
# Development: Mainly written with GPT-5.5 Medium/GPT-5.6 Luna-XHigh on Codex, with Jiawei Zhao's human
# review and revisions.
#########################################################

from __future__ import annotations

import base64
import hashlib
import io
import json

import pytest

from VAEQL_plus.util.AWS_Batch_interface import (
    SSE_CUSTOMER_KEY_ENV_VAR,
    AWS_Batch_Interface,
)
from VAEQL_plus.util.AWS_S3_interface import AWS_S3_Interface

# 256 bits (symmetric key required by AES-256) / 8 = 32 bytes
SSE_C_KEY = bytes(range(32))
SSE_C_KEY_B64 = base64.b64encode(SSE_C_KEY).decode("ascii")
SSE_C_KEY_MD5_B64 = base64.b64encode(
    hashlib.md5(SSE_C_KEY, usedforsecurity=False).digest()
).decode("ascii")


class FakeBatchClient:
    def __init__(self) -> None:
        self.submitted: dict = {}

    def submit_job(self, **request):
        self.submitted = request
        return {"jobId": "job-123", "jobName": request["jobName"]}


class FakeS3Client:
    def __init__(self) -> None:
        self.uploaded: tuple | None = None
        self.downloaded: tuple | None = None
        self.put_request: dict | None = None

    def upload_file(self, *args, **kwargs) -> None:
        self.uploaded = (args, kwargs)

    def put_object(self, **request) -> None:
        self.put_request = request

    def download_file(self, *args, **kwargs) -> None:
        self.downloaded = (args, kwargs)

    def get_object(self, **request):
        assert request == {
            "Bucket": "bucket",
            "Key": "metadata.json",
            "SSECustomerAlgorithm": "AES256",
            "SSECustomerKey": SSE_C_KEY_B64,
            "SSECustomerKeyMD5": SSE_C_KEY_MD5_B64,
        }
        return {"Body": io.BytesIO(b'{"run_id": "run-1"}')}


def test_batch_interface_preserves_local_aws_configuration() -> None:
    client = FakeBatchClient()
    batch = AWS_Batch_Interface(
        region_name="eu-central-1",
        profile_name="vaeql-research",
        client=client,
    )

    assert batch.region_name == "eu-central-1"
    assert batch.profile_name == "vaeql-research"
    assert batch._batch_client() is client


def test_batch_request_contains_step_command_and_environment() -> None:
    assert SSE_CUSTOMER_KEY_ENV_VAR == "VAEQL_S3_SSE_CUSTOMER_KEY_B64"
    batch = AWS_Batch_Interface()
    request = batch.build_submit_job_request(
        job_name="run-1-step1",
        job_queue="research-queue",
        job_definition="vaeql-step",
        command=batch.build_step_command(
            "VAEQL_plus.step1_preprocessing",
            arguments=("--run-id", "run-1"),
        ),
        environment={"VAEQL_RUN_ID": "run-1"},
        retry_attempts=2,
    )

    assert request["containerOverrides"]["command"] == [
        "python",
        "-m",
        "VAEQL_plus.step1_preprocessing",
        "--run-id",
        "run-1",
    ]
    assert request["containerOverrides"]["environment"] == [
        {"name": "VAEQL_RUN_ID", "value": "run-1"}
    ]
    assert request["retryStrategy"] == {"attempts": 2}

    assert batch.build_step_command("VAEQL_plus.step0_SLM_metadata_profiling") == [
        "python",
        "-m",
        "VAEQL_plus.step0_SLM_metadata_profiling",
    ]


@pytest.mark.parametrize("step_module", ["", "step1", "VAEQL_plus.step1", "-step1_preprocessing"])
def test_build_step_command_rejects_invalid_step_modules(step_module: str) -> None:
    with pytest.raises(ValueError, match="dotted VAEQL step module path"):
        AWS_Batch_Interface.build_step_command(step_module)


def test_submit_training_step_uses_injected_batch_client() -> None:
    client = FakeBatchClient()
    batch = AWS_Batch_Interface(client=client)

    response = batch.submit_training_step(
        job_name="run-1-step2",
        job_queue="research-queue",
        job_definition="vaeql-step",
        step_module="VAEQL_plus.step2_beta_C_tuning",
    )

    assert response["jobId"] == "job-123"
    assert client.submitted["containerOverrides"]["command"] == [
        "python",
        "-m",
        "VAEQL_plus.step2_beta_C_tuning",
    ]


def test_submit_training_step_forwards_sse_customer_key(monkeypatch) -> None:
    monkeypatch.setenv(SSE_CUSTOMER_KEY_ENV_VAR, SSE_C_KEY_B64)
    client = FakeBatchClient()
    batch = AWS_Batch_Interface(client=client)

    batch.submit_training_step(
        job_name="run-1-step2",
        job_queue="research-queue",
        job_definition="vaeql-step",
        step_module="VAEQL_plus.step2_beta_C_tuning",
    )

    assert client.submitted["containerOverrides"]["environment"] == [
        {"name": SSE_CUSTOMER_KEY_ENV_VAR, "value": SSE_C_KEY_B64}
    ]


def test_s3_helpers_use_sse_c() -> None:
    s3 = AWS_S3_Interface(sse_customer_key_b64=SSE_C_KEY_B64)
    assert s3.build_s3_uri("bucket", "/metadata.json") == "s3://bucket/metadata.json"
    assert s3.parse_s3_uri("s3://bucket/metadata.json") == ("bucket", "metadata.json")


def test_s3_writes_use_injected_client_and_sse_c(tmp_path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    client = FakeS3Client()
    s3 = AWS_S3_Interface(sse_customer_key_b64=SSE_C_KEY_B64, client=client)

    uri = s3.upload_file(
        source,
        bucket="bucket",
        key="runs/run-1/artifact.bin",
        extra_args={"SSECustomerAlgorithm": "invalid"},
    )

    assert uri == "s3://bucket/runs/run-1/artifact.bin"
    _, kwargs = client.uploaded
    assert kwargs["ExtraArgs"]["SSECustomerAlgorithm"] == "AES256"
    assert kwargs["ExtraArgs"]["SSECustomerKey"] == SSE_C_KEY_B64
    assert kwargs["ExtraArgs"]["SSECustomerKeyMD5"] == SSE_C_KEY_MD5_B64
    assert kwargs["ExtraArgs"]["ChecksumAlgorithm"] == "SHA256"

    metadata_uri = s3.write_json_metadata(
        {"run_id": "run-1"},
        bucket="bucket",
        key="metadata.json",
    )
    assert metadata_uri == "s3://bucket/metadata.json"
    assert json.loads(client.put_request["Body"]) == {"run_id": "run-1"}
    assert client.put_request["SSECustomerAlgorithm"] == "AES256"
    assert client.put_request["SSECustomerKey"] == SSE_C_KEY_B64
    assert client.put_request["SSECustomerKeyMD5"] == SSE_C_KEY_MD5_B64
    assert client.put_request["ChecksumAlgorithm"] == "SHA256"


def test_s3_json_reads_use_injected_client() -> None:
    s3 = AWS_S3_Interface(sse_customer_key_b64=SSE_C_KEY_B64, client=FakeS3Client())
    assert s3.get_json("s3://bucket/metadata.json") == {"run_id": "run-1"}


def test_s3_file_download_uses_sse_c(tmp_path) -> None:
    client = FakeS3Client()
    s3 = AWS_S3_Interface(sse_customer_key_b64=SSE_C_KEY_B64, client=client)
    destination = s3.download_file("s3://bucket/artifact.bin", tmp_path / "artifact.bin")

    assert destination == tmp_path / "artifact.bin"
    args, kwargs = client.downloaded
    assert args == ("bucket", "artifact.bin", str(destination))
    assert kwargs["ExtraArgs"] == {
        "SSECustomerAlgorithm": "AES256",
        "SSECustomerKey": SSE_C_KEY_B64,
        "SSECustomerKeyMD5": SSE_C_KEY_MD5_B64,
    }


def test_s3_rejects_non_256_bit_sse_c_keys() -> None:
    with pytest.raises(ValueError, match="exactly 32 bytes"):
        AWS_S3_Interface(sse_customer_key_b64=base64.b64encode(b"short").decode("ascii"))
