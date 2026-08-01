from __future__ import annotations

import io
import json

from VAEQL_plus.util.AWS_Batch_interface import (
    build_step_command,
    build_submit_job_request,
    submit_training_step,
)
from VAEQL_plus.util.AWS_S3_interface import (
    build_s3_uri,
    build_sse_s3_args,
    get_json,
    parse_s3_uri,
    put_json,
    upload_file,
)


class FakeBatchClient:
    def __init__(self) -> None:
        self.submitted: dict = {}

    def submit_job(self, **request):
        self.submitted = request
        return {"jobId": "job-123", "jobName": request["jobName"]}


class FakeS3Client:
    def __init__(self) -> None:
        self.uploaded: tuple | None = None
        self.put_request: dict | None = None

    def upload_file(self, *args, **kwargs) -> None:
        self.uploaded = (args, kwargs)

    def put_object(self, **request) -> None:
        self.put_request = request

    def get_object(self, **request):
        assert request == {"Bucket": "bucket", "Key": "metadata.json"}
        return {"Body": io.BytesIO(b'{"run_id": "run-1"}')}


def test_batch_request_contains_step_command_and_environment() -> None:
    request = build_submit_job_request(
        job_name="run-1-step1",
        job_queue="research-queue",
        job_definition="vaeql-step",
        command=build_step_command(
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


def test_submit_training_step_uses_injected_batch_client() -> None:
    client = FakeBatchClient()

    response = submit_training_step(
        job_name="run-1-step2",
        job_queue="research-queue",
        job_definition="vaeql-step",
        step_module="VAEQL_plus.step2_beta_C_tuning",
        client=client,
    )

    assert response["jobId"] == "job-123"
    assert client.submitted["containerOverrides"]["command"] == [
        "python",
        "-m",
        "VAEQL_plus.step2_beta_C_tuning",
    ]


def test_s3_helpers_use_native_sse_s3() -> None:
    assert build_s3_uri("bucket", "/metadata.json") == "s3://bucket/metadata.json"
    assert parse_s3_uri("s3://bucket/metadata.json") == ("bucket", "metadata.json")
    assert build_sse_s3_args() == {
        "ServerSideEncryption": "AES256",
        "ChecksumAlgorithm": "SHA256",
    }


def test_s3_writes_use_injected_client_and_kms(tmp_path) -> None:
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"artifact")
    client = FakeS3Client()

    uri = upload_file(
        source,
        bucket="bucket",
        key="runs/run-1/artifact.bin",
        extra_args={"ServerSideEncryption": "AES256"},
        client=client,
    )

    assert uri == "s3://bucket/runs/run-1/artifact.bin"
    _, kwargs = client.uploaded
    assert kwargs["ExtraArgs"]["ServerSideEncryption"] == "AES256"

    metadata_uri = put_json(
        {"run_id": "run-1"},
        bucket="bucket",
        key="metadata.json",
        client=client,
    )
    assert metadata_uri == "s3://bucket/metadata.json"
    assert json.loads(client.put_request["Body"]) == {"run_id": "run-1"}
    assert client.put_request["ServerSideEncryption"] == "AES256"


def test_s3_json_reads_use_injected_client() -> None:
    assert get_json("s3://bucket/metadata.json", client=FakeS3Client()) == {"run_id": "run-1"}
