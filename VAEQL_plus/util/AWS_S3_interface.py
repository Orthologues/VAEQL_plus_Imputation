#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-08-01
# Description: S3 helpers for protected VAEQL run artifacts.
# Development: Mainly written with GPT-5.5 Medium/GPT-5.6 Luna-XHigh on Codex, with Jiawei Zhao's human
# review and revisions.
#########################################################

"""S3 helpers for protected VAEQL run artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


def _s3_client(*, region_name: str | None = None, client: Any | None = None) -> Any:
    """Return an injected client or lazily create a boto3 S3 client."""
    if client is not None:
        return client
    try:
        import boto3
    except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "boto3 is required for AWS S3 operations; install the project "
            "environment first."
        ) from exc
    return boto3.client("s3", region_name=region_name)


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/key`` URI into bucket and object key."""
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
        raise ValueError(f"Expected an S3 URI in the form s3://bucket/key, got {uri!r}")
    return parsed.netloc, parsed.path.lstrip("/")


def build_s3_uri(bucket: str, key: str) -> str:
    """Build an S3 URI after validating its bucket and object key."""
    bucket = bucket.strip()
    key = key.lstrip("/")
    if not bucket or not key:
        raise ValueError("bucket and key must be non-empty")
    return f"s3://{bucket}/{key}"


def build_sse_s3_args(*, checksum_algorithm: str | None = "SHA256") -> dict[str, str]:
    """Build native S3 server-side encryption and checksum arguments."""
    args = {
        "ServerSideEncryption": "AES256",
    }
    if checksum_algorithm is not None:
        if checksum_algorithm not in {"CRC32", "CRC32C", "SHA1", "SHA256"}:
            raise ValueError("Unsupported checksum algorithm")
        args["ChecksumAlgorithm"] = checksum_algorithm
    return args


def upload_file(
    file_path: str | Path,
    *,
    bucket: str,
    key: str,
    extra_args: Mapping[str, Any] | None = None,
    region_name: str | None = None,
    client: Any | None = None,
) -> str:
    """Upload a file with mandatory SSE-S3 protection and return its S3 URI."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    upload_args: dict[str, Any] = {}
    if extra_args:
        upload_args.update(dict(extra_args))
    # Apply the protection setting last so callers cannot downgrade the write.
    upload_args.update(build_sse_s3_args())
    _s3_client(region_name=region_name, client=client).upload_file(
        str(path),
        bucket,
        key.lstrip("/"),
        ExtraArgs=upload_args,
    )
    return build_s3_uri(bucket, key)


def download_file(
    uri: str,
    destination: str | Path,
    *,
    region_name: str | None = None,
    client: Any | None = None,
) -> Path:
    """Download an S3 object to a local path and return the resulting path."""
    bucket, key = parse_s3_uri(uri)
    destination_path = Path(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    _s3_client(region_name=region_name, client=client).download_file(
        bucket,
        key,
        str(destination_path),
    )
    return destination_path


def put_json(
    payload: Mapping[str, Any],
    *,
    bucket: str,
    key: str,
    region_name: str | None = None,
    client: Any | None = None,
) -> str:
    """Write JSON metadata to S3 with SSE-S3 and a SHA-256 checksum."""
    body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    args: dict[str, Any] = build_sse_s3_args()
    args.update(
        {
            "Body": body,
            "Bucket": bucket,
            "Key": key.lstrip("/"),
            "ContentType": "application/json",
        }
    )
    _s3_client(region_name=region_name, client=client).put_object(**args)
    return build_s3_uri(bucket, key)


def get_json(
    uri: str,
    *,
    region_name: str | None = None,
    client: Any | None = None,
) -> dict[str, Any]:
    """Read a JSON object from S3."""
    bucket, key = parse_s3_uri(uri)
    response = _s3_client(region_name=region_name, client=client).get_object(
        Bucket=bucket,
        Key=key,
    )
    payload = json.loads(response["Body"].read())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object at {uri}")
    return payload
