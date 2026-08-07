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

import base64
import binascii
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

# at runtime, `typing.TYPE_CHECKING` is always `False`, so this block is skipped during normal execution
if TYPE_CHECKING:
    from botocore.client import BaseClient


# Constants
_SSE_C_ALGORITHM = "AES256"
_SSE_C_KEY_BYTES = 32
_CHECKSUM_ALGORITHM = "SHA256"


class AWS_S3_Interface:
    def __init__(
        self,
        *,
        sse_customer_key_b64: str,
        region_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.region_name = region_name
        self.client = client
        self.sse_customer_key_b64 = sse_customer_key_b64
        self.sse_customer_key, self.sse_customer_key_md5 = self._prepare_sse_c_key(
            sse_customer_key_b64
        )

    # Boilerplate methods
    @staticmethod
    def _prepare_sse_c_key(sse_customer_key_b64: str) -> tuple[bytes, str]:
        """Validate a base64 key, decode it for boto3, and calculate its MD5."""
        if not isinstance(sse_customer_key_b64, str) or not sse_customer_key_b64:
            raise ValueError("sse_customer_key_b64 must be a non-empty base64 string")
        try:
            key_bytes = base64.b64decode(sse_customer_key_b64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("sse_customer_key_b64 must be valid base64") from exc
        if len(key_bytes) != _SSE_C_KEY_BYTES:
            raise ValueError("sse_customer_key_b64 must decode to exactly 32 bytes")
        # Convert the raw MD5 digest to the Base64 format required by S3.
        key_md5 = base64.b64encode(
            # .digest() returns the raw binary MD5 hash
            hashlib.md5(key_bytes, usedforsecurity=False).digest()
        ).decode("ascii")
        # key_bytes is the 32-byte AES-256 customer key.
        # key_md5 is the key's Base64-encoded MD5 digest,
        # represented as a Python string for S3 request validation.
        return key_bytes, key_md5

    def _sse_c_args(self) -> dict[str, Any]:
        """Return the SSE-C headers required for each S3 request."""
        return {
            "SSECustomerAlgorithm": _SSE_C_ALGORITHM,
            # boto3 expects the customer key as its original Base64 string.
            "SSECustomerKey": self.sse_customer_key_b64,
            "SSECustomerKeyMD5": self.sse_customer_key_md5,
        }

    def _s3_client(self) -> "BaseClient":
        """Return an injected client or lazily create a boto3 S3 client.

        S3 client API:
        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
        """
        if self.client is not None:
            return self.client
        try:
            import boto3
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "boto3 is required for AWS S3 operations; install the project "
                "environment first."
            ) from exc
        self.client = boto3.client("s3", region_name=self.region_name)
        return self.client

    @staticmethod
    def parse_s3_uri(uri: str) -> tuple[str, str]:
        """Split an ``s3://bucket/key`` URI into bucket and object key."""
        parsed = urlparse(uri)
        # Require the S3 scheme, a non-empty bucket, and a non-empty object key.
        if parsed.scheme != "s3" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(f"Expected an S3 URI in the form s3://bucket/key, got {uri!r}")
        return parsed.netloc, parsed.path.lstrip("/")

    @staticmethod
    def build_s3_uri(bucket: str, key: str) -> str:
        """Build an S3 URI after validating its bucket and object key."""
        bucket = bucket.strip()
        key = key.lstrip("/")
        if not bucket or not key:
            raise ValueError("bucket and key must be non-empty")
        return f"s3://{bucket}/{key}"

    # Main operational methods
    def upload_file(
        self,
        file_path: str | Path,
        *,
        bucket: str,
        key: str,
        extra_args: Mapping[str, Any] | None = None,
    ) -> str:
        """Upload a file with mandatory SSE-C protection and return its S3 URI."""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        upload_args: dict[str, Any] = {}
        if extra_args:
            upload_args.update(dict(extra_args))
        # Apply the protection setting last so callers cannot downgrade the write.
        upload_args.update(self._sse_c_args())
        upload_args["ChecksumAlgorithm"] = _CHECKSUM_ALGORITHM
        self._s3_client().upload_file(
            str(path),
            bucket,
            key.lstrip("/"),
            ExtraArgs=upload_args,
        )
        return self.build_s3_uri(bucket, key)

    def download_file(self, uri: str, destination: str | Path) -> Path:
        """Download an SSE-C object to a local path and return the resulting path."""
        bucket, key = self.parse_s3_uri(uri)
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        self._s3_client().download_file(
            bucket,
            key,
            str(destination_path),
            ExtraArgs=self._sse_c_args(),
        )
        return destination_path

    def write_json_metadata(
        self,
        payload: Mapping[str, Any],
        bucket: str,
        key: str,
    ) -> str:
        """Write JSON metadata to S3 with SSE-C and a SHA-256 checksum."""
        body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        args: dict[str, Any] = self._sse_c_args()
        args.update(
            {
                "Body": body,
                "Bucket": bucket,
                "Key": key.lstrip("/"),
                "ContentType": "application/json",
                "ChecksumAlgorithm": _CHECKSUM_ALGORITHM,
            }
        )
        self._s3_client().put_object(**args)
        return self.build_s3_uri(bucket, key)

    def get_json(self, uri: str) -> dict[str, Any]:
        """Read an SSE-C JSON object from S3."""
        bucket, key = self.parse_s3_uri(uri)
        response = self._s3_client().get_object(
            Bucket=bucket,
            Key=key,
            **self._sse_c_args(),
        )
        payload = json.loads(response["Body"].read())
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object at {uri}")
        return payload
