#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-08-01
# Description: Small, import-safe AWS Batch helpers for VAEQL training steps.
# Development: Mainly written with GPT-5.5 Medium/GPT-5.6 Luna-XHigh on Codex, with Jiawei Zhao's human
# review and revisions.
#########################################################

"""AWS Batch helpers for VAEQL training steps."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any


# Documentation: https://docs.aws.amazon.com/batch/latest/userguide/specifying-sensitive-data-secrets.html
# Local setup: store one Base64-encoded 32-byte AES key in an ignored env file:
# VAEQL_S3_SSE_CUSTOMER_KEY_B64=<base64-encoded-key>
# Load it before submitting the Batch job; never commit or print the key.
# Example:
# .env.local: VAEQL_S3_SSE_CUSTOMER_KEY_B64=<base64-encoded-key>
# shell: set -a; source .env.local; set +a
SSE_CUSTOMER_KEY_ENV_VAR = "VAEQL_S3_SSE_CUSTOMER_KEY_B64"
# Example step module paths: `VAEQL_plus.step0_SLM_metadata_profiling.feature_type_profiling` and `VAEQL_plus.step1_preprocessing`.
_STEP_MODULE_PATTERN = re.compile(
    r"(?:[A-Za-z_]\w*\.)*step[0-9]\d*_[A-Za-z0-9_]+(?:\.[A-Za-z_]\w*)*"
)


class AWS_Batch_Interface:
    """Submit jobs to a previously provisioned AWS Batch environment.

    The AWS account, IAM roles, networking, compute environment, job queue, and
    job definition must be configured before this interface submits a job.
    """

    def __init__(
        self,
        *,
        region_name: str | None = None,
        profile_name: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.region_name = region_name
        self.profile_name = profile_name
        self.client = client

    # Boilerplate methods
    def _batch_client(self) -> Any:
        """Return an injected client or lazily create a boto3 Batch client."""
        if self.client is not None:
            return self.client
        try:
            import boto3
            from botocore.exceptions import NoRegionError, ProfileNotFound
        except ModuleNotFoundError as exc:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "boto3 is required for AWS Batch operations; install the project "
                "environment before submitting a job."
            ) from exc
        try:
            session = boto3.Session(
                profile_name=self.profile_name,
                region_name=self.region_name,
            )
            self.client = session.client("batch")
        except (NoRegionError, ProfileNotFound) as exc:  # pragma: no cover - local setup
            raise RuntimeError(
                "AWS Batch client setup failed; configure a valid AWS profile or "
                "IAM role and region before submitting a job. See "
                "'https://docs.aws.amazon.com/batch/latest/userguide/get-set-up-for-aws-batch.html'"
            ) from exc
        return self.client

    @staticmethod
    def build_step_command(
        step_module: str,
        *,
        arguments: Sequence[str] = (),
    ) -> list[str]:
        """Build the default command for a Python ``stepX_X`` module."""
        if not _STEP_MODULE_PATTERN.fullmatch(step_module):
            raise ValueError("step_module must be a dotted VAEQL step module path")
        return ["python", "-m", step_module, *[str(argument) for argument in arguments]]

    @staticmethod
    def build_submit_job_request(
        *,
        job_name: str,
        job_queue: str,
        job_definition: str,
        command: Sequence[str] | None = None,
        environment: Mapping[str, str] | None = None,
        parameters: Mapping[str, str] | None = None,
        depends_on: Sequence[str] = (),
        array_size: int | None = None,
        retry_attempts: int | None = None,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Build a validated ``batch.submit_job`` request without network access."""
        for name, value in (
            ("job_name", job_name),
            ("job_queue", job_queue),
            ("job_definition", job_definition),
        ):
            if not value:
                raise ValueError(f"{name} must be non-empty")

        request: dict[str, Any] = {
            "jobName": job_name,
            "jobQueue": job_queue,
            "jobDefinition": job_definition,
        }

        container_overrides: dict[str, Any] = {}
        if command is not None:
            if not command:
                raise ValueError("command must contain at least one argument")
            container_overrides["command"] = [str(argument) for argument in command]
        if environment:
            container_overrides["environment"] = [
                {"name": str(name), "value": str(value)}
                for name, value in environment.items()
            ]
        if container_overrides:
            request["containerOverrides"] = container_overrides

        if parameters:
            request["parameters"] = {str(name): str(value) for name, value in parameters.items()}
        if depends_on:
            request["dependsOn"] = [{"jobId": str(job_id)} for job_id in depends_on]
        # AWS Batch array jobs must contain 2 to 10,000 child jobs; 
        # an array of one is not considered a valid array job
        if array_size is not None:
            if array_size < 2 or array_size > 10_000:
                raise ValueError("array_size must be between 2 and 10000")
            request["arrayProperties"] = {"size": int(array_size)}
        if retry_attempts is not None:
            if retry_attempts < 1 or retry_attempts > 10:
                raise ValueError("retry_attempts must be between 1 and 10")
            request["retryStrategy"] = {"attempts": int(retry_attempts)}
        if timeout_seconds is not None:
            if timeout_seconds <= 0:
                raise ValueError("timeout_seconds must be greater than zero")
            request["timeout"] = {"attemptDurationSeconds": int(timeout_seconds)}

        return request

    # Main operational methods
    def submit_training_step(
        self,
        *,
        job_name: str,
        job_queue: str,
        job_definition: str,
        step_module: str,
        arguments: Sequence[str] = (),
        environment: Mapping[str, str] | None = None,
        sse_customer_key_b64: str | None = None,
        parameters: Mapping[str, str] | None = None,
        depends_on: Sequence[str] = (),
        array_size: int | None = None,
        retry_attempts: int | None = 1,
        timeout_seconds: int | None = None,
    ) -> dict[str, Any]:
        """Submit one VAEQL ``stepX_X`` module to AWS Batch.

        If ``sse_customer_key_b64`` is omitted, the launcher environment is
        checked for ``SSE_CUSTOMER_KEY_ENV_VAR`` and the value is forwarded to
        the container environment.
        """
        job_environment = dict(environment or {})
        if sse_customer_key_b64 is None:
            sse_customer_key_b64 = os.environ.get(SSE_CUSTOMER_KEY_ENV_VAR)
        if sse_customer_key_b64 is not None:
            job_environment[SSE_CUSTOMER_KEY_ENV_VAR] = sse_customer_key_b64
        request = self.build_submit_job_request(
            job_name=job_name,
            job_queue=job_queue,
            job_definition=job_definition,
            command=self.build_step_command(step_module, arguments=arguments),
            environment=job_environment or None,
            parameters=parameters,
            depends_on=depends_on,
            array_size=array_size,
            retry_attempts=retry_attempts,
            timeout_seconds=timeout_seconds,
        )
        return self._batch_client().submit_job(**request)

    def describe_jobs(self, job_ids: Sequence[str]) -> dict[str, Any]:
        """Return AWS Batch state for the supplied job IDs."""
        ids = [str(job_id) for job_id in job_ids if str(job_id)]
        if not ids:
            raise ValueError("job_ids must contain at least one job ID")
        return self._batch_client().describe_jobs(jobs=ids)

    def cancel_job(
        self,
        job_id: str,
        *,
        reason: str = "Cancelled by VAEQL job controller",
    ) -> dict[str, Any]:
        """Cancel a submitted job with an auditable reason."""
        if not job_id:
            raise ValueError("job_id must be non-empty")
        return self._batch_client().cancel_job(
            jobId=job_id,
            reason=reason,
        )
