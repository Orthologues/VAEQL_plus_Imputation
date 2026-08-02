"""Public, lazily loaded interface for Step 0 feature-type profiling."""

from __future__ import annotations

__all__ = [
    "DEFAULT_MODEL_NAME",
    "JOB_MODULE",
    "MinistralFeatureTypeProfiler",
    "build_profile_manifest",
    "parse_profile_response",
    "run_feature_type_profiling",
    "submit_feature_type_profiling_job",
]


def __getattr__(name: str):
    """Load Step 0 helpers only when they are requested."""
    if name in __all__:
        from . import feature_type_profiling as _feature_type_profiling

        return getattr(_feature_type_profiling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
