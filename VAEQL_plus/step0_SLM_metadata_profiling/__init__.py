"""Public, lazily loaded interface for Step 0 feature-type profiling."""

from __future__ import annotations

__constants__ = ["DEFAULT_MODEL_NAME", "ALTERNATIVE_MODEL_NAMES", "ALLOWED_MODEL_TYPES"]
__classes__ = ["FeatureTypeProfiler", "FeatureTypeProfilingBySLM"]
__methods__ = [
    "run_feature_type_profiling",
    "submit_feature_type_profiling_job",
]
__all__ = __constants__ + __classes__ + __methods__


def __getattr__(name: str):
    """Load Step 0 helpers only when they are requested."""
    if name in __all__:
        from . import feature_type_profiling as _feature_type_profiling

        return getattr(_feature_type_profiling, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
