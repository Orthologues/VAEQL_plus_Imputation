"""Public, lazily loaded interface for the preprocessing step."""

from __future__ import annotations

__classes__ = ["FeaturePreprocessor", "OrderedFeature"]
__all__ = __classes__


def __getattr__(name: str):
    """Load preprocessing objects only when they are requested."""
    if name in __classes__:
        from . import feat_preprocessor as _feat_preprocessor

        return getattr(_feat_preprocessor, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
