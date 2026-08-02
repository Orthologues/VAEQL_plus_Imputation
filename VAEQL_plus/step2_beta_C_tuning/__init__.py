"""Public, lazily loaded interface for beta/C tuning."""

from __future__ import annotations

__methods__ = ["iterative_halving_search", "train_and_save_best_model"]
__all__ = __methods__


def __getattr__(name: str):
    """Load tuning entry points only when they are requested."""
    if name in __methods__:
        from . import fine_tuner as _fine_tuner

        return getattr(_fine_tuner, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
