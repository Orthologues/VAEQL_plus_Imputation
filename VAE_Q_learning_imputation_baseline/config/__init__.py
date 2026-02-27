"""
Convenience re-exports for VAEQL config objects with lazy loading.

Lazy import avoids double-loading `config.py` when executing
`python -m VAE_Q_learning_imputation_baseline.config.config`.
"""


__all__ = ["FeaturesTypeDict", "VaeQlConfig", "DisentangledBetaVaeTuningConfig"]


def __getattr__(name: str):
    if name in __all__:
        from . import config as _config  # local import to avoid eager loading
        return getattr(_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)

