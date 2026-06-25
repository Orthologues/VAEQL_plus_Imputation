"""
Convenience re-exports for VAEQL config objects with lazy loading.

Designed to avoid double-importing `config.py` when running
`python -m VAE_Q_learning_imputation_baseline.conf.config`.
"""

__methods__ = []
__classes__ = ["FeaturesTypeDict", "VaeQlConfig", "DisentangledBetaVaeTuningConfig"]
__all__ = __methods__ + __classes__

# non-essential methods to support IDE autocompletion and dir() introspection without eager loading

def __getattr__(name: str):
    if name in __all__:
        from . import config as _config  # local import to avoid eager loading
        return getattr(_config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
