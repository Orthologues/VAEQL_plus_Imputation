"""
Convenience re-exports for VAEQL utility helpers with lazy loading.
"""

__methods__ = ["load_dataset", "load_pandas", "load_pyspark"]
__classes__ = ["ReplayBuffer"]
__all__ = __methods__ + __classes__

# non-essential methods to support IDE autocompletion and dir() introspection without eager loading

def __getattr__(name: str):
    if name in __classes__:
        from . import dataset_loader as _dataset_loader  # local import to avoid eager loading
        return getattr(_dataset_loader, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
