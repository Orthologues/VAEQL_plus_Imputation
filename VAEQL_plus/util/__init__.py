"""
Convenience re-exports for VAEQL utility helpers with lazy loading.
"""

from importlib import import_module

## CHANGELOG: add AWS interface classes to the static set of classes here
__methods__ = ["load_dataset", "load_pandas", "load_pyspark"]
__classes__ = [
    "ReplayBuffer",
    "FeaturePreprocessor",
    "AWS_S3_Interface",
    "AWS_Batch_Interface",
]
__all__ = __methods__ + __classes__

## QUESTIONLOG: so the feat_preprocessor class could be imported from both `.step1_preprocessing.FeaturePreprocessor` and `utils.FeaturePreprocessor`?
_EXPORT_MODULES = {
    "load_dataset": ".dataset_loader",
    "load_pandas": ".dataset_loader",
    "load_pyspark": ".dataset_loader",
    "ReplayBuffer": ".replay_buffer",
    "FeaturePreprocessor": "..step1_preprocessing.feat_preprocessor",
    "AWS_S3_Interface": ".AWS_S3_interface",
    "AWS_Batch_Interface": ".AWS_Batch_interface",
}

# non-essential methods to support IDE autocompletion and dir() introspection without eager loading


def __getattr__(name: str):
    if name in _EXPORT_MODULES:
        module = import_module(_EXPORT_MODULES[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
