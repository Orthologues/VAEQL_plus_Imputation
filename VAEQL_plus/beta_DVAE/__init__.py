"""
Public package exports for disentangled beta-VAE.

Author: Jiawei Zhao (jiz@imada.sdu.dk)
Date: 2026-04-01
"""

from importlib import import_module

__module_exports__ = []
__class_exports__ = [
    "BetaGausMixedDVAE",
    "BetaGausMixedDVAETrainer",
    "BetaGausMixedDVAEUtils"
]
__method_exports__ = ["iterative_halving_search", "train_and_save_best_model"]

__all__ = __module_exports__ + __class_exports__ + __method_exports__

def __getattr__(name: str):
    if name in __module_exports__:
        return import_module(f".{name}", __name__)
    if name == "BetaGausMixedDVAE":
        return getattr(import_module(".torch_nn", __name__), name)
    if name == "BetaGausMixedDVAETrainer":
        return getattr(import_module(".lightning_mod", __name__), name)
    if name == "BetaGausMixedDVAEUtils":
        return getattr(import_module(".utils", __name__), name)
    if name in __method_exports__:
        mod = import_module(".lightning_mod", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
