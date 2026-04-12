"""
Convenience re-exports for disentangled beta-VAE API objects with lazy loading.

Designed to avoid eager-import side effects while allowing direct package-level imports.
"""

from importlib import import_module

__modules__ = ["DisentangledBetaVAE", "DisentangledBetaVaeUtil"]
__methods__ = ["iterative_halving_search", "train_and_save_best_model"]
__classes__ = []
__all__ = __modules__ + __methods__ + __classes__


def __getattr__(name: str):
    if name in __modules__:
        return import_module(f".{name}", __name__)
    if name in __methods__:
        vae_module = import_module(".DisentangledBetaVAE", __name__)
        return getattr(vae_module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + __all__)
