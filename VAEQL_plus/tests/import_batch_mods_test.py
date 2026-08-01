#########################################################
# Author： Jiawei Zhao
# Email1: jiz@sdu.dk
# Email2: jwz.student.bmc.lu@gmail.com
# Date: 2026-08-01
# Description: Batch import smoke tests for VAEQL_plus modules.
# Development: Mainly written with GPT-5.5 Medium/GPT-5.6 Luna-XHigh on Codex, with Jiawei Zhao's human
# review and revisions.
#########################################################

from __future__ import annotations

import importlib
import pkgutil
import sys
from pathlib import Path

import pytest

PKG_NAME = "VAEQL_plus"
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Add fully-qualified modules here if a module is intentionally not import-safe
# in a test environment due to side effects or required runtime context.
SKIP_MODULES = {
    # "VAEQL_plus.some_module",
}


def iter_pkg_mods(package_name: str):
    pkg = importlib.import_module(package_name)
    if not hasattr(pkg, "__path__"):
        return
    for module_info in pkgutil.walk_packages(pkg.__path__, prefix=f"{package_name}."):
        mod_name = module_info.name
        if mod_name in SKIP_MODULES:
            continue
        yield mod_name


ALL_MODULES = sorted(set(iter_pkg_mods(PKG_NAME)))


class Module_import_test:
    """Batch smoke tests for importing all modules under the `VAEQL_plus` package."""

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_import_all_modules(self, module_name: str) -> None:
        """Each discovered module should be importable."""
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            # Keep batch import tests robust when optional dependencies are absent.
            missing_pkg = exc.name or "unknown"
            pytest.skip(f"Optional dependency missing for {module_name}: {missing_pkg}")

    @pytest.mark.parametrize("module_name", ALL_MODULES)
    def test_module_has_file(self, module_name: str) -> None:
        """Imported module should map to an existing file when `__file__` is present."""
        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing_pkg = exc.name or "unknown"
            pytest.skip(f"Optional dependency missing for {module_name}: {missing_pkg}")
        module_file = getattr(mod, "__file__", None)
        if module_file is not None:
            assert Path(module_file).exists()
