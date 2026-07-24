"""Contract test: the documented public API surface of `nba_edge` is stable.

This guards against accidentally breaking imports for downstream consumers of
the published package (the CLV dashboard, line-shop tooling, etc.). It is meant
to fail loudly if a public symbol is renamed, dropped, or omitted from
``__all__``, or if ``__version__`` ever stops being a valid version string.
"""
from __future__ import annotations

import inspect
import re

import nba_edge


def test_version_is_valid_string() -> None:
    assert isinstance(nba_edge.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", nba_edge.__version__), nba_edge.__version__


def test_all_is_sequence_of_public_strings() -> None:
    assert isinstance(nba_edge.__all__, (list, tuple))
    assert all(isinstance(name, str) for name in nba_edge.__all__)
    # No duplicates and no dunder/private names leaked into the contract.
    assert len(set(nba_edge.__all__)) == len(nba_edge.__all__)
    assert all(not name.startswith("_") for name in nba_edge.__all__)


def test_every_exported_name_is_importable_and_comes_from_package() -> None:
    for name in nba_edge.__all__:
        obj = getattr(nba_edge, name, None)
        assert obj is not None, f"{name} listed in __all__ but not importable"
        module = getattr(obj, "__module__", "")
        assert module.startswith("nba_edge"), (
            f"{name} is exported but originates from {module!r}, not nba_edge"
        )


def test_no_public_symbol_missing_from_all() -> None:
    # Any function/class defined inside the nba_edge package and exposed at the
    # top level must be declared in __all__, so the public contract is explicit
    # and nothing leaks by accident.
    expected = set(nba_edge.__all__)
    for name in dir(nba_edge):
        if name.startswith("_"):
            continue
        obj = getattr(nba_edge, name)
        if not (inspect.isfunction(obj) or inspect.isclass(obj)):
            continue
        if getattr(obj, "__module__", "").startswith("nba_edge"):
            assert name in expected, f"public symbol {name!r} is missing from __all__"
