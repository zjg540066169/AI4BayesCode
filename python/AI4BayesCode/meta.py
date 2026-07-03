"""Package metadata + resource-path helpers, mirroring the R ai4bayescode_* API:

    version()        # -> R ai4bayescode_version()
    include_path()   # -> R ai4bayescode_include_path()  (alias of vendored_include_path)
    list_skills()    # -> R ai4bayescode_list_skills()
"""
from __future__ import annotations

import importlib.resources
from pathlib import Path

__all__ = ["version", "include_path", "list_skills"]


def _pkg_dir() -> Path:
    return Path(str(importlib.resources.files("AI4BayesCode")))


def version() -> str:
    """Installed AI4BayesCode version string. Mirrors R ``ai4bayescode_version()``."""
    import AI4BayesCode
    return AI4BayesCode.__version__


def include_path() -> str:
    """Absolute path to the bundled C++ header tree. Mirrors R
    ``ai4bayescode_include_path()``; same target as ``vendored_include_path()``.

    Mirrors R's ``system.file()`` contract: returns ``""`` (empty string) when
    the vendored include directory does not exist."""
    from .source import vendored_include_path
    p = vendored_include_path()
    return str(p) if p.is_dir() else ""


def list_skills() -> list[str]:
    """Names of the shipped skill files. Mirrors R ``ai4bayescode_list_skills()``:
    ``start.md``, the top-level ``_skills/*.md``, and ``block_catalogue/index.md``."""
    pkg = _pkg_dir()
    d = pkg / "_skills"
    files: list[str] = []
    if (pkg / "start.md").is_file():
        files.append("start.md")
    if d.is_dir():
        files += sorted(p.name for p in d.glob("*.md"))
        if (d / "block_catalogue" / "index.md").is_file():
            files.append("block_catalogue/index.md")
    seen: set[str] = set()
    out: list[str] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out
