"""Load a SHIPPED AI4BayesCode example BY NAME.

Python analogue of the R-side ``ai4bayescode_example()``: resolve the bundled
``_examples/<name>.cpp``, compile it via :func:`source`, and return the imported
pybind11 module. Use this when you want a shipped example by name; use
:func:`source` when you have an arbitrary ``.cpp`` file path.

    import AI4BayesCode
    mod = AI4BayesCode.example("GBartMultinomial")   # by name, no path, no .cpp
    m = mod.GBartMultinomial(...)                    # construct the model
"""
from __future__ import annotations

from ._blocks import _example_names, _examples_dir
from .source import source

__all__ = ["example", "list_examples", "examples_path"]


def list_examples() -> list[str]:
    """Names of the shipped examples (without the ``.cpp`` extension).

    Python analogue of R's ``ai4bayescode_list_examples()``.
    """
    return _example_names()


def examples_path(name: str | None = None) -> str:
    """Absolute path to the bundled examples directory, or to ``<name>.cpp`` in it.

    Python analogue of R's ``ai4bayescode_examples_path()``. ``example("Foo")`` is
    the same as ``source(examples_path("Foo"))``.
    """
    d = _examples_dir()
    if name is None:
        return str(d)
    fname = name if name.endswith(".cpp") else f"{name}.cpp"
    return str(d / fname)


def example(name: str, *, rebuild: bool = False, quiet: bool = False, **kwargs):
    """Compile + import a SHIPPED example BY NAME, e.g. ``example("GBartMultinomial")``.

    Python analogue of R's ``ai4bayescode_example()``. Returns the imported
    pybind11 module; construct the model with ``mod.<ClassName>(...)``. Pass a
    bare name (no directory, ``.cpp`` optional). For an arbitrary file path use
    :func:`source` instead.

    Raises ``FileNotFoundError`` (listing the available examples) if ``name`` is
    not a shipped example.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            "`name` must be a non-empty string, e.g. example('GBartMultinomial')")
    stem = name[:-4] if name.endswith(".cpp") else name
    cpp = _examples_dir() / f"{stem}.cpp"
    if not cpp.is_file():
        avail = list_examples()
        raise FileNotFoundError(
            f"Example '{stem}' not found.\n"
            f"Available examples ({len(avail)}):\n  " + ", ".join(avail))
    return source(str(cpp), rebuild=rebuild, quiet=quiet, **kwargs)
