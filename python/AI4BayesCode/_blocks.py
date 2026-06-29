"""``AI4BayesCode.blocks`` -- IPython-friendly block/example cards.

The IPython/Jupyter ``?`` operator shows an object's ``__doc__``. This module
exposes each shipped example as a small object whose ``__doc__`` *is* its card,
so::

    AI4BayesCode.blocks.BartNoise?     # IPython -> the card
    AI4BayesCode.blocks.BartNoise      # printing it shows the card too
    AI4BayesCode.blocks.<TAB>          # tab-completes all example names

This is the Python parallel to R's ``?BartNoise`` and to
``AI4BayesCode.doc("BartNoise")``.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path


def _examples_dir() -> Path:
    import importlib.resources
    return Path(str(importlib.resources.files("AI4BayesCode"))) / "_examples"


def _example_names() -> list[str]:
    d = _examples_dir()
    return sorted(p.stem for p in d.glob("*.cpp")) if d.is_dir() else []


def _render_card(name: str) -> str:
    from .doc import doc
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        doc(name)
    return buf.getvalue().rstrip("\n")


class _BlockDoc:
    """One block/example card. ``?`` shows it; printing it shows it too."""

    def __init__(self, name: str):
        self._name = name
        self.__doc__ = _render_card(name)

    def __repr__(self) -> str:
        return self.__doc__

    def show(self) -> None:
        print(self.__doc__)


class _Blocks:
    """``AI4BayesCode.blocks.<Name>`` -> the block's card (use ``?`` in IPython)."""

    def __getattr__(self, name: str) -> _BlockDoc:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in _example_names():
            return _BlockDoc(name)
        raise AttributeError(
            f"no AI4BayesCode example named {name!r}; "
            f"see  dir(AI4BayesCode.blocks)  for the list"
        )

    def __dir__(self) -> list[str]:
        return _example_names()

    def __repr__(self) -> str:
        names = _example_names()
        return (f"AI4BayesCode blocks/examples ({len(names)} available) -- use  "
                f"AI4BayesCode.blocks.<Name>?  for a card:\n  " + ", ".join(names))


blocks = _Blocks()
