"""Console output that cannot fail on a narrow character set.

On Windows, output that is redirected to a file or a pipe (as when a coding
agent runs Python) is encoded with the ANSI code page, e.g. cp1252, which has
no check mark, no em dash and no Greek letters. The built-in print() then
raises UnicodeEncodeError in the middle of a run. The modules of this package
that print import ``safe_print as print`` from here.
"""
from __future__ import annotations

import sys


def _encoding(stream) -> str:
    return getattr(stream, "encoding", None) or "ascii"


def safe_print(*args, **kwargs) -> None:
    """print() that shows characters the stream cannot encode as '?'."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        stream = kwargs.get("file") or sys.stdout
        enc = _encoding(stream)
        sep = kwargs.pop("sep", " ")
        text = sep.join(str(a) for a in args)
        print(text.encode(enc, errors="replace").decode(enc, errors="replace"), **kwargs)


def glyph(preferred: str, fallback: str) -> str:
    """``preferred`` when standard output can show it, else ``fallback``."""
    try:
        preferred.encode(_encoding(sys.stdout))
        return preferred
    except (UnicodeEncodeError, LookupError):
        return fallback
