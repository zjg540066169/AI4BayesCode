"""CRAN-style installation of contributed blocks from the hub registry.

Mirror of the R `ai4bayescode_install_block` family. The registry
(github.com/zjg540066169/AI4BayesCode-hub/registry/) is the "CRAN repo": each
subfolder is a reviewed, pre-validated bundle (``manifest.dcf`` + ``<block>.hpp``
+ ``skills/`` + ``test_<block>.cpp`` + optional ``examples/`` + ``vendor/``).
``install_block`` downloads a bundle into a per-user block library that
``AI4BayesCode.source()`` adds to the compile ``-I`` path, so
``#include "<block>.hpp"`` (and its vendored headers) resolve.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import urllib.request

_HUB_REPO = "zjg540066169/AI4BayesCode-hub"
_HUB_REF = "main"


# ---- per-user block library -------------------------------------------------

def _blocks_dir() -> str:
    # User-global, language-agnostic block store SHARED with the R package (same
    # AI4BAYESCODE_DATA_HOME override). One dir across R / Python / C++ and all
    # projects, so a block installed once is found everywhere.
    base = os.environ.get("AI4BAYESCODE_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".AI4BayesCode")
    return os.path.join(base, "blocks_download")


def blocks_path(name: str | None = None) -> str:
    """Path to the per-user contributed-block library (or one block under it)."""
    d = _blocks_dir()
    return d if name is None else os.path.join(d, name)


def _block_include_flags() -> list[str]:
    """``-I`` flags for every contributed block (block dir + each vendored dep dir).

    Covers BOTH tiers: user-global installed blocks (``~/.AI4BayesCode/blocks_download/``)
    AND project-local blocks (``./blocks_local/``, relative to the working directory).
    """
    flags: list[str] = []
    seen: set[str] = set()
    # project-local FIRST, then user-global; DEDUP by block name so a locally
    # developed block shadows a same-named downloaded copy (develop -> upload ->
    # re-download) — local is the live working copy, download the published snapshot.
    for bdir in (os.path.join(os.getcwd(), "blocks_local"), _blocks_dir()):
        if not os.path.isdir(bdir):
            continue
        for b in sorted(os.listdir(bdir)):
            bp = os.path.join(bdir, b)
            if not os.path.isdir(bp) or b in seen:
                continue
            seen.add(b)
            flags.append(f"-I{bp}")
            vdir = os.path.join(bp, "vendor")
            if os.path.isdir(vdir):
                for v in sorted(os.listdir(vdir)):
                    vp = os.path.join(vdir, v)
                    if os.path.isdir(vp):
                        flags.append(f"-I{vp}")
    return flags


# ---- registry access --------------------------------------------------------

def _raw_url(path: str) -> str:
    return f"https://raw.githubusercontent.com/{_HUB_REPO}/{_HUB_REF}/{path}"


def _tree_url() -> str:
    return f"https://api.github.com/repos/{_HUB_REPO}/git/trees/{_HUB_REF}?recursive=1"


def _fetch(url: str, binary: bool = False):
    req = urllib.request.Request(url, headers={"User-Agent": "AI4BayesCode-install_block"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (trusted host)
        data = r.read()
    return data if binary else data.decode("utf-8")


def _hub_tree() -> list[dict]:
    try:
        tr = json.loads(_fetch(_tree_url()))
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not reach the AI4BayesCode block registry: {e}") from e
    if tr.get("truncated"):
        print("warning: registry listing truncated by GitHub; some blocks may be hidden.",
              file=sys.stderr)
    return tr.get("tree", [])


def _read_dcf(text: str) -> dict:
    """Minimal Debian-control-format parser (key: value, continuation lines)."""
    out: dict[str, str] = {}
    key = None
    for line in text.splitlines():
        if not line.strip():
            continue
        if line[0].isspace() and key is not None:
            out[key] += " " + line.strip()
        elif ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            out[key] = val.strip()
    return out


def _core_block_names() -> set[str]:
    try:
        from .sourceCpp import _pkg_root  # type: ignore
        inc = os.path.join(str(_pkg_root()), "include", "AI4BayesCode")
    except Exception:  # noqa: BLE001
        root = os.path.dirname(os.path.abspath(__file__))
        inc = os.path.join(root, "_vendored_include", "AI4BayesCode")
    if not os.path.isdir(inc):
        return set()
    return {f[:-4] for f in os.listdir(inc) if f.endswith("_block.hpp")}


def _check_core_dep(depends: str, name: str) -> None:
    import re
    m = re.search(r"core\s*\(\s*>=\s*([0-9.]+)\s*\)", depends or "")
    if not m:
        return
    need = m.group(1)
    try:
        from . import __version__ as have
    except Exception:  # noqa: BLE001
        return

    def _v(s):
        # Parse each dot-separated segment's LEADING-INTEGER run so mixed
        # segments keep the patch level ("1.2.3a" -> (1, 2, 3)), matching R's
        # utils::compareVersion (which compares numeric prefixes segment-wise).
        # The old `if x.isdigit()` silently DROPPED any non-numeric segment,
        # collapsing "1.2.3a" -> (1, 2) and losing the patch level.
        out = []
        for seg in s.split("."):
            m = re.match(r"\d+", seg)
            out.append(int(m.group(0)) if m else 0)
        return tuple(out)
    if _v(have) < _v(need):
        print(f"warning: block '{name}' wants AI4BayesCode core (>= {need}) "
              f"but you have {have}; it may not compile.", file=sys.stderr)


# ---- public CRAN-family API -------------------------------------------------

def available_blocks() -> list[str]:
    """Blocks available in the hub registry (like ``available.packages()``)."""
    tree = _hub_tree()
    names = []
    for e in tree:
        p = e.get("path", "")
        if e.get("type") == "tree" and p.startswith("registry/") and p.count("/") == 1:
            names.append(p.split("/", 1)[1])
    return sorted(names)


def installed_blocks() -> list[str]:
    """Contributed blocks already installed (like ``installed.packages()``)."""
    bdir = _blocks_dir()
    if not os.path.isdir(bdir):
        return []
    return sorted(b for b in os.listdir(bdir) if os.path.isdir(os.path.join(bdir, b)))


def install_block(name: str, force: bool = False, quiet: bool = False) -> str:
    """Install a contributed block from the hub registry (like ``install.packages()``).

    Downloads a reviewed bundle into the per-user block library; afterwards the
    header + vendored deps are on ``AI4BayesCode.source()``'s include path.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty block name")
    dest = blocks_path(name)
    # A project-local block of the same name shadows the download (local > download):
    # the develop -> upload -> re-download case. Flag it so the user is not surprised
    # that edits to ./blocks_local/ still win over the freshly downloaded snapshot.
    if not quiet and os.path.isdir(os.path.join(os.getcwd(), "blocks_local", name)):
        print(f"Note: a project-local block '{name}' exists in ./blocks_local/ and will take "
              f"PRECEDENCE over this download (local > download). The download is kept as the "
              f"published snapshot; remove the local copy to use it.")
    if os.path.isdir(dest) and not force:
        if not quiet:
            print(f"Block '{name}' is already installed ({dest}).\n"
                  f"  Use force=True to reinstall, or AI4BayesCode.remove_block('{name}').")
        return dest
    # 1. manifest (also the existence check)
    try:
        man_txt = _fetch(_raw_url(f"registry/{name}/manifest.dcf"))
    except Exception:  # noqa: BLE001
        try:
            avail = available_blocks()
        except Exception:  # noqa: BLE001
            avail = []
        extra = f"\nAvailable ({len(avail)}): {', '.join(avail)}" if avail else ""
        raise ValueError(f"Block '{name}' is not in the registry.{extra}") from None
    man = _read_dcf(man_txt)
    # 2. reserved core name
    if name in _core_block_names():
        raise ValueError(f"'{name}' is a reserved core block name.")
    # 3. core-version dependency
    _check_core_dep(man.get("Depends", ""), name)
    # 4. download every file in registry/<name>/
    if not quiet:
        print(f"Installing '{name}' from the AI4BayesCode registry ...")
    prefix = f"registry/{name}/"
    files = [e["path"] for e in _hub_tree()
             if e.get("type") == "blob" and e.get("path", "").startswith(prefix)]
    if not files:
        raise RuntimeError(f"Registry bundle for '{name}' is empty.")
    tmp = tempfile.mkdtemp(prefix=f"ai4b_blk_{name}_")
    try:
        for f in files:
            out = os.path.join(tmp, f[len(prefix):])
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "wb") as fh:
                fh.write(_fetch(_raw_url(f), binary=True))
        # 5. swap into the library
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.move(tmp, dest)
    finally:
        if os.path.isdir(tmp):
            shutil.rmtree(tmp, ignore_errors=True)
    if not quiet:
        _report(man, dest)
    return dest


def remove_block(name: str) -> bool:
    """Remove an installed contributed block (like ``remove.packages()``)."""
    dest = blocks_path(name)
    if not os.path.isdir(dest):
        print(f"Block '{name}' is not installed.")
        return False
    # A symlinked block dir must be unlinked, not rmtree'd: shutil.rmtree raises
    # on a symlink to a directory, whereas R's unlink(recursive = TRUE) removes
    # it. Guard to match R behavior (and avoid deleting through the symlink).
    if os.path.islink(dest):
        os.unlink(dest)
    else:
        shutil.rmtree(dest)
    print(f"Removed block '{name}'.")
    return True


def _report(man: dict, dest: str) -> None:
    def line(label, key):
        v = man.get(key, "")
        if v:
            print(f"  {label}: {v}")
    print(f"Installed block '{man.get('Block', '?')}':")
    line("version", "Version")
    line("title", "Title")
    line("license", "License")
    line("vendored", "Vendored")
    print(f"  location: {dest}")
    ex = man.get("Example", "")
    if ex:
        print(f"  example:  see {os.path.join(dest, ex)}  (compile with AI4BayesCode.source())")
    print("  -> usable now: the header + vendored deps are on the source() include path.")
