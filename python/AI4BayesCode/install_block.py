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
import warnings
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


def _check_block_name(name: str) -> str:
    """A block name is ONE directory name under the block library.

    Never a path, never empty. ``os.path.join(d, "")`` returns the library root,
    which passes ``isdir()`` and made ``remove_block("")`` delete every
    installed block; ``"../x"`` escapes the library and an ABSOLUTE path makes
    ``os.path.join`` discard the library prefix entirely. Validated here, at the
    single point install / remove / lookup all resolve through.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(
            f"block name must be a non-empty string; got {name!r}")
    if os.path.isabs(name) or "/" in name or "\\" in name or name in (".", ".."):
        raise ValueError(
            f"block name must be a plain block name, not a path: {name!r}. "
            "Call AI4BayesCode.installed_blocks() to see what is installed.")
    return name


def blocks_path(name: str | None = None) -> str:
    """Path to the per-user contributed-block library (or one block under it)."""
    d = _blocks_dir()
    return d if name is None else os.path.join(d, _check_block_name(name))


#: The manifest fields skills/block_design_skills/skill.md Sec.3 requires of a
#: contributed block. A bundle missing any of them, or missing its header, was
#: previously accepted in silence and put on the compile -I path anyway -- an
#: empty directory dropped into the library reported as "installed".
_REQUIRED_MANIFEST_FIELDS = (
    "Block", "Version", "Title", "Description", "Author", "License",
    "EngineKind", "ConstraintKinds", "RoutingKey", "SelectWhen", "Skill",
    "Tests", "Depends",
)


def validate_bundle(path: str) -> list[str]:
    """Problems with the block bundle at `path`; empty list means it is well formed.

    Checks what can be checked without compiling: the manifest parses, carries
    every required field, and names the block consistently with the directory
    and the header file that has to exist.
    """
    problems: list[str] = []
    name = os.path.basename(os.path.normpath(path))
    if not os.path.isdir(path):
        return [f"{path} is not a directory"]
    man_path = os.path.join(path, "manifest.dcf")
    if not os.path.isfile(man_path):
        problems.append("manifest.dcf is missing")
    else:
        try:
            with open(man_path, encoding="utf-8", errors="replace") as fh:
                man = _read_dcf(fh.read())
        except Exception as e:  # noqa: BLE001
            return [f"manifest.dcf could not be parsed ({e})"]
        missing = [f for f in _REQUIRED_MANIFEST_FIELDS if not man.get(f)]
        if missing:
            problems.append("manifest.dcf is missing required field(s): "
                            + ", ".join(missing))
        declared = man.get("Block", "")
        if declared and declared != name:
            problems.append(
                f"manifest.dcf says Block: {declared!r} but the directory is "
                f"{name!r} -- they must match")
    if not os.path.isfile(os.path.join(path, f"{name}.hpp")):
        problems.append(f"{name}.hpp is missing (the header the block is used by)")
    return problems


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
            # A malformed bundle used to go on the -I path in silence -- an
            # empty directory, or one with a manifest missing 12 of its 13
            # required fields, was indistinguishable from a real block until
            # the compile failed with a missing-header error.
            problems = validate_bundle(bp)
            if problems:
                warnings.warn(
                    f"skipping block bundle {bp!r}: " + "; ".join(problems),
                    RuntimeWarning, stacklevel=3)
                continue
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


def _github_token() -> str:
    return os.environ.get("GITHUB_PAT") or os.environ.get("GITHUB_TOKEN") or ""


def _fetch(url: str, binary: bool = False):
    # The GitHub API allows 60 unauthenticated requests per hour PER IP, which a
    # shared or institutional address can exhaust without the user doing anything
    # unusual. A token raises that to 5000/hr.
    headers = {"User-Agent": "AI4BayesCode-install_block"}
    tok = _github_token()
    if tok:
        headers["Authorization"] = f"Bearer {tok}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 (trusted host)
        data = r.read()
    return data if binary else data.decode("utf-8")


def _hub_tree() -> list[dict]:
    try:
        tr = json.loads(_fetch(_tree_url()))
    except Exception as e:  # noqa: BLE001
        hint = ""
        if not _github_token() and ("403" in str(e) or "rate limit" in str(e).lower()):
            hint = ("\n  This looks like GitHub's unauthenticated rate limit "
                    "(60 requests/hour per IP). Set GITHUB_PAT to a personal "
                    "access token to raise it.")
        raise RuntimeError(
            f"Could not reach the AI4BayesCode block registry: {e}{hint}") from e
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
    # The core block headers ship inside the installed package.
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


def installed_blocks(tier: str = "all") -> list[str]:
    """Contributed blocks visible to the compiler (like ``installed.packages()``).

    Two tiers reach the compile ``-I`` path: project-local blocks under
    ``./blocks_local/`` and downloaded ones under the block library. This used
    to report only the download tier, so a user with four live blocks under
    ``blocks_local/`` was told ``[]`` while those four were being compiled in.

    ``tier`` is ``"all"`` (default), ``"local"`` or ``"download"``. A block
    present in both is reported once: local shadows download, matching what
    the compile path does.
    """
    if tier not in ("all", "local", "download"):
        raise ValueError(
            f"tier must be 'all', 'local' or 'download'; got {tier!r}")
    out: list[str] = []
    seen: set[str] = set()
    dirs = []
    if tier in ("all", "local"):
        dirs.append(os.path.join(os.getcwd(), "blocks_local"))
    if tier in ("all", "download"):
        dirs.append(_blocks_dir())
    for bdir in dirs:
        if not os.path.isdir(bdir):
            continue
        for b in sorted(os.listdir(bdir)):
            if b in seen or not os.path.isdir(os.path.join(bdir, b)):
                continue
            seen.add(b); out.append(b)
    return sorted(out)


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
    except Exception as man_err:  # noqa: BLE001
        # The fetch failing does NOT mean the block is absent -- being offline,
        # behind a proxy, or hitting a GitHub outage fails identically. The
        # registry INDEX settles it: if that is reachable, the block is really
        # not there; if it is not, this is a connectivity problem and saying
        # "not in the registry" sends the user hunting for a typo.
        try:
            avail = available_blocks()
        except Exception as idx_err:  # noqa: BLE001
            raise ConnectionError(
                f"Cannot reach the AI4BayesCode block registry, so whether "
                f"'{name}' exists is unknown. Check your network connection "
                f"(and any proxy), then retry."
                f"\n  manifest fetch: {man_err}"
                f"\n  registry index: {idx_err}") from None
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
    _check_block_name(name)   # explicit, so remove_block(None) reports the
                              # name problem rather than the containment one
    dest = blocks_path(name)
    # Defence in depth: whatever the name resolved to, it has to be a direct
    # child of the block library before anything is deleted recursively.
    root = os.path.realpath(_blocks_dir())
    here = os.path.realpath(dest)
    if os.path.dirname(here) != root or here == root:
        raise ValueError(
            f"refusing to remove {dest!r}: not a block directory under {root}")
    if not os.path.isdir(dest):
        # A project-local block IS installed -- it is on the compile include
        # path and installed_blocks() lists it -- but it lives in the user's own
        # project, not the download cache, so this function does not delete it.
        # Saying "not installed" would flatly contradict installed_blocks().
        local_dir = os.path.join("blocks_local", name)
        if os.path.isdir(local_dir):
            print(f"Block '{name}' is a project-local block at "
                  f"{os.path.abspath(local_dir)}.\n  remove_block() manages the "
                  f"download cache only; delete that directory to remove it.")
            return False
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
