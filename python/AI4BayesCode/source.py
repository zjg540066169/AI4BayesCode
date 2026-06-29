"""Self-contained pybind11 source/compile for AI4BayesCode ``.cpp`` files.

Mirror of the R-side ``ai4bayescode_source()``. After ``pip install AI4BayesCode``::

    import AI4BayesCode
    mod = AI4BayesCode.source("MyModel.cpp")   # file path ...
    mod = AI4BayesCode.source(cpp_source_string) # ... or a string
    m = mod.MyModel(y, X, seed=1, keep_history=True)

compiles a generated sampler against the header tree **vendored inside the
installed package** (``_vendored_include/``) -- there is **no dependency on the
location of an AI4BayesCode checkout**. The only flag that differs from the R
build is the module macro (``-DAI4BAYESCODE_PYBIND_MODULE`` instead of
``-DAI4BAYESCODE_RCPP_MODULE``).

Standalone Armadillo: unlike R (which gets Armadillo via RcppArmadillo), the
Python build needs a standalone Armadillo header tree. It is discovered from
``ARMADILLO_INCLUDE_DIR`` or common system/brew locations; if not found, pass
``extra_cppflags=["-I/path/to/armadillo/include"]``. This is a documented
system dependency for v0.10.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Iterable

import pybind11

__all__ = ["source", "source_AI4BayesCode", "vendored_include_path"]


# ---------------------------------------------------------------------------
# Path / environment helpers
# ---------------------------------------------------------------------------
def vendored_include_path() -> Path:
    """Absolute path to the header tree bundled inside the installed package."""
    base = importlib.resources.files("AI4BayesCode")
    return Path(str(base)) / "_vendored_include"


def _user_cache_dir() -> Path:
    """A writable per-user cache dir (no platformdirs dependency)."""
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Caches"
    elif os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "AI4BayesCode"


def _ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def _compiler() -> str:
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError(
            "No C++ compiler found. Set the CXX environment variable or install "
            "clang++ / g++."
        )
    return cxx


def _compiler_id(cxx: str) -> str:
    """A short, stable identifier for the compiler + its version (cache key)."""
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True)
        first = (out.stdout or out.stderr or "").splitlines()
        return f"{cxx}|{first[0]}" if first else cxx
    except Exception:
        return cxx


def _find_armadillo() -> Path | None:
    candidates = [
        os.environ.get("ARMADILLO_INCLUDE_DIR"),
        "/opt/homebrew/opt/armadillo/include",   # brew, Apple silicon
        "/usr/local/opt/armadillo/include",      # brew, Intel
        "/usr/include",                          # Linux system
        "/opt/local/include",                    # MacPorts
    ]
    for cand in candidates:
        if cand and (Path(cand) / "armadillo").is_file():
            return Path(cand)
    return None


# ---------------------------------------------------------------------------
# Source resolution + flags
# ---------------------------------------------------------------------------
def _resolve_code(code: str | os.PathLike) -> tuple[Path, str]:
    """Return (cpp_path, source_text) for a file path OR a source string."""
    # 1) An existing file path? (A long source string may overflow path limits
    #    when probed as a path -- treat that as "not a file".)
    try:
        p = Path(code)
        if p.is_file():
            return p.resolve(), p.read_text()
    except (TypeError, ValueError, OSError):
        pass

    # 2) A source string?
    if not isinstance(code, str):
        raise TypeError("`code` must be a .cpp file path or a .cpp source string.")
    looks_like_source = ("\n" in code) or ("#include" in code) or any(
        ch in code for ch in "{};"
    )
    if not looks_like_source:
        raise FileNotFoundError(
            "`code` is neither an existing file nor recognizable C++ source.\n"
            f"If you meant a file path, it does not exist: {code}"
        )
    fd, tmp_name = tempfile.mkstemp(prefix="ai4bayes_", suffix=".cpp")
    os.close(fd)
    tmp = Path(tmp_name)
    tmp.write_text(code)
    return tmp.resolve(), code


def _read_pybind_module_name(text: str, where: str) -> str:
    needle = "PYBIND11_MODULE("
    idx = text.find(needle)
    if idx < 0:
        raise ValueError(
            f"No PYBIND11_MODULE(<name>, ...) block found in {where}.\n"
            "Python compilation requires a pybind11 module declaration. If the "
            "file has only RCPP_MODULE (Rcpp-only), add a PYBIND11_MODULE inside "
            "an `#ifdef AI4BAYESCODE_PYBIND_MODULE ... #endif` guard. See the "
            "dual-module reference examples/ODE_SIR.cpp."
        )
    start = idx + len(needle)
    end = text.find(",", start)
    if end < 0:
        raise ValueError(
            f"Malformed PYBIND11_MODULE(...) in {where} -- expected two arguments."
        )
    return text[start:end].strip()


def _compile_flags(include_root: Path) -> list[str]:
    """The -I / -D flags for the *packaged* (vendored) header nesting."""
    inc = include_root
    roots = [
        inc,
        inc / "mcmclib",
        inc / "mcmclib" / "BaseMatrixOps" / "include",
        inc / "eigen",
        inc / "celerite" / "include",
        inc / "libgp_kernels",
    ]
    missing = [str(r) for r in roots if not r.is_dir()]
    if missing:
        raise FileNotFoundError(
            "AI4BayesCode include root(s) missing from the install:\n  "
            + "\n  ".join(missing)
            + "\nThe package install looks incomplete -- reinstall AI4BayesCode."
        )
    flags = [f"-I{r}" for r in roots]
    arma = _find_armadillo()
    if arma is not None:
        flags.append(f"-I{arma}")
    flags += [
        "-DMCMC_ENABLE_ARMA_WRAPPERS",
        "-DARMA_DONT_USE_WRAPPER",
        "-DAI4BAYESCODE_PYBIND_MODULE",   # the only macro that differs from R
        "-std=c++17",
        "-O2",
        "-fvisibility=hidden",
    ]
    flags.append(f"-I{pybind11.get_include()}")
    return flags


def _link_flags() -> list[str]:
    if platform.system() == "Darwin":
        return ["-framework", "Accelerate"]
    return ["-lblas", "-llapack", "-lm"]


def _signature(text: str, flags: Iterable[str], libs: Iterable[str], cxx_id: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode())
    for f in list(flags) + list(libs):
        h.update(b"\x00")
        h.update(str(f).encode())
    h.update(b"\x00")
    h.update(cxx_id.encode())   # compiler-version-aware (toolchain bump invalidates)
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def source(
    code: str | os.PathLike,
    *,
    rebuild: bool = False,
    verbose: bool = False,
    cache_dir: str | os.PathLike | None = None,
    extra_cppflags: Iterable[str] = (),
    extra_libs: Iterable[str] = (),
    include_root: str | os.PathLike | None = None,
    quiet: bool = False,
):
    """Compile ``code`` against the installed AI4BayesCode headers and import it.

    Parameters
    ----------
    code:
        A ``.cpp`` file path, or a string containing the ``.cpp`` source. The
        file must contain a ``PYBIND11_MODULE(<name>, m)`` block.
    rebuild:
        If True, ignore the build cache and recompile.
    verbose:
        Echo the compiler command + output.
    cache_dir:
        Where to store the compiled ``.so``. Defaults to a per-user cache dir.
    extra_cppflags, extra_libs:
        Extra compiler / linker flags (e.g. a custom Armadillo ``-I``).
    include_root:
        Override the header root. Defaults to the vendored tree shipped in the
        installed package -- you should not normally need this.

    Returns
    -------
    module
        The imported module with the pybind11 class bindings.
    """
    cpp_path, text = _resolve_code(code)
    module_name = _read_pybind_module_name(text, str(cpp_path))

    inc = Path(include_root) if include_root is not None else vendored_include_path()
    if not inc.is_dir():
        raise FileNotFoundError(
            f"AI4BayesCode header tree not found at {inc}. The package install "
            "looks incomplete -- reinstall AI4BayesCode."
        )

    flags = _compile_flags(inc) + list(extra_cppflags)
    libs = _link_flags() + list(extra_libs)
    cxx = _compiler()
    sig = _signature(text, flags, libs, _compiler_id(cxx))

    cache = Path(cache_dir) if cache_dir is not None else _user_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    so_path = cache / f"{module_name}_{sig}{_ext_suffix()}"

    if so_path.exists() and not rebuild:
        if verbose:
            print(f"[source] using cached {so_path}")
    else:
        _compile(cpp_path, so_path, flags, libs, cxx, verbose=verbose)

    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    # Register class -> source for AI4BayesCode.doc(), and hint the user.
    try:
        from .doc import _register_source
        _register_source(str(cpp_path))
    except Exception:
        pass
    if not quiet:
        print(f"✓ Loaded '{module_name}' — run  AI4BayesCode.doc(\"{module_name}\")  for usage.")
    return module


def _compile(cpp_file: Path, so_path: Path, cppflags: list[str], libs: list[str],
             cxx: str, *, verbose: bool) -> None:
    py_include = sysconfig.get_path("include")
    py_lib = sysconfig.get_config_var("LIBDIR")
    cmd = [cxx, *cppflags, f"-I{py_include}"]
    if platform.system() == "Darwin":
        cmd += ["-bundle", "-undefined", "dynamic_lookup"]
    else:
        cmd += ["-shared", "-fPIC"]
    cmd += [str(cpp_file), "-o", str(so_path), *libs]
    if py_lib:
        cmd += [f"-L{py_lib}"]
    if verbose:
        print("[source] compile command:\n  " + " ".join(map(str, cmd)))
    res = subprocess.run(cmd, capture_output=not verbose, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Compilation failed (exit {res.returncode}).\n"
            f"command:\n  {' '.join(map(str, cmd))}\n"
            f"stderr:\n{res.stderr if not verbose else '(see above)'}"
        )


# Back-compat alias (pre-0.1.3 name). Canonical name is source().
source_AI4BayesCode = source
