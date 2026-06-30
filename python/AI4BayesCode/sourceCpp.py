"""Compile-on-demand pybind11 wrapper for AI4BayesCode `.cpp` files.

Parallels the R-side `AI4BayesCode_sourceCpp(cpp_file, AI4BayesCode_path)`:
Rcpp's sourceCpp mangles include paths via a temp Makevars, while the
Python equivalent here spawns a minimal setuptools + pybind11 build,
produces a `.so` under a cache directory, imports it, and returns the
module handle.

Why a custom builder instead of cppimport / cppyy:
- cppimport needs a magic comment at the top of every `.cpp` (too
  invasive; breaks Rcpp compatibility).
- cppyy JIT parses headers but struggles with Armadillo template
  instantiation and large header sets.
- Our blockmcmc tree is header-only Armadillo + template C++, which
  setuptools + pybind11 handles cleanly with the right include paths.

Cache:
- The compiled `.so` is stored at
  `<ai4bayescode_path>/../generated/<module_name>_<hash>.so`
- On subsequent calls, we reuse the cached build if source mtime
  and flags haven't changed.
"""

from __future__ import annotations

import hashlib
import importlib
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


def _read_pybind_module_name(cpp_file: Path) -> str:
    """Scan source for `PYBIND11_MODULE(<name>, ...)` and return <name>.

    Fails loudly if no PYBIND11_MODULE block is found — the user must
    add one (Python equivalent of RCPP_MODULE).
    """
    text = cpp_file.read_text()
    # Simple regex-free scan — find 'PYBIND11_MODULE(' then the identifier.
    needle = "PYBIND11_MODULE("
    idx = text.find(needle)
    if idx < 0:
        raise ValueError(
            f"No PYBIND11_MODULE(<name>, ...) block found in {cpp_file}.\n"
            "Python compilation requires a pybind11 module declaration. "
            "See examples_py/ODE_SIR.cpp for a reference. If the file has "
            "only RCPP_MODULE (Rcpp-only), add a PYBIND11_MODULE inside an "
            "`#ifdef AI4BAYESCODE_PYBIND_MODULE ... #endif` guard."
        )
    start = idx + len(needle)
    end = text.find(",", start)
    if end < 0:
        raise ValueError(
            f"Malformed PYBIND11_MODULE(...) macro in {cpp_file} — expected "
            "two arguments (module_name, module_var)."
        )
    return text[start:end].strip()


def _compile_flags(ai4bayescode_path: Path) -> list[str]:
    """Return the -I / -D flags for the AI4BayesCode core library tree."""
    root = ai4bayescode_path
    inc_root = root / "include"
    inc_mc = inc_root / "mcmclib"
    inc_bmo = inc_mc / "BaseMatrixOps" / "include"
    inc_eigen = inc_root / "eigen"
    inc_libgp = root / "libgp_kernels"
    inc_celerite = root / "celerite" / "include"
    for p in (inc_root, inc_mc, inc_bmo):
        if not p.is_dir():
            raise FileNotFoundError(
                f"Expected AI4BayesCode include directory not found: {p}\n"
                "Check that `ai4bayescode_path` points at the AI4BayesCode root "
                "(the folder containing `include/AI4BayesCode/`)."
            )
    flags = [
        f"-I{inc_root}",
        f"-I{inc_mc}",
        f"-I{inc_bmo}",
    ]
    if inc_eigen.is_dir():
        flags.append(f"-I{inc_eigen}")
    if inc_libgp.is_dir():
        flags.append(f"-I{inc_libgp}")
    if inc_celerite.is_dir():
        flags.append(f"-I{inc_celerite}")
    # installed contributed blocks (AI4BayesCode.install_block)
    from .install_block import _block_include_flags
    flags.extend(_block_include_flags())
    # Standalone Armadillo (Python does not go through RcppArmadillo).
    # Try common install locations; user can override with extra_cppflags.
    for arma_guess in (
        Path("/opt/homebrew/opt/armadillo/include"),      # brew macOS arm64
        Path("/usr/local/opt/armadillo/include"),         # brew macOS intel
        Path("/usr/include"),                             # Linux system
    ):
        if (arma_guess / "armadillo").is_file():
            flags.append(f"-I{arma_guess}")
            break
    # Match the R Makevars: Armadillo configuration
    flags += [
        "-DMCMC_ENABLE_ARMA_WRAPPERS",
        "-DARMA_DONT_USE_WRAPPER",
        "-DAI4BAYESCODE_PYBIND_MODULE",     # so .cpp can use `#ifdef` to route
        "-std=c++17",
        "-O2",
        "-fvisibility=hidden",
    ]
    # pybind11 include (headers are under its install package)
    flags.append(f"-I{pybind11.get_include()}")
    return flags


def _link_flags() -> list[str]:
    """BLAS/LAPACK linking, platform-dependent (mirrors R Makevars)."""
    if platform.system() == "Darwin":
        return ["-framework", "Accelerate"]
    # Linux default: use whatever the Python install is linked against.
    # Users can override via LDFLAGS env var.
    return ["-lblas", "-llapack", "-lm"]


def _compute_signature(cpp_file: Path, flags: Iterable[str]) -> str:
    """Hash source content + compile flags to detect cache staleness."""
    h = hashlib.sha256()
    h.update(cpp_file.read_bytes())
    for f in flags:
        h.update(f.encode())
    return h.hexdigest()[:16]


def _get_ext_suffix() -> str:
    """Extension suffix for the current Python (e.g. .cpython-311-darwin.so)."""
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def sourceCpp(
    cpp_file: str | os.PathLike,
    ai4bayescode_path: str | os.PathLike | None = None,
    *,
    rebuild: bool = False,
    verbose: bool = False,
    extra_cppflags: Iterable[str] = (),
    extra_libs: Iterable[str] = (),
    cache_dir: str | os.PathLike | None = None,
):
    """Compile `cpp_file` against the AI4BayesCode headers and import the module.

    Parameters
    ----------
    cpp_file : path-like
        Path to the `.cpp` file. Must contain a `PYBIND11_MODULE(<name>, m)`
        block (guarded by `#ifdef AI4BAYESCODE_PYBIND_MODULE` if the same file
        also has RCPP_MODULE).
    ai4bayescode_path : path-like
        Path to the AI4BayesCode root (the folder containing `include/AI4BayesCode/`).
    rebuild : bool, default False
        If True, ignore the cache and force a fresh compile.
    verbose : bool, default False
        Echo compiler invocation and output.
    extra_cppflags, extra_libs : iterable of str
        Extra flags to pass through to the compiler / linker.
    cache_dir : path-like, optional
        Where to store compiled `.so`. Defaults to `<ai4bayescode_path>/../generated`.

    Returns
    -------
    module
        The imported Python module containing the pybind11 class bindings.
    """
    # Self-contained path (preferred): no checkout needed -- compile against the
    # headers vendored inside the installed package via source().
    if ai4bayescode_path is None:
        from .source import source
        return source(
            cpp_file,
            rebuild=rebuild,
            verbose=verbose,
            cache_dir=cache_dir,
            extra_cppflags=extra_cppflags,
            extra_libs=extra_libs,
        )

    # Legacy checkout mode (back-compat): compile against an on-disk AI4BayesCode tree.
    cpp_file = Path(cpp_file).resolve()
    ai4bayescode_path = Path(ai4bayescode_path).resolve()

    if not cpp_file.is_file():
        raise FileNotFoundError(f"cpp_file not found: {cpp_file}")
    if not ai4bayescode_path.is_dir():
        raise FileNotFoundError(f"ai4bayescode_path not found: {ai4bayescode_path}")

    module_name = _read_pybind_module_name(cpp_file)

    if cache_dir is None:
        cache_dir = ai4bayescode_path.parent / "generated"
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    flags = _compile_flags(ai4bayescode_path) + list(extra_cppflags)
    libs = _link_flags() + list(extra_libs)

    sig = _compute_signature(cpp_file, flags + libs)
    so_name = f"{module_name}_{sig}{_get_ext_suffix()}"
    so_path = cache_dir / so_name

    if so_path.exists() and not rebuild:
        if verbose:
            print(f"[AI4BayesCode.sourceCpp] using cached {so_path}")
    else:
        _compile(cpp_file, so_path, flags, libs, verbose=verbose)

    # Import the freshly-built module from the .so file.
    spec = importlib.util.spec_from_file_location(module_name, so_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create spec for {so_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _compile(cpp_file: Path, so_path: Path, cppflags: list[str],
             libs: list[str], *, verbose: bool) -> None:
    """Invoke the C++ compiler to produce a `.so` from `cpp_file`."""
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        raise RuntimeError(
            "No C++ compiler found. Set the CXX environment variable or "
            "install clang++ / g++."
        )
    # Python's extension compiler flags (helps find Python headers)
    py_include = sysconfig.get_path("include")
    py_lib = sysconfig.get_config_var("LIBDIR")
    cmd = [cxx]
    cmd += cppflags
    cmd += [f"-I{py_include}"]
    # Platform-specific shared-library build flags
    if platform.system() == "Darwin":
        cmd += ["-bundle", "-undefined", "dynamic_lookup"]
    else:
        cmd += ["-shared", "-fPIC"]
    cmd += [str(cpp_file), "-o", str(so_path)]
    cmd += libs
    if py_lib:
        cmd += [f"-L{py_lib}"]
    if verbose:
        print("[AI4BayesCode.sourceCpp] compile command:")
        print("  " + " ".join(str(c) for c in cmd))
    res = subprocess.run(cmd, capture_output=not verbose, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Compilation failed (exit {res.returncode}).\n"
            f"stderr:\n{res.stderr if not verbose else '(see above)'}"
        )
