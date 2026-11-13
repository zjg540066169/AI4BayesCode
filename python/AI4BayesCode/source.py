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
system dependency.

Windows: the module is built with the MinGW-w64 g++ of MSYS2's UCRT64
environment, which also provides Armadillo and OpenBLAS. ``C:\\msys64`` (and the
Scoop and Chocolatey install folders) are found automatically; set ``CXX`` to a
``g++.exe`` to use another MinGW-w64 toolchain.
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
import time
from pathlib import Path
from typing import Iterable

import pybind11

__all__ = ["source", "vendored_include_path"]


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


_WINDOWS_TOOLCHAIN_HELP = (
    "On Windows, AI4BayesCode compiles samplers with the MinGW-w64 toolchain of "
    "MSYS2, which also provides Armadillo and OpenBLAS:\n"
    "  1. Install MSYS2 from https://www.msys2.org (keep the default folder "
    "C:\\msys64).\n"
    "  2. Open the 'MSYS2 UCRT64' terminal and run:\n"
    "       pacman -S --needed mingw-w64-ucrt-x86_64-gcc "
    "mingw-w64-ucrt-x86_64-armadillo mingw-w64-ucrt-x86_64-openblas\n"
    "AI4BayesCode then finds C:\\msys64\\ucrt64 by itself. For a toolchain in "
    "another folder, set the CXX environment variable to its g++.exe.")


def _is_windows() -> bool:
    return os.name == "nt"


def _msys2_ucrt64_prefixes() -> list[Path]:
    """MSYS2 UCRT64 prefixes that hold a g++: the installer default first, then
    the Scoop and Chocolatey install folders."""
    homes = [Path(r"C:\msys64")]
    if os.environ.get("USERPROFILE"):
        homes.append(Path(os.environ["USERPROFILE"]) / "scoop" / "apps" / "msys2" / "current")
    homes.append(Path(r"C:\tools\msys64"))
    return [h / "ucrt64" for h in homes if (h / "ucrt64" / "bin" / "g++.exe").is_file()]


def _toolchain_prefix(cxx: str) -> Path | None:
    """The prefix (holding bin/, include/ and lib/) of a MinGW-w64 compiler."""
    found = shutil.which(cxx)
    if found is None:
        return None
    prefix = Path(found).resolve().parent.parent
    return prefix if (prefix / "lib").is_dir() else None


def _compiler() -> str:
    cxx = os.environ.get("CXX")
    if cxx:
        return cxx
    if _is_windows():
        # clang++ on Windows targets the MSVC toolchain, so only a MinGW-w64 g++
        # can build the module without Visual Studio. MSYS2's comes first because
        # the same environment also supplies Armadillo and OpenBLAS.
        for prefix in _msys2_ucrt64_prefixes():
            return str(prefix / "bin" / "g++.exe")
        cxx = shutil.which("g++")
        if cxx is None:
            raise RuntimeError("No MinGW-w64 C++ compiler found.\n" + _WINDOWS_TOOLCHAIN_HELP)
        return cxx
    cxx = shutil.which("clang++") or shutil.which("g++")
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


def _find_armadillo(cxx: str | None = None) -> Path | None:
    candidates = [
        os.environ.get("ARMADILLO_INCLUDE_DIR"),
        "/opt/homebrew/opt/armadillo/include",   # brew, Apple silicon
        "/usr/local/opt/armadillo/include",      # brew, Intel
        "/usr/include",                          # Linux system
        "/opt/local/include",                    # MacPorts
    ]
    if _is_windows() and cxx is not None:
        prefix = _toolchain_prefix(cxx)          # e.g. C:\msys64\ucrt64
        if prefix is not None:
            candidates.append(str(prefix / "include"))
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
            return p.resolve(), p.read_text(encoding="utf-8")
    except (TypeError, ValueError, OSError):
        pass

    # 2) A source string?
    if not isinstance(code, str):
        raise TypeError("`code` must be a .cpp file path or a .cpp source string.")
    looks_like_source = ("\n" in code) or ("#include" in code) or any(
        ch in code for ch in "{};"
    )
    if not looks_like_source:
        # If it looks like a SHIPPED example name (e.g. "GBartMultinomial" or
        # "GBartMultinomial.cpp"), point the user at example() -- source() takes a
        # file PATH, example() takes a NAME (the R-side ai4bayescode_example() split).
        stem = os.path.basename(code)
        stem = stem[:-4] if stem.endswith(".cpp") else stem
        hint = ""
        try:
            from ._blocks import _example_names
            if stem in _example_names():
                hint = (f"\nThat is a SHIPPED example -- load it BY NAME with "
                        f"AI4BayesCode.example(\"{stem}\")  (source() takes a file path).")
        except Exception:  # noqa: BLE001
            pass
        raise FileNotFoundError(
            "`code` is neither an existing file nor recognizable C++ source.\n"
            f"If you meant a file path, it does not exist: {code}" + hint
        )
    fd, tmp_name = tempfile.mkstemp(prefix="ai4bayes_", suffix=".cpp")
    os.close(fd)
    tmp = Path(tmp_name)
    tmp.write_text(code, encoding="utf-8")
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
            "dual-module reference AI4BayesCode.examples_path(\"ODE_SIR\")."
        )
    start = idx + len(needle)
    end = text.find(",", start)
    if end < 0:
        raise ValueError(
            f"Malformed PYBIND11_MODULE(...) in {where} -- expected two arguments."
        )
    return text[start:end].strip()


def _compile_flags(include_root: Path, cxx: str | None = None) -> list[str]:
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
    from .install_block import _block_include_flags   # installed contributed blocks
    flags.extend(_block_include_flags())
    arma = _find_armadillo(cxx)

    if arma is None:
        # Detected and then ignored: the -I was simply skipped and the
        # user got the whole clang++ command line plus
        # "fatal error: 'armadillo' file not found".
        raise FileNotFoundError(
            "Armadillo headers not found. AI4BayesCode compiles generated "
            "samplers against Armadillo, so it has to be installed first:\n"
            "  macOS:          brew install armadillo\n"
            "  Debian/Ubuntu:  apt-get install libarmadillo-dev\n"
            "  conda:          conda install -c conda-forge armadillo\n"
            "  Windows:        see below\n"
            "If it is installed somewhere unusual, set ARMADILLO_INCLUDE_DIR "
            "to the directory containing the 'armadillo' header."
            + ("\n\n" + _WINDOWS_TOOLCHAIN_HELP if _is_windows() else ""))
    if arma is not None:
        flags.append(f"-I{arma}")
    flags += [
        "-DMCMC_ENABLE_ARMA_WRAPPERS",
        # Trailing "=" makes this an EMPTY replacement, matching
        # RcppArmadillo's own `#define ARMA_DONT_USE_WRAPPER`. A bare -D
        # expands to "1", a different token sequence, and every single user
        # compile then opens with a macro-redefinition warning.
        "-DARMA_DONT_USE_WRAPPER=",
        "-DAI4BAYESCODE_PYBIND_MODULE",   # the only macro that differs from R
        # MinGW's math.h hides M_PI and the other M_* constants in strict ISO
        # mode, so Windows builds use the GNU dialect, as R itself does.
        "-std=gnu++17" if _is_windows() else "-std=c++17",
        "-O2",
    ]
    if _is_windows():
        flags += [
            # Python's pyconfig.h sets MS_WIN64 only for MSVC; without it a
            # 64-bit GCC build sees a 32-bit Py_ssize_t.
            "-DMS_WIN64=",
            # Template-heavy samplers exceed the COFF section limit of 2^15.
            "-Wa,-mbig-obj",
        ]
    else:
        flags.append("-fvisibility=hidden")
    flags.append(f"-I{pybind11.get_include()}")
    return flags


def _python_link_args() -> list[str]:
    """Windows only: link the CPython DLL, through its import library when the
    installation ships one (python.org and conda do), else the DLL itself."""
    base = Path(sys.base_prefix)
    name = f"python{sys.version_info.major}{sys.version_info.minor}"
    if (base / "libs" / f"{name}.lib").is_file():
        return [f"-L{base / 'libs'}", f"-l{name}"]
    if (base / f"{name}.dll").is_file():
        return [str(base / f"{name}.dll")]
    raise RuntimeError(
        f"Cannot find {name}.lib or {name}.dll under {base}; the Python "
        "installation cannot be linked against.")


def _link_flags(cxx: str | None = None) -> list[str]:
    if platform.system() == "Darwin":
        return ["-framework", "Accelerate"]
    if _is_windows():
        prefix = _toolchain_prefix(cxx) if cxx else None
        libdir = [f"-L{prefix / 'lib'}"] if prefix is not None else []
        # OpenBLAS carries LAPACK. libstdc++ and libgcc go in statically, so a
        # different copy of them already loaded in the process cannot clash.
        return libdir + ["-lopenblas", "-static-libgcc", "-static-libstdc++"] + _python_link_args()
    return ["-lblas", "-llapack", "-lm"]


# Windows: DLL directories added for compiled modules. The handles are kept so
# the directories stay on the search path for the life of the process.
_DLL_DIRS: dict[str, object] = {}


def _add_dll_directory(cxx: str) -> None:
    """Windows: let the module find the toolchain's DLLs (OpenBLAS and its
    Fortran runtime). Python 3.8+ ignores PATH for extension-module
    dependencies, so the directory has to be registered explicitly."""
    found = shutil.which(cxx)
    if found is None:
        return
    bindir = str(Path(found).resolve().parent)
    if bindir not in _DLL_DIRS:
        _DLL_DIRS[bindir] = os.add_dll_directory(bindir)


def _contributed_block_fingerprint(flags: Iterable[str]) -> str:
    """Content fingerprint of every header tree on the -I path.

    The flags alone are not enough. Reinstalling a block -- install_block(force=
    True) fetching a new version, or editing a header under ./blocks_local/ --
    leaves the -I paths byte-identical, so the signature did not move and
    source() silently handed back the binary built from the PREVIOUS header.
    A user who updated a block and re-ran kept executing the old sampler, which
    is exactly the silent-wrong-posterior failure the contrib design exists to
    prevent.

    The core and vendored trees are covered for the same reason: reinstalling
    the package (a new version, a changed core header) also leaves the -I path
    and the example source byte-identical, and the cached binary built against
    the OLD headers would be loaded -- a wrapper compiled before a field was
    added to a core struct then reports the old layout.

    Fingerprints (path, size, mtime_ns) of every header under each -I'd
    directory. Cheap: a stat per file, no reading (a few tens of ms for the
    vendored tree).
    """
    h = hashlib.sha256()
    roots = sorted({f[2:] for f in flags
                    if isinstance(f, str) and f.startswith("-I")})
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            for fn in sorted(filenames):
                if not fn.endswith((".hpp", ".h", ".hh", ".hxx", ".ipp",
                                    ".cpp", ".cc", ".dcf")):
                    continue
                fp = os.path.join(dirpath, fn)
                try:
                    st = os.stat(fp)
                except OSError:
                    continue
                h.update(os.path.relpath(fp, root).encode())
                h.update(f"\x00{st.st_size}\x00{st.st_mtime_ns}\x00".encode())
    return h.hexdigest()


def _signature(text: str, flags: Iterable[str], libs: Iterable[str], cxx_id: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode())
    flags = list(flags)
    for f in flags + list(libs):
        h.update(b"\x00")
        h.update(str(f).encode())
    h.update(b"\x00")
    h.update(cxx_id.encode())   # compiler-version-aware (toolchain bump invalidates)
    h.update(b"\x00")
    h.update(_contributed_block_fingerprint(flags).encode())
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

    cxx = _compiler()
    flags = _compile_flags(inc, cxx) + list(extra_cppflags)
    libs = _link_flags(cxx) + list(extra_libs)
    sig = _signature(text, flags, libs, _compiler_id(cxx))

    cache = Path(cache_dir) if cache_dir is not None else _user_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    so_path = cache / f"{module_name}_{sig}{_ext_suffix()}"

    if so_path.exists() and not rebuild:
        if verbose:
            print(f"[source] using cached {so_path}")
    else:
        if _is_windows() and so_path.exists():
            try:
                so_path.unlink()
            except PermissionError:
                # Loaded earlier in this process: Windows locks the file, so the
                # rebuild goes to a new name.
                so_path = so_path.with_name(
                    f"{module_name}_{sig}_{os.getpid()}_{time.time_ns()}{_ext_suffix()}")
        _compile(cpp_path, so_path, flags, libs, cxx, verbose=verbose)

    if _is_windows():
        _add_dll_directory(cxx)

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
    elif _is_windows():
        cmd += ["-shared"]
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
