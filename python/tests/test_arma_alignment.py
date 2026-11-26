"""source() compiles every sampler with -include and the absolute path of the
vendored AI4BayesCode/arma_aligned_alloc.hpp, so Armadillo takes its heap
arrays from memory aligned to 64 bytes.

The flag is part of the build signature, so a module built without it is not
loaded from the cache. The compile tests build a small module and check the
address of Armadillo heap arrays of 17 to 400 elements modulo 64, once in a
working directory that holds a file AI4BayesCode/arma_aligned_alloc.hpp of its
own (the compiler looks for a relative -include name there first)."""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest

src = importlib.import_module("AI4BayesCode.source")

HEADER = "AI4BayesCode/arma_aligned_alloc.hpp"


def test_the_header_ships_in_the_vendored_tree():
    assert (src.vendored_include_path() / HEADER).is_file()


def test_compile_flags_force_include_the_header():
    try:
        cxx = src._compiler()
        flags = src._compile_flags(src.vendored_include_path(), cxx)
    except (RuntimeError, FileNotFoundError) as exc:   # no toolchain / Armadillo
        pytest.skip(str(exc))
    i = flags.index("-include", flags.index("-DAI4BAYESCODE_PYBIND_MODULE"))
    path = Path(flags[i + 1])
    assert path.is_absolute()
    assert path == (src.vendored_include_path() / HEADER).absolute()


def test_the_flag_moves_the_signature():
    flags = ["-Iinc", "-DAI4BAYESCODE_PYBIND_MODULE"]
    without = src._signature("// code\n", flags, [], "c++|1")
    with_it = src._signature("// code\n", flags + ["-include", HEADER], [], "c++|1")
    assert with_it != without


PROBE = r"""
#include <armadillo>
#include <cstdint>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
static std::vector<int> remainders() {
    std::vector<int> out;
    std::vector<arma::vec> keep;
    for (arma::uword n = 17; n <= 400; ++n) {
        keep.emplace_back(n, arma::fill::zeros);
        arma::mat A(n, 2, arma::fill::ones);
        arma::vec v = A.col(0) + 1.0;
        out.push_back(int(reinterpret_cast<std::uintptr_t>(keep.back().memptr()) % 64));
        out.push_back(int(reinterpret_cast<std::uintptr_t>(A.memptr()) % 64));
        out.push_back(int(reinterpret_cast<std::uintptr_t>(v.memptr()) % 64));
    }
    return out;
}
PYBIND11_MODULE(ai4b_align_probe, m) { m.def("remainders", &remainders); }
"""


@pytest.mark.compile
def test_armadillo_heap_arrays_are_64_byte_aligned(tmp_path):
    try:
        cxx = src._compiler()
    except RuntimeError as exc:                         # no C++ compiler
        pytest.skip(str(exc))
    if src._find_armadillo(cxx) is None:
        pytest.skip("Armadillo headers not found")
    mod = src.source(PROBE, cache_dir=tmp_path, quiet=True)
    rem = mod.remainders()
    assert len(rem) == 3 * 384
    assert set(rem) == {0}


@pytest.mark.compile
def test_a_header_of_the_same_name_in_the_working_directory_is_not_read(
        tmp_path, monkeypatch):
    try:
        cxx = src._compiler()
    except RuntimeError as exc:                         # no C++ compiler
        pytest.skip(str(exc))
    if src._find_armadillo(cxx) is None:
        pytest.skip("Armadillo headers not found")
    decoy = tmp_path / "cwd" / HEADER
    decoy.parent.mkdir(parents=True)
    decoy.write_text('#error "the header in the working directory was read"\n')
    monkeypatch.chdir(tmp_path / "cwd")
    # another module name, so this test compiles a module of its own
    code = PROBE.replace("ai4b_align_probe", "ai4b_align_probe_cwd")
    mod = src.source(code, cache_dir=tmp_path / "cache", quiet=True)
    rem = mod.remainders()
    assert len(rem) == 3 * 384
    assert set(rem) == {0}
