"""source() compiles every sampler with -include AI4BayesCode/arma_aligned_alloc.hpp,
so Armadillo takes its heap arrays from memory aligned to 64 bytes.

The flag is part of the build signature, so a module built without it is not
loaded from the cache. The compile test builds a small module and checks the
address of Armadillo heap arrays of 17 to 400 elements modulo 64."""
from __future__ import annotations

import importlib

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
    assert flags[i + 1] == HEADER


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
