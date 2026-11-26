"""source() compiles every sampler with -include and the absolute path of the
vendored AI4BayesCode/arma_aligned_alloc.hpp, so Armadillo allocates its heap
arrays on arma_heap_alignment-byte boundaries.

The flag is part of the build signature, so a module built without it is not
loaded from the cache. The compile tests build a small module and check
  (a) the addresses of Armadillo heap arrays of 17 to 400 elements modulo
      arma_heap_alignment, once in a working directory that holds a file
      AI4BayesCode/arma_aligned_alloc.hpp of its own (the compiler resolves a
      relative -include name against its working directory first);
  (b) that ddot (n = 5000, 14400), dtrtrs (lower, n = 110, one right-hand side)
      and dgetrf + dgetrs (n = 110, one right-hand side), with every operand at
      offset k * 64 bytes from a 16384-byte boundary, k = 0, ..., 255, give one
      result per class of the offset modulo arma_heap_alignment. On an Apple
      M2 Max with macOS 27.0.1 these Accelerate routines give results that
      depend on the offset modulo 256 bytes, so there (b) fails for an
      alignment of 64 or 128 bytes."""
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


def test_windows_flags_force_include_both_headers_by_absolute_path(monkeypatch):
    # The Windows branch of _compile_flags, run on this system: the compiler
    # resolves a relative -include name against its working directory first,
    # so both force-included headers are passed by absolute path.
    try:
        cxx = src._compiler()
    except RuntimeError as exc:                          # no C++ compiler
        pytest.skip(str(exc))
    monkeypatch.setattr(src, "_is_windows", lambda: True)
    try:
        flags = src._compile_flags(src.vendored_include_path(), cxx)
    except FileNotFoundError as exc:                     # no Armadillo
        pytest.skip(str(exc))
    paths = [Path(flags[i + 1]) for i, f in enumerate(flags) if f == "-include"]
    assert [p.name for p in paths] == ["arma_aligned_alloc.hpp", "win_ucrt_math.hpp"]
    for p in paths:
        assert p.is_absolute() and p.is_file()
        assert p.parent == (src.vendored_include_path() / "AI4BayesCode").absolute()


def test_the_flag_moves_the_signature():
    flags = ["-Iinc", "-DAI4BAYESCODE_PYBIND_MODULE"]
    without = src._signature("// code\n", flags, [], "c++|1")
    with_it = src._signature("// code\n", flags + ["-include", HEADER], [], "c++|1")
    assert with_it != without


PROBE = r"""
#include <armadillo>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

// AI4BayesCode::arma_heap_alignment if the build force-included
// AI4BayesCode/arma_aligned_alloc.hpp, 0 otherwise.
static int alignment() {
#ifdef AI4BAYESCODE_ARMA_ALIGNED_ALLOC_HPP
    return static_cast<int>(AI4BayesCode::arma_heap_alignment);
#else
    return 0;
#endif
}

// Addresses modulo 65536 of Armadillo heap arrays of 17 to 400 elements.
static std::vector<int> addresses() {
    std::vector<int> out;
    std::vector<arma::vec> keep;
    for (arma::uword n = 17; n <= 400; ++n) {
        keep.emplace_back(n, arma::fill::zeros);
        arma::mat A(n, 2, arma::fill::ones);
        arma::vec v = A.col(0) + 1.0;
        for (const double* p : {keep.back().memptr(), A.memptr(), v.memptr()})
            out.push_back(int(reinterpret_cast<std::uintptr_t>(p) % 65536));
    }
    return out;
}

// room for n_bytes at every offset k * 64, k = 0..255, from a 16384-byte boundary
struct OffsetBuffer {
    std::vector<unsigned char> raw;
    unsigned char* base;
    explicit OffsetBuffer(std::size_t n_bytes) : raw(n_bytes + 2 * 16384) {
        const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(raw.data());
        base = raw.data() + (16384 - p % 16384) % 16384;
    }
    double* at(std::size_t off) { return reinterpret_cast<double*>(base + off); }
};

static std::vector<double> draws(std::size_t n, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd;
    std::vector<double> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
}

static py::bytes bits(const double* x, std::size_t n) {
    return py::bytes(reinterpret_cast<const char*>(x), n * sizeof(double));
}

// For each routine, the bits of the result at offset offsets[k], k = 0..255.
static py::dict offset_results() {
    const int K = 256;
    std::vector<int> offsets(K);
    for (int k = 0; k < K; ++k) offsets[k] = 64 * k;
    py::dict out;
    out["offsets"] = offsets;
    for (const std::size_t n : {std::size_t(5000), std::size_t(14400)}) {
        const std::vector<double> x0 = draws(n, 21), y0 = draws(n, 22);
        OffsetBuffer bx(n * 8), by(n * 8);
        py::list r;
        for (int k = 0; k < K; ++k) {
            double* x = bx.at(64 * k); double* y = by.at(64 * k);
            std::memcpy(x, x0.data(), n * 8); std::memcpy(y, y0.data(), n * 8);
            const double d = arma::blas::dot(arma::uword(n), x, y);
            r.append(bits(&d, 1));
        }
        out[n == 5000 ? "ddot_5000" : "ddot_14400"] = r;
    }
    // not const: Armadillo releases before 15.8 declare the LAPACK wrappers
    // with non-const pointer parameters
    arma::blas_int n = 110, n_rhs = 1;
    const std::size_t nn = 110, n2 = nn * nn;
    const std::vector<double> a0 = draws(n2, 23), b0 = draws(nn, 24);
    std::vector<double> L0(n2, 0.0), G0(a0);
    for (std::size_t j = 0; j < nn; ++j) {
        for (std::size_t i = j; i < nn; ++i)
            L0[i + j * nn] = (i == j) ? 5.0 + std::fabs(a0[i + j * nn]) : 0.3 * a0[i + j * nn];
        G0[j + j * nn] += double(nn);
    }
    py::list tr, ge;
    OffsetBuffer bL(n2 * 8), bG(n2 * 8), bb(nn * 8), bp(nn * sizeof(arma::blas_int));
    for (int k = 0; k < K; ++k) {
        const std::size_t off = 64 * k;
        char uplo = 'L', trans = 'N', diag = 'N';
        arma::blas_int info = 0;
        double* L = bL.at(off); double* b = bb.at(off);
        std::memcpy(L, L0.data(), n2 * 8); std::memcpy(b, b0.data(), nn * 8);
        arma::lapack::trtrs<double>(&uplo, &trans, &diag, &n, &n_rhs, L, &n, b, &n, &info);
        tr.append(bits(b, nn));
        double* G = bG.at(off);
        arma::blas_int* ipiv = reinterpret_cast<arma::blas_int*>(bp.at(off));
        std::memcpy(G, G0.data(), n2 * 8); std::memcpy(b, b0.data(), nn * 8);
        arma::lapack::getrf<double>(&n, &n, G, &n, ipiv, &info);
        arma::lapack::getrs<double>(&trans, &n, &n_rhs, G, &n, ipiv, b, &n, &info);
        ge.append(bits(b, nn));
    }
    out["dtrtrs_110"] = tr;
    out["dgetrs_110"] = ge;
    return out;
}

PYBIND11_MODULE(ai4b_align_probe, m) {
    m.def("alignment", &alignment);
    m.def("addresses", &addresses);
    m.def("offset_results", &offset_results);
}
"""


def _needs_toolchain():
    try:
        cxx = src._compiler()
    except RuntimeError as exc:                         # no C++ compiler
        pytest.skip(str(exc))
    if src._find_armadillo(cxx) is None:
        pytest.skip("Armadillo headers not found")


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    _needs_toolchain()
    return src.source(PROBE, cache_dir=tmp_path_factory.mktemp("align_cache"), quiet=True)


@pytest.mark.compile
def test_armadillo_heap_arrays_lie_on_arma_heap_alignment_boundaries(probe):
    alignment = probe.alignment()
    assert alignment > 0
    addr = probe.addresses()
    assert len(addr) == 3 * 384
    assert all(a % alignment == 0 for a in addr)


@pytest.mark.compile
def test_routines_give_one_result_per_class_of_the_offset(probe):
    alignment = probe.alignment()
    assert alignment > 0
    res = probe.offset_results()
    offsets = res["offsets"]
    assert len(offsets) == 256
    for name, results in res.items():
        if name == "offsets":
            continue
        first = {}
        for off, r in zip(offsets, results):
            ref = first.setdefault(off % alignment, r)
            assert r == ref, (f"{name}: the offsets {off % alignment} modulo {alignment} "
                              "give different results")


@pytest.mark.compile
def test_a_header_of_the_same_name_in_the_working_directory_is_not_read(
        tmp_path, monkeypatch):
    _needs_toolchain()
    decoy = tmp_path / "cwd" / HEADER
    decoy.parent.mkdir(parents=True)
    decoy.write_text('#error "the header in the working directory was read"\n')
    monkeypatch.chdir(tmp_path / "cwd")
    # another module name, so this test compiles a module of its own
    code = PROBE.replace("ai4b_align_probe", "ai4b_align_probe_cwd")
    mod = src.source(code, cache_dir=tmp_path / "cache", quiet=True)
    alignment = mod.alignment()
    assert alignment > 0
    addr = mod.addresses()
    assert len(addr) == 3 * 384
    assert all(a % alignment == 0 for a in addr)
