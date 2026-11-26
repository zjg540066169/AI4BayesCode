// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// arma_aligned_alloc.hpp
//
//  Armadillo heap memory on 256-byte boundaries.
//
//  Armadillo stores the elements of a Mat, Col or Row with more than 16
//  elements, of a Cube with more than 64 elements, and of every SpMat, in
//  memory from the heap: from posix_memalign with 16-byte (below 1024 bytes)
//  or 32-byte alignment where <unistd.h> sets _POSIX_ADVISORY_INFO (glibc),
//  from _aligned_malloc with the same alignments under MSVC, and from malloc
//  otherwise (macOS, MinGW-w64), which aligns to 16 bytes. The address of an
//  array modulo 256 then depends on the state of the heap.
//
//  Measured on an Apple M2 Max, macOS 27.0.1, Accelerate's BLAS with
//  Accelerate's LAPACK (one and two threads) and with R 4.5.2's reference
//  LAPACK (one thread). All operands of one call (inputs, outputs, work
//  arrays) were placed at the same offset k * 64 bytes from 65536-byte
//  boundaries, k = 0..255, with n x n matrices for n in {17, 33, 50, 64, 100,
//  110, 118, 120, 128, 200, 256, 400} and vectors of 33 to 100000 elements.
//  The result depended on the offset modulo 256 bytes for ddot (n = 1000,
//  5000, 14400, 40000, 100000); dtrtrs (lower or upper triangle, not
//  transposed, 1 or 3 right-hand sides), dpotrs and dgetrs (1 right-hand
//  side) at n >= 64 (dtrtrs with the upper triangle or 3 right-hand sides at
//  some of these n); dgetri at n = 33, 50, 64; dsyev and dsyevd at n >= 100;
//  dgesdd at n >= 200 (with Accelerate's LAPACK at n >= 100); and dgemv('T')
//  at n = 400. It depended on the offset modulo 128 bytes for ddot at
//  n = 2000, dgemv('T') at n = 128, 200, 256 and dsymv at n >= 200. No
//  dependence was found for dnrm2, dasum, dgemv('N'), dgemm, dsyrk, dtrsm,
//  dtrsv, dpotrf, dpotri, dtrtri, dtrcon, dgecon, dtrtrs with the triangle
//  transposed or with n right-hand sides, or for any routine at n <= 50
//  except dgetri. With each operand at its own multiple of 256 bytes (48
//  random placements in [0, 65536)), every routine and size gave one result.
//  Armadillo calls ddot for dot(a, b) and accu(A % B) with more than 32
//  elements, and dtrtrs for solve(trimatl(L), b) and solve(trimatu(U), b).
//  With 64-byte alignment, examples/GPRegression.cpp with N = 100 or 110,
//  built several times with one seed in one process, gave different draws in
//  R and in Python; with 256-byte alignment the draws were bit-identical.
//
//  When this header precedes the first Armadillo header, Armadillo allocates
//  every such array through the two functions below, which return memory
//  aligned to arma_heap_alignment (256) bytes:
//      POSIX                                   posix_memalign / free
//      Windows (MinGW-w64, Rtools, MSYS2, MSVC) _aligned_malloc / _aligned_free
//  In R, ai4bayescode_source(), ai4bayescode_source_checkout() and the Rcpp
//  plugin (a file with // [[Rcpp::depends(AI4BayesCode)]] compiled by
//  Rcpp::sourceCpp), and in Python, AI4BayesCode.source(), compile every
//  sampler with -include and this header, so the preprocessor processes the
//  header before the first line of the sampler source, and therefore before
//  <RcppArmadillo.h> or <armadillo>.
//
//  Memory from _aligned_malloc must be released by _aligned_free. On Windows
//  every translation unit that creates or destroys the Armadillo objects of
//  one module must therefore be compiled with this header; the packages
//  compile each sampler as one translation unit.
//
//  Not covered: a Mat with at most 16 elements (arma_config::mat_prealloc)
//  and a Cube with at most 64 elements (Cube_prealloc::mem_n_elem) store
//  them inside the object, at the alignment of the object; an Armadillo
//  object built on memory it does not own (copy_aux_mem = false) uses that
//  memory; and memory not allocated by Armadillo (std::vector, R or NumPy
//  arrays) has the alignment of the allocator that provided it.
//
//  If ARMA_ALIEN_MEM_ALLOC_FUNCTION or ARMA_ALIEN_MEM_FREE_FUNCTION is
//  already defined (with -D, say), this header defines neither and the
//  existing definitions apply. In C translation units it defines nothing.
// ============================================================================
#ifndef AI4BAYESCODE_ARMA_ALIGNED_ALLOC_HPP
#define AI4BAYESCODE_ARMA_ALIGNED_ALLOC_HPP

#if defined(__cplusplus) && !defined(ARMA_ALIEN_MEM_ALLOC_FUNCTION) \
                         && !defined(ARMA_ALIEN_MEM_FREE_FUNCTION)

#if defined(ARMA_INCLUDES)
#  error "AI4BayesCode/arma_aligned_alloc.hpp must precede every Armadillo header: \
the allocator macros take effect only if they are defined when <armadillo> is preprocessed."
#endif

#include <cstddef>
#include <stdlib.h>
#if defined(_WIN32)
#  include <malloc.h>
#endif

namespace AI4BayesCode {

/// Alignment, in bytes, of every array that Armadillo allocates on the heap.
constexpr std::size_t arma_heap_alignment = 256;

/// n_bytes of memory aligned to arma_heap_alignment, or nullptr on failure
/// (Armadillo then throws std::bad_alloc). Armadillo calls this with
/// n_bytes > 0 only.
inline void* arma_aligned_malloc(std::size_t n_bytes) noexcept {
#if defined(_WIN32)
    return _aligned_malloc(n_bytes, arma_heap_alignment);
#else
    void* mem = nullptr;
    const int status = posix_memalign(&mem, arma_heap_alignment, n_bytes);
    return (status == 0) ? mem : nullptr;
#endif
}

/// Releases memory from arma_aligned_malloc. Armadillo calls this with a
/// non-null pointer only.
inline void arma_aligned_free(void* mem) noexcept {
#if defined(_WIN32)
    _aligned_free(mem);
#else
    free(mem);
#endif
}

}  // namespace AI4BayesCode

#define ARMA_ALIEN_MEM_ALLOC_FUNCTION ::AI4BayesCode::arma_aligned_malloc
#define ARMA_ALIEN_MEM_FREE_FUNCTION  ::AI4BayesCode::arma_aligned_free

#endif  // C++, no definition made before

#endif  // AI4BAYESCODE_ARMA_ALIGNED_ALLOC_HPP
