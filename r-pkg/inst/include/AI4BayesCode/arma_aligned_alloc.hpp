// Copyright (C) 2026 AI4BayesCode contributors. GPL-3.0-or-later. See LICENSE at the repository root.
// arma_aligned_alloc.hpp
//
//  Armadillo heap memory on 64-byte boundaries.
//
//  Armadillo stores the elements of a Mat, Col or Row with more than 16
//  elements, of a Cube with more than 64 elements, and of every SpMat, in
//  memory from the heap: from posix_memalign with 16-byte (below 1024 bytes)
//  or 32-byte alignment where <unistd.h> sets _POSIX_ADVISORY_INFO (glibc),
//  from _aligned_malloc with the same alignments under MSVC, and from malloc
//  otherwise (macOS, MinGW-w64), which aligns to 16 bytes. The address of an
//  array modulo 64 then depends on the state of the heap. In some BLAS and
//  LAPACK implementations (macOS Accelerate among them) the order in which
//  ddot and the triangular solves add up their terms depends on the
//  alignment of the operands, so the same values stored at addresses with
//  different remainders modulo 64 give results that differ in the last bits.
//  Armadillo calls ddot for dot(a, b) and accu(A % B) with more than 32
//  elements, and LAPACK's dtrtrs for solve(trimatl(L), b). Two models built
//  with the same seed in one process can then produce different draws.
//
//  When this header precedes the first Armadillo header, Armadillo takes
//  every such array from the two functions below, which return memory
//  aligned to 64 bytes:
//      POSIX                                   posix_memalign / free
//      Windows (MinGW-w64, Rtools, MSYS2, MSVC) _aligned_malloc / _aligned_free
//  In R, ai4bayescode_source(), ai4bayescode_source_checkout() and the Rcpp
//  plugin (a file with // [[Rcpp::depends(AI4BayesCode)]] compiled by
//  Rcpp::sourceCpp), and in Python, AI4BayesCode.source(), compile every
//  sampler with -include and this header, so the header is read before the
//  first line of the sampler source, and therefore before <RcppArmadillo.h>
//  or <armadillo>.
//
//  Memory from _aligned_malloc must be released by _aligned_free. On Windows
//  every translation unit that creates or destroys the Armadillo objects of
//  one module must therefore be compiled with this header; the packages
//  compile each sampler as one translation unit.
//
//  Not affected: a Mat with at most 16 elements (arma_config::mat_prealloc)
//  and a Cube with at most 64 elements (Cube_prealloc::mem_n_elem) store
//  them inside the object, at the alignment of the object, and an Armadillo
//  object built on memory it does not own (copy_aux_mem = false) uses that
//  memory.
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

/// Alignment, in bytes, of every array Armadillo takes from the heap.
constexpr std::size_t arma_heap_alignment = 64;

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
