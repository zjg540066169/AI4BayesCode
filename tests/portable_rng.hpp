#pragma once
/*================================================================================
 *  portable_rng.hpp -- draws that are identical on every standard library
 *
 *  std::mt19937_64 is specified bit-exactly by the C++ standard. The
 *  DISTRIBUTIONS are not. libstdc++ (Linux, and CI) and libc++ (macOS) both use
 *  the polar method for std::normal_distribution and return its paired variates
 *  in OPPOSITE order, so the same seed yields the same numbers in a different
 *  sequence:
 *
 *      libc++    : 1.2938  0.7050  0.3980 -0.5741
 *      libstdc++ : 0.7050  1.2938 -0.5741  0.3980
 *
 *  std::uniform_int_distribution is worse -- the two libraries emit genuinely
 *  different bit sequences, not a reordering.
 *
 *  A test that simulates its data through those templates therefore fits a
 *  DIFFERENT dataset on Linux than on macOS, and no absolute tolerance on the
 *  fitted parameters can be calibrated for both. Every generator below is
 *  built from the engine's raw 64-bit output, so the simulated data is
 *  identical everywhere and a tolerance means one thing.
 *
 *  These are for TEST DATA GENERATION. They are not meant to replace the
 *  samplers inside the library, whose own std:: usage is a separate matter.
 *
 *  Each generator is checked against its std:: counterpart, distributionally,
 *  by tests/test_portable_rng.cpp.
 *
 *  Copyright (C) 2026 AI4BayesCode
 *  SPDX-License-Identifier: Apache-2.0
 *================================================================================*/

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <random>
#include <vector>

namespace prng {

/// Uniform on [0, 1) with 53-bit resolution, straight from the engine.
inline double u01(std::mt19937_64& g) {
    return static_cast<double>(g() >> 11) * (1.0 / 9007199254740992.0);
}

/// Uniform on [lo, hi).
inline double uniform(std::mt19937_64& g, double lo, double hi) {
    return lo + (hi - lo) * u01(g);
}

/// N(0, 1) by Box-Muller.
///
/// One variate per call, and deliberately so: caching the second of the pair
/// is exactly what the two standard libraries disagree about, and a cached
/// value would also make the stream depend on how many times the generator has
/// been called for other purposes.
inline double n01(std::mt19937_64& g) {
    double u1 = u01(g);
    const double u2 = u01(g);
    if (u1 < 1e-300) u1 = 1e-300;                 // log(0) guard
    return std::sqrt(-2.0 * std::log(u1)) *
           std::cos(2.0 * 3.14159265358979323846 * u2);
}

/// N(mu, sd^2).
inline double normal(std::mt19937_64& g, double mu, double sd) {
    return mu + sd * n01(g);
}

/// Uniform integer on [lo, hi], inclusive, without modulo bias.
///
/// Lemire-style rejection on the raw 64-bit output: draw until the value falls
/// in the largest multiple of the range that fits, so every outcome has exactly
/// equal probability.
inline std::uint64_t uniform_int(std::mt19937_64& g, std::uint64_t lo,
                                 std::uint64_t hi) {
    const std::uint64_t range = hi - lo + 1;
    if (range == 0) return g();                   // full 64-bit range
    const std::uint64_t limit = UINT64_MAX - (UINT64_MAX % range) - 1;
    std::uint64_t x;
    do { x = g(); } while (x > limit);
    return lo + (x % range);
}

/// Bernoulli(p).
inline int bernoulli(std::mt19937_64& g, double p) {
    return u01(g) < p ? 1 : 0;
}

/// Gamma(shape, scale), Marsaglia-Tsang (2000) squeeze method.
///
/// For shape < 1 it uses the standard boost: draw Gamma(shape + 1) and scale by
/// u^(1/shape), which is exact.
inline double gamma(std::mt19937_64& g, double shape, double scale) {
    if (shape < 1.0) {
        const double u = u01(g);
        return gamma(g, shape + 1.0, scale) *
               std::pow(u < 1e-300 ? 1e-300 : u, 1.0 / shape);
    }
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        double x, v;
        do {
            x = n01(g);
            v = 1.0 + c * x;
        } while (v <= 0.0);
        v = v * v * v;
        const double u = u01(g);
        const double x2 = x * x;
        if (u < 1.0 - 0.0331 * x2 * x2)      return d * v * scale;
        if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v)))
            return d * v * scale;
    }
}

/// Poisson(lambda). Knuth's product method below 30, where all our tests sit;
/// above that it falls back to a normal approximation with continuity
/// correction, which is adequate for generating test COVARIATES but is not a
/// general-purpose Poisson sampler -- say so if you ever need one.
inline int poisson(std::mt19937_64& g, double lambda) {
    if (lambda < 30.0) {
        const double L = std::exp(-lambda);
        int k = 0;
        double p = 1.0;
        do { ++k; p *= u01(g); } while (p > L);
        return k - 1;
    }
    const double v = lambda + std::sqrt(lambda) * n01(g);
    const int k = static_cast<int>(v + 0.5);
    return k < 0 ? 0 : k;
}

/// Categorical draw from unnormalised weights; returns an index in [0, w.size()).
inline std::size_t discrete(std::mt19937_64& g, const std::vector<double>& w) {
    double total = 0.0;
    for (double v : w) total += v;
    const double t = u01(g) * total;
    double acc = 0.0;
    for (std::size_t i = 0; i < w.size(); ++i) {
        acc += w[i];
        if (t < acc) return i;
    }
    return w.empty() ? 0 : w.size() - 1;          // only reachable on rounding
}


// ----------------------------------------------------------------------------
//  Drop-in replacements for the std:: distribution templates.
//
//  Same construction arguments and the same operator()(engine), so converting a
//  test file is a one-word change on the DECLARATION and every call site stays
//  as it is. That is the whole point: the smaller the edit, the less chance of
//  silently changing what a test measures while making it portable.
// ----------------------------------------------------------------------------

template <class RealType = double>
class normal_distribution {
public:
    explicit normal_distribution(RealType mean = 0.0, RealType stddev = 1.0)
        : mean_(mean), sd_(stddev) {}
    template <class G> RealType operator()(G& g) const { return normal(g, mean_, sd_); }
    RealType mean() const { return mean_; }
    RealType stddev() const { return sd_; }
private:
    RealType mean_, sd_;
};

template <class RealType = double>
class uniform_real_distribution {
public:
    explicit uniform_real_distribution(RealType a = 0.0, RealType b = 1.0)
        : a_(a), b_(b) {}
    template <class G> RealType operator()(G& g) const { return uniform(g, a_, b_); }
    RealType a() const { return a_; }
    RealType b() const { return b_; }
private:
    RealType a_, b_;
};

template <class IntType = int>
class uniform_int_distribution {
public:
    explicit uniform_int_distribution(IntType a = 0, IntType b = 1) : a_(a), b_(b) {}
    template <class G> IntType operator()(G& g) const {
        return static_cast<IntType>(
            uniform_int(g, static_cast<std::uint64_t>(a_),
                           static_cast<std::uint64_t>(b_)));
    }
private:
    IntType a_, b_;
};

template <class RealType = double>
class gamma_distribution {
public:
    explicit gamma_distribution(RealType shape = 1.0, RealType scale = 1.0)
        : shape_(shape), scale_(scale) {}
    template <class G> RealType operator()(G& g) const { return gamma(g, shape_, scale_); }
private:
    RealType shape_, scale_;
};

template <class IntType = int>
class poisson_distribution {
public:
    explicit poisson_distribution(double mean = 1.0) : lambda_(mean) {}
    template <class G> IntType operator()(G& g) const {
        return static_cast<IntType>(poisson(g, lambda_));
    }
private:
    double lambda_;
};

template <class IntType = int>
class discrete_distribution {
public:
    discrete_distribution() : w_{1.0} {}
    template <class It> discrete_distribution(It first, It last) : w_(first, last) {}
    discrete_distribution(std::initializer_list<double> w) : w_(w) {}
    template <class G> IntType operator()(G& g) const {
        return static_cast<IntType>(discrete(g, w_));
    }
private:
    std::vector<double> w_;
};

}  // namespace prng
