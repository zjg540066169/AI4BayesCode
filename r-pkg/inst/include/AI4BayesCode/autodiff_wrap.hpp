/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  autodiff_wrap.hpp  --  bridge between natural-scale scalar log-density
 *                         functions written with autodiff::var types and
 *                         the (theta_unc, grad_unc) interface that
 *                         nuts_block expects.
 *
 *  THE IDEA
 *  ========
 *  For a user's Bayesian model the AI-generated code writes only the
 *  scalar natural-scale log-density, using templated types that work
 *  with both `double` (for value-only eval) and `autodiff::var` (for
 *  value + gradient eval). The wrap functions in this header handle:
 *
 *    1. unpacking theta_unc -> theta_nat via the constraint's inverse
 *       transform (identity for real, exp for positive, stick-breaking
 *       for simplex, etc.), using autodiff::var so the transform itself
 *       is differentiable.
 *
 *    2. computing log|Jacobian| of the constrain transform as a var
 *       expression -- the AD reverse pass therefore picks up its
 *       contribution to grad_unc automatically via the chain rule.
 *
 *    3. calling the user-supplied natural-scale log-density on the
 *       VectorXvar theta_nat, getting a single var lp_nat.
 *
 *    4. building lp_total = lp_nat + log_jac and asking autodiff for
 *       gradient(lp_total, theta_unc_var).
 *
 *    5. copying the resulting Eigen::VectorXd gradient back into an
 *       arma::vec for nuts_block's oracle signature.
 *
 *  The AI never writes a gradient, never writes a Jacobian, never
 *  touches the unconstrained scale. Validator Check #5 ("no hand-
 *  written Jacobian") is enforced structurally: the user lambda sees
 *  only theta_nat and returns a scalar.
 *
 *  USAGE from generated code
 *  -------------------------
 *    template <typename T>
 *    T gauss_lp(const Eigen::Matrix<T, -1, 1>& theta_nat,
 *               const block_context& ctx) {
 *        T mu    = theta_nat[0];
 *        T sigma = theta_nat[1];
 *        const arma::vec& y = ctx.at("y");
 *        T lp = T(0);
 *        for (std::size_t i = 0; i < y.n_elem; ++i) {
 *            T r = y[i] - mu;
 *            lp += -0.5 * r * r / (sigma * sigma) - log(sigma);
 *        }
 *        lp += -0.5 * mu * mu / 10000.0;
 *        lp += -0.5 * sigma * sigma / 100.0;
 *        return lp;
 *    }
 *
 *    cfg.log_density_grad =
 *        [](const arma::vec& theta_unc, const block_context& ctx,
 *           arma::vec* grad) {
 *            return autodiff_wrap::mixed<autodiff_wrap::real,
 *                                        autodiff_wrap::positive>(
 *                theta_unc, grad,
 *                [&](const auto& theta_nat) {
 *                    return gauss_lp(theta_nat, ctx);
 *                });
 *        };
 *
 *  V1 constraint kinds supported here: real, positive, lower_bounded,
 *  upper_bounded, interval. Simplex / ordered / cholesky_corr /
 *  unit_vector are in V2 and live in a separate overload set once the
 *  V1 interfaces prove out.
 *================================================================================*/

#ifndef AI4BAYESCODE_AUTODIFF_WRAP_HPP
#define AI4BAYESCODE_AUTODIFF_WRAP_HPP

#include "block_sampler.hpp"

#include <autodiff/reverse/var.hpp>
#include <autodiff/reverse/var/eigen.hpp>

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {
namespace autodiff_wrap {

// Tag types for constraint kinds. Used to dispatch the per-slice
// transform + log-Jacobian inside `mixed` below. Each tag provides:
//   - static constexpr int unc_dim(int nat_dim) — for v1 all kinds have
//     unc_dim == nat_dim (no stick-breaking-style shrinkage). When we add
//     simplex we'll generalise.
//   - template <typename T> static T constrain_and_add_log_jac(
//         const Eigen::Matrix<T,-1,1>& unc_slice,
//         T& log_jac_accumulator,
//         Eigen::Matrix<T,-1,1>& nat_slice_out);
//
// The transform writes the natural-scale slice and *adds* to
// log_jac_accumulator; the outer `mixed` sums across slices.

struct real_tag {
    template <typename T>
    static void apply(const Eigen::Matrix<T, -1, 1>& unc,
                      T& log_jac,
                      Eigen::Matrix<T, -1, 1>& nat)
    {
        // identity; no Jacobian contribution
        nat = unc;
        // log_jac unchanged
        (void)log_jac;
    }
};

struct positive_tag {
    template <typename T>
    static void apply(const Eigen::Matrix<T, -1, 1>& unc,
                      T& log_jac,
                      Eigen::Matrix<T, -1, 1>& nat)
    {
        // nat = exp(unc); log|J| per element = unc[i]  (because d nat/d unc = exp(unc) = nat)
        const Eigen::Index n = unc.size();
        nat.resize(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            nat[i] = exp(unc[i]);
            log_jac = log_jac + unc[i];
        }
    }
};

struct lower_bounded_tag {
    double lo;
    template <typename T>
    void apply(const Eigen::Matrix<T, -1, 1>& unc,
               T& log_jac,
               Eigen::Matrix<T, -1, 1>& nat) const
    {
        // nat = lo + exp(unc); log|J| per element = unc[i]
        const Eigen::Index n = unc.size();
        nat.resize(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            nat[i] = T(lo) + exp(unc[i]);
            log_jac = log_jac + unc[i];
        }
    }
};

struct upper_bounded_tag {
    double hi;
    template <typename T>
    void apply(const Eigen::Matrix<T, -1, 1>& unc,
               T& log_jac,
               Eigen::Matrix<T, -1, 1>& nat) const
    {
        // nat = hi - exp(unc); log|J| per element = unc[i]
        const Eigen::Index n = unc.size();
        nat.resize(n);
        for (Eigen::Index i = 0; i < n; ++i) {
            nat[i] = T(hi) - exp(unc[i]);
            log_jac = log_jac + unc[i];
        }
    }
};

struct interval_tag {
    double lo;
    double hi;
    template <typename T>
    void apply(const Eigen::Matrix<T, -1, 1>& unc,
               T& log_jac,
               Eigen::Matrix<T, -1, 1>& nat) const
    {
        // nat = lo + (hi - lo) * sigmoid(unc);
        //   sigmoid(x) = 1 / (1 + exp(-x))
        // log|J| = log(hi - lo) - unc - 2 log(1 + exp(-unc))   (Stan reference)
        //        = log(hi - lo) + log(sigmoid(unc)) + log(1 - sigmoid(unc))
        const Eigen::Index n = unc.size();
        nat.resize(n);
        const T span = T(hi - lo);
        const T log_span = T(std::log(hi - lo));
        for (Eigen::Index i = 0; i < n; ++i) {
            // numerically stable sigmoid log
            T s = 1.0 / (1.0 + exp(-unc[i]));
            nat[i] = T(lo) + span * s;
            // log|J_i| = log(hi-lo) + log(s) + log(1-s)
            log_jac = log_jac + log_span + log(s) + log(1.0 - s);
        }
    }
};

// ---- Helper: copy arma::vec → VectorXvar, assign scalars one-by-one
//      so each entry becomes an independent var leaf suitable for
//      autodiff::gradient(...) wrt that vector.
inline void arma_to_varvec(const arma::vec& src, autodiff::VectorXvar& dst)
{
    dst.resize(static_cast<Eigen::Index>(src.n_elem));
    for (std::size_t i = 0; i < src.n_elem; ++i) {
        dst[static_cast<Eigen::Index>(i)] = src[i];
    }
}

inline void eigen_vecd_to_arma(const Eigen::VectorXd& src, arma::vec& dst)
{
    const std::size_t n = static_cast<std::size_t>(src.size());
    dst.set_size(n);
    for (std::size_t i = 0; i < n; ++i) {
        dst[i] = src[static_cast<Eigen::Index>(i)];
    }
}

// ---- Single-constraint wraps ---------------------------------------
//
// For a single constraint over the whole theta_unc, these are the simple
// entry points. For mixed constraints (e.g. real + positive in one
// block) use `mixed<...>` below.

/**
 * Wrap a user-supplied natural-scale log-density with the REAL
 * constraint (identity, no Jacobian). The user lambda f_nat takes a
 * VectorXvar theta_nat (same length as theta_unc) and returns a var.
 *
 * Signature: F : const autodiff::VectorXvar& -> autodiff::var
 */
template <typename F>
double wrap_real(const arma::vec& theta_unc,
                 arma::vec* grad_unc,
                 F f_nat)
{
    const std::size_t n = theta_unc.n_elem;
    autodiff::VectorXvar theta_var;
    arma_to_varvec(theta_unc, theta_var);

    // real: nat = unc (identity)
    autodiff::var lp = f_nat(theta_var);

    if (grad_unc) {
        Eigen::VectorXd g = autodiff::gradient(lp, theta_var);
        eigen_vecd_to_arma(g, *grad_unc);
        if (grad_unc->n_elem != n) {
            throw std::runtime_error(
                "autodiff_wrap::wrap_real: gradient length mismatch");
        }
    }
    return static_cast<double>(lp);
}

/**
 * Wrap a user-supplied natural-scale log-density with the SIMPLEX
 * constraint via stick-breaking. theta_unc has length K-1; theta_nat
 * has length K with entries > 0 summing to 1. Stan's stick-breaking
 * transform and associated Jacobian are implemented in autodiff::var
 * expressions so that reverse-mode differentiation walks through them
 * automatically.
 *
 * Stan reference:
 *   z_i      = inv_logit(y_i + log(1 / (K - i)))     for i = 1..K-1
 *   theta_1  = z_1
 *   theta_k  = (1 - sum_{j<k} theta_j) * z_k         for k = 2..K-1
 *   theta_K  = 1 - sum_{j<K} theta_j
 *   log|J|   = sum_{k=1..K-1} [log(z_k) + log(1 - z_k)
 *                              + log(1 - sum_{j<k} theta_j)]
 *
 * The user's f_nat is given theta_nat (length K, on the natural
 * simplex).
 */
template <typename F>
double wrap_simplex(const arma::vec& theta_unc,
                    arma::vec* grad_unc,
                    F f_nat)
{
    const std::size_t K_minus_1 = theta_unc.n_elem;
    const std::size_t K         = K_minus_1 + 1;

    autodiff::VectorXvar y_var;
    arma_to_varvec(theta_unc, y_var);

    autodiff::VectorXvar theta_nat_var(K);
    autodiff::var log_jac    = 0.0;
    autodiff::var sum_so_far = 0.0;

    for (std::size_t i = 0; i < K_minus_1; ++i) {
        const Eigen::Index ii = static_cast<Eigen::Index>(i);
        // Stan uses Stan's k = 1..K-1 with offset log(1/(K-k)).
        // In 0-based i = k-1, that is log(1/(K-i-1)) = -log(K-i-1).
        const double offset = -std::log(static_cast<double>(K - i - 1));

        // inv_logit(y_i + offset)
        autodiff::var arg = y_var[ii] + offset;
        autodiff::var z   = 1.0 / (1.0 + exp(-arg));

        // theta_i = (1 - sum_so_far) * z
        autodiff::var remaining = 1.0 - sum_so_far;
        theta_nat_var[ii]       = remaining * z;

        // log|J_i|
        log_jac = log_jac + log(z) + log(1.0 - z) + log(remaining);

        // update sum_so_far
        sum_so_far = sum_so_far + theta_nat_var[ii];
    }
    // last component
    theta_nat_var[static_cast<Eigen::Index>(K_minus_1)] = 1.0 - sum_so_far;

    autodiff::var lp_nat   = f_nat(theta_nat_var);
    autodiff::var lp_total = lp_nat + log_jac;

    if (grad_unc) {
        Eigen::VectorXd g = autodiff::gradient(lp_total, y_var);
        eigen_vecd_to_arma(g, *grad_unc);
    }
    return static_cast<double>(lp_total);
}

/**
 * Wrap a user-supplied natural-scale log-density with the POSITIVE
 * constraint. theta_unc on the log scale; theta_nat = exp(theta_unc);
 * log|J| = sum(theta_unc). The user lambda f_nat sees theta_nat > 0.
 */
template <typename F>
double wrap_positive(const arma::vec& theta_unc,
                     arma::vec* grad_unc,
                     F f_nat)
{
    const std::size_t n = theta_unc.n_elem;
    autodiff::VectorXvar theta_unc_var;
    arma_to_varvec(theta_unc, theta_unc_var);

    autodiff::VectorXvar theta_nat_var(n);
    autodiff::var log_jac = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const Eigen::Index ii = static_cast<Eigen::Index>(i);
        theta_nat_var[ii] = exp(theta_unc_var[ii]);
        log_jac = log_jac + theta_unc_var[ii];
    }

    autodiff::var lp_nat   = f_nat(theta_nat_var);
    autodiff::var lp_total = lp_nat + log_jac;

    if (grad_unc) {
        Eigen::VectorXd g = autodiff::gradient(lp_total, theta_unc_var);
        eigen_vecd_to_arma(g, *grad_unc);
    }
    return static_cast<double>(lp_total);
}

// ---- Mixed-constraint wrap -----------------------------------------
//
// `mixed<T1, T2, ...>(theta_unc, grad, {t1, t2, ...}, f_nat)` applies
// constraint t1 to the first slice of theta_unc, t2 to the second, etc.
// Each tag specifies its own slice length in the accompanying
// `slice_sizes` argument. The user's natural-scale lambda sees a single
// VectorXvar of the concatenated natural-scale parameters.
//
// Stubbed as a slice-based function (not variadic templates) so the
// implementation is easy to audit. If you have (real, real, positive) in
// that order with sizes (1, P, 1), call:
//
//    mixed_wrap(theta_unc, grad, {real_slice(1), real_slice(P), positive_slice(1)},
//               f_nat);
//
// This is one concrete ergonomic API; variadic-template sugar can be
// layered on top later without changing the semantics.

struct slice_spec {
    enum kind_t { REAL, POSITIVE, LOWER_BOUNDED, UPPER_BOUNDED, INTERVAL } kind;
    std::size_t dim;
    double lo = 0.0;  // lower_bounded / interval
    double hi = 0.0;  // upper_bounded / interval

    static slice_spec real(std::size_t d)     { return {REAL, d, 0.0, 0.0}; }
    static slice_spec positive(std::size_t d) { return {POSITIVE, d, 0.0, 0.0}; }
    static slice_spec lower(std::size_t d, double lo)
        { return {LOWER_BOUNDED, d, lo, 0.0}; }
    static slice_spec upper(std::size_t d, double hi)
        { return {UPPER_BOUNDED, d, 0.0, hi}; }
    static slice_spec interval(std::size_t d, double lo, double hi)
        { return {INTERVAL, d, lo, hi}; }
};

template <typename F>
double wrap_mixed(const arma::vec& theta_unc,
                  arma::vec* grad_unc,
                  const std::vector<slice_spec>& slices,
                  F f_nat)
{
    // Total dimension sanity check
    std::size_t total = 0;
    for (const auto& s : slices) total += s.dim;
    if (total != theta_unc.n_elem) {
        throw std::invalid_argument(
            "autodiff_wrap::wrap_mixed: sum of slice dims (" +
            std::to_string(total) + ") does not match theta_unc.n_elem (" +
            std::to_string(theta_unc.n_elem) + ")");
    }

    autodiff::VectorXvar theta_unc_var;
    arma_to_varvec(theta_unc, theta_unc_var);

    autodiff::VectorXvar theta_nat_var(total);
    autodiff::var log_jac = 0.0;

    std::size_t off = 0;
    for (const auto& s : slices) {
        // Extract the unconstrained slice as a VectorXvar sub-block.
        autodiff::VectorXvar unc_slice(s.dim);
        for (std::size_t i = 0; i < s.dim; ++i) {
            unc_slice[static_cast<Eigen::Index>(i)] =
                theta_unc_var[static_cast<Eigen::Index>(off + i)];
        }
        autodiff::VectorXvar nat_slice(s.dim);
        switch (s.kind) {
            case slice_spec::REAL: {
                real_tag::apply(unc_slice, log_jac, nat_slice);
                break;
            }
            case slice_spec::POSITIVE: {
                positive_tag::apply(unc_slice, log_jac, nat_slice);
                break;
            }
            case slice_spec::LOWER_BOUNDED: {
                lower_bounded_tag t{s.lo};
                t.apply(unc_slice, log_jac, nat_slice);
                break;
            }
            case slice_spec::UPPER_BOUNDED: {
                upper_bounded_tag t{s.hi};
                t.apply(unc_slice, log_jac, nat_slice);
                break;
            }
            case slice_spec::INTERVAL: {
                interval_tag t{s.lo, s.hi};
                t.apply(unc_slice, log_jac, nat_slice);
                break;
            }
        }
        for (std::size_t i = 0; i < s.dim; ++i) {
            theta_nat_var[static_cast<Eigen::Index>(off + i)] =
                nat_slice[static_cast<Eigen::Index>(i)];
        }
        off += s.dim;
    }

    autodiff::var lp_nat   = f_nat(theta_nat_var);
    autodiff::var lp_total = lp_nat + log_jac;

    if (grad_unc) {
        Eigen::VectorXd g = autodiff::gradient(lp_total, theta_unc_var);
        eigen_vecd_to_arma(g, *grad_unc);
    }
    return static_cast<double>(lp_total);
}

} // namespace autodiff_wrap
} // namespace AI4BayesCode

#endif // AI4BAYESCODE_AUTODIFF_WRAP_HPP
