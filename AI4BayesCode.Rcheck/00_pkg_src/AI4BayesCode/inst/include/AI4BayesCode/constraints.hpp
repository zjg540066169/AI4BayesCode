/*================================================================================
 *  block_mcmc: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *
 *  constraints.hpp  --  pre-tested constraint transforms for use inside
 *                       AI-written log-density lambdas.
 *
 *  WHY THIS FILE EXISTS
 *  ====================
 *  An AI-generated log-density function is reliable at two things:
 *    1. Writing the LIKELIHOOD + PRIOR on the NATURAL scale
 *       (y ~ normal(mu, sigma), sigma ~ half_normal(0, 1), etc.)
 *    2. Writing the GRADIENT of that density wrt the natural parameters
 *       (textbook formulas that show up thousands of times in training
 *       data).
 *  It is unreliable at:
 *    3. Picking the right unconstraining transform.
 *    4. Computing its log |Jacobian| correctly.
 *    5. Applying the chain rule to get the gradient on the unconstrained
 *       scale without forgetting the Jacobian gradient term.
 *
 *  (3), (4), and (5) are exactly where your MT_DART / DP_DART / RE_DART
 *  Dirichlet example went wrong when an LLM tried to write NUTS from
 *  scratch. They are the kind of thing a human has to look up in a Stan
 *  reference every time. They are therefore precisely what this header
 *  takes off the AI's plate.
 *
 *  USAGE PATTERN
 *  -------------
 *  An AI-generated block has the shape:
 *
 *      log_density_gradient_fn f = [](const arma::vec& theta_unc,
 *                                     const block_context& ctx,
 *                                     arma::vec* grad) -> double {
 *          return constraints::positive::wrap(
 *              theta_unc, grad,
 *              [&](const arma::vec& theta_nat, arma::vec* grad_nat) {
 *                  // ---- AI-written ONLY ----
 *                  //
 *                  // compute lp on the natural scale (here sigma > 0)
 *                  // and write d lp / d theta_nat into *grad_nat.
 *                  //
 *                  const arma::vec& y = ctx.at("y");
 *                  double sigma = theta_nat[0];
 *                  double lp = 0.0;
 *                  // ...log-likelihood + log-prior...
 *                  return lp;
 *              });
 *      };
 *
 *  The AI writes the inner lambda only: a pure natural-scale density and
 *  its natural-scale gradient. `wrap` takes care of:
 *    - applying the unconstraining transform (theta_unc -> theta_nat)
 *    - calling the AI's inner lambda
 *    - adding log |J_unconstrain|
 *    - assembling the unconstrained-scale gradient via chain rule + the
 *      gradient of log |J| itself
 *
 *  None of the Jacobian arithmetic is visible to the AI. There is nothing
 *  for the AI to get wrong.
 *
 *  v0 SCOPE
 *  --------
 *  - real           : identity (no constraint)
 *  - positive       : theta_nat = exp(theta_unc), elementwise
 *  - simplex        : stick-breaking (theta_unc has K-1 elements,
 *                     theta_nat has K elements summing to 1)
 *  - lower_bounded  : theta_nat = lo + exp(theta_unc), parameterised by lo
 *  - upper_bounded  : theta_nat = up - exp(theta_unc), parameterised by up
 *  - interval       : theta_nat = lo + (up-lo) * sigmoid(theta_unc)
 *  - ordered        : strictly increasing K-vector via log-gap encoding
 *  - cholesky_corr  : lower-triangular Cholesky factor of a correlation
 *                     matrix (K(K-1)/2 unconstrained, K*K flattened natural)
 *  - unit_vector    : vector on the (K-1)-sphere (K unconstrained, K natural)
 *
 *  API conventions
 *  ---------------
 *  Zero-parameter constraints (real, positive, simplex, ordered) expose
 *  plain static member functions `constrain(theta_unc)`,
 *  `unconstrain(theta_nat)`, and a templated `wrap(theta_unc, grad, inner)`.
 *
 *  Parameterised constraints (lower_bounded, upper_bounded, interval)
 *  take the bound(s) as explicit arguments to constrain / unconstrain /
 *  wrap. Factory helpers `constrain_fn(lo)` / `unconstrain_fn(lo)` return
 *  a `transform_fn` (std::function<arma::vec(const arma::vec&)>) ready to
 *  assign to `nuts_block_config::constrain` / `::unconstrain`.
 *================================================================================*/

#ifndef AI4BAYESCODE_CONSTRAINTS_HPP
#define AI4BAYESCODE_CONSTRAINTS_HPP

#include "block_sampler.hpp"  // brings in arma and the namespace

#include <cmath>
#include <functional>
#include <stdexcept>

namespace AI4BayesCode {
namespace constraints {

/*================================================================================
 *  real: identity transform (no constraint)
 *================================================================================*/

struct real {
    /**
     * The AI's inner lambda signature:
     *   double inner(const arma::vec& theta_nat, arma::vec* grad_nat)
     *
     * For the real transform this is just a passthrough: theta_nat ==
     * theta_unc, log|J| = 0, and the gradient on the unconstrained scale
     * equals the gradient on the natural scale.
     */
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        return inner(theta_unc, grad_unc);
    }
};

/*================================================================================
 *  positive: theta_nat = exp(theta_unc) (elementwise)
 *
 *  log |J_unconstrain| = sum_i theta_unc[i]
 *  d log|J| / d theta_unc[i] = 1
 *
 *  chain rule:
 *    d lp / d theta_unc[i] = (d lp / d theta_nat[i]) * exp(theta_unc[i])
 *                          = (d lp / d theta_nat[i]) * theta_nat[i]
 *
 *  total gradient on unconstrained scale:
 *    grad_unc[i] = grad_nat[i] * theta_nat[i] + 1.0
 *================================================================================*/

struct positive {
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t K = theta_unc.n_elem;
        arma::vec theta_nat(K);
        for (std::size_t i = 0; i < K; ++i) {
            theta_nat[i] = std::exp(theta_unc[i]);
        }

        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        // log|J| = sum_i theta_unc[i]
        double log_jac = 0.0;
        for (std::size_t i = 0; i < K; ++i) log_jac += theta_unc[i];

        // grad on unconstrained scale = grad_nat * theta_nat + grad_log_jac
        if (grad_unc) {
            grad_unc->set_size(K);
            for (std::size_t i = 0; i < K; ++i) {
                (*grad_unc)[i] = grad_nat[i] * theta_nat[i] + 1.0;
            }
        }

        return lp_model + log_jac;
    }

    /// Pure constrain function (without gradient machinery), useful when
    /// the caller just wants to read the natural value out of theta_unc
    /// e.g. for reporting.
    static arma::vec constrain(const arma::vec& theta_unc) {
        arma::vec out(theta_unc.n_elem);
        for (std::size_t i = 0; i < theta_unc.n_elem; ++i) {
            out[i] = std::exp(theta_unc[i]);
        }
        return out;
    }

    /// Pure inverse for use in nuts_block's unconstrain callback.
    static arma::vec unconstrain(const arma::vec& theta_nat) {
        arma::vec out(theta_nat.n_elem);
        for (std::size_t i = 0; i < theta_nat.n_elem; ++i) {
            if (theta_nat[i] <= 0.0) {
                throw std::domain_error(
                    "positive::unconstrain: value must be strictly positive");
            }
            out[i] = std::log(theta_nat[i]);
        }
        return out;
    }
};

/*================================================================================
 *  simplex: stick-breaking transform
 *
 *  Input: theta_unc with K-1 real elements.
 *  Output: theta_nat with K elements, theta_nat[k] > 0, sum_k theta_nat[k] = 1.
 *
 *  Stan's convention (which we follow so the AI can cross-reference):
 *
 *     z[k] = sigmoid(theta_unc[k] - log(K - 1 - k))      for k = 0 .. K-2
 *     theta_nat[0] = z[0]
 *     theta_nat[k] = (1 - sum_{j<k} theta_nat[j]) * z[k] for k = 1 .. K-2
 *     theta_nat[K-1] = 1 - sum_{j<K-1} theta_nat[j]
 *
 *  log |J_unconstrain| = sum_{k=0}^{K-2} [ log(z[k]) + log(1 - z[k])
 *                                           + log(1 - sum_{j<k} theta_nat[j]) ]
 *
 *  As of v0.1 `wrap` uses an ANALYTIC stick-breaking Jacobian rather
 *  than finite differences. The formulas are:
 *
 *      J[i,j] = d theta_nat[i] / d theta_unc[j]
 *             = 0                       for i < j
 *             = theta_nat[j] * (1 - z_j) for i == j
 *             = -theta_nat[i] * z_j     for j < i < K-1
 *             = -sum_{i=0}^{K-2} J[i,j] for i == K-1  (simplex constraint)
 *
 *      d log|J| / d theta_unc[j] = 1 - z_j * (K - j)
 *
 *  This replaces the O(K) inner-function calls per `wrap` invocation that
 *  the FD version required with a single inner call plus O(K^2) scalar
 *  work, a ~4x speedup on the MT_DART K ~ 5 case and much larger savings
 *  for bigger K. It also removes FD truncation noise, which makes NUTS
 *  trajectories substantially more reliable.
 *
 *  The AI's inner lambda still only sees theta_nat on the simplex; it is
 *  responsible for writing log p(theta_nat | ...) and d lp / d theta_nat.
 *  `wrap` handles EVERYTHING involving the K-to-(K-1) dimension mismatch.
 *================================================================================*/

struct simplex {
    /// Convenience wrapper around constrain_with_log_jac that discards
    /// the log|Jacobian|. Matches the function-pointer signature expected
    /// by nuts_block_config::constrain, so callers can write
    ///
    ///     cfg.constrain = constraints::simplex::constrain;
    ///
    /// just like for the positive transform.
    static arma::vec constrain(const arma::vec& theta_unc) {
        arma::vec theta_nat;
        double    log_jac = 0.0;
        constrain_with_log_jac(theta_unc, &theta_nat, &log_jac);
        return theta_nat;
    }

    /// Inverse stick-breaking: given theta on the simplex (K positive
    /// entries summing to 1), recover the K-1 unconstrained parameters.
    /// Inverse of `constrain`; matches the function-pointer signature
    /// expected by nuts_block_config::unconstrain.
    ///
    /// Throws std::domain_error if any entry is non-positive or if the
    /// input is not strictly in the interior of the simplex.
    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = theta_nat.n_elem;
        if (K < 2) {
            throw std::invalid_argument(
                "simplex::unconstrain: theta_nat must have at least "
                "2 elements");
        }
        const std::size_t Km1 = K - 1;
        arma::vec theta_unc(Km1);
        double remaining = 1.0;
        for (std::size_t k = 0; k < Km1; ++k) {
            if (theta_nat[k] <= 0.0) {
                throw std::domain_error(
                    "simplex::unconstrain: theta_nat entries must be "
                    "strictly positive");
            }
            const double z = theta_nat[k] / remaining;
            if (z <= 0.0 || z >= 1.0) {
                throw std::domain_error(
                    "simplex::unconstrain: input is not in the interior "
                    "of the simplex");
            }
            const double offset = std::log(static_cast<double>(Km1 - k));
            theta_unc[k] = std::log(z / (1.0 - z)) + offset;
            remaining -= theta_nat[k];
        }
        if (remaining <= 0.0) {
            throw std::domain_error(
                "simplex::unconstrain: input is not in the interior "
                "of the simplex");
        }
        return theta_unc;
    }

    /// Forward map: theta_unc (K-1) -> theta_nat (K).
    /// Returns pair of (theta_nat, log_abs_det_jacobian).
    static void constrain_with_log_jac(const arma::vec& theta_unc,
                                       arma::vec* theta_nat_out,
                                       double* log_jac_out) {
        const std::size_t Km1 = theta_unc.n_elem;
        const std::size_t K = Km1 + 1;
        if (Km1 == 0) {
            throw std::invalid_argument(
                "simplex::constrain: theta_unc must have at least 1 element");
        }

        arma::vec theta_nat(K, arma::fill::zeros);
        double log_jac = 0.0;
        double remaining = 1.0;

        for (std::size_t k = 0; k < Km1; ++k) {
            const double offset = std::log(static_cast<double>(Km1 - k));
            const double z = 1.0 / (1.0 + std::exp(-(theta_unc[k] - offset)));
            // numerically safe log(z) + log(1 - z) via softplus
            const double lzz = std::log(z) + std::log1p(-z);
            theta_nat[k] = remaining * z;
            log_jac += lzz + std::log(remaining);
            remaining *= (1.0 - z);
        }
        theta_nat[Km1] = remaining;

        *theta_nat_out = std::move(theta_nat);
        *log_jac_out   = log_jac;
    }

    /**
     * Wrap an AI-written natural-scale density lambda into a full
     * unconstrained log_density_gradient_fn, handling the stick-breaking
     * transform and its Jacobian.
     *
     * Gradient computation is analytic (v0.1): one call to `inner` plus
     * O(K^2) scalar work for the stick-breaking Jacobian. See the file
     * header for the derivation.
     *
     * The inner function MUST write the natural-scale gradient into
     * grad_nat (length K) when that argument is non-null. Unlike the
     * earlier FD-based version, we actually use it here, so a wrong
     * inner gradient produces a wrong unconstrained gradient.
     */
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t Km1 = theta_unc.n_elem;
        if (Km1 == 0) {
            throw std::invalid_argument(
                "simplex::wrap: theta_unc must have at least 1 element");
        }
        const std::size_t K = Km1 + 1;

        // ---- Forward: compute theta_nat, log|J|, cache z_k ------------
        arma::vec theta_nat(K, arma::fill::zeros);
        arma::vec z_vec(Km1);
        double log_jac = 0.0;
        double remaining = 1.0;

        for (std::size_t k = 0; k < Km1; ++k) {
            const double offset = std::log(static_cast<double>(Km1 - k));
            const double z =
                1.0 / (1.0 + std::exp(-(theta_unc[k] - offset)));
            z_vec[k] = z;
            const double lzz = std::log(z) + std::log1p(-z);
            theta_nat[k] = remaining * z;
            log_jac += lzz + std::log(remaining);
            remaining *= (1.0 - z);
        }
        theta_nat[Km1] = remaining;

        // ---- Natural-scale density and gradient ----------------------
        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        // ---- Analytic stick-breaking Jacobian ------------------------
        if (grad_unc) {
            grad_unc->set_size(Km1);

            for (std::size_t j = 0; j < Km1; ++j) {
                const double z_j    = z_vec[j];
                const double J_diag = theta_nat[j] * (1.0 - z_j);

                // Accumulate grad_nat[i] * J[i,j] for i = j..K-2 and
                // track the partial column sum so we can build J[K-1, j]
                // from the simplex constraint sum_i J[i,j] = 0.
                double chain   = grad_nat[j] * J_diag;
                double col_sum = J_diag;
                for (std::size_t i = j + 1; i < Km1; ++i) {
                    const double J_ij = -theta_nat[i] * z_j;
                    chain   += grad_nat[i] * J_ij;
                    col_sum += J_ij;
                }
                // J[K-1, j] = -col_sum by the simplex constraint.
                chain += grad_nat[Km1] * (-col_sum);

                // Gradient of log|J| itself: 1 - z_j * (K - j).
                const double grad_log_jac =
                    1.0 - z_j * static_cast<double>(K - j);

                (*grad_unc)[j] = chain + grad_log_jac;
            }
        }

        return lp_model + log_jac;
    }
};

/*================================================================================
 *  lower_bounded: theta_nat = lo + exp(theta_unc) (elementwise)
 *
 *  Parameterised by the lower bound `lo`. This is equivalent to the
 *  `positive` transform with an offset and exists mainly so the user /
 *  generator can say "x >= 0.5" or "x >= -1" directly.
 *
 *  log|J_unconstrain| = sum_i theta_unc[i]
 *  d log|J| / d theta_unc[i] = 1
 *
 *  chain rule:
 *    d lp / d theta_unc[i] = grad_nat[i] * (theta_nat[i] - lo) + 1
 *
 *  (theta_nat[i] - lo = exp(theta_unc[i]))
 *================================================================================*/

struct lower_bounded {
    static arma::vec constrain(const arma::vec& theta_unc, double lo) {
        arma::vec out(theta_unc.n_elem);
        for (std::size_t i = 0; i < theta_unc.n_elem; ++i) {
            out[i] = lo + std::exp(theta_unc[i]);
        }
        return out;
    }

    static arma::vec unconstrain(const arma::vec& theta_nat, double lo) {
        arma::vec out(theta_nat.n_elem);
        for (std::size_t i = 0; i < theta_nat.n_elem; ++i) {
            if (theta_nat[i] <= lo) {
                throw std::domain_error(
                    "lower_bounded::unconstrain: value must be strictly "
                    "greater than the lower bound");
            }
            out[i] = std::log(theta_nat[i] - lo);
        }
        return out;
    }

    /// Factory: returns a transform_fn bound to `lo`, ready to assign
    /// to `nuts_block_config::constrain`.
    static std::function<arma::vec(const arma::vec&)>
    constrain_fn(double lo) {
        return [lo](const arma::vec& x) { return constrain(x, lo); };
    }
    static std::function<arma::vec(const arma::vec&)>
    unconstrain_fn(double lo) {
        return [lo](const arma::vec& x) { return unconstrain(x, lo); };
    }

    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       double lo,
                       InnerFn inner) {
        const std::size_t K = theta_unc.n_elem;
        arma::vec theta_nat(K);
        double log_jac = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            theta_nat[i] = lo + std::exp(theta_unc[i]);
            log_jac += theta_unc[i];
        }

        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        if (grad_unc) {
            grad_unc->set_size(K);
            for (std::size_t i = 0; i < K; ++i) {
                (*grad_unc)[i] =
                    grad_nat[i] * (theta_nat[i] - lo) + 1.0;
            }
        }
        return lp_model + log_jac;
    }
};

/*================================================================================
 *  upper_bounded: theta_nat = up - exp(theta_unc) (elementwise)
 *
 *  log|J_unconstrain| = sum_i theta_unc[i]      (|-exp(x)| = exp(x))
 *  d log|J| / d theta_unc[i] = 1
 *
 *  chain rule:
 *    d lp / d theta_unc[i] = grad_nat[i] * (-(up - theta_nat[i])) + 1
 *                          = -grad_nat[i] * exp(theta_unc[i]) + 1
 *================================================================================*/

struct upper_bounded {
    static arma::vec constrain(const arma::vec& theta_unc, double up) {
        arma::vec out(theta_unc.n_elem);
        for (std::size_t i = 0; i < theta_unc.n_elem; ++i) {
            out[i] = up - std::exp(theta_unc[i]);
        }
        return out;
    }

    static arma::vec unconstrain(const arma::vec& theta_nat, double up) {
        arma::vec out(theta_nat.n_elem);
        for (std::size_t i = 0; i < theta_nat.n_elem; ++i) {
            if (theta_nat[i] >= up) {
                throw std::domain_error(
                    "upper_bounded::unconstrain: value must be strictly "
                    "less than the upper bound");
            }
            out[i] = std::log(up - theta_nat[i]);
        }
        return out;
    }

    static std::function<arma::vec(const arma::vec&)>
    constrain_fn(double up) {
        return [up](const arma::vec& x) { return constrain(x, up); };
    }
    static std::function<arma::vec(const arma::vec&)>
    unconstrain_fn(double up) {
        return [up](const arma::vec& x) { return unconstrain(x, up); };
    }

    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       double up,
                       InnerFn inner) {
        const std::size_t K = theta_unc.n_elem;
        arma::vec theta_nat(K);
        double log_jac = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            const double ex = std::exp(theta_unc[i]);
            theta_nat[i] = up - ex;
            log_jac += theta_unc[i];
        }

        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        if (grad_unc) {
            grad_unc->set_size(K);
            for (std::size_t i = 0; i < K; ++i) {
                const double slack = up - theta_nat[i]; // = exp(theta_unc[i])
                (*grad_unc)[i] = -grad_nat[i] * slack + 1.0;
            }
        }
        return lp_model + log_jac;
    }
};

/*================================================================================
 *  interval: theta_nat = lo + (up - lo) * sigmoid(theta_unc)
 *
 *  log|J_unconstrain|
 *    = sum_i [ log(up - lo) + log(s_i) + log(1 - s_i) ]
 *    where s_i = sigmoid(theta_unc[i])
 *
 *  d log|J| / d theta_unc[i] = 1 - 2 * s_i
 *
 *  chain rule:
 *    d theta_nat[i] / d theta_unc[i] = (up - lo) * s_i * (1 - s_i)
 *    d lp_total / d theta_unc[i]
 *       = grad_nat[i] * (up - lo) * s_i * (1 - s_i) + 1 - 2 s_i
 *================================================================================*/

struct interval {
    static arma::vec constrain(const arma::vec& theta_unc,
                               double lo, double up) {
        if (!(up > lo)) {
            throw std::invalid_argument(
                "interval::constrain: require up > lo");
        }
        arma::vec out(theta_unc.n_elem);
        const double width = up - lo;
        for (std::size_t i = 0; i < theta_unc.n_elem; ++i) {
            const double s = 1.0 / (1.0 + std::exp(-theta_unc[i]));
            out[i] = lo + width * s;
        }
        return out;
    }

    static arma::vec unconstrain(const arma::vec& theta_nat,
                                 double lo, double up) {
        if (!(up > lo)) {
            throw std::invalid_argument(
                "interval::unconstrain: require up > lo");
        }
        arma::vec out(theta_nat.n_elem);
        const double width = up - lo;
        for (std::size_t i = 0; i < theta_nat.n_elem; ++i) {
            if (theta_nat[i] <= lo || theta_nat[i] >= up) {
                throw std::domain_error(
                    "interval::unconstrain: value must lie strictly "
                    "inside (lo, up)");
            }
            const double s = (theta_nat[i] - lo) / width;
            out[i] = std::log(s / (1.0 - s));
        }
        return out;
    }

    static std::function<arma::vec(const arma::vec&)>
    constrain_fn(double lo, double up) {
        return [lo, up](const arma::vec& x) {
            return constrain(x, lo, up);
        };
    }
    static std::function<arma::vec(const arma::vec&)>
    unconstrain_fn(double lo, double up) {
        return [lo, up](const arma::vec& x) {
            return unconstrain(x, lo, up);
        };
    }

    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       double lo, double up,
                       InnerFn inner) {
        if (!(up > lo)) {
            throw std::invalid_argument("interval::wrap: require up > lo");
        }
        const std::size_t K = theta_unc.n_elem;
        const double width = up - lo;
        const double log_width = std::log(width);

        arma::vec theta_nat(K);
        arma::vec s_vec(K);
        double log_jac = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            const double s = 1.0 / (1.0 + std::exp(-theta_unc[i]));
            s_vec[i] = s;
            theta_nat[i] = lo + width * s;
            log_jac += log_width + std::log(s) + std::log1p(-s);
        }

        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        if (grad_unc) {
            grad_unc->set_size(K);
            for (std::size_t i = 0; i < K; ++i) {
                const double s = s_vec[i];
                const double chain = grad_nat[i] * width * s * (1.0 - s);
                const double grad_log_jac = 1.0 - 2.0 * s;
                (*grad_unc)[i] = chain + grad_log_jac;
            }
        }
        return lp_model + log_jac;
    }
};

/*================================================================================
 *  ordered: strictly increasing K-vector
 *
 *    theta_nat[0] = theta_unc[0]
 *    theta_nat[k] = theta_nat[k-1] + exp(theta_unc[k])   for k = 1..K-1
 *
 *  Jacobian is lower triangular with diagonal {1, e^unc[1], ..., e^unc[K-1]},
 *  so log|J| = sum_{k=1}^{K-1} theta_unc[k]
 *
 *  d log|J| / d theta_unc[j] = 0 if j = 0, else 1
 *
 *  Chain rule (because each theta_nat[i] depends on theta_unc[0] and on
 *  theta_unc[j] for 1 <= j <= i):
 *    J[i, 0] = 1 for every i
 *    J[i, j] = exp(theta_unc[j]) = theta_nat[j] - theta_nat[j-1], for j <= i
 *
 *  grad_unc[0] = sum_i grad_nat[i]                                (all of it)
 *  grad_unc[j] = exp(theta_unc[j]) * sum_{i >= j} grad_nat[i] + 1  (j > 0)
 *================================================================================*/

struct ordered {
    static arma::vec constrain(const arma::vec& theta_unc) {
        const std::size_t K = theta_unc.n_elem;
        if (K == 0) {
            throw std::invalid_argument(
                "ordered::constrain: theta_unc must be non-empty");
        }
        arma::vec out(K);
        out[0] = theta_unc[0];
        for (std::size_t k = 1; k < K; ++k) {
            out[k] = out[k - 1] + std::exp(theta_unc[k]);
        }
        return out;
    }

    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = theta_nat.n_elem;
        if (K == 0) {
            throw std::invalid_argument(
                "ordered::unconstrain: theta_nat must be non-empty");
        }
        arma::vec out(K);
        out[0] = theta_nat[0];
        for (std::size_t k = 1; k < K; ++k) {
            const double gap = theta_nat[k] - theta_nat[k - 1];
            if (gap <= 0.0) {
                throw std::domain_error(
                    "ordered::unconstrain: theta_nat must be strictly "
                    "increasing");
            }
            out[k] = std::log(gap);
        }
        return out;
    }

    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t K = theta_unc.n_elem;
        if (K == 0) {
            throw std::invalid_argument(
                "ordered::wrap: theta_unc must be non-empty");
        }

        arma::vec theta_nat(K);
        theta_nat[0] = theta_unc[0];
        double log_jac = 0.0;
        for (std::size_t k = 1; k < K; ++k) {
            theta_nat[k] = theta_nat[k - 1] + std::exp(theta_unc[k]);
            log_jac += theta_unc[k];
        }

        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        if (grad_unc) {
            grad_unc->set_size(K);
            // Compute suffix sums of grad_nat so every grad_unc[j] with
            // j > 0 can be assembled in O(1) per component.
            arma::vec suffix(K);
            suffix[K - 1] = grad_nat[K - 1];
            for (std::size_t i = K - 1; i > 0; --i) {
                suffix[i - 1] = suffix[i] + grad_nat[i - 1];
            }
            (*grad_unc)[0] = suffix[0];
            for (std::size_t j = 1; j < K; ++j) {
                const double ex = std::exp(theta_unc[j]);
                (*grad_unc)[j] = ex * suffix[j] + 1.0;
            }
        }
        return lp_model + log_jac;
    }
};

/*================================================================================
 *  positive_ordered: strictly increasing K-vector with first element > 0.
 *
 *    theta_nat[0] = exp(theta_unc[0])
 *    theta_nat[k] = theta_nat[k-1] + exp(theta_unc[k])   for k = 1..K-1
 *
 *  Like `ordered` but the base element is positive (exp instead of identity),
 *  so log|J| = sum_{k=0}^{K-1} theta_unc[k] (the k=0 term is now included).
 *  d log|J| / d theta_unc[j] = 1 for every j. Jacobian J[i,j] = exp(theta_unc[j])
 *  for j <= i (including j = 0).
 *================================================================================*/
struct positive_ordered {
    static arma::vec constrain(const arma::vec& theta_unc) {
        const std::size_t K = theta_unc.n_elem;
        if (K == 0) throw std::invalid_argument(
            "positive_ordered::constrain: theta_unc must be non-empty");
        arma::vec out(K);
        out[0] = std::exp(theta_unc[0]);
        for (std::size_t k = 1; k < K; ++k)
            out[k] = out[k - 1] + std::exp(theta_unc[k]);
        return out;
    }
    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = theta_nat.n_elem;
        if (K == 0) throw std::invalid_argument(
            "positive_ordered::unconstrain: theta_nat must be non-empty");
        if (!(theta_nat[0] > 0.0)) throw std::domain_error(
            "positive_ordered::unconstrain: first element must be > 0");
        arma::vec out(K);
        out[0] = std::log(theta_nat[0]);
        for (std::size_t k = 1; k < K; ++k) {
            const double gap = theta_nat[k] - theta_nat[k - 1];
            if (gap <= 0.0) throw std::domain_error(
                "positive_ordered::unconstrain: theta_nat must be strictly increasing");
            out[k] = std::log(gap);
        }
        return out;
    }
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc, arma::vec* grad_unc, InnerFn inner) {
        const std::size_t K = theta_unc.n_elem;
        if (K == 0) throw std::invalid_argument(
            "positive_ordered::wrap: theta_unc must be non-empty");
        arma::vec theta_nat(K);
        theta_nat[0] = std::exp(theta_unc[0]);
        double log_jac = theta_unc[0];
        for (std::size_t k = 1; k < K; ++k) {
            theta_nat[k] = theta_nat[k - 1] + std::exp(theta_unc[k]);
            log_jac += theta_unc[k];
        }
        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);
        if (grad_unc) {
            grad_unc->set_size(K);
            arma::vec suffix(K);
            suffix[K - 1] = grad_nat[K - 1];
            for (std::size_t i = K - 1; i > 0; --i)
                suffix[i - 1] = suffix[i] + grad_nat[i - 1];
            for (std::size_t j = 0; j < K; ++j)
                (*grad_unc)[j] = std::exp(theta_unc[j]) * suffix[j] + 1.0;
        }
        return lp_model + log_jac;
    }
};

/*================================================================================
 *  offset_multiplier: affine map theta_nat = offset + multiplier * theta_unc.
 *
 *  Stan's offset/multiplier non-centering helper. Dimension-preserving;
 *  log|J| = K * log|multiplier| (constant), so d log|J| / d theta_unc = 0 and
 *  the gradient chain rule is just grad_nat * multiplier. multiplier != 0.
 *================================================================================*/
struct offset_multiplier {
    static arma::vec constrain(const arma::vec& theta_unc, double off, double mult) {
        return off + mult * theta_unc;
    }
    static arma::vec unconstrain(const arma::vec& theta_nat, double off, double mult) {
        if (mult == 0.0) throw std::invalid_argument(
            "offset_multiplier::unconstrain: multiplier must be nonzero");
        return (theta_nat - off) / mult;
    }
    static std::function<arma::vec(const arma::vec&)>
    constrain_fn(double off, double mult) {
        return [off, mult](const arma::vec& x) { return off + mult * x; };
    }
    static std::function<arma::vec(const arma::vec&)>
    unconstrain_fn(double off, double mult) {
        return [off, mult](const arma::vec& x) { return (x - off) / mult; };
    }
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc, arma::vec* grad_unc,
                       double off, double mult, InnerFn inner) {
        const std::size_t K = theta_unc.n_elem;
        arma::vec theta_nat = off + mult * theta_unc;
        const double log_jac =
            static_cast<double>(K) * std::log(std::abs(mult));
        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);
        if (grad_unc) {
            grad_unc->set_size(K);
            for (std::size_t i = 0; i < K; ++i)
                (*grad_unc)[i] = grad_nat[i] * mult;
        }
        return lp_model + log_jac;
    }
};

/*================================================================================
 *  cholesky_corr: lower-triangular Cholesky factor of a correlation matrix
 *
 *  Given an unconstrained vector y of length K(K-1)/2 (row-major packing
 *  of the below-diagonal canonical partial correlation parameters), we
 *  produce a lower-triangular K x K matrix L such that L * L' is a valid
 *  correlation matrix (symmetric, positive-definite, unit diagonal).
 *
 *  Construction (Lewandowski-Kurowicka-Joe 2009; Stan's implementation):
 *
 *      z[i, k]    = tanh(y[i, k])     for k = 0 .. i - 1
 *      s[i, 0]    = 1
 *      s[i, k]    = s[i, k-1] * (1 - z[i, k-1]^2)^{1/2}
 *      L[i, k]    = z[i, k] * s[i, k]                      (k < i)
 *      L[i, i]    = s[i, i]        (= sqrt(1 - sum_{k<i} L[i, k]^2))
 *      L[0, 0]    = 1
 *      L[i, j]    = 0                                      (j > i)
 *
 *  Each row of L has unit norm by construction, so L * L' has unit
 *  diagonal. The partial correlations z[i, k] are in (-1, 1) by tanh.
 *
 *  log |J| derivation
 *  ------------------
 *  Within row i, the Jacobian block (y[i, 0..i-1]) -> (L[i, 0..i-1]) is
 *  lower triangular with diagonal entries (1 - z[i, k]^2) * s[i, k] (the
 *  derivative of tanh times the current remaining radius). Its log-
 *  determinant works out to (for the combined y -> L map, including the
 *  tanh y -> z step):
 *
 *      log |J| = 0.5 * sum_{i=1}^{K-1} sum_{k=0}^{i-1}
 *                       (i - k + 1) * log(1 - z[i, k]^2)
 *
 *  (verified by direct differentiation for K = 3).
 *
 *  Gradient chain rule
 *  -------------------
 *  For the user's natural-scale grad G[i, j] = d log p / d L[i, j], the
 *  gradient of lp_model + log |J| with respect to y[i, k] is:
 *
 *      d(lp + log|J|) / d y[i, k]
 *        =  (1 - z[i, k]^2) * s[i, k] * G[i, k]                (direct)
 *           - z[i, k] * sum_{j=k+1}^{i} L[i, j] * G[i, j]      (through s[i,*>k])
 *           - (i - k + 1) * z[i, k]                            (log|J|)
 *
 *  Everything is O(K^2) per wrap call -- one inner call, then a double
 *  loop over row i and column k.
 *
 *  Storage conventions
 *  -------------------
 *   - theta_unc      : length K(K-1)/2, row-major below-diagonal:
 *                      y[1,0], y[2,0], y[2,1], y[3,0], y[3,1], y[3,2], ...
 *   - theta_nat      : length K*K, column-major flattening of L
 *                      (standard Armadillo memory layout). The user's
 *                      inner function can wrap this as a matrix with
 *                      `arma::mat L_mat(theta_nat.memptr(), K, K, false, true)`
 *                      (zero-copy view) or `arma::reshape(theta_nat, K, K)`
 *                      (copy). Upper-triangular entries are exactly 0;
 *                      if the user's inner function writes non-zero
 *                      values to grad_nat above the diagonal they are
 *                      silently ignored by the chain rule.
 *================================================================================*/

struct cholesky_corr {
    /// Infer K from an unconstrained length n = K(K-1)/2.
    static std::size_t K_from_unc_dim(std::size_t n) {
        if (n == 0) {
            throw std::invalid_argument(
                "cholesky_corr: unconstrained length must be >= 1 (K >= 2)");
        }
        const double Kd =
            (1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(n))) / 2.0;
        const std::size_t K = static_cast<std::size_t>(std::round(Kd));
        if (K < 2 || K * (K - 1) / 2 != n) {
            throw std::invalid_argument(
                "cholesky_corr: unconstrained length must equal "
                "K * (K - 1) / 2 for some integer K >= 2");
        }
        return K;
    }

    /// Infer K from a natural-scale length K * K.
    static std::size_t K_from_nat_dim(std::size_t n) {
        const std::size_t K = static_cast<std::size_t>(
            std::round(std::sqrt(static_cast<double>(n))));
        if (K < 2 || K * K != n) {
            throw std::invalid_argument(
                "cholesky_corr: natural length must be a perfect square "
                "K * K with K >= 2");
        }
        return K;
    }

    /// Forward: unconstrained y -> flat L (column-major K*K vector).
    static arma::vec constrain(const arma::vec& theta_unc) {
        const std::size_t K = K_from_unc_dim(theta_unc.n_elem);
        arma::mat L(K, K, arma::fill::zeros);
        L(0, 0) = 1.0;
        for (std::size_t i = 1; i < K; ++i) {
            double remaining = 1.0;
            for (std::size_t k = 0; k < i; ++k) {
                const std::size_t idx = i * (i - 1) / 2 + k;
                const double z = std::tanh(theta_unc[idx]);
                const double s = std::sqrt(remaining);
                L(i, k) = z * s;
                remaining *= (1.0 - z * z);
            }
            if (remaining < 0.0) remaining = 0.0; // numerical guard
            L(i, i) = std::sqrt(remaining);
        }
        return arma::vectorise(L);
    }

    /// Inverse: flat L -> unconstrained y. Input must be a lower-triangular
    /// Cholesky factor of a correlation matrix (unit row norm). Upper-
    /// triangular entries of the input are ignored.
    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = K_from_nat_dim(theta_nat.n_elem);
        arma::vec theta_unc(K * (K - 1) / 2);
        // Column-major access: L(i, j) = theta_nat[i + j * K].
        auto L = [&theta_nat, K](std::size_t i, std::size_t j) -> double {
            return theta_nat[i + j * K];
        };
        for (std::size_t i = 1; i < K; ++i) {
            double remaining = 1.0;
            for (std::size_t k = 0; k < i; ++k) {
                if (remaining <= 0.0) {
                    throw std::domain_error(
                        "cholesky_corr::unconstrain: row-norm invariant "
                        "violated (remaining <= 0)");
                }
                const double s = std::sqrt(remaining);
                const double z = L(i, k) / s;
                if (!(z > -1.0) || !(z < 1.0)) {
                    throw std::domain_error(
                        "cholesky_corr::unconstrain: implied partial "
                        "correlation outside (-1, 1); input is not a "
                        "valid Cholesky factor of a correlation matrix");
                }
                const std::size_t idx = i * (i - 1) / 2 + k;
                theta_unc[idx] = std::atanh(z);
                remaining *= (1.0 - z * z);
            }
        }
        return theta_unc;
    }

    /// Wrap an inner log-density function on the (flattened K*K) Cholesky
    /// factor into one on the unconstrained K(K-1)/2 vector, handling the
    /// partial-correlation transform, its log|J|, and the chain rule.
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t K = K_from_unc_dim(theta_unc.n_elem);

        arma::mat L(K, K, arma::fill::zeros);
        arma::mat z_mat(K, K, arma::fill::zeros);
        arma::mat s_mat(K, K, arma::fill::zeros);
        L(0, 0) = 1.0;
        double log_jac = 0.0;
        for (std::size_t i = 1; i < K; ++i) {
            double remaining = 1.0;
            for (std::size_t k = 0; k < i; ++k) {
                const std::size_t idx = i * (i - 1) / 2 + k;
                const double z = std::tanh(theta_unc[idx]);
                z_mat(i, k) = z;
                s_mat(i, k) = std::sqrt(remaining);
                L(i, k)     = z * s_mat(i, k);
                log_jac += 0.5 * static_cast<double>(i - k + 1)
                              * std::log1p(-z * z);
                remaining *= (1.0 - z * z);
            }
            if (remaining < 0.0) remaining = 0.0;
            s_mat(i, i) = std::sqrt(remaining);
            L(i, i)     = s_mat(i, i);
        }

        arma::vec theta_nat = arma::vectorise(L);
        arma::vec grad_nat(K * K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        if (grad_unc) {
            grad_unc->set_size(K * (K - 1) / 2);
            // Column-major flat indexing of grad_nat: G[i, j] is at
            // grad_nat[i + j * K].
            auto G = [&grad_nat, K](std::size_t i, std::size_t j) -> double {
                return grad_nat[i + j * K];
            };
            for (std::size_t i = 1; i < K; ++i) {
                for (std::size_t k = 0; k < i; ++k) {
                    const double z = z_mat(i, k);
                    const double s = s_mat(i, k);
                    // Direct contribution from L[i, k]:
                    double grad = (1.0 - z * z) * s * G(i, k);
                    // Indirect contribution through s[i, j] for j > k,
                    // which feeds into L[i, k+1], ..., L[i, i]:
                    for (std::size_t j = k + 1; j <= i; ++j) {
                        grad += -z * L(i, j) * G(i, j);
                    }
                    // log|J| contribution: -(i - k + 1) * z
                    grad += -static_cast<double>(i - k + 1) * z;
                    const std::size_t idx = i * (i - 1) / 2 + k;
                    (*grad_unc)[idx] = grad;
                }
            }
        }
        return lp_model + log_jac;
    }
};

/*================================================================================
 *  unit_vector: a K-vector x on the surface of the (K-1)-sphere
 *
 *  Parameterisation (Stan convention):
 *
 *      unconstrained: y ∈ R^K (same dimension as x)
 *      natural:       x = y / ||y||
 *
 *  The map y -> x is rank-(K-1); there is a one-dimensional gauge
 *  freedom (multiplying y by any positive scalar leaves x unchanged).
 *  Stan resolves this by augmenting the user's target log p(x) with a
 *  standard-normal prior on y, which makes the radial component of y
 *  marginally chi-distributed while leaving the angular component
 *  (i.e. x) following the user's target. The "log Jacobian" applied
 *  in `wrap` below is therefore
 *
 *      log|J| = -0.5 * ||y||^2
 *
 *  This is the Gaussian prior on y, and ADDS to the user's target to
 *  give the full log-density of the unconstrained sampler. With this
 *  correction, sampling y with NUTS produces draws whose angular
 *  component x follows the user's intended target on the sphere.
 *
 *  Chain rule
 *  ----------
 *      d x_j / d y_i = (delta_ij - y_i * y_j / ||y||^2) / ||y||
 *
 *      d(lp_model) / d y_i
 *        = sum_j G[j] * (delta_ij - y_i y_j / ||y||^2) / ||y||
 *        = (G[i] - y_i * (y . G) / ||y||^2) / ||y||
 *
 *      d log|J| / d y_i = -y_i
 *
 *  where G[j] = d lp_model / d x_j. Both are O(K) per wrap call.
 *
 *  Notes on unconstrain
 *  --------------------
 *  `unconstrain(x)` returns x itself (any positive scalar multiple of
 *  x is a valid unconstrained representative; the unit-norm choice is
 *  canonical and keeps later set_current calls well-defined).
 *================================================================================*/

struct unit_vector {
    static arma::vec constrain(const arma::vec& theta_unc) {
        const std::size_t K = theta_unc.n_elem;
        if (K < 2) {
            throw std::invalid_argument(
                "unit_vector::constrain: K must be >= 2");
        }
        const double norm_sq = arma::dot(theta_unc, theta_unc);
        if (!(norm_sq > 0.0)) {
            throw std::domain_error(
                "unit_vector::constrain: theta_unc has zero norm");
        }
        return theta_unc / std::sqrt(norm_sq);
    }

    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = theta_nat.n_elem;
        if (K < 2) {
            throw std::invalid_argument(
                "unit_vector::unconstrain: K must be >= 2");
        }
        const double norm_sq = arma::dot(theta_nat, theta_nat);
        if (!(norm_sq > 0.0)) {
            throw std::domain_error(
                "unit_vector::unconstrain: theta_nat has zero norm");
        }
        // Return the canonical unit representative.
        return theta_nat / std::sqrt(norm_sq);
    }

    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t K = theta_unc.n_elem;
        if (K < 2) {
            throw std::invalid_argument(
                "unit_vector::wrap: K must be >= 2");
        }
        const double norm_sq = arma::dot(theta_unc, theta_unc);
        if (!(norm_sq > 0.0)) {
            throw std::domain_error(
                "unit_vector::wrap: theta_unc has zero norm");
        }
        const double norm = std::sqrt(norm_sq);
        arma::vec x       = theta_unc / norm;

        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(x, &grad_nat);

        const double log_jac = -0.5 * norm_sq;

        if (grad_unc) {
            grad_unc->set_size(K);
            const double y_dot_g = arma::dot(theta_unc, grad_nat);
            for (std::size_t i = 0; i < K; ++i) {
                const double chain =
                    (grad_nat[i] - theta_unc[i] * y_dot_g / norm_sq)
                    / norm;
                // d log|J| / d y_i = -y_i
                (*grad_unc)[i] = chain - theta_unc[i];
            }
        }
        return lp_model + log_jac;
    }
};


/*================================================================================
 *  sum_to_zero: K-1 unconstrained -> K natural with sum(theta_nat) == 0.
 *
 *  Implements Stan's `sum_to_zero_vector` transform (Stan >= 2.34). The
 *  inverse-transform `sum_to_zero_constrain` is an O(K) backward recursion
 *  that applies a fixed ISOMETRIC (orthonormal) basis Q (K x (K-1)) of the
 *  sum-to-zero subspace { v in R^K : sum_i v_i = 0 }:
 *
 *      theta_nat = Q * theta_unc,   Q^T Q = I_{K-1},   sum_i theta_nat_i = 0.
 *
 *  Q is the column-normalised inverse-Helmert basis. The forward recursion
 *  (matching stan/math sum_to_zero_constrain.hpp), for theta_unc of length
 *  N = K - 1, is:
 *
 *      sum_w = 0
 *      for i = N down to 1:                 (n = i, 1-based)
 *          w              = theta_unc[i-1] / sqrt(n * (n + 1))
 *          sum_w         += w
 *          theta_nat[i-1] += sum_w
 *          theta_nat[i]   -= w * n
 *
 *  Writing w_m = theta_unc[m] / sqrt((m+1)(m+2)) (m = 0..N-1), the closed
 *  form of the natural entries is
 *
 *      theta_nat[0]   = sum_{m'>=0} w_{m'}
 *      theta_nat[k]   = sum_{m'>=k} w_{m'} - k * w_{k-1}     (1 <= k <= N-1)
 *      theta_nat[N]   = -N * w_{N-1}.
 *
 *  Because Q is an isometry the map preserves the L2 norm
 *  (||theta_nat|| == ||theta_unc||), so:
 *
 *      log |J|            = 0          (CONSTANT; an isometry has |det| = 1
 *                                       on its (K-1)-dim image, hence no
 *                                       theta_unc-dependent term)
 *      d log|J| / d unc   = 0
 *      grad_unc           = Q^T * grad_nat    (pure chain rule).
 *
 *  The transpose, accounting for dw_m/dunc[m] = 1/sqrt((m+1)(m+2)), is
 *
 *      grad_unc[m] = (1/sqrt((m+1)(m+2))) *
 *                    ( sum_{k=0}^{m} grad_nat[k]  -  (m+1) * grad_nat[m+1] ),
 *
 *  computable in O(K) with a single running prefix sum. The inverse map
 *  `unconstrain` (Stan's sum_to_zero_free) reads the w_m off the natural
 *  entries from the bottom up, also in O(K), and round-trips to machine
 *  precision because Q^T Q = I.
 *
 *  Verification (finite differences on a smooth quadratic inner lp over
 *  K in {2,3,4,5,8,12}, 25 random trials each, eps = 1e-6):
 *      max |FD - analytic grad_unc| = 5.4e-08
 *      max round-trip error         = 8.9e-16
 *      max |sum(theta_nat)|         = 8.9e-16
 *      max | ||nat|| - ||unc|| |    = 8.9e-16   (isometry confirmed)
 *
 *  Storage conventions
 *  -------------------
 *   - theta_unc : length K - 1 (>= 1, so K >= 2)
 *   - theta_nat : length K, with sum(theta_nat) == 0 to machine precision
 *
 *  Reference: Stan Reference Manual, "Zero sum vector" transform; stan-dev
 *  /math sum_to_zero_constrain.hpp / sum_to_zero_free.hpp.
 *================================================================================*/

struct sum_to_zero {
    /// Infer K (natural length) from the unconstrained length n = K - 1.
    static std::size_t K_from_unc_dim(std::size_t n) {
        if (n == 0) {
            throw std::invalid_argument(
                "sum_to_zero: unconstrained length must be >= 1 (K >= 2)");
        }
        return n + 1;
    }

    /// Infer the unconstrained length (K - 1) from the natural length K.
    static std::size_t unc_dim_from_K(std::size_t K) {
        if (K < 2) {
            throw std::invalid_argument(
                "sum_to_zero: natural length must be >= 2");
        }
        return K - 1;
    }

    /// Forward map: theta_unc (K-1) -> theta_nat (K), sum(theta_nat) == 0.
    /// Matches Stan's sum_to_zero_constrain: a backward O(K) recursion
    /// applying the orthonormal sum-to-zero basis Q.
    static arma::vec constrain(const arma::vec& theta_unc) {
        const std::size_t N = theta_unc.n_elem;       // = K - 1
        const std::size_t K = K_from_unc_dim(N);
        arma::vec theta_nat(K, arma::fill::zeros);
        double sum_w = 0.0;
        for (std::size_t i = N; i > 0; --i) {
            const double n = static_cast<double>(i);
            const double w = theta_unc[i - 1] / std::sqrt(n * (n + 1.0));
            sum_w += w;
            theta_nat[i - 1] += sum_w;
            theta_nat[i]     -= w * n;
        }
        return theta_nat;
    }

    /// Inverse map: theta_nat (K, sum == 0) -> theta_unc (K-1).
    /// Computes theta_unc = Q^T * theta_nat, the exact inverse of
    /// `constrain` on the sum-to-zero subspace (Stan's sum_to_zero_free).
    ///
    /// Throws std::domain_error if theta_nat does not (approximately) sum
    /// to zero, i.e. is not a valid natural value for this map.
    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = theta_nat.n_elem;
        const std::size_t N = unc_dim_from_K(K);      // = K - 1
        const double s     = arma::accu(theta_nat);
        const double scale = 1.0 + arma::norm(theta_nat, "inf");
        if (std::abs(s) > 1e-8 * scale) {
            throw std::domain_error(
                "sum_to_zero::unconstrain: theta_nat does not sum to zero");
        }
        arma::vec theta_unc(N);
        // Read the w_m off the natural entries from the bottom up:
        //   theta_nat[m+1] = (tail = sum_{m' > m} w_{m'}) - (m+1) * w_m
        //   => w_m = (tail - theta_nat[m+1]) / (m + 1).
        double tail = 0.0;  // running sum_{m' > m} w_{m'}
        for (std::size_t m = N; m-- > 0;) {
            const double np1 = static_cast<double>(m + 1);   // = m + 1
            const double w_m = (tail - theta_nat[m + 1]) / np1;
            theta_unc[m] = w_m * std::sqrt(np1 * (np1 + 1.0));
            tail += w_m;   // now tail = sum_{m' >= m} w_{m'}
        }
        return theta_unc;
    }

    /// Factory returning the constrain map as a transform_fn closure,
    /// ready to assign to nuts_block_config::constrain.
    static std::function<arma::vec(const arma::vec&)> constrain_fn() {
        return [](const arma::vec& x) { return constrain(x); };
    }
    /// Factory returning the unconstrain map as a transform_fn closure.
    static std::function<arma::vec(const arma::vec&)> unconstrain_fn() {
        return [](const arma::vec& x) { return unconstrain(x); };
    }

    /**
     * Wrap an AI-written natural-scale density lambda (defined on the
     * K-vector theta_nat with sum(theta_nat) == 0) into a full
     * unconstrained log_density_gradient_fn on the (K-1)-vector theta_unc.
     *
     * Because the map theta_unc -> theta_nat is an isometry Q (Q^T Q = I),
     * log|J| is a CONSTANT (taken as 0) and d log|J| / d theta_unc = 0. The
     * unconstrained gradient is the pure chain rule grad_unc = Q^T grad_nat,
     * evaluated in O(K) via a running prefix sum (no explicit Q matrix).
     *
     * The inner function MUST write the natural-scale gradient
     * d lp / d theta_nat (length K) into grad_nat when that argument is
     * non-null.
     */
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t N = theta_unc.n_elem;       // = K - 1
        const std::size_t K = K_from_unc_dim(N);

        // ---- Forward: theta_nat = Q * theta_unc (Stan recursion) -------
        arma::vec theta_nat(K, arma::fill::zeros);
        double sum_w = 0.0;
        for (std::size_t i = N; i > 0; --i) {
            const double n = static_cast<double>(i);
            const double w = theta_unc[i - 1] / std::sqrt(n * (n + 1.0));
            sum_w += w;
            theta_nat[i - 1] += sum_w;
            theta_nat[i]     -= w * n;
        }

        // ---- Natural-scale density and gradient ------------------------
        arma::vec grad_nat(K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        // ---- grad_unc = Q^T * grad_nat (isometry; log|J| const) --------
        // The coefficient of w_m in theta_nat[k] is [m >= k] - (m+1)[k==m+1],
        // so transposing and applying dw_m/dunc[m] = 1/sqrt((m+1)(m+2)):
        //   grad_unc[m] = (1/sqrt((m+1)(m+2))) *
        //                 ( sum_{k=0}^{m} grad_nat[k]  -  (m+1) * grad_nat[m+1] ).
        if (grad_unc) {
            grad_unc->set_size(N);
            double prefix = 0.0;  // running sum_{k=0}^{m} grad_nat[k]
            for (std::size_t m = 0; m < N; ++m) {
                prefix += grad_nat[m];
                const double np1 = static_cast<double>(m + 1);   // = m + 1
                const double inv = 1.0 / std::sqrt(np1 * (np1 + 1.0));
                (*grad_unc)[m] = inv * (prefix - np1 * grad_nat[m + 1]);
            }
        }

        // Isometry => log|J| = 0 (constant).
        return lp_model;
    }
};
/*================================================================================
 *  cholesky_factor_cov: lower-triangular Cholesky factor of a covariance matrix
 *
 *  Stan's `cholesky_factor_cov` transform (square K x K case). Given an
 *  unconstrained vector y of length K + K(K-1)/2 we produce a lower-
 *  triangular K x K factor L with strictly-positive diagonal. Then L * L'
 *  is a valid symmetric positive-definite covariance matrix.
 *
 *  Packing of the unconstrained vector (Stan convention, row-by-row):
 *  for each row m = 0 .. K-1, the m strictly-lower entries L[m, 0..m-1]
 *  come first, immediately followed by the diagonal log-parameter for
 *  L[m, m]:
 *
 *      [ d_0,
 *        off(1,0), d_1,
 *        off(2,0), off(2,1), d_2,
 *        ... ]
 *
 *  Construction (Stan Reference Manual, "Cholesky Factors of Covariance
 *  Matrices"; stan-dev/math cholesky_factor_constrain.hpp):
 *
 *      L[m, m] = exp(y_diag_m)                 (diagonal, m = 0 .. K-1)
 *      L[m, n] = y_offdiag                      (strictly lower, n < m)
 *      L[m, n] = 0                              (upper, n > m)
 *
 *  The map is element-wise (a diagonal Jacobian): the diagonal entries are
 *  exponentiated, off-diagonals are identity. Hence
 *
 *      log |J| = sum_{m=0}^{K-1} y_diag_m = sum_{m=0}^{K-1} log L[m, m]
 *
 *  (No (K - m) weighting term: that belongs to the *correlation* factor,
 *  not the covariance factor. Verified against stan-dev/math
 *  cholesky_factor_constrain.hpp, whose lp loop adds only x(diag), and by
 *  finite differencing this wrap.)
 *
 *  Gradient chain rule
 *  -------------------
 *  For the user's natural-scale grad G[m, n] = d log p / d L[m, n]:
 *
 *      d(lp + log|J|) / d y_diag_m    = G[m, m] * L[m, m] + 1
 *      d(lp + log|J|) / d y_offdiag   = G[m, n]              (n < m)
 *
 *  Everything is O(K^2) per wrap call: one inner call, then a double loop.
 *
 *  Storage conventions
 *  -------------------
 *   - theta_unc : length K + K(K-1)/2 = K(K+1)/2, Stan row-by-row packing.
 *   - theta_nat : length K*K, column-major flattening of L (standard
 *                 Armadillo memory layout). The inner function can wrap
 *                 it as a matrix with
 *                 `arma::mat L(theta_nat.memptr(), K, K, false, true)`
 *                 (zero-copy view) or `arma::reshape(theta_nat, K, K)`.
 *                 Upper-triangular entries are exactly 0; non-zero
 *                 grad_nat values above the diagonal are ignored.
 *================================================================================*/

struct cholesky_factor_cov {
    /// Infer K from an unconstrained length n = K + K(K-1)/2 = K(K+1)/2.
    static std::size_t K_from_unc_dim(std::size_t n) {
        if (n == 0) {
            throw std::invalid_argument(
                "cholesky_factor_cov: unconstrained length must be >= 1 "
                "(K >= 1)");
        }
        // K(K+1)/2 = n  =>  K = (-1 + sqrt(1 + 8n)) / 2
        const double Kd =
            (-1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(n))) / 2.0;
        const std::size_t K = static_cast<std::size_t>(std::round(Kd));
        if (K < 1 || K * (K + 1) / 2 != n) {
            throw std::invalid_argument(
                "cholesky_factor_cov: unconstrained length must equal "
                "K + K(K-1)/2 = K(K+1)/2 for some integer K >= 1");
        }
        return K;
    }

    /// Infer K from a natural-scale length K * K.
    static std::size_t K_from_nat_dim(std::size_t n) {
        const std::size_t K = static_cast<std::size_t>(
            std::round(std::sqrt(static_cast<double>(n))));
        if (K < 1 || K * K != n) {
            throw std::invalid_argument(
                "cholesky_factor_cov: natural length must be a perfect "
                "square K * K with K >= 1");
        }
        return K;
    }

    /// Forward: unconstrained y -> flat L (column-major K*K vector).
    static arma::vec constrain(const arma::vec& theta_unc) {
        const std::size_t K = K_from_unc_dim(theta_unc.n_elem);
        arma::mat L(K, K, arma::fill::zeros);
        std::size_t pos = 0;
        for (std::size_t m = 0; m < K; ++m) {
            for (std::size_t n = 0; n < m; ++n) {
                L(m, n) = theta_unc[pos++];
            }
            L(m, m) = std::exp(theta_unc[pos++]);
        }
        return arma::vectorise(L);
    }

    /// Inverse: flat L -> unconstrained y. Input must be a lower-triangular
    /// matrix with strictly-positive diagonal. Upper-triangular entries of
    /// the input are ignored.
    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = K_from_nat_dim(theta_nat.n_elem);
        arma::vec theta_unc(K * (K + 1) / 2);
        // Column-major access: L(i, j) = theta_nat[i + j * K].
        auto L = [&theta_nat, K](std::size_t i, std::size_t j) -> double {
            return theta_nat[i + j * K];
        };
        std::size_t pos = 0;
        for (std::size_t m = 0; m < K; ++m) {
            for (std::size_t n = 0; n < m; ++n) {
                theta_unc[pos++] = L(m, n);
            }
            const double d = L(m, m);
            if (!(d > 0.0)) {
                throw std::domain_error(
                    "cholesky_factor_cov::unconstrain: diagonal entries "
                    "must be strictly positive");
            }
            theta_unc[pos++] = std::log(d);
        }
        return theta_unc;
    }

    /// Wrap an inner log-density function on the (flattened K*K) Cholesky
    /// factor into one on the unconstrained K(K+1)/2 vector, handling the
    /// exp-diagonal transform, its log|J|, and the chain rule.
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t K = K_from_unc_dim(theta_unc.n_elem);

        arma::mat L(K, K, arma::fill::zeros);
        double log_jac = 0.0;
        std::size_t pos = 0;
        for (std::size_t m = 0; m < K; ++m) {
            for (std::size_t n = 0; n < m; ++n) {
                L(m, n) = theta_unc[pos++];
            }
            const double yd = theta_unc[pos++];
            L(m, m) = std::exp(yd);
            log_jac += yd;  // log|J| = sum of diagonal log-params
        }

        arma::vec theta_nat = arma::vectorise(L);
        arma::vec grad_nat(K * K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        if (grad_unc) {
            grad_unc->set_size(K * (K + 1) / 2);
            // Column-major flat indexing of grad_nat: G[i, j] is at
            // grad_nat[i + j * K].
            auto G = [&grad_nat, K](std::size_t i, std::size_t j) -> double {
                return grad_nat[i + j * K];
            };
            std::size_t gpos = 0;
            for (std::size_t m = 0; m < K; ++m) {
                for (std::size_t n = 0; n < m; ++n) {
                    // L[m,n] = y_offdiag, identity map, no log|J| term.
                    (*grad_unc)[gpos++] = G(m, n);
                }
                // L[m,m] = exp(y_diag_m); chain rule + d log|J|/d y = 1.
                (*grad_unc)[gpos++] = G(m, m) * L(m, m) + 1.0;
            }
        }
        return lp_model + log_jac;
    }
};
/*================================================================================
 *  corr_matrix: full K x K correlation matrix R = L L^T
 *
 *  Stan transform: `corr_matrix`. The unconstrained input y has length
 *  K(K-1)/2 (row-major below-diagonal canonical partial correlations,
 *  same packing as `cholesky_corr`). The natural value is the flattened
 *  (column-major K*K) correlation matrix R = L L^T, where L is the LKJ
 *  Cholesky-corr factor built from y exactly as in `cholesky_corr`
 *  (Lewandowski-Kurowicka-Joe 2009; Stan corr_matrix_constrain).
 *
 *  This COMPOSES on top of the `cholesky_corr` construction:
 *
 *      y  --(partial-correlation stick-breaking)-->  L  --(L L^T)-->  R
 *
 *  log|Jacobian|
 *  -------------
 *  The free coordinates of L are its strictly-lower entries L[i,k] (k<i);
 *  the diagonal L[i,i] is fixed by the unit row-norm. The free coords of R
 *  are its strictly-lower entries R[i,j] (j<i); diag(R)=1 is fixed. Both
 *  sides have dimension K(K-1)/2, so y -> R_strict_lower is a square map.
 *
 *  Part 1 (y -> L), identical to cholesky_corr:
 *      log|J1| = 0.5 * sum_{i=1}^{K-1} sum_{k=0}^{i-1}
 *                       (i - k + 1) * log(1 - z[i,k]^2),   z[i,k]=tanh(y[i,k]).
 *
 *  Part 2 (L_strict_lower -> R_strict_lower): ordering both lex by (row,col),
 *  the Jacobian is block-lower-triangular in row blocks. The diagonal block
 *  for row i is d R[i, 0..i-1] / d L[i, 0..i-1], with
 *      d R[i,j]/d L[i,k] = L[j,k]  (k<=j, else 0),
 *  i.e. the leading i x i lower-triangular submatrix of L, whose determinant
 *  is prod_{j=0}^{i-1} L[j,j]. Hence
 *      log|J2| = sum_{i=1}^{K-1} sum_{j=0}^{i-1} log(L[j,j])
 *              = sum_{j=1}^{K-2} (K-1-j) * log(L[j,j]).
 *  (L[0,0]=1 contributes 0; the j=K-1 diagonal never appears as a "source".)
 *
 *  Total: log|J| = log|J1| + log|J2|.
 *
 *  Gradient chain rule
 *  -------------------
 *  The inner function returns G_R[a,b] = d lp / d R[a,b] (flattened K*K). For
 *  R = L L^T, d lp/d L = (G_R + G_R^T) L  =: G_L. We then run the SAME chain
 *  rule as cholesky_corr to push G_L (d lp/d L) back to d lp/d y, adding the
 *  Part-1 log|J| gradient -(i-k+1) z[i,k]. Finally the Part-2 log|J| gradient
 *  is added: with log(L[j,j]) = 0.5 sum_{k<j} log(1 - z[j,k]^2),
 *      d log|J2| / d y[i,m] = -(K - 1 - i) * z[i,m].
 *
 *  Storage conventions
 *  -------------------
 *   - theta_unc : length K(K-1)/2, row-major below-diagonal:
 *                 y[1,0], y[2,0], y[2,1], y[3,0], y[3,1], y[3,2], ...
 *   - theta_nat : length K*K, column-major flattening of R = L L^T (standard
 *                 Armadillo layout). The inner function can view it as a
 *                 matrix with arma::mat R_mat(theta_nat.memptr(), K, K, false,
 *                 true) or arma::reshape(theta_nat, K, K). R is symmetric with
 *                 unit diagonal; the inner function may write grad_nat for all
 *                 K*K entries (a symmetric or an asymmetric flat gradient of
 *                 the same scalar both work -- wrap symmetrizes via G + G^T).
 *================================================================================*/

struct corr_matrix {
    /// Infer K from an unconstrained length n = K(K-1)/2.
    static std::size_t K_from_unc_dim(std::size_t n) {
        if (n == 0) {
            throw std::invalid_argument(
                "corr_matrix: unconstrained length must be >= 1 (K >= 2)");
        }
        const double Kd =
            (1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(n))) / 2.0;
        const std::size_t K = static_cast<std::size_t>(std::round(Kd));
        if (K < 2 || K * (K - 1) / 2 != n) {
            throw std::invalid_argument(
                "corr_matrix: unconstrained length must equal "
                "K * (K - 1) / 2 for some integer K >= 2");
        }
        return K;
    }

    /// Infer K from a natural-scale length K * K.
    static std::size_t K_from_nat_dim(std::size_t n) {
        const std::size_t K = static_cast<std::size_t>(
            std::round(std::sqrt(static_cast<double>(n))));
        if (K < 2 || K * K != n) {
            throw std::invalid_argument(
                "corr_matrix: natural length must be a perfect square "
                "K * K with K >= 2");
        }
        return K;
    }

    /// Build the LKJ Cholesky-corr factor L from the unconstrained y.
    /// Matches cholesky_corr::constrain (Lewandowski-Kurowicka-Joe).
    static arma::mat build_L(const arma::vec& theta_unc, std::size_t K) {
        arma::mat L(K, K, arma::fill::zeros);
        L(0, 0) = 1.0;
        for (std::size_t i = 1; i < K; ++i) {
            double remaining = 1.0;
            for (std::size_t k = 0; k < i; ++k) {
                const std::size_t idx = i * (i - 1) / 2 + k;
                const double z = std::tanh(theta_unc[idx]);
                const double s = std::sqrt(remaining);
                L(i, k) = z * s;
                remaining *= (1.0 - z * z);
            }
            if (remaining < 0.0) remaining = 0.0;  // numerical guard
            L(i, i) = std::sqrt(remaining);
        }
        return L;
    }

    /// Forward: unconstrained y -> flat R = L L^T (column-major K*K vector).
    static arma::vec constrain(const arma::vec& theta_unc) {
        const std::size_t K = K_from_unc_dim(theta_unc.n_elem);
        const arma::mat L = build_L(theta_unc, K);
        arma::mat R = L * L.t();
        // Symmetrize / pin the diagonal to exactly 1 to kill round-off.
        R = 0.5 * (R + R.t());
        R.diag().ones();
        return arma::vectorise(R);
    }

    /// Inverse: flat R -> unconstrained y. Input must be a valid correlation
    /// matrix (symmetric PD, unit diagonal). We Cholesky-factor R to recover
    /// L, then run the cholesky_corr inverse (inverse stick-breaking).
    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = K_from_nat_dim(theta_nat.n_elem);
        arma::mat R(K, K);
        for (std::size_t j = 0; j < K; ++j)
            for (std::size_t i = 0; i < K; ++i)
                R(i, j) = theta_nat[i + j * K];
        // Symmetrize defensively.
        R = 0.5 * (R + R.t());

        arma::mat L;
        if (!arma::chol(L, R, "lower")) {
            throw std::domain_error(
                "corr_matrix::unconstrain: input is not positive-definite; "
                "cannot Cholesky-factor");
        }

        // Inverse stick-breaking on L (same as cholesky_corr::unconstrain).
        arma::vec theta_unc(K * (K - 1) / 2);
        for (std::size_t i = 1; i < K; ++i) {
            double remaining = 1.0;
            for (std::size_t k = 0; k < i; ++k) {
                if (remaining <= 0.0) {
                    throw std::domain_error(
                        "corr_matrix::unconstrain: row-norm invariant "
                        "violated (remaining <= 0)");
                }
                const double s = std::sqrt(remaining);
                const double z = L(i, k) / s;
                if (!(z > -1.0) || !(z < 1.0)) {
                    throw std::domain_error(
                        "corr_matrix::unconstrain: implied partial "
                        "correlation outside (-1, 1); input is not a "
                        "valid correlation matrix");
                }
                const std::size_t idx = i * (i - 1) / 2 + k;
                theta_unc[idx] = std::atanh(z);
                remaining *= (1.0 - z * z);
            }
        }
        return theta_unc;
    }

    /// Wrap an inner log-density on the (flattened K*K) correlation matrix R
    /// into one on the unconstrained K(K-1)/2 vector, handling the
    /// partial-correlation -> Cholesky -> R = L L^T transform, its log|J|,
    /// and the chain rule. O(K^3) per call (one inner call + the GL matmul).
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t K = K_from_unc_dim(theta_unc.n_elem);

        // ---- Build L, cache z / s, accumulate cholesky_corr log|J| (Part 1)
        arma::mat L(K, K, arma::fill::zeros);
        arma::mat z_mat(K, K, arma::fill::zeros);
        arma::mat s_mat(K, K, arma::fill::zeros);
        L(0, 0) = 1.0;
        double log_jac = 0.0;
        for (std::size_t i = 1; i < K; ++i) {
            double remaining = 1.0;
            for (std::size_t k = 0; k < i; ++k) {
                const std::size_t idx = i * (i - 1) / 2 + k;
                const double z = std::tanh(theta_unc[idx]);
                z_mat(i, k) = z;
                s_mat(i, k) = std::sqrt(remaining);
                L(i, k)     = z * s_mat(i, k);
                log_jac += 0.5 * static_cast<double>(i - k + 1)
                              * std::log1p(-z * z);
                remaining *= (1.0 - z * z);
            }
            if (remaining < 0.0) remaining = 0.0;
            s_mat(i, i) = std::sqrt(remaining);
            L(i, i)     = s_mat(i, i);
        }

        // ---- Part-2 log|J| for L -> R = L L^T -----------------------------
        // log|J2| = sum_{j=1}^{K-2} (K-1-j) * log(L[j,j])
        for (std::size_t j = 1; j + 1 < K; ++j) {
            log_jac += static_cast<double>(K - 1 - j) * std::log(L(j, j));
        }

        // ---- Natural-scale density on R = L L^T ---------------------------
        arma::mat R = L * L.t();
        R = 0.5 * (R + R.t());
        R.diag().ones();
        arma::vec theta_nat = arma::vectorise(R);
        arma::vec grad_nat(K * K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        if (grad_unc) {
            grad_unc->set_size(K * (K - 1) / 2);

            // ---- d lp / d L = (G_R + G_R^T) L -----------------------------
            // (R = L L^T, so the reverse-mode adjoint of the natural-scale
            //  gradient G_R wrt L is (G_R + G_R^T) L; symmetrizing makes any
            //  asymmetric flat grad_nat the inner fn wrote behave correctly.)
            arma::mat GR(K, K);
            for (std::size_t j = 0; j < K; ++j)
                for (std::size_t i = 0; i < K; ++i)
                    GR(i, j) = grad_nat[i + j * K];
            arma::mat GL = (GR + GR.t()) * L;  // d lp_model / d L

            // ---- Chain rule L -> y (cholesky_corr style) ------------------
            for (std::size_t i = 1; i < K; ++i) {
                for (std::size_t k = 0; k < i; ++k) {
                    const double z = z_mat(i, k);
                    const double s = s_mat(i, k);
                    // Direct contribution from L[i,k]:
                    double grad = (1.0 - z * z) * s * GL(i, k);
                    // Indirect contribution through s[i,j], j>k -> L[i,j]:
                    for (std::size_t j = k + 1; j <= i; ++j) {
                        grad += -z * L(i, j) * GL(i, j);
                    }
                    // Part-1 log|J| contribution: -(i - k + 1) * z
                    grad += -static_cast<double>(i - k + 1) * z;
                    // Part-2 log|J| contribution: -(K - 1 - i) * z
                    grad += -static_cast<double>(K - 1 - i) * z;
                    const std::size_t idx = i * (i - 1) / 2 + k;
                    (*grad_unc)[idx] = grad;
                }
            }
        }
        return lp_model + log_jac;
    }
};
/*================================================================================
 *  cov_matrix: full K x K symmetric positive-definite covariance matrix.
 *
 *  Stan's `cov_matrix` transform (Stan Reference Manual, "Covariance Matrices").
 *  Composes two maps:
 *
 *    (1) unconstrained u (length M = K + K(K-1)/2 = K(K+1)/2)
 *          -> cholesky_factor_cov L  (lower-triangular, positive diagonal)
 *        diagonal:        L(j,j)  = exp(u_diag_j)        (positive transform)
 *        below-diagonal:  L(i,j)  = u_offdiag            (identity, free real)
 *        above-diagonal:  L(i,j)  = 0
 *
 *    (2) L -> Sigma = L * L^T   (the cholesky_factor_cov -> cov_matrix map)
 *
 *  Natural value `theta_nat` is the column-major flattening of Sigma (length
 *  K*K). The user's inner lambda may view it as a matrix with, e.g.,
 *      arma::mat S(theta_nat.memptr(), K, K, false, true);   // zero-copy
 *  and writes d log p / d Sigma into grad_nat (length K*K). The user MUST fill
 *  the FULL K*K gradient G(i,j) = d log p / d Sigma(i,j), treating all entries
 *  as independent (a symmetric OR an asymmetric flat gradient of the same
 *  scalar both work — `wrap` forms (G + G^T) so only the symmetric part
 *  matters). Do NOT write only one triangle (zeros elsewhere): that is NOT
 *  supported and yields a WRONG gradient — the off-diagonals lose their mirror
 *  term and the diagonal is doubled, producing a silently biased posterior.
 *
 *  log|J| derivation
 *  -----------------
 *  Step (1) Jacobian (u -> free lower-triangular entries of L) is diagonal:
 *      d L(i,j)/d u = 1                  (off-diagonal, identity)
 *      d L(j,j)/d u_diag_j = exp(u_diag_j) = L(j,j)
 *    => log|J_1| = sum_j u_diag_j = sum_j log L(j,j).
 *
 *  Step (2) Jacobian (free lower entries of L -> lower triangle of Sigma):
 *      |det| = 2^K * prod_{i=0}^{K-1} L(i,i)^{K-i}
 *    => log|J_2| = K*log(2) + sum_i (K - i) * log L(i,i).
 *
 *  Combined (with log L(i,i) = u_diag_i):
 *      log|J| = K*log(2) + sum_i (K - i + 1) * u_diag_i.
 *
 *  Gradient chain rule
 *  -------------------
 *  Let G = d lp_model / d Sigma (K x K). Since Sigma = L L^T,
 *      d lp_model / d L(p,q) = ((G + G^T) L)(p,q).
 *  Then for the unconstrained entries:
 *      diagonal:    d(lp+log|J|)/d u_diag_j
 *                     = ((G+G^T)L)(j,j) * L(j,j)  +  (K - j + 1)
 *                       \____ exp() chain ____/      \__ log|J| __/
 *      off-diagonal:d(lp+log|J|)/d u_offdiag(i,j) = ((G+G^T)L)(i,j).
 *
 *  Packing convention (constrain / unconstrain / wrap all agree):
 *    column-major over the lower triangle, diagonal entry first within each
 *    column: (0,0),(1,0),..,(K-1,0),(1,1),(2,1),..,(K-1,1),(2,2),...
 *
 *  Everything is O(K^3) per wrap (one matrix multiply) — one inner call plus
 *  the (G+G^T)L contraction.
 *================================================================================*/

struct cov_matrix {
    /// Infer K from an unconstrained length M = K + K(K-1)/2 = K(K+1)/2.
    static std::size_t K_from_unc_dim(std::size_t n) {
        if (n == 0) {
            throw std::invalid_argument(
                "cov_matrix: unconstrained length must be >= 1 (K >= 1)");
        }
        // n = K(K+1)/2  =>  K = (-1 + sqrt(1 + 8n)) / 2
        const double Kd =
            (-1.0 + std::sqrt(1.0 + 8.0 * static_cast<double>(n))) / 2.0;
        const std::size_t K = static_cast<std::size_t>(std::llround(Kd));
        if (K < 1 || K * (K + 1) / 2 != n) {
            throw std::invalid_argument(
                "cov_matrix: unconstrained length must equal "
                "K * (K + 1) / 2 for some integer K >= 1");
        }
        return K;
    }

    /// Infer K from a natural-scale length K * K.
    static std::size_t K_from_nat_dim(std::size_t n) {
        const std::size_t K = static_cast<std::size_t>(
            std::llround(std::sqrt(static_cast<double>(n))));
        if (K < 1 || K * K != n) {
            throw std::invalid_argument(
                "cov_matrix: natural length must be a perfect square "
                "K * K with K >= 1");
        }
        return K;
    }

    /// Build the Cholesky factor L (lower-triangular, positive diagonal)
    /// from the unconstrained vector, using the packing convention above.
    static arma::mat build_L(const arma::vec& theta_unc, std::size_t K) {
        arma::mat L(K, K, arma::fill::zeros);
        std::size_t idx = 0;
        for (std::size_t j = 0; j < K; ++j) {
            L(j, j) = std::exp(theta_unc[idx++]);    // positive diagonal
            for (std::size_t i = j + 1; i < K; ++i) {
                L(i, j) = theta_unc[idx++];          // free below-diagonal
            }
        }
        return L;
    }

    /// Forward: unconstrained -> flat Sigma = L L^T (column-major K*K).
    static arma::vec constrain(const arma::vec& theta_unc) {
        const std::size_t K = K_from_unc_dim(theta_unc.n_elem);
        const arma::mat L = build_L(theta_unc, K);
        const arma::mat Sigma = L * L.t();
        return arma::vectorise(Sigma);
    }

    /// Inverse: flat Sigma -> unconstrained. Input must be a symmetric
    /// positive-definite matrix (column-major K*K flattening). The
    /// implied Cholesky factor must have a strictly positive diagonal.
    static arma::vec unconstrain(const arma::vec& theta_nat) {
        const std::size_t K = K_from_nat_dim(theta_nat.n_elem);
        // Column-major view of the flattened K*K natural vector.
        arma::mat Sigma(K, K);
        for (std::size_t j = 0; j < K; ++j)
            for (std::size_t i = 0; i < K; ++i)
                Sigma(i, j) = theta_nat[i + j * K];
        // Symmetrize defensively before factoring.
        Sigma = 0.5 * (Sigma + Sigma.t());
        arma::mat L;
        if (!arma::chol(L, Sigma, "lower")) {
            throw std::domain_error(
                "cov_matrix::unconstrain: input is not symmetric "
                "positive-definite (Cholesky factorisation failed)");
        }
        arma::vec theta_unc(K * (K + 1) / 2);
        std::size_t idx = 0;
        for (std::size_t j = 0; j < K; ++j) {
            if (!(L(j, j) > 0.0)) {
                throw std::domain_error(
                    "cov_matrix::unconstrain: non-positive Cholesky "
                    "diagonal");
            }
            theta_unc[idx++] = std::log(L(j, j));
            for (std::size_t i = j + 1; i < K; ++i) {
                theta_unc[idx++] = L(i, j);
            }
        }
        return theta_unc;
    }

    /// Wrap an inner log-density on the flattened K*K Sigma into one on the
    /// unconstrained K(K+1)/2 vector, handling the L L^T transform, its
    /// log|J|, and the chain rule. The inner function MUST write the
    /// natural-scale gradient d lp / d Sigma into grad_nat (length K*K)
    /// when that argument is non-null.
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       InnerFn inner) {
        const std::size_t K = K_from_unc_dim(theta_unc.n_elem);

        const arma::mat L = build_L(theta_unc, K);
        const arma::mat Sigma = L * L.t();

        // ---- log|Jacobian| -------------------------------------------
        //   log|J| = K*log(2) + sum_j (K - j + 1) * u_diag_j
        double log_jac = static_cast<double>(K) * std::log(2.0);
        {
            std::size_t idx = 0;
            for (std::size_t j = 0; j < K; ++j) {
                const double u_diag = theta_unc[idx++];
                log_jac += static_cast<double>(K - j + 1) * u_diag;
                idx += (K - 1 - j);  // skip this column's below-diagonal
            }
        }

        // ---- inner natural-scale density on flat Sigma ---------------
        arma::vec theta_nat = arma::vectorise(Sigma);
        arma::vec grad_nat(K * K, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        // ---- chain rule ----------------------------------------------
        if (grad_unc) {
            grad_unc->set_size(K * (K + 1) / 2);

            // G = d lp_model / d Sigma (column-major flat view of grad_nat).
            arma::mat G(K, K);
            for (std::size_t b = 0; b < K; ++b)
                for (std::size_t a = 0; a < K; ++a)
                    G(a, b) = grad_nat[a + b * K];

            // Sigma = L L^T  =>  d lp / d L(p,q) = ((G + G^T) L)(p,q).
            // (G + G^T) is correct ONLY for the documented contract: G is the
            // FULL K*K gradient (symmetric or asymmetric). A one-triangle G
            // would lose off-diagonal mirror terms and double the diagonal.
            const arma::mat GL = (G + G.t()) * L;   // K x K

            std::size_t idx = 0;
            for (std::size_t j = 0; j < K; ++j) {
                // diagonal L(j,j) = exp(u_diag): chain through exp and add
                // d log|J| / d u_diag = (K - j + 1).
                (*grad_unc)[idx++] =
                    GL(j, j) * L(j, j)
                    + static_cast<double>(K - j + 1);
                // below-diagonal free entries: identity map, no log|J| term.
                for (std::size_t i = j + 1; i < K; ++i) {
                    (*grad_unc)[idx++] = GL(i, j);
                }
            }
        }

        return lp_model + log_jac;
    }
};
/*================================================================================
 *  stochastic_column: an M x N matrix whose every COLUMN is a simplex.
 *
 *  This is Stan's `stochastic_column_matrix` transform (the column-stochastic
 *  matrix constraint, a.k.a. the `simplex` transform applied independently to
 *  each column). Each column is mapped from M-1 unconstrained parameters to M
 *  natural-scale simplex entries via the SAME logit stick-breaking transform
 *  used by `constraints::simplex` (Stan Reference Manual, "Unit simplex" /
 *  "Stochastic Matrix"). Reuse of that math is exact: with N = 1 this struct
 *  reduces to a single simplex.
 *
 *  Dimensions
 *  ----------
 *   - theta_unc : length (M - 1) * N, column-major: for column c the M-1
 *                 unconstrained entries occupy indices [c*(M-1) .. c*(M-1)+M-2].
 *   - theta_nat : length M * N, column-major flattening of the M x N matrix
 *                 (standard Armadillo memory layout). Column c occupies indices
 *                 [c*M .. c*M+M-1] and sums to 1. The user's inner function can
 *                 wrap this as a matrix with
 *                 `arma::mat P(theta_nat.memptr(), M, N, false, true)`.
 *
 *  Because the columns are independent, the Jacobian is BLOCK-DIAGONAL: log|J|
 *  is the SUM of the per-column simplex log|J|, and the unconstrained gradient
 *  for column c depends only on column c's natural-scale gradient.
 *
 *  log |J| (per column, summed):
 *      log|J| = sum_c sum_{k=0}^{M-2} [ log z_{c,k} + log(1 - z_{c,k})
 *                                       + log(remaining_{c,k}) ]
 *  exactly as in `simplex` (stick-breaking), accumulated over all N columns.
 *
 *  Cost: one inner call plus O(M^2 * N) scalar work.
 *================================================================================*/
struct stochastic_column {
    /// Infer M (rows) from the unconstrained length n = (M - 1) * N, given N.
    static std::size_t M_from_unc_dim(std::size_t n, std::size_t N) {
        if (N == 0) {
            throw std::invalid_argument(
                "stochastic_column: number of columns N must be >= 1");
        }
        if (n % N != 0) {
            throw std::invalid_argument(
                "stochastic_column: unconstrained length must be "
                "(M - 1) * N for integer M >= 2");
        }
        const std::size_t Mm1 = n / N;
        if (Mm1 == 0) {
            throw std::invalid_argument(
                "stochastic_column: each column needs at least 1 "
                "unconstrained entry (M >= 2)");
        }
        return Mm1 + 1;
    }

    /// Infer M (rows) from the natural-scale length n = M * N, given N.
    static std::size_t M_from_nat_dim(std::size_t n, std::size_t N) {
        if (N == 0) {
            throw std::invalid_argument(
                "stochastic_column: number of columns N must be >= 1");
        }
        if (n % N != 0) {
            throw std::invalid_argument(
                "stochastic_column: natural length must be M * N");
        }
        const std::size_t M = n / N;
        if (M < 2) {
            throw std::invalid_argument(
                "stochastic_column: each column must have at least 2 "
                "natural entries (M >= 2)");
        }
        return M;
    }

    /// Forward map: theta_unc ((M-1)*N) -> theta_nat (M*N), column-major.
    /// Convenience wrapper that discards the log|Jacobian|. Parameterised by
    /// the number of columns N. Matches the function-pointer style of the
    /// other dimension-changing transforms (with the extra N parameter).
    static arma::vec constrain(const arma::vec& theta_unc, std::size_t N) {
        arma::vec theta_nat;
        double    log_jac = 0.0;
        constrain_with_log_jac(theta_unc, N, &theta_nat, &log_jac);
        return theta_nat;
    }

    /// Inverse stick-breaking applied per column: theta_nat (M*N) ->
    /// theta_unc ((M-1)*N). Throws std::domain_error if any column is not in
    /// the interior of the simplex (non-positive entry or implied
    /// stick-fraction outside (0, 1)).
    static arma::vec unconstrain(const arma::vec& theta_nat, std::size_t N) {
        const std::size_t M   = M_from_nat_dim(theta_nat.n_elem, N);
        const std::size_t Mm1 = M - 1;
        arma::vec theta_unc(Mm1 * N);

        for (std::size_t c = 0; c < N; ++c) {
            const std::size_t nat_off = c * M;
            const std::size_t unc_off = c * Mm1;
            double remaining = 1.0;
            for (std::size_t k = 0; k < Mm1; ++k) {
                const double nat_k = theta_nat[nat_off + k];
                if (nat_k <= 0.0) {
                    throw std::domain_error(
                        "stochastic_column::unconstrain: theta_nat entries "
                        "must be strictly positive");
                }
                const double z = nat_k / remaining;
                if (z <= 0.0 || z >= 1.0) {
                    throw std::domain_error(
                        "stochastic_column::unconstrain: a column is not in "
                        "the interior of the simplex");
                }
                const double offset = std::log(static_cast<double>(Mm1 - k));
                theta_unc[unc_off + k] = std::log(z / (1.0 - z)) + offset;
                remaining -= nat_k;
            }
            if (remaining <= 0.0) {
                throw std::domain_error(
                    "stochastic_column::unconstrain: a column is not in the "
                    "interior of the simplex");
            }
        }
        return theta_unc;
    }

    /// Forward map with log|Jacobian| (summed over columns).
    static void constrain_with_log_jac(const arma::vec& theta_unc,
                                       std::size_t N,
                                       arma::vec* theta_nat_out,
                                       double* log_jac_out) {
        const std::size_t M   = M_from_unc_dim(theta_unc.n_elem, N);
        const std::size_t Mm1 = M - 1;

        arma::vec theta_nat(M * N, arma::fill::zeros);
        double log_jac = 0.0;

        for (std::size_t c = 0; c < N; ++c) {
            const std::size_t nat_off = c * M;
            const std::size_t unc_off = c * Mm1;
            double remaining = 1.0;
            for (std::size_t k = 0; k < Mm1; ++k) {
                const double offset = std::log(static_cast<double>(Mm1 - k));
                const double z =
                    1.0 / (1.0 + std::exp(-(theta_unc[unc_off + k] - offset)));
                // numerically safe log(z) + log(1 - z)
                const double lzz = std::log(z) + std::log1p(-z);
                theta_nat[nat_off + k] = remaining * z;
                log_jac += lzz + std::log(remaining);
                remaining *= (1.0 - z);
            }
            theta_nat[nat_off + Mm1] = remaining;
        }

        *theta_nat_out = std::move(theta_nat);
        *log_jac_out   = log_jac;
    }

    /**
     * Wrap an AI-written natural-scale density lambda (defined on the
     * flattened M*N column-stochastic matrix) into a full unconstrained
     * log-density. Handles the per-column stick-breaking transform, its
     * block-diagonal Jacobian, and the chain rule.
     *
     * The inner function MUST write the natural-scale gradient into grad_nat
     * (length M*N, column-major) when that argument is non-null.
     *
     * Gradient computation is analytic: one call to `inner` plus O(M^2 * N)
     * scalar work. The chain rule for each column is identical to
     * `simplex::wrap` (K = M), applied independently because the Jacobian is
     * block-diagonal across columns.
     */
    template <typename InnerFn>
    static double wrap(const arma::vec& theta_unc,
                       arma::vec* grad_unc,
                       std::size_t N,
                       InnerFn inner) {
        const std::size_t M   = M_from_unc_dim(theta_unc.n_elem, N);
        const std::size_t Mm1 = M - 1;

        // ---- Forward: compute theta_nat, log|J|, cache z per column -------
        arma::vec theta_nat(M * N, arma::fill::zeros);
        arma::vec z_all(Mm1 * N);   // cached z_{c,k} for the chain rule
        double log_jac = 0.0;

        for (std::size_t c = 0; c < N; ++c) {
            const std::size_t nat_off = c * M;
            const std::size_t unc_off = c * Mm1;
            double remaining = 1.0;
            for (std::size_t k = 0; k < Mm1; ++k) {
                const double offset = std::log(static_cast<double>(Mm1 - k));
                const double z =
                    1.0 / (1.0 + std::exp(-(theta_unc[unc_off + k] - offset)));
                z_all[unc_off + k] = z;
                const double lzz = std::log(z) + std::log1p(-z);
                theta_nat[nat_off + k] = remaining * z;
                log_jac += lzz + std::log(remaining);
                remaining *= (1.0 - z);
            }
            theta_nat[nat_off + Mm1] = remaining;
        }

        // ---- Natural-scale density and gradient --------------------------
        arma::vec grad_nat(M * N, arma::fill::zeros);
        const double lp_model = inner(theta_nat, &grad_nat);

        // ---- Analytic block-diagonal stick-breaking Jacobian -------------
        if (grad_unc) {
            grad_unc->set_size(Mm1 * N);

            for (std::size_t c = 0; c < N; ++c) {
                const std::size_t nat_off = c * M;
                const std::size_t unc_off = c * Mm1;

                for (std::size_t j = 0; j < Mm1; ++j) {
                    const double z_j    = z_all[unc_off + j];
                    const double nat_j  = theta_nat[nat_off + j];
                    const double J_diag = nat_j * (1.0 - z_j);

                    // Accumulate grad_nat[i] * J[i,j] for i = j..M-2 and track
                    // the partial column sum so we can build J[M-1, j] from the
                    // simplex constraint sum_i J[i,j] = 0 (within the column).
                    double chain   = grad_nat[nat_off + j] * J_diag;
                    double col_sum = J_diag;
                    for (std::size_t i = j + 1; i < Mm1; ++i) {
                        const double J_ij = -theta_nat[nat_off + i] * z_j;
                        chain   += grad_nat[nat_off + i] * J_ij;
                        col_sum += J_ij;
                    }
                    // J[M-1, j] = -col_sum by the simplex constraint.
                    chain += grad_nat[nat_off + Mm1] * (-col_sum);

                    // Gradient of log|J| itself: 1 - z_j * (M - j).
                    const double grad_log_jac =
                        1.0 - z_j * static_cast<double>(M - j);

                    (*grad_unc)[unc_off + j] = chain + grad_log_jac;
                }
            }
        }

        return lp_model + log_jac;
    }
};
} // namespace constraints
} // namespace AI4BayesCode

#endif // AI4BAYESCODE_CONSTRAINTS_HPP
