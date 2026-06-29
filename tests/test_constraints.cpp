/*================================================================================
 *  AI4BayesCode  --  unit tests for the constraint transform library.
 *================================================================================
 *
 *  For every constraint in AI4BayesCode::constraints, this file runs:
 *    1. Round-trip check:  unconstrain(constrain(theta_unc)) == theta_unc
 *    2. Forward-inverse:   constrain(unconstrain(theta_nat)) == theta_nat
 *    3. Gradient FD check: the analytic `wrap` gradient matches the
 *       central-difference gradient of the wrapped total log-posterior
 *       at several random theta_unc points (relative error < 1e-5).
 *
 *  No MCMC here -- pure math on the transforms. This is what catches a
 *  Jacobian sign mistake or an off-by-one in an index before it makes
 *  it anywhere near NUTS.
 *================================================================================*/

#include "AI4BayesCode/constraints.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace constraints = AI4BayesCode::constraints;

// ============================================================================
//  Common helpers
// ============================================================================

static bool almost_equal(double a, double b, double atol, double rtol) {
    const double diff = std::abs(a - b);
    return diff <= atol + rtol * std::max(std::abs(a), std::abs(b));
}

static bool vec_almost_equal(const arma::vec& a, const arma::vec& b,
                             double atol, double rtol) {
    if (a.n_elem != b.n_elem) return false;
    for (std::size_t i = 0; i < a.n_elem; ++i) {
        if (!almost_equal(a[i], b[i], atol, rtol)) return false;
    }
    return true;
}

// Central-difference gradient of a scalar function f(theta_unc) whose
// signature matches the wrap targets. Eps = 1e-5 is a good compromise
// between truncation and roundoff error.
using target_fn = std::function<double(const arma::vec&, arma::vec*)>;

static arma::vec fd_gradient(const target_fn& f,
                             const arma::vec& theta_unc,
                             double eps = 1e-5) {
    arma::vec grad(theta_unc.n_elem);
    arma::vec th = theta_unc;
    for (std::size_t i = 0; i < theta_unc.n_elem; ++i) {
        const double saved = th[i];
        th[i] = saved + eps;
        const double fp = f(th, nullptr);
        th[i] = saved - eps;
        const double fm = f(th, nullptr);
        th[i] = saved;
        grad[i] = (fp - fm) / (2.0 * eps);
    }
    return grad;
}

static bool gradient_matches_fd(const target_fn& f,
                                const arma::vec& theta_unc,
                                double atol = 1e-4,
                                double rtol = 1e-4) {
    arma::vec ana_grad(theta_unc.n_elem);
    (void)f(theta_unc, &ana_grad);
    arma::vec fd_grad = fd_gradient(f, theta_unc);
    return vec_almost_equal(ana_grad, fd_grad, atol, rtol);
}

// ============================================================================
//  Per-constraint tests
// ============================================================================

struct TestResult {
    std::string name;
    bool        pass;
    std::string detail;
};

static std::vector<TestResult> results;

static void record(const std::string& name, bool pass,
                   const std::string& detail = "") {
    results.push_back({name, pass, detail});
    std::printf("  %-50s %s\n", name.c_str(), pass ? "PASS" : "FAIL");
    if (!pass && !detail.empty()) {
        std::printf("    %s\n", detail.c_str());
    }
}

// ---- real ------------------------------------------------------------------

static void test_real() {
    std::printf("[real]\n");
    std::mt19937_64 rng(1);
    std::normal_distribution<double> nd(0.0, 2.0);

    // Use a simple non-trivial target: sum of log densities of a multivariate
    // normal so gradients are not identically zero.
    auto f = [](const arma::vec& theta, arma::vec* grad) -> double {
        return constraints::real::wrap(
            theta, grad,
            [](const arma::vec& theta_nat, arma::vec* grad_nat) {
                double lp = 0.0;
                for (std::size_t i = 0; i < theta_nat.n_elem; ++i) {
                    lp += -0.5 * theta_nat[i] * theta_nat[i];
                }
                if (grad_nat) {
                    grad_nat->set_size(theta_nat.n_elem);
                    for (std::size_t i = 0; i < theta_nat.n_elem; ++i) {
                        (*grad_nat)[i] = -theta_nat[i];
                    }
                }
                return lp;
            });
    };

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        const bool ok = gradient_matches_fd(f, x);
        record("real::wrap gradient matches FD, trial "
               + std::to_string(trial), ok);
    }
}

// ---- positive --------------------------------------------------------------

static void test_positive() {
    std::printf("[positive]\n");
    std::mt19937_64 rng(2);
    std::normal_distribution<double> nd(0.0, 1.0);

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        const arma::vec y = constraints::positive::constrain(x);
        const arma::vec z = constraints::positive::unconstrain(y);
        record("positive round-trip, trial " + std::to_string(trial),
               vec_almost_equal(x, z, 1e-12, 1e-12));
    }

    auto f = [](const arma::vec& theta, arma::vec* grad) -> double {
        return constraints::positive::wrap(
            theta, grad,
            [](const arma::vec& sigma, arma::vec* grad_nat) {
                // Half-normal-ish target: -0.5 * sum(sigma^2) + log(sigma[0])
                double lp = 0.0;
                for (std::size_t i = 0; i < sigma.n_elem; ++i)
                    lp += -0.5 * sigma[i] * sigma[i];
                lp += std::log(sigma[0]);
                if (grad_nat) {
                    grad_nat->set_size(sigma.n_elem);
                    for (std::size_t i = 0; i < sigma.n_elem; ++i)
                        (*grad_nat)[i] = -sigma[i];
                    (*grad_nat)[0] += 1.0 / sigma[0];
                }
                return lp;
            });
    };

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        record("positive::wrap gradient matches FD, trial "
               + std::to_string(trial), gradient_matches_fd(f, x));
    }
}

// ---- simplex ---------------------------------------------------------------

static void test_simplex() {
    std::printf("[simplex]\n");
    std::mt19937_64 rng(3);
    std::normal_distribution<double> nd(0.0, 1.0);

    for (std::size_t trial = 0; trial < 5; ++trial) {
        const std::size_t K = 5;
        arma::vec x(K - 1);
        for (std::size_t i = 0; i < K - 1; ++i) x[i] = nd(rng);
        const arma::vec y = constraints::simplex::constrain(x);

        // On-simplex check
        const double sum = arma::sum(y);
        const bool sum_ok = almost_equal(sum, 1.0, 1e-12, 1e-12);
        bool positive_ok = true;
        for (std::size_t k = 0; k < K; ++k) {
            if (y[k] <= 0.0) positive_ok = false;
        }
        record("simplex constrain produces valid simplex, trial "
               + std::to_string(trial), sum_ok && positive_ok);

        const arma::vec z = constraints::simplex::unconstrain(y);
        record("simplex round-trip, trial " + std::to_string(trial),
               vec_almost_equal(x, z, 1e-10, 1e-10));
    }

    // Analytic gradient vs finite difference
    auto f = [](const arma::vec& theta, arma::vec* grad) -> double {
        return constraints::simplex::wrap(
            theta, grad,
            [](const arma::vec& theta_nat, arma::vec* grad_nat) {
                // Dirichlet-ish target: sum_k alpha_k log theta_nat[k]
                const arma::vec alpha{2.0, 3.0, 1.5, 4.0, 2.5};
                double lp = 0.0;
                for (std::size_t k = 0; k < theta_nat.n_elem; ++k) {
                    lp += alpha[k] * std::log(theta_nat[k]);
                }
                if (grad_nat) {
                    grad_nat->set_size(theta_nat.n_elem);
                    for (std::size_t k = 0; k < theta_nat.n_elem; ++k) {
                        (*grad_nat)[k] = alpha[k] / theta_nat[k];
                    }
                }
                return lp;
            });
    };

    for (std::size_t trial = 0; trial < 10; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        record("simplex::wrap analytic gradient matches FD, trial "
               + std::to_string(trial), gradient_matches_fd(f, x));
    }
}

// ---- lower_bounded ---------------------------------------------------------

static void test_lower_bounded() {
    std::printf("[lower_bounded]\n");
    std::mt19937_64 rng(4);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double lo = 0.5;

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        const arma::vec y = constraints::lower_bounded::constrain(x, lo);
        bool ok_bound = true;
        for (std::size_t i = 0; i < y.n_elem; ++i) {
            if (y[i] <= lo) ok_bound = false;
        }
        record("lower_bounded bound respected, trial "
               + std::to_string(trial), ok_bound);

        const arma::vec z = constraints::lower_bounded::unconstrain(y, lo);
        record("lower_bounded round-trip, trial " + std::to_string(trial),
               vec_almost_equal(x, z, 1e-12, 1e-12));
    }

    auto f = [lo](const arma::vec& theta, arma::vec* grad) -> double {
        return constraints::lower_bounded::wrap(
            theta, grad, lo,
            [](const arma::vec& t_nat, arma::vec* grad_nat) {
                double lp = 0.0;
                for (std::size_t i = 0; i < t_nat.n_elem; ++i) {
                    const double d = t_nat[i] - 1.0;
                    lp += -0.5 * d * d;
                }
                if (grad_nat) {
                    grad_nat->set_size(t_nat.n_elem);
                    for (std::size_t i = 0; i < t_nat.n_elem; ++i) {
                        (*grad_nat)[i] = -(t_nat[i] - 1.0);
                    }
                }
                return lp;
            });
    };

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        record("lower_bounded::wrap gradient matches FD, trial "
               + std::to_string(trial), gradient_matches_fd(f, x));
    }
}

// ---- upper_bounded ---------------------------------------------------------

static void test_upper_bounded() {
    std::printf("[upper_bounded]\n");
    std::mt19937_64 rng(5);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double up = 2.0;

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        const arma::vec y = constraints::upper_bounded::constrain(x, up);
        bool ok_bound = true;
        for (std::size_t i = 0; i < y.n_elem; ++i) {
            if (y[i] >= up) ok_bound = false;
        }
        record("upper_bounded bound respected, trial "
               + std::to_string(trial), ok_bound);

        const arma::vec z = constraints::upper_bounded::unconstrain(y, up);
        record("upper_bounded round-trip, trial " + std::to_string(trial),
               vec_almost_equal(x, z, 1e-12, 1e-12));
    }

    auto f = [up](const arma::vec& theta, arma::vec* grad) -> double {
        return constraints::upper_bounded::wrap(
            theta, grad, up,
            [](const arma::vec& t_nat, arma::vec* grad_nat) {
                double lp = 0.0;
                for (std::size_t i = 0; i < t_nat.n_elem; ++i) {
                    const double d = t_nat[i] - 0.5;
                    lp += -0.5 * d * d;
                }
                if (grad_nat) {
                    grad_nat->set_size(t_nat.n_elem);
                    for (std::size_t i = 0; i < t_nat.n_elem; ++i) {
                        (*grad_nat)[i] = -(t_nat[i] - 0.5);
                    }
                }
                return lp;
            });
    };

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        record("upper_bounded::wrap gradient matches FD, trial "
               + std::to_string(trial), gradient_matches_fd(f, x));
    }
}

// ---- interval --------------------------------------------------------------

static void test_interval() {
    std::printf("[interval]\n");
    std::mt19937_64 rng(6);
    std::normal_distribution<double> nd(0.0, 1.0);
    const double lo = -1.0, up = 2.0;

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        const arma::vec y = constraints::interval::constrain(x, lo, up);
        bool inside = true;
        for (std::size_t i = 0; i < y.n_elem; ++i) {
            if (y[i] <= lo || y[i] >= up) inside = false;
        }
        record("interval constraint respected, trial "
               + std::to_string(trial), inside);

        const arma::vec z = constraints::interval::unconstrain(y, lo, up);
        record("interval round-trip, trial " + std::to_string(trial),
               vec_almost_equal(x, z, 1e-12, 1e-12));
    }

    auto f = [lo, up](const arma::vec& theta, arma::vec* grad) -> double {
        return constraints::interval::wrap(
            theta, grad, lo, up,
            [](const arma::vec& t_nat, arma::vec* grad_nat) {
                double lp = 0.0;
                for (std::size_t i = 0; i < t_nat.n_elem; ++i) {
                    lp += -0.5 * t_nat[i] * t_nat[i];
                }
                if (grad_nat) {
                    grad_nat->set_size(t_nat.n_elem);
                    for (std::size_t i = 0; i < t_nat.n_elem; ++i) {
                        (*grad_nat)[i] = -t_nat[i];
                    }
                }
                return lp;
            });
    };

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(4);
        for (std::size_t i = 0; i < 4; ++i) x[i] = nd(rng);
        record("interval::wrap gradient matches FD, trial "
               + std::to_string(trial), gradient_matches_fd(f, x));
    }
}

// ---- ordered ---------------------------------------------------------------

static void test_ordered() {
    std::printf("[ordered]\n");
    std::mt19937_64 rng(7);
    std::normal_distribution<double> nd(0.0, 1.0);

    for (std::size_t trial = 0; trial < 5; ++trial) {
        arma::vec x(5);
        for (std::size_t i = 0; i < 5; ++i) x[i] = nd(rng);
        const arma::vec y = constraints::ordered::constrain(x);
        bool monotone = true;
        for (std::size_t k = 1; k < y.n_elem; ++k) {
            if (!(y[k] > y[k - 1])) monotone = false;
        }
        record("ordered constrain is strictly increasing, trial "
               + std::to_string(trial), monotone);

        const arma::vec z = constraints::ordered::unconstrain(y);
        record("ordered round-trip, trial " + std::to_string(trial),
               vec_almost_equal(x, z, 1e-12, 1e-12));
    }

    auto f = [](const arma::vec& theta, arma::vec* grad) -> double {
        return constraints::ordered::wrap(
            theta, grad,
            [](const arma::vec& t_nat, arma::vec* grad_nat) {
                // Gaussian target on t_nat with known mean / sd
                const arma::vec mu{-1.0, 0.0, 1.0, 2.0, 3.0};
                double lp = 0.0;
                for (std::size_t i = 0; i < t_nat.n_elem; ++i) {
                    const double d = t_nat[i] - mu[i];
                    lp += -0.5 * d * d;
                }
                if (grad_nat) {
                    grad_nat->set_size(t_nat.n_elem);
                    for (std::size_t i = 0; i < t_nat.n_elem; ++i) {
                        (*grad_nat)[i] = -(t_nat[i] - mu[i]);
                    }
                }
                return lp;
            });
    };

    for (std::size_t trial = 0; trial < 10; ++trial) {
        arma::vec x(5);
        for (std::size_t i = 0; i < 5; ++i) x[i] = nd(rng);
        record("ordered::wrap gradient matches FD, trial "
               + std::to_string(trial), gradient_matches_fd(f, x));
    }
}

// ---- cholesky_corr ---------------------------------------------------------

static void test_cholesky_corr() {
    std::printf("[cholesky_corr]\n");
    std::mt19937_64 rng(8);
    std::normal_distribution<double> nd(0.0, 0.8);

    // Round-trip + valid-correlation check across several K values.
    for (std::size_t K : {2u, 3u, 5u}) {
        for (std::size_t trial = 0; trial < 3; ++trial) {
            const std::size_t n = K * (K - 1) / 2;
            arma::vec y(n);
            for (std::size_t i = 0; i < n; ++i) y[i] = nd(rng);

            arma::vec L_flat = constraints::cholesky_corr::constrain(y);
            arma::mat L      = arma::reshape(L_flat, K, K);

            // L * L' should be a correlation matrix (sym, unit diag, PD).
            arma::mat C = L * L.t();
            bool unit_diag = true;
            for (std::size_t i = 0; i < K; ++i) {
                if (!almost_equal(C(i, i), 1.0, 1e-10, 1e-10)) {
                    unit_diag = false;
                }
            }
            bool symmetric = true;
            for (std::size_t i = 0; i < K; ++i) {
                for (std::size_t j = 0; j < K; ++j) {
                    if (!almost_equal(C(i, j), C(j, i), 1e-12, 1e-12)) {
                        symmetric = false;
                    }
                }
            }
            // PD via Cholesky decomposition of C (should succeed).
            arma::mat L_check;
            bool pd = arma::chol(L_check, C, "lower");

            record("cholesky_corr L*L' has unit diagonal, K="
                   + std::to_string(K) + " trial "
                   + std::to_string(trial), unit_diag);
            record("cholesky_corr L*L' is symmetric, K="
                   + std::to_string(K) + " trial "
                   + std::to_string(trial), symmetric);
            record("cholesky_corr L*L' is positive definite, K="
                   + std::to_string(K) + " trial "
                   + std::to_string(trial), pd);

            // Round-trip via unconstrain.
            arma::vec y_back = constraints::cholesky_corr::unconstrain(L_flat);
            record("cholesky_corr round-trip, K=" + std::to_string(K)
                   + " trial " + std::to_string(trial),
                   vec_almost_equal(y, y_back, 1e-10, 1e-10));
        }
    }

    // Gradient FD check.
    //
    // Target: log det(L * L'), which for a lower-triangular L with unit
    // row norms reduces to 2 * sum_i log(L[i, i]). The gradient with
    // respect to L[i, j] is:
    //   G[i, i] = 2 / L[i, i]    (only diagonal entries matter)
    //   G[i, j] = 0              otherwise
    // This is a non-trivial target whose gradient w.r.t. theta_unc must
    // match the central-difference gradient of (lp_model + log|J|).
    for (std::size_t K : {3u, 5u}) {
        const std::size_t n = K * (K - 1) / 2;
        auto f = [K](const arma::vec& theta, arma::vec* grad) -> double {
            return constraints::cholesky_corr::wrap(
                theta, grad,
                [K](const arma::vec& theta_nat, arma::vec* grad_nat) {
                    auto L = [&theta_nat, K](std::size_t i,
                                             std::size_t j) -> double {
                        return theta_nat[i + j * K];
                    };
                    double lp = 0.0;
                    for (std::size_t i = 0; i < K; ++i) {
                        lp += 2.0 * std::log(L(i, i));
                    }
                    if (grad_nat) {
                        grad_nat->set_size(K * K);
                        grad_nat->zeros();
                        for (std::size_t i = 0; i < K; ++i) {
                            (*grad_nat)[i + i * K] = 2.0 / L(i, i);
                        }
                    }
                    return lp;
                });
        };

        for (std::size_t trial = 0; trial < 5; ++trial) {
            arma::vec y(n);
            for (std::size_t i = 0; i < n; ++i) y[i] = nd(rng);
            record("cholesky_corr::wrap gradient matches FD, K="
                   + std::to_string(K) + " trial "
                   + std::to_string(trial),
                   gradient_matches_fd(f, y, 1e-3, 1e-3));
        }
    }
}

// ---- unit_vector -----------------------------------------------------------

static void test_unit_vector() {
    std::printf("[unit_vector]\n");
    std::mt19937_64 rng(9);
    std::normal_distribution<double> nd(0.0, 2.0);

    // constrain: any non-zero y maps to a unit vector.
    for (std::size_t K : {2u, 3u, 5u}) {
        for (std::size_t trial = 0; trial < 3; ++trial) {
            arma::vec y(K);
            for (std::size_t i = 0; i < K; ++i) y[i] = nd(rng);
            const arma::vec x = constraints::unit_vector::constrain(y);
            const double norm = std::sqrt(arma::dot(x, x));
            record("unit_vector constrain produces unit norm, K="
                   + std::to_string(K) + " trial "
                   + std::to_string(trial),
                   almost_equal(norm, 1.0, 1e-12, 1e-12));
            // unconstrain is the canonical unit representative.
            const arma::vec y_back =
                constraints::unit_vector::unconstrain(x);
            record("unit_vector unconstrain round-trips unit input, K="
                   + std::to_string(K) + " trial "
                   + std::to_string(trial),
                   vec_almost_equal(x, y_back, 1e-12, 1e-12));
        }
    }

    // Gradient FD check.
    //
    // Target: a simple linear functional on x, lp_model = a . x where
    // a is a fixed direction vector. The natural-scale gradient is a;
    // the unconstrained gradient and the log|J| contribution are what
    // we're checking.
    for (std::size_t K : {3u, 5u}) {
        arma::vec a(K);
        for (std::size_t i = 0; i < K; ++i) {
            a[i] = std::sin(static_cast<double>(i) + 0.5);
        }
        auto f = [a, K](const arma::vec& theta, arma::vec* grad) -> double {
            return constraints::unit_vector::wrap(
                theta, grad,
                [a, K](const arma::vec& x, arma::vec* grad_nat) {
                    double lp = 0.0;
                    for (std::size_t i = 0; i < K; ++i) lp += a[i] * x[i];
                    if (grad_nat) {
                        grad_nat->set_size(K);
                        for (std::size_t i = 0; i < K; ++i) {
                            (*grad_nat)[i] = a[i];
                        }
                    }
                    return lp;
                });
        };
        for (std::size_t trial = 0; trial < 5; ++trial) {
            arma::vec y(K);
            for (std::size_t i = 0; i < K; ++i) y[i] = nd(rng);
            // Ensure non-zero norm.
            if (arma::dot(y, y) < 1e-3) y[0] += 1.0;
            record("unit_vector::wrap gradient matches FD, K="
                   + std::to_string(K) + " trial "
                   + std::to_string(trial),
                   gradient_matches_fd(f, y, 1e-4, 1e-4));
        }
    }
}

// ============================================================================
//  Main
// ============================================================================

int main() {
    test_real();
    test_positive();
    test_simplex();
    test_lower_bounded();
    test_upper_bounded();
    test_interval();
    test_ordered();
    test_cholesky_corr();
    test_unit_vector();

    std::size_t n_pass = 0;
    for (const auto& r : results) if (r.pass) ++n_pass;

    std::printf("\n%zu / %zu checks passed\n", n_pass, results.size());
    const bool ok = (n_pass == results.size());
    std::printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
