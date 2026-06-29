// Correctness: wrap_autodiff_simplex vs constraints::simplex::wrap
// On a Dirichlet log-density over y_counts. Gradients must match.

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>

#include "AI4BayesCode/autodiff_wrap.hpp"
#include "AI4BayesCode/constraints.hpp"

namespace adw = AI4BayesCode::autodiff_wrap;
namespace cs  = AI4BayesCode::constraints;
using AI4BayesCode::block_context;

// Hand-written natural-scale log-density for a Dirichlet posterior:
//   log p(theta | y_counts, alpha) = sum_k (alpha_k + y_k - 1) log(theta_k)
//   grad_k = (alpha_k + y_k - 1) / theta_k
static double dirichlet_hand(const arma::vec& theta_nat,
                             const block_context& ctx,
                             arma::vec* grad_nat) {
    const arma::vec& y     = ctx.at("y");
    const arma::vec& alpha = ctx.at("alpha");
    const std::size_t K = theta_nat.n_elem;
    double lp = 0.0;
    if (grad_nat) grad_nat->set_size(K);
    for (std::size_t k = 0; k < K; ++k) {
        const double t = theta_nat[k];
        if (t <= 0.0) return -std::numeric_limits<double>::infinity();
        const double ak = alpha[k] + y[k] - 1.0;
        lp += ak * std::log(t);
        if (grad_nat) (*grad_nat)[k] = ak / t;
    }
    return lp;
}

// Templated AD version, same math, no grad.
template <typename Vec>
auto dirichlet_ad(const Vec& theta_nat,
                  const block_context& ctx) {
    const arma::vec& y     = ctx.at("y");
    const arma::vec& alpha = ctx.at("alpha");
    const std::size_t K = theta_nat.size();
    using T = typename std::decay<decltype(theta_nat[0])>::type;
    T lp = T(0);
    for (std::size_t k = 0; k < K; ++k) {
        auto t = theta_nat[k];
        const double ak = alpha[k] + y[k] - 1.0;
        lp = lp + ak * log(t);
    }
    return lp;
}

// [[Rcpp::export]]
Rcpp::List check_simplex_grad(const arma::vec& y_counts,
                              const arma::vec& alpha,
                              int n_points = 5,
                              int seed = 12345) {
    block_context ctx;
    ctx["y"]     = y_counts;
    ctx["alpha"] = alpha;

    // Unconstrained dim = K - 1
    const std::size_t K = y_counts.n_elem;
    const std::size_t K_minus_1 = K - 1;

    std::mt19937_64 rng(static_cast<std::uint64_t>(seed));
    std::uniform_real_distribution<double> unif(-2.0, 2.0);

    double max_diff = 0.0;
    for (int k = 0; k < n_points; ++k) {
        arma::vec theta_unc(K_minus_1);
        for (std::size_t i = 0; i < K_minus_1; ++i) theta_unc[i] = unif(rng);

        arma::vec grad_hw(K_minus_1);
        double lp_hw = cs::simplex::wrap(
            theta_unc, &grad_hw,
            [&](const arma::vec& th, arma::vec* g) {
                return dirichlet_hand(th, ctx, g);
            });

        arma::vec grad_ad(K_minus_1);
        double lp_ad = adw::wrap_simplex(
            theta_unc, &grad_ad,
            [&](const auto& th) {
                return dirichlet_ad(th, ctx);
            });

        double d = std::max(std::abs(lp_hw - lp_ad),
                            arma::max(arma::abs(grad_hw - grad_ad)));
        if (d > max_diff) max_diff = d;
    }
    return Rcpp::List::create(
        Rcpp::Named("max_diff") = max_diff,
        Rcpp::Named("K")        = static_cast<int>(K),
        Rcpp::Named("n_points") = n_points);
}
