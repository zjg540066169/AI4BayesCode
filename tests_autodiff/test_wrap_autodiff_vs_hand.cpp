// M2/M3 correctness: AD grad must equal hand-written grad for
//   (a) real constraint (mu ~ Normal(0, 100))
//   (b) positive constraint (sigma ~ Half-Normal(0, 10))
//   (c) mixed constraint: full Gaussian location-scale log-density
//
// Compares outputs of autodiff_wrap::wrap_* vs the constraints::*::wrap
// path on the same problem. Max abs diff must be < 1e-10.

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include "AI4BayesCode/autodiff_wrap.hpp"
#include "AI4BayesCode/constraints.hpp"

namespace adw = AI4BayesCode::autodiff_wrap;
namespace cs  = AI4BayesCode::constraints;

using AI4BayesCode::block_context;

// ----------------------------------------------------------------------------
//  (a) REAL: scalar mu with standard-normal-like prior.
//     log p(mu) = -0.5 * mu^2 / 100    (prior N(0, sqrt(100)=10))
//     grad      = -mu / 100
// ----------------------------------------------------------------------------

// hand-written grad version (operates on theta_nat — identical to unc for real)
static double mu_lp_handwritten(const arma::vec& mu_nat,
                                const block_context& /*ctx*/,
                                arma::vec* grad_nat) {
    double mu = mu_nat[0];
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -mu / 100.0;
    }
    return -0.5 * mu * mu / 100.0;
}

// AD version: take Matrix<var,-1,1> theta_nat, return var lp
template <typename Vec>
auto mu_lp_ad(const Vec& theta_nat) {
    auto mu = theta_nat[0];
    return -0.5 * mu * mu / 100.0;
}

// [[Rcpp::export]]
Rcpp::List check_real_grad(double mu) {
    arma::vec theta_unc{mu};
    block_context ctx;

    arma::vec grad_hand(1), grad_ad(1);
    double lp_hand = cs::real::wrap(theta_unc, &grad_hand,
        [&](const arma::vec& th, arma::vec* g) {
            return mu_lp_handwritten(th, ctx, g);
        });
    double lp_ad = adw::wrap_real(theta_unc, &grad_ad,
        [&](const auto& th) { return mu_lp_ad(th); });

    return Rcpp::List::create(
        Rcpp::Named("lp_hand")  = lp_hand,
        Rcpp::Named("lp_ad")    = lp_ad,
        Rcpp::Named("grad_hand")= grad_hand[0],
        Rcpp::Named("grad_ad")  = grad_ad[0]);
}

// ----------------------------------------------------------------------------
//  (b) POSITIVE: scalar sigma with Half-Normal(0, 10) prior.
//     Natural scale: sigma > 0
//     log p(sigma) = -0.5 * sigma^2 / 100     (normalized const dropped)
//     d/dsigma      = -sigma / 100
//     The log|Jacobian| of log-transform adds +log(sigma) on nat scale,
//     equivalently +theta_unc on unc scale.
// ----------------------------------------------------------------------------

static double sigma_lp_handwritten(const arma::vec& sigma_nat,
                                   const block_context& /*ctx*/,
                                   arma::vec* grad_nat) {
    double sigma = sigma_nat[0];
    if (grad_nat) {
        grad_nat->set_size(1);
        (*grad_nat)[0] = -sigma / 100.0;
    }
    return -0.5 * sigma * sigma / 100.0;
}

template <typename Vec>
auto sigma_lp_ad(const Vec& theta_nat) {
    auto sigma = theta_nat[0];
    return -0.5 * sigma * sigma / 100.0;
}

// [[Rcpp::export]]
Rcpp::List check_positive_grad(double sigma_nat) {
    double sigma_unc = std::log(sigma_nat);
    arma::vec theta_unc{sigma_unc};
    block_context ctx;

    arma::vec grad_hand(1), grad_ad(1);
    double lp_hand = cs::positive::wrap(theta_unc, &grad_hand,
        [&](const arma::vec& th, arma::vec* g) {
            return sigma_lp_handwritten(th, ctx, g);
        });
    double lp_ad = adw::wrap_positive(theta_unc, &grad_ad,
        [&](const auto& th) { return sigma_lp_ad(th); });

    return Rcpp::List::create(
        Rcpp::Named("lp_hand")  = lp_hand,
        Rcpp::Named("lp_ad")    = lp_ad,
        Rcpp::Named("grad_hand")= grad_hand[0],
        Rcpp::Named("grad_ad")  = grad_ad[0]);
}

// ----------------------------------------------------------------------------
//  (c) MIXED: (mu real, sigma positive) Gaussian log-density on N
//     observations.
// ----------------------------------------------------------------------------

// Hand-written (mu, log_sigma) as unconstrained vec. Uses cs::real for mu
// and cs::positive for sigma, combined by hand.
//
// For comparability we implement it as a monolithic wrap: unconstrained
// theta = [mu; log_sigma]; constrain: sigma = exp(log_sigma); Jacobian
// contributes log_sigma. Gradient components:
//   d/dmu          = sum((y - mu)) / sigma^2 - mu/100
//   d/d_log_sigma  = (-N + sum((y-mu)^2)/sigma^2) + (-sigma^2/100) * sigma + 1
//     (combining: d/d_sigma = -N/sigma + sum_sq/sigma^3 - sigma/100;
//      then chain rule d_sigma/d_log_sigma = sigma;
//      plus the log-Jacobian term d/d_log_sigma(log_sigma) = 1)
static double gauss_lp_handwritten(const arma::vec& theta_unc,
                                   const block_context& ctx,
                                   arma::vec* grad_unc) {
    double mu        = theta_unc[0];
    double log_sigma = theta_unc[1];
    double sigma     = std::exp(log_sigma);

    const arma::vec& y = ctx.at("y");
    const std::size_t N = y.n_elem;

    double sum_res = 0.0, sum_sq = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double r = y[i] - mu;
        sum_res += r;
        sum_sq  += r * r;
    }

    // Likelihood (natural scale, sigma):
    //   lp_lik = -N log(sigma) - 0.5 sum_sq / sigma^2
    // Priors:
    //   mu    ~ N(0, 10^2): -0.5 * mu^2 / 100
    //   sigma ~ HN(0, 10) : -0.5 * sigma^2 / 100
    // Jacobian: +log(sigma) = +log_sigma
    double lp =
        - static_cast<double>(N) * log_sigma
        - 0.5 * sum_sq / (sigma * sigma)
        - 0.5 * mu * mu / 100.0
        - 0.5 * sigma * sigma / 100.0
        + log_sigma;  // Jacobian for log-transform

    if (grad_unc) {
        grad_unc->set_size(2);
        // d/dmu (everything in unc space; only lp_lik and mu-prior depend on mu)
        (*grad_unc)[0] = sum_res / (sigma * sigma) - mu / 100.0;
        // d/d_log_sigma:
        //   from -N log(sigma): -N
        //   from -0.5 sum_sq/sigma^2:   sum_sq / sigma^2     (chain: d(sigma^-2)/d(log_sigma) = -2/sigma^2)
        //   from -0.5 sigma^2/100:      -sigma^2 / 100
        //   from +log_sigma (Jacobian): +1
        (*grad_unc)[1] = -static_cast<double>(N)
                         + sum_sq / (sigma * sigma)
                         - sigma * sigma / 100.0
                         + 1.0;
    }
    return lp;
}

// AD version. Receives theta_nat (VectorXvar of natural-scale values).
template <typename Vec>
auto gauss_lp_ad(const Vec& theta_nat, const block_context& ctx) {
    using T = typename std::decay<decltype(theta_nat[0])>::type;
    auto mu    = theta_nat[0];
    auto sigma = theta_nat[1];

    const arma::vec& y = ctx.at("y");
    const std::size_t N = y.n_elem;

    T lp = T(0);
    for (std::size_t i = 0; i < N; ++i) {
        auto r = y[i] - mu;
        lp = lp - 0.5 * r * r / (sigma * sigma) - log(sigma);
    }
    lp = lp - 0.5 * mu * mu / 100.0;
    lp = lp - 0.5 * sigma * sigma / 100.0;
    return lp;
}

// [[Rcpp::export]]
Rcpp::List check_mixed_grad(const arma::vec& y, double mu, double sigma) {
    arma::vec theta_unc{mu, std::log(sigma)};
    block_context ctx;
    ctx["y"] = y;

    arma::vec grad_hand(2), grad_ad(2);
    double lp_hand = gauss_lp_handwritten(theta_unc, ctx, &grad_hand);

    double lp_ad = adw::wrap_mixed(
        theta_unc, &grad_ad,
        { adw::slice_spec::real(1), adw::slice_spec::positive(1) },
        [&](const auto& th) { return gauss_lp_ad(th, ctx); });

    return Rcpp::List::create(
        Rcpp::Named("lp_hand")   = lp_hand,
        Rcpp::Named("lp_ad")     = lp_ad,
        Rcpp::Named("grad_hand") = Rcpp::NumericVector(grad_hand.begin(),
                                                       grad_hand.end()),
        Rcpp::Named("grad_ad")   = Rcpp::NumericVector(grad_ad.begin(),
                                                       grad_ad.end()));
}
