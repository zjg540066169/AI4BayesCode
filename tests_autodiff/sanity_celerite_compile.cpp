// Sanity: confirm vendored celerite compiles + computes a Cholesky +
// log-determinant for a simple exponential (Matern 1/2) kernel.
//
// celerite 1-D time-series GP, exponential kernel k(Δt) = a * exp(-c Δt).
// Set N=50 regularly-spaced times, compute Cholesky, log|K|, verify
// against a direct dense Cholesky via Armadillo.

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <RcppArmadillo.h>
#include <celerite/celerite.h>

// [[Rcpp::export]]
Rcpp::List sanity_celerite_compile() {
    const int N = 50;
    Eigen::VectorXd t(N);
    for (int i = 0; i < N; ++i) t[i] = i * 0.1;   // regularly spaced

    // Single real exponential term: k(Δt) = a * exp(-c Δt)
    Eigen::VectorXd a_real(1), c_real(1);
    a_real[0] = 1.0;
    c_real[0] = 0.5;

    // Empty complex terms
    Eigen::VectorXd a_comp(0), b_comp(0), c_comp(0), d_comp(0);

    // Empty general terms
    Eigen::VectorXd A_empty(0);
    Eigen::MatrixXd U_empty(0, 0), V_empty(0, 0);

    // Diagonal (noise)
    Eigen::VectorXd diag(N);
    diag.setConstant(0.01);

    celerite::solver::CholeskySolver<double> solver;
    double jitter = 1e-10;
    try {
        solver.compute(jitter, a_real, c_real, a_comp, b_comp, c_comp, d_comp,
                       A_empty, U_empty, V_empty, t, diag);
    } catch (const std::exception& e) {
        return Rcpp::List::create(
            Rcpp::Named("error")    = std::string(e.what()),
            Rcpp::Named("all_pass") = false);
    }
    double log_det_celerite = solver.log_determinant();

    // Compute the same via direct dense Cholesky
    arma::mat K(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double dt = std::abs(t[i] - t[j]);
            K(i, j) = 1.0 * std::exp(-0.5 * dt);
        }
        K(i, i) += 0.01 + jitter;
    }
    arma::mat L;
    arma::chol(L, K, "lower");
    double log_det_direct = 2.0 * arma::sum(arma::log(L.diag()));

    double diff = std::abs(log_det_celerite - log_det_direct);

    return Rcpp::List::create(
        Rcpp::Named("log_det_celerite") = log_det_celerite,
        Rcpp::Named("log_det_direct")   = log_det_direct,
        Rcpp::Named("abs_diff")         = diff,
        Rcpp::Named("N")                = N,
        Rcpp::Named("all_pass")         = diff < 1e-8);
}
