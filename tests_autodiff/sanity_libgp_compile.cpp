// Sanity: confirm the vendored libgp kernel subsystem compiles + links
// inside an Rcpp sourceCpp translation unit via the unity header.
//
// Builds a CovSEiso + CovNoise kernel pair, sets hyperparams, evaluates
// k(x1, x2). Round-trip through Eigen arithmetic.

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <RcppArmadillo.h>

// Include all the libgp kernel machinery (headers + .cc sources) in one go.
#include "libgp_kernels_unity.h"

// [[Rcpp::export]]
Rcpp::List sanity_libgp_compile() {
    // Construct CovSEiso with input_dim = 3
    libgp::CovSEiso cf;
    cf.init(3);

    // Set log hyperparameters [log(ell), log(sf)]
    Eigen::VectorXd hyper(2);
    hyper[0] = std::log(1.5);   // lengthscale 1.5
    hyper[1] = std::log(2.0);   // signal sd 2.0
    cf.set_loghyper(hyper);

    // Evaluate pairs
    Eigen::VectorXd x1(3); x1 << 0.0, 0.0, 0.0;
    Eigen::VectorXd x2(3); x2 << 1.0, 0.0, 0.0;
    Eigen::VectorXd x3(3); x3 << 0.5, 0.5, 0.0;

    double k11 = cf.get(x1, x1);
    double k12 = cf.get(x1, x2);
    double k13 = cf.get(x1, x3);

    // Analytical: k(x,x) = sf^2; k(x1,x2) = sf^2 * exp(-0.5*|x1-x2|^2/ell^2)
    double sf2 = 4.0;  // 2.0^2
    double ell = 1.5;
    double d12 = 1.0; // euclidean distance
    double d13 = std::sqrt(0.5);
    double exp_k11 = sf2;
    double exp_k12 = sf2 * std::exp(-0.5 * d12*d12 / (ell*ell));
    double exp_k13 = sf2 * std::exp(-0.5 * d13*d13 / (ell*ell));

    // Also test CovNoise
    libgp::CovNoise cfn;
    cfn.init(3);
    Eigen::VectorXd noise_hyper(1);
    noise_hyper[0] = std::log(0.1); // noise sd 0.1
    cfn.set_loghyper(noise_hyper);
    double kn_same = cfn.get(x1, x1);
    double kn_diff = cfn.get(x1, x2);
    double exp_kn_same = 0.01;  // 0.1^2
    double exp_kn_diff = 0.0;

    return Rcpp::List::create(
        Rcpp::Named("k11") = k11,
        Rcpp::Named("exp_k11") = exp_k11,
        Rcpp::Named("k12") = k12,
        Rcpp::Named("exp_k12") = exp_k12,
        Rcpp::Named("k13") = k13,
        Rcpp::Named("exp_k13") = exp_k13,
        Rcpp::Named("kn_same") = kn_same,
        Rcpp::Named("exp_kn_same") = exp_kn_same,
        Rcpp::Named("kn_diff") = kn_diff,
        Rcpp::Named("exp_kn_diff") = exp_kn_diff,
        Rcpp::Named("all_pass") =
            std::abs(k11 - exp_k11) < 1e-10 &&
            std::abs(k12 - exp_k12) < 1e-10 &&
            std::abs(k13 - exp_k13) < 1e-10 &&
            std::abs(kn_same - exp_kn_same) < 1e-10 &&
            std::abs(kn_diff - exp_kn_diff) < 1e-10);
}
