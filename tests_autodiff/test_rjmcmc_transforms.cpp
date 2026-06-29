// Regression test for the five library-provided RJMCMC transforms
// in include/AI4BayesCode/rjmcmc_transforms.hpp (ported from librjmcmc).
// Run via Rcpp::sourceCpp; prints a PASS/FAIL line.
//
// For each transform class:
//   (1) verify abs_jacobian<0>() matches the analytic expectation
//       (|det M| for linear; prod of diagonal for diagonal variants).
//   (2) verify apply<0> followed by apply<1> round-trips to identity
//       within floating-point precision (< 1e-12).

// [[Rcpp::depends(RcppArmadillo)]]
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <RcppArmadillo.h>

#include "AI4BayesCode/rjmcmc_transforms.hpp"

#include <array>
#include <cmath>

namespace T = AI4BayesCode::rjmcmc_transforms;

// [[Rcpp::export]]
Rcpp::List test_rjmcmc_transforms() {
    bool all_ok = true;
    std::vector<std::string> lines;
    auto record = [&](const std::string& name, double diff, bool ok) {
        lines.push_back(name + ": diff=" + std::to_string(diff) +
                        " (" + (ok ? "PASS" : "FAIL") + ")");
        if (!ok) all_ok = false;
    };

    // (1) identity_transform<3, double>
    {
        T::identity_transform<3, double> id;
        double in[3]  = {1.5, -2.0, 3.0};
        double out[3] = {0, 0, 0};
        double jac = id.apply<0>(in, out);
        double jac_err = std::abs(jac - 1.0);
        double roundtrip_err = 0;
        for (int i = 0; i < 3; ++i)
            roundtrip_err += std::abs(in[i] - out[i]);
        record("identity_transform jac_err", jac_err, jac_err < 1e-14);
        record("identity_transform roundtrip", roundtrip_err,
               roundtrip_err < 1e-14);
    }

    // (2) linear_transform<2, double>
    {
        // M = [[2, 1], [-1, 3]],  det = 7.
        double M[4] = {2.0, -1.0, 1.0, 3.0};   // column-major
        T::linear_transform<2, double> Lt(M);
        double in[2]  = {1.0, -2.0};
        double out[2] = {0, 0};
        double jac = Lt.apply<0>(in, out);
        double jac_err = std::abs(jac - 7.0);
        // round-trip:
        double back[2] = {0, 0};
        Lt.apply<1>(out, back);
        double roundtrip_err =
            std::abs(in[0] - back[0]) + std::abs(in[1] - back[1]);
        record("linear_transform jac_err (|det|=7)", jac_err,
               jac_err < 1e-12);
        record("linear_transform roundtrip", roundtrip_err,
               roundtrip_err < 1e-12);
    }

    // (3) diagonal_linear_transform<4, double>
    {
        double diag[4] = {2.0, 0.5, 3.0, -1.0};
        T::diagonal_linear_transform<4, double> Dt(diag);
        double in[4]  = {1, 2, -1, 4};
        double out[4] = {0, 0, 0, 0};
        double jac = Dt.apply<0>(in, out);
        double exp_jac = std::abs(2.0 * 0.5 * 3.0 * -1.0);  // = 3.0
        double jac_err = std::abs(jac - exp_jac);
        double back[4] = {0, 0, 0, 0};
        Dt.apply<1>(out, back);
        double roundtrip_err = 0;
        for (int i = 0; i < 4; ++i)
            roundtrip_err += std::abs(in[i] - back[i]);
        record("diagonal_linear_transform jac_err (=3)", jac_err,
               jac_err < 1e-14);
        record("diagonal_linear_transform roundtrip", roundtrip_err,
               roundtrip_err < 1e-14);
    }

    // (4) diagonal_affine_transform<3, double>: y_i = D_i x_i + b_i
    //     Constructor order: (diagonal, offset) — D first, b second.
    {
        double diag[3] = {2.0, -0.5, 3.0};
        double b[3]    = {1.0, -2.0, 0.5};
        T::diagonal_affine_transform<3, double> At(diag, b);
        double in[3]  = {0.5, 4.0, -1.0};
        double out[3] = {0, 0, 0};
        double jac = At.apply<0>(in, out);
        double exp_jac = std::abs(2.0 * -0.5 * 3.0);  // = 3.0
        double jac_err = std::abs(jac - exp_jac);
        // Verify y_i = D_i * x_i + b_i manually.
        double expected_out[3] = {
            2.0 * 0.5 + 1.0,
            -0.5 * 4.0 + -2.0,
            3.0 * -1.0 + 0.5};
        double out_err = 0;
        for (int i = 0; i < 3; ++i)
            out_err += std::abs(out[i] - expected_out[i]);
        double back[3] = {0, 0, 0};
        At.apply<1>(out, back);
        double roundtrip_err = 0;
        for (int i = 0; i < 3; ++i)
            roundtrip_err += std::abs(in[i] - back[i]);
        record("diagonal_affine_transform jac_err (=3)", jac_err,
               jac_err < 1e-14);
        record("diagonal_affine_transform forward out", out_err,
               out_err < 1e-14);
        record("diagonal_affine_transform roundtrip", roundtrip_err,
               roundtrip_err < 1e-14);
    }

    // (5) affine_transform<2, double>: y = M x + b.
    {
        double M[4] = {2.0, 1.0, 0.0, 3.0};   // column-major, det = 6
        double b[2] = {0.5, -1.0};
        T::affine_transform<2, double> Af(M, b);
        double in[2]  = {1.0, 2.0};
        double out[2] = {0, 0};
        double jac = Af.apply<0>(in, out);
        double jac_err = std::abs(jac - 6.0);
        double back[2] = {0, 0};
        Af.apply<1>(out, back);
        double roundtrip_err =
            std::abs(in[0] - back[0]) + std::abs(in[1] - back[1]);
        record("affine_transform jac_err (|det|=6)", jac_err,
               jac_err < 1e-12);
        record("affine_transform roundtrip", roundtrip_err,
               roundtrip_err < 1e-12);
    }

    Rcpp::CharacterVector out(lines.size());
    for (std::size_t i = 0; i < lines.size(); ++i)
        out[i] = lines[i];
    return Rcpp::List::create(
        Rcpp::Named("all_pass") = all_ok,
        Rcpp::Named("details")  = out);
}
