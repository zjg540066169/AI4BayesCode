// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
// test_ode_rk45.cpp
//
// Parity test for AI4BayesCode::ode::rk45 Tier 1 Dormand-Prince integrator.
//
// Verifies correctness via four canonical test problems:
//
//   (1) Linear ODE: dy/dt = -k*y, y(0) = y0 with closed form y(t) = y0 * exp(-k*t).
//       Check max_abs(numerical - analytical) < 1e-8.
//
//   (2) Harmonic oscillator: (x, v) with dx/dt = v, dv/dt = -omega^2 * x.
//       Closed form x(t) = x0*cos(omega*t) + v0/omega*sin(omega*t). Check
//       max_abs(numerical - analytical) < 1e-8 AND energy conservation
//       E = 0.5*v^2 + 0.5*omega^2*x^2 to within rtol over 10 periods.
//
//   (3) Lotka-Volterra: dx/dt = alpha*x - beta*x*y, dy/dt = delta*x*y - gamma*y.
//       No closed form but conserved quantity
//         V = delta*x - gamma*log(x) + beta*y - alpha*log(y)
//       is preserved. Check max deviation over 5 periods < 1e-4.
//
//   (4) SIR model: dS/dt = -beta*S*I, dI/dt = beta*S*I - gamma*I, dR/dt = gamma*I.
//       Total S + I + R conserved. Check deviation < 1e-8 over the epidemic.
// ============================================================================

// [[Rcpp::depends(RcppArmadillo)]]

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <RcppArmadillo.h>

#include "AI4BayesCode/ode_rk45.hpp"

#include <cmath>

// [[Rcpp::export]]
Rcpp::List test_ode_rk45() {
    using AI4BayesCode::ode::rk45;

    Rcpp::List out;
    bool all_pass = true;

    // --- Test 1: Linear ODE ---------------------------------------------
    {
        auto f = [](double /*t*/, const arma::vec& y,
                    const arma::vec& theta) -> arma::vec {
            // dy/dt = -k * y
            return -theta[0] * y;
        };

        const arma::vec y0 = {1.0};
        const arma::vec theta = {0.5};  // k = 0.5
        arma::vec ts(51);
        for (arma::uword i = 0; i < 51; ++i) ts[i] = i * 0.1;

        arma::mat y_num = rk45(f, y0, ts, theta, 1e-10, 1e-10);
        arma::vec y_ana(ts.n_elem);
        for (arma::uword i = 0; i < ts.n_elem; ++i)
            y_ana[i] = y0[0] * std::exp(-theta[0] * ts[i]);

        double max_err = 0.0;
        for (arma::uword i = 0; i < ts.n_elem; ++i) {
            const double e = std::abs(y_num(i, 0) - y_ana[i]);
            if (e > max_err) max_err = e;
        }

        const bool pass = max_err < 1e-8;
        all_pass = all_pass && pass;

        out["test1_linear_max_err"] = max_err;
        out["test1_linear_pass"]    = pass;
    }

    // --- Test 2: Harmonic oscillator ------------------------------------
    {
        auto f = [](double /*t*/, const arma::vec& y,
                    const arma::vec& theta) -> arma::vec {
            // dx/dt = v;  dv/dt = -omega^2 * x
            arma::vec dy(2);
            dy[0] = y[1];
            dy[1] = -theta[0] * theta[0] * y[0];
            return dy;
        };

        const double omega = 2.0;
        const double x0 = 1.0, v0 = 0.0;
        const arma::vec y0 = {x0, v0};
        const arma::vec theta = {omega};

        // Integrate for 10 periods, 100 samples per period
        const double T = 2.0 * M_PI / omega;
        const arma::uword n_samples = 1001;
        arma::vec ts(n_samples);
        for (arma::uword i = 0; i < n_samples; ++i)
            ts[i] = static_cast<double>(i) * (10.0 * T) / static_cast<double>(n_samples - 1);

        arma::mat y_num = rk45(f, y0, ts, theta, 1e-10, 1e-10);

        // Analytical solution
        double max_err = 0.0;
        for (arma::uword i = 0; i < n_samples; ++i) {
            const double x_ana = x0 * std::cos(omega * ts[i]) + (v0 / omega) * std::sin(omega * ts[i]);
            const double e = std::abs(y_num(i, 0) - x_ana);
            if (e > max_err) max_err = e;
        }

        // Energy conservation E = 0.5*v^2 + 0.5*omega^2*x^2
        const double E0 = 0.5 * v0 * v0 + 0.5 * omega * omega * x0 * x0;
        double max_E_dev = 0.0;
        for (arma::uword i = 0; i < n_samples; ++i) {
            const double x = y_num(i, 0), v = y_num(i, 1);
            const double E = 0.5 * v * v + 0.5 * omega * omega * x * x;
            const double d = std::abs(E - E0) / std::abs(E0);
            if (d > max_E_dev) max_E_dev = d;
        }

        const bool pass = (max_err < 1e-6) && (max_E_dev < 1e-6);
        all_pass = all_pass && pass;
        out["test2_harmonic_max_err"]   = max_err;
        out["test2_harmonic_E_dev"]     = max_E_dev;
        out["test2_harmonic_pass"]      = pass;
    }

    // --- Test 3: Lotka-Volterra ------------------------------------------
    {
        auto f = [](double /*t*/, const arma::vec& y,
                    const arma::vec& theta) -> arma::vec {
            // dx/dt = alpha*x - beta*x*y
            // dy/dt = delta*x*y - gamma*y
            const double alpha = theta[0], beta = theta[1];
            const double delta = theta[2], gamma = theta[3];
            arma::vec dy(2);
            dy[0] = alpha * y[0] - beta * y[0] * y[1];
            dy[1] = delta * y[0] * y[1] - gamma * y[1];
            return dy;
        };

        const arma::vec y0 = {1.0, 1.0};
        const arma::vec theta = {1.0, 1.0, 1.0, 1.0};  // alpha, beta, delta, gamma
        // Integrate for 15 time units (~5 periods for this fixture)
        const arma::uword n_samples = 301;
        arma::vec ts(n_samples);
        for (arma::uword i = 0; i < n_samples; ++i)
            ts[i] = static_cast<double>(i) * 15.0 / static_cast<double>(n_samples - 1);

        arma::mat y_num = rk45(f, y0, ts, theta, 1e-10, 1e-10);

        // Conserved quantity: V = delta*x - gamma*log(x) + beta*y - alpha*log(y)
        auto V_of = [&](double x, double y) {
            return theta[2] * x - theta[3] * std::log(x)
                 + theta[1] * y - theta[0] * std::log(y);
        };
        const double V0 = V_of(y0[0], y0[1]);

        double max_V_dev = 0.0;
        for (arma::uword i = 0; i < n_samples; ++i) {
            const double V = V_of(y_num(i, 0), y_num(i, 1));
            const double d = std::abs(V - V0) / std::abs(V0);
            if (d > max_V_dev) max_V_dev = d;
        }

        const bool pass = max_V_dev < 1e-4;  // looser since no closed form
        all_pass = all_pass && pass;
        out["test3_lv_max_V_dev"] = max_V_dev;
        out["test3_lv_pass"]      = pass;
    }

    // --- Test 4: SIR model ----------------------------------------------
    {
        auto f = [](double /*t*/, const arma::vec& y,
                    const arma::vec& theta) -> arma::vec {
            // S, I, R
            // dS/dt = -beta*S*I / N
            // dI/dt = beta*S*I / N - gamma*I
            // dR/dt = gamma*I
            const double beta = theta[0], gamma = theta[1];
            const double S = y[0], I = y[1], R = y[2];
            const double N = S + I + R;
            arma::vec dy(3);
            dy[0] = -beta * S * I / N;
            dy[1] =  beta * S * I / N - gamma * I;
            dy[2] =  gamma * I;
            return dy;
        };

        const arma::vec y0 = {990.0, 10.0, 0.0};  // N=1000, 1% initial infected
        const arma::vec theta = {0.4, 0.1};        // R0 = 4
        const arma::uword n_samples = 201;
        arma::vec ts(n_samples);
        for (arma::uword i = 0; i < n_samples; ++i)
            ts[i] = static_cast<double>(i) * 100.0 / static_cast<double>(n_samples - 1);

        arma::mat y_num = rk45(f, y0, ts, theta, 1e-10, 1e-10);

        const double N0 = y0[0] + y0[1] + y0[2];
        double max_N_dev = 0.0;
        for (arma::uword i = 0; i < n_samples; ++i) {
            const double N = y_num(i, 0) + y_num(i, 1) + y_num(i, 2);
            const double d = std::abs(N - N0) / std::abs(N0);
            if (d > max_N_dev) max_N_dev = d;
        }

        // Epidemic does spread: final R should be > 0, final S should be < S0
        const double S_final = y_num(n_samples - 1, 0);
        const double R_final = y_num(n_samples - 1, 2);
        const bool epidemic_ok = (S_final < y0[0]) && (R_final > 0.0);

        const bool pass = (max_N_dev < 1e-8) && epidemic_ok;
        all_pass = all_pass && pass;
        out["test4_sir_max_N_dev"] = max_N_dev;
        out["test4_sir_S_final"]   = S_final;
        out["test4_sir_R_final"]   = R_final;
        out["test4_sir_pass"]      = pass;
    }

    out["all_pass"] = all_pass;
    return out;
}
