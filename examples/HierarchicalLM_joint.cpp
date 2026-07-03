// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  HierarchicalLM_joint.cpp
//
//  Hierarchical linear model with a non-centered reparameterization (NCR):
//  tau and sigma are FOLDED INTO the single joint_nuts_block alongside the
//  non-centered random effects z_u.
//
//  Model
//  -----
//      y_n | alpha, beta, u, sigma  ~ Normal(alpha + X_n' beta + u_{g[n]}, sigma^2)
//      u_g | tau                     ~ Normal(0, tau^2),   g = 1..G
//      alpha                         ~ Normal(0, 10^2)
//      beta_p                        ~ Normal(0, 10^2)
//      sigma                         ~ Half-Normal(0, 5)
//      tau                           ~ Half-Normal(0, 5)
//
//  Non-centered reparameterization (NCR)
//  --------------------------------------
//      z_u_g ~ Normal(0, 1)  (SAMPLED)
//      u_g   = tau * z_u_g   (DERIVED, written to shared_data after each step)
//
//  The log-density in terms of the sampled parameters (alpha, beta, z_u, tau, sigma):
//
//    log p = -0.5 * sum_n (y_n - alpha - X_n' beta - tau * z_u_{g[n]})^2 / sigma^2
//           - N * log(sigma)                                                 [Gaussian normalizer, N terms]
//           - 0.5 * sum_g z_u_g^2                                           [z_u ~ N(0,1)]
//           - 0.5 * alpha^2 / (10^2)                                        [alpha prior]
//           - 0.5 * sum_p beta_p^2 / (10^2)                                 [beta prior]
//           - 0.5 * sigma^2 / (5^2)                                         [sigma HalfNormal]
//           - 0.5 * tau^2 / (5^2)                                           [tau HalfNormal]
//
//  No Jacobians in this function: the block adds +log(sigma) + log(tau)
//  internally for the POSITIVE slices.
//
//  Block decomposition (ONE block)
//  --------------------------------
//      (alpha, beta, z_u, tau, sigma) : ONE joint_nuts_block
//          sub_params = [ {alpha, 1, REAL}, {beta, p, REAL},
//                         {z_u, G, REAL}, {tau, 1, POSITIVE}, {sigma, 1, POSITIVE} ]
//
//      No separate nuts_blocks for tau or sigma.
//
//  get_current()/get_history() expose NATURAL parameters:
//      alpha, beta, u (= tau * z_u), tau, sigma.
//  z_u is also exposed so the reparameterized space can be inspected.
//
//  Class name:  HierarchicalLM_joint
//  Module name: HierarchicalLM_joint_module
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates hierarchical
// data, fits the joint-NUTS block, and checks recovery. No R / Python binding
// is built or required.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/constraints.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::joint_nuts_block;
using AI4BayesCode::joint_nuts_block_config;
using AI4BayesCode::joint_nuts_sub_param;
using AI4BayesCode::joint_constraint;

namespace constraints = AI4BayesCode::constraints;

namespace {

// ----------------------------------------------------------------------------
//  Joint NCR log-density on the NATURAL scale.
//
//  Layout of theta_cat (length 1 + p + G + 1 + 1):
//      index 0           -> alpha
//      index 1 .. p      -> beta_{1..p}
//      index p+1 .. p+G  -> z_u_{1..G}    (non-centered; u_g = tau * z_u_g)
//      index p+G+1       -> tau   (> 0, block enforces; NO Jacobian here)
//      index p+G+2       -> sigma (> 0, block enforces; NO Jacobian here)
//
//  Context reads: y, X (flat column-major), g_idx (0-based), p, G.
//  tau and sigma come from theta_cat (they are sub-params of this block).
// ----------------------------------------------------------------------------
double ncr_joint_log_density(const arma::vec& theta_cat,
                              const block_context& ctx,
                              arma::vec* grad_nat) {
    const arma::vec& y      = ctx.at("y");
    const arma::vec& X_flat = ctx.at("X");
    const arma::vec& g_idx  = ctx.at("g_idx");   // 0-based
    const std::size_t N = static_cast<std::size_t>(y.n_elem);
    const std::size_t p = static_cast<std::size_t>(ctx.at("p")[0]);
    const std::size_t G = static_cast<std::size_t>(ctx.at("G")[0]);

    const std::size_t dim_expected = 1 + p + G + 2;
    if (theta_cat.n_elem != dim_expected) {
        if (grad_nat) { grad_nat->set_size(dim_expected); grad_nat->zeros(); }
        return -std::numeric_limits<double>::infinity();
    }

    // Slice the joint vector.
    const double alpha   = theta_cat[0];
    // beta: indices 1..p (length p, may be zero-length)
    // z_u: indices p+1..p+G (length G)
    // tau:   index p+G+1
    // sigma: index p+G+2
    const double tau   = theta_cat[p + G + 1];
    const double sigma = theta_cat[p + G + 2];

    if (tau <= 0.0 || sigma <= 0.0) {
        if (grad_nat) { grad_nat->set_size(dim_expected); grad_nat->zeros(); }
        return -std::numeric_limits<double>::infinity();
    }

    const double sigma2 = sigma * sigma;
    const double tau2   = tau   * tau;

    constexpr double prior_sd_alpha_beta = 10.0;
    const double prior_var = prior_sd_alpha_beta * prior_sd_alpha_beta;
    constexpr double prior_sd_scale = 5.0;
    const double prior_var_scale = prior_sd_scale * prior_sd_scale;

    double lp = 0.0;

    if (grad_nat) {
        grad_nat->set_size(dim_expected);
        grad_nat->zeros();
    }

    // ---- Prior on alpha --------------------------------------------------
    lp += -0.5 * alpha * alpha / prior_var;
    if (grad_nat) (*grad_nat)[0] += -alpha / prior_var;

    // ---- Prior on beta ---------------------------------------------------
    for (std::size_t j = 0; j < p; ++j) {
        const double bj = theta_cat[1 + j];
        lp += -0.5 * bj * bj / prior_var;
        if (grad_nat) (*grad_nat)[1 + j] += -bj / prior_var;
    }

    // ---- Prior on z_u: z_u_g ~ N(0,1) -----------------------------------
    for (std::size_t g = 0; g < G; ++g) {
        const double zg = theta_cat[1 + p + g];
        lp += -0.5 * zg * zg;
        if (grad_nat) (*grad_nat)[1 + p + g] += -zg;
    }

    // ---- HalfNormal prior on tau -----------------------------------------
    lp += -0.5 * tau2 / prior_var_scale;
    if (grad_nat) (*grad_nat)[p + G + 1] += -tau / prior_var_scale;

    // ---- HalfNormal prior on sigma ---------------------------------------
    lp += -0.5 * sigma2 / prior_var_scale;
    if (grad_nat) (*grad_nat)[p + G + 2] += -sigma / prior_var_scale;

    // ---- Gaussian likelihood -N*log(sigma) - 0.5*sum_r^2/sigma^2 --------
    // We also accumulate gradients for alpha, beta, z_u, tau, sigma here.
    lp += -static_cast<double>(N) * std::log(sigma);
    if (grad_nat) (*grad_nat)[p + G + 2] += -static_cast<double>(N) / sigma;

    double sum_r2 = 0.0;

    for (std::size_t n = 0; n < N; ++n) {
        // X_flat is column-major: X_flat[n + k * N] = X(n, k)
        double xb = 0.0;
        for (std::size_t k = 0; k < p; ++k) {
            xb += X_flat[n + k * N] * theta_cat[1 + k];
        }
        const std::size_t gn = static_cast<std::size_t>(g_idx[n]);
        if (gn >= G) {
            if (grad_nat) { grad_nat->zeros(); }
            return -std::numeric_limits<double>::infinity();
        }
        const double z_gn = theta_cat[1 + p + gn];
        const double u_gn = tau * z_gn;                    // u_g = tau * z_u_g
        const double mu_n = alpha + xb + u_gn;
        const double r    = y[n] - mu_n;
        sum_r2 += r * r;

        if (grad_nat) {
            const double g_mu = r / sigma2;                // d log p / d mu_n
            (*grad_nat)[0]          += g_mu;               // d/d alpha
            for (std::size_t k = 0; k < p; ++k)
                (*grad_nat)[1 + k]  += g_mu * X_flat[n + k * N];  // d/d beta_k
            (*grad_nat)[1 + p + gn]+= g_mu * tau;         // d/d z_u_gn (chain rule via u_gn=tau*z_gn)
            (*grad_nat)[p + G + 1] += g_mu * z_gn;        // d/d tau (chain rule via u_gn=tau*z_gn)
        }
    }

    lp += -0.5 * sum_r2 / sigma2;
    if (grad_nat) {
        // d (-0.5 sum_r^2/sigma^2) / d sigma = sum_r^2 / sigma^3
        (*grad_nat)[p + G + 2] += sum_r2 / (sigma2 * sigma);
    }

    if (!std::isfinite(lp)) {
        if (grad_nat) grad_nat->zeros();
        return -std::numeric_limits<double>::infinity();
    }
    return lp;
}

} // anonymous namespace

class HierarchicalLM_joint {
public:
    HierarchicalLM_joint(const arma::vec& y,
                            const arma::mat& X,
                            const arma::ivec& g_idx_1based,   // 1-based group index
                            int G,
                            double sigma_init,
                            double tau_init,
                            int rng_seed,
                            bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          readapt_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0xBF58476D1CE4E5B9ULL}),
          impl_(std::make_unique<composite_block>("HierarchicalLM_joint")),
          keep_history_(keep_history)
    {
        const std::size_t N = y.n_elem;
        const std::size_t p = X.n_cols;
        if (X.n_rows != N) throw std::runtime_error("X rows must equal length(y)");
        if (g_idx_1based.n_elem != static_cast<arma::uword>(N))
            throw std::runtime_error("g_idx length must equal length(y)");
        if (G < 1) throw std::runtime_error("G must be >= 1");
        if (sigma_init <= 0.0 || tau_init <= 0.0)
            throw std::runtime_error("sigma_init and tau_init must be > 0");

        // Convert 1-based to 0-based.
        arma::vec g_idx(N);
        for (std::size_t n = 0; n < N; ++n) {
            const int gn = g_idx_1based[n] - 1;
            if (gn < 0 || gn >= G) throw std::runtime_error("g_idx entries must be in 1..G");
            g_idx[n] = static_cast<double>(gn);
        }

        // Initial values.
        const double alpha_init = arma::mean(y);
        arma::vec beta_init(p, arma::fill::zeros);
        // z_u_init = 0 => u_init = tau_init * 0 = 0.
        arma::vec z_u_init(G, arma::fill::zeros);
        arma::vec u_init(G, arma::fill::zeros);  // u = tau * z_u

        // Store data in shared_data.
        impl_->data().set("y",     y);
        impl_->data().set("X",     arma::vectorise(X));
        impl_->data().set("g_idx", g_idx);
        impl_->data().set("p",     arma::vec{static_cast<double>(p)});
        impl_->data().set("G",     arma::vec{static_cast<double>(G)});

        // Natural-scale parameter slots.
        impl_->data().set("alpha", arma::vec{alpha_init});
        impl_->data().set("beta",  beta_init);
        impl_->data().set("z_u",   z_u_init);
        impl_->data().set("u",     u_init);    // derived: u_g = tau * z_u_g
        impl_->data().set("sigma", arma::vec{sigma_init});
        impl_->data().set("tau",   arma::vec{tau_init});

        // u is derived from (tau, z_u) after each step. Register refresher.
        impl_->data().register_refresher(
            "u",
            [G](const AI4BayesCode::shared_data_t& d) {
                const double tau      = d.get("tau")[0];
                const arma::vec& z_u  = d.get("z_u");
                arma::vec u(G);
                for (std::size_t g = 0; g < static_cast<std::size_t>(G); ++g)
                    u[g] = tau * z_u[g];
                return u;
            });

        // mu_fixed = alpha + X*beta (no random effect); used for predict.
        impl_->data().set("mu_fixed",
                          arma::vec(N, arma::fill::value(alpha_init)));
        impl_->data().register_refresher(
            "mu_fixed",
            [p, N](const AI4BayesCode::shared_data_t& d) -> arma::vec {
                const double alpha      = d.get("alpha")[0];
                const arma::vec& beta   = d.get("beta");
                const arma::vec& X_flat = d.get("X");
                arma::vec mf(N);
                for (std::size_t n = 0; n < N; ++n) {
                    double xb = 0.0;
                    for (std::size_t k = 0; k < p; ++k)
                        xb += X_flat[n + k * N] * beta[k];
                    mf[n] = alpha + xb;
                }
                return mf;
            });

        // Dependency DAG: joint block's context needs y, X, g_idx, p, G.
        // tau and sigma come from the block's own sub-params, NOT from ctx.
        // Key dependencies under the JOINT BLOCK NAME.
        impl_->data().declare_dependencies(
            "ncr_joint", {"y", "X", "g_idx", "p", "G"});

        // The joint block invalidates mu_fixed and u when it updates.
        impl_->data().declare_invalidates(
            "ncr_joint", {"mu_fixed", "u"});

        // Predict DAG (edges keyed by sub-param name or derived node name).
        impl_->data().declare_data_input("X");
        impl_->data().declare_data_input("g_idx");
        impl_->data().declare_predict_edges("X",        {"mu_fixed"});
        impl_->data().declare_predict_edges("alpha",    {"mu_fixed"});
        impl_->data().declare_predict_edges("beta",     {"mu_fixed"});
        impl_->data().declare_predict_edges("mu_fixed", {"y_rep"});
        impl_->data().declare_predict_edges("u",        {"y_rep"});
        impl_->data().declare_predict_edges("g_idx",    {"y_rep"});
        impl_->data().declare_predict_edges("tau",      {"y_rep"});
        impl_->data().declare_predict_edges("sigma",    {"y_rep"});

        // Generative DAG: tau -> u (VIZ-ONLY context edge).
        impl_->data().declare_context_edges("tau", {"u"});

        // y_rep stochastic refresher: reads alpha, beta, u (natural scale), sigma.
        impl_->data().set("y_rep", arma::vec(N, arma::fill::zeros));
        impl_->data().register_stochastic_refresher(
            "y_rep",
            [N](const AI4BayesCode::shared_data_t& d,
                std::mt19937_64& rng) {
                const arma::vec& mf  = d.get("mu_fixed");   // alpha + X*beta
                const arma::vec& u   = d.get("u");          // natural-scale u, length G
                const arma::vec& gix = d.get("g_idx");      // 0-based, length N
                const double tau     = d.get("tau")[0];
                const double s       = d.get("sigma")[0];
                const std::size_t G  = u.n_elem;
                std::normal_distribution<double> norm(0.0, 1.0);
                arma::vec y_rep(N);
                for (std::size_t i = 0; i < N; ++i) {
                    const std::size_t g = static_cast<std::size_t>(gix[i]);
                    const double re = (g < G) ? u[g] : tau * norm(rng);
                    y_rep[i] = mf[i] + re + s * norm(rng);
                }
                return y_rep;
            });

        // ---- ONE joint_nuts_block over (alpha, beta, z_u, tau, sigma) ----
        {
            joint_nuts_block_config cfg;
            cfg.name = "ncr_joint";

            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "alpha", 1,
                                      joint_constraint::REAL });
            if (p > 0) {
                cfg.sub_params.push_back(
                    joint_nuts_sub_param{ "beta", p,
                                          joint_constraint::REAL });
            }
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "z_u", static_cast<std::size_t>(G),
                                      joint_constraint::REAL });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "tau", 1,
                                      joint_constraint::POSITIVE });
            cfg.sub_params.push_back(
                joint_nuts_sub_param{ "sigma", 1,
                                      joint_constraint::POSITIVE });

            // Initial natural-scale vector: [alpha, beta, z_u, tau, sigma].
            const std::size_t dim = 1 + p + G + 2;
            arma::vec init_cat(dim);
            init_cat[0] = alpha_init;
            for (std::size_t k = 0; k < p; ++k) init_cat[1 + k] = 0.0;
            for (std::size_t g = 0; g < static_cast<std::size_t>(G); ++g)
                init_cat[1 + p + g] = 0.0;
            init_cat[p + G + 1] = tau_init;
            init_cat[p + G + 2] = sigma_init;
            cfg.initial_cat = init_cat;

            cfg.log_density_grad    = &ncr_joint_log_density;
            // RULE: cfg.use_diagonal_metric = true (per mandate).
            cfg.use_diagonal_metric = true;
            cfg.n_warmup_first_call = 800;

            impl_->add_child(std::make_unique<joint_nuts_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // get_current: exposes NATURAL parameters alpha, beta, u, tau, sigma.
    // Also exposes z_u for cross-validation of the NCR parameterization.
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["alpha"] = impl_->data().get("alpha");
        out["beta"]  = impl_->data().get("beta");
        out["z_u"]   = impl_->data().get("z_u");
        out["u"]     = impl_->data().get("u");
        out["sigma"] = impl_->data().get("sigma");
        out["tau"]   = impl_->data().get("tau");
        return out;
    }

    // set_current: overwrite any subset of {alpha, beta, z_u, tau, sigma}.
    // u is derived from (tau, z_u) and is recomputed automatically.
    // Setting u directly is NOT supported (set z_u instead).
    void set_current(const AI4BayesCode::state_map& params) {
        auto& joint_blk = dynamic_cast<joint_nuts_block&>(impl_->child(0));
        arma::vec cat_cur = joint_blk.current();

        const std::size_t p = impl_->data().get("beta").n_elem;
        const std::size_t G = impl_->data().get("z_u").n_elem;

        bool touched = false;

        auto it_alpha = params.find("alpha");
        if (it_alpha != params.end()) {
            cat_cur[0] = it_alpha->second[0];
            touched = true;
        }
        auto it_beta = params.find("beta");
        if (it_beta != params.end()) {
            const arma::vec& b = it_beta->second;
            if (b.n_elem != p) throw std::runtime_error("set_current: beta length must equal p");
            for (std::size_t k = 0; k < p; ++k) cat_cur[1 + k] = b[k];
            touched = true;
        }
        auto it_zu = params.find("z_u");
        if (it_zu != params.end()) {
            const arma::vec& z = it_zu->second;
            if (z.n_elem != G) throw std::runtime_error("set_current: z_u length must equal G");
            for (std::size_t g = 0; g < G; ++g) cat_cur[1 + p + g] = z[g];
            touched = true;
        }
        auto it_tau = params.find("tau");
        if (it_tau != params.end()) {
            const double tu = it_tau->second[0];
            if (!(tu > 0.0)) throw std::runtime_error("tau must be > 0");
            cat_cur[p + G + 1] = tu;
            touched = true;
        }
        auto it_sigma = params.find("sigma");
        if (it_sigma != params.end()) {
            const double sg = it_sigma->second[0];
            if (!(sg > 0.0)) throw std::runtime_error("sigma must be > 0");
            cat_cur[p + G + 2] = sg;
            touched = true;
        }

        if (touched) {
            joint_blk.set_current(cat_cur);
            impl_->data().set("alpha", arma::vec{cat_cur[0]});
            arma::vec beta_new(p);
            for (std::size_t k = 0; k < p; ++k) beta_new[k] = cat_cur[1 + k];
            impl_->data().set("beta", beta_new);
            arma::vec z_u_new(G);
            for (std::size_t g = 0; g < G; ++g) z_u_new[g] = cat_cur[1 + p + g];
            impl_->data().set("z_u", z_u_new);
            const double tau_new   = cat_cur[p + G + 1];
            const double sigma_new = cat_cur[p + G + 2];
            impl_->data().set("tau",   arma::vec{tau_new});
            impl_->data().set("sigma", arma::vec{sigma_new});
            // Recompute derived u = tau * z_u.
            arma::vec u_new(G);
            for (std::size_t g = 0; g < G; ++g) u_new[g] = tau_new * z_u_new[g];
            impl_->data().set("u", u_new);
        }
    }

    AI4BayesCode::history_map get_history() const { return impl_->get_history(); }

    AI4BayesCode::history_map predict_at(const AI4BayesCode::state_map& new_data) const {
        if (!new_data.empty()) {
            throw std::runtime_error(
                "HierarchicalLM_joint::predict_at: does not accept "
                "replaced data inputs. Call with an empty map for "
                "posterior-predictive y_rep at training X / g_idx.");
        }

        AI4BayesCode::history_map out;

        if (!keep_history_) {
            block_context replaced;
            block_context result = impl_->predict_at(replaced, predict_rng_);
            for (const auto& kv : result) {
                arma::mat m(1, kv.second.n_elem);
                for (std::size_t j = 0; j < kv.second.n_elem; ++j)
                    m(0, j) = kv.second[j];
                out.emplace(kv.first, std::move(m));
            }
            return out;
        }

        // History mode: replay per draw.
        AI4BayesCode::history_map hist = impl_->get_history();
        const arma::mat& alpha_hist = hist.at("alpha");   // n_draws x 1
        const arma::mat& beta_hist  = hist.at("beta");    // n_draws x p
        const arma::mat& u_hist     = hist.at("u");       // n_draws x G
        const arma::mat& sigma_hist = hist.at("sigma");   // n_draws x 1
        const std::size_t n_draws = alpha_hist.n_rows;
        const std::size_t p_dim   = beta_hist.n_cols;

        const arma::vec& X_flat = impl_->data().get("X");
        const arma::vec& g_idx  = impl_->data().get("g_idx");
        const std::size_t N     = g_idx.n_elem;

        arma::mat yrep_mat(n_draws, N);
        std::normal_distribution<double> norm01(0.0, 1.0);
        for (std::size_t d = 0; d < n_draws; ++d) {
            const double alpha_d = alpha_hist(d, 0);
            const double sigma_d = sigma_hist(d, 0);
            for (std::size_t i = 0; i < N; ++i) {
                double xb = 0.0;
                for (std::size_t j = 0; j < p_dim; ++j)
                    xb += X_flat[i + j * N] * beta_hist(d, j);
                const std::size_t gi = static_cast<std::size_t>(g_idx[i]);
                const double mu = alpha_d + xb + u_hist(d, gi);
                yrep_mat(d, i) = mu + sigma_d * norm01(predict_rng_);
            }
        }
        out.emplace("y_rep", std::move(yrep_mat));
        return out;
    }

    AI4BayesCode::dag_info get_dag() const { return impl_->get_dag(); }

    void readapt_NUTS(int n, bool reset = false, int max_tree_depth = -1) {
        if (n < 0) throw std::runtime_error("readapt_NUTS: n must be non-negative");
        impl_->readapt_NUTS(static_cast<std::size_t>(n), reset, readapt_rng_, max_tree_depth < 0 ? std::size_t(0) : static_cast<std::size_t>(max_tree_depth));
    }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    mutable std::mt19937_64          readapt_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a hierarchical linear model with G groups from a KNOWN truth,
//  fits via the joint-NUTS block (NCR over alpha, beta, z_u, tau, sigma), and
//  checks posterior-mean recovery of the fixed effects (alpha, beta), the
//  residual scale (sigma), and the random-effect scale (tau). Also confirms
//  the fit beats a naive intercept-only baseline (pooled mean of y) in RMSE.
//==============================================================================
#include <cstdio>
int main() {
    // ---- KNOWN truth -------------------------------------------------------
    const std::size_t G          = 8;     // number of groups
    const std::size_t p          = 2;     // number of covariates
    const std::size_t per_group  = 30;    // obs per group
    const std::size_t N          = G * per_group;

    const double alpha_true      = 1.5;
    const arma::vec beta_true     = {2.0, -1.0};
    const double sigma_true      = 0.7;   // residual scale
    const double tau_true        = 1.2;   // random-effect scale

    std::mt19937_64 sim_rng(2024);
    std::normal_distribution<double> norm(0.0, 1.0);

    // Group random effects u_g ~ N(0, tau^2).
    arma::vec u_true(G);
    for (std::size_t g = 0; g < G; ++g) u_true[g] = tau_true * norm(sim_rng);

    // Design matrix X (N x p), group index g_idx (1-based), response y.
    arma::mat  X(N, p);
    arma::ivec g_idx_1based(N);
    arma::vec  y(N);
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t g = n / per_group;     // balanced design
        g_idx_1based[n] = static_cast<int>(g) + 1;
        double mu = alpha_true + u_true[g];
        for (std::size_t k = 0; k < p; ++k) {
            const double xnk = norm(sim_rng);
            X(n, k) = xnk;
            mu += beta_true[k] * xnk;
        }
        y[n] = mu + sigma_true * norm(sim_rng);
    }

    // ---- Fit ---------------------------------------------------------------
    HierarchicalLM_joint model(y, X, g_idx_1based, static_cast<int>(G),
                               /*sigma_init=*/1.0, /*tau_init=*/1.0,
                               /*rng_seed=*/11, /*keep_history=*/false);
    model.step(800);   // extra warmup beyond the block's internal adaptation

    // ---- Sample posterior, accumulate means --------------------------------
    const int M = 3000;
    double alpha_bar = 0.0, sigma_bar = 0.0, tau_bar = 0.0;
    arma::vec beta_bar(p, arma::fill::zeros);
    arma::vec u_bar(G, arma::fill::zeros);
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        alpha_bar += cur.at("alpha")[0];
        sigma_bar += cur.at("sigma")[0];
        tau_bar   += cur.at("tau")[0];
        beta_bar  += cur.at("beta");
        u_bar     += cur.at("u");
    }
    const double Md = static_cast<double>(M);
    alpha_bar /= Md;
    sigma_bar /= Md;
    tau_bar   /= Md;
    beta_bar  /= Md;
    u_bar     /= Md;

    // ---- In-sample fit RMSE vs naive pooled-mean baseline ------------------
    double sse_fit = 0.0, sse_naive = 0.0;
    const double y_mean = arma::mean(y);
    for (std::size_t n = 0; n < N; ++n) {
        const std::size_t g = static_cast<std::size_t>(g_idx_1based[n]) - 1;
        double mu = alpha_bar + u_bar[g];
        for (std::size_t k = 0; k < p; ++k) mu += beta_bar[k] * X(n, k);
        const double r_fit   = y[n] - mu;
        const double r_naive = y[n] - y_mean;
        sse_fit   += r_fit * r_fit;
        sse_naive += r_naive * r_naive;
    }
    const double rmse_fit   = std::sqrt(sse_fit   / static_cast<double>(N));
    const double rmse_naive = std::sqrt(sse_naive / static_cast<double>(N));

    // ---- Report ------------------------------------------------------------
    std::printf("HierarchicalLM_joint demo (G=%zu, p=%zu, N=%zu)\n",
                G, p, N);
    std::printf("  alpha_hat=%.3f (truth %.2f)\n", alpha_bar, alpha_true);
    for (std::size_t k = 0; k < p; ++k)
        std::printf("  beta[%zu]_hat=%.3f (truth %.2f)\n",
                    k, beta_bar[k], beta_true[k]);
    std::printf("  sigma_hat=%.3f (truth %.2f)\n", sigma_bar, sigma_true);
    std::printf("  tau_hat=%.3f (truth %.2f)\n",   tau_bar,   tau_true);
    std::printf("  RMSE fit=%.3f  vs  naive pooled-mean=%.3f\n",
                rmse_fit, rmse_naive);

    // ---- PASS criteria (derived from actual computed comparisons) ----------
    bool ok_alpha = std::abs(alpha_bar - alpha_true) < 0.5;
    bool ok_beta  = true;
    for (std::size_t k = 0; k < p; ++k)
        ok_beta = ok_beta && (std::abs(beta_bar[k] - beta_true[k]) < 0.3);
    bool ok_sigma = std::abs(sigma_bar - sigma_true) < 0.3;
    bool ok_tau   = std::abs(tau_bar   - tau_true)   < 0.8;  // tau noisier (G=8)
    bool ok_rmse  = rmse_fit < rmse_naive;                   // beats baseline

    const bool ok = ok_alpha && ok_beta && ok_sigma && ok_tau && ok_rmse;
    std::printf("%s\n",
                ok ? "[demo PASS] joint-NUTS recovers hierarchical LM params"
                   : "[demo FAIL]");
    return ok ? 0 : 1;
}
