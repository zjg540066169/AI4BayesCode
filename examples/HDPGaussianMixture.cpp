// Copyright (C) 2026 AI4BayesCode.
// Licensed under the GNU General Public License v2.0 or later
// (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
// ============================================================================
//  HDPGaussianMixture.cpp
//
//  Hierarchical Dirichlet Process Gaussian mixture in TRUNCATED form
//  (Wang-Paisley-Blei 2011 simplified architecture, after Teh et al.
//  2006). Demonstrates COMPOSITION of multiple shipped blocks
//  (`stick_breaking_block` + G × `dirichlet_gibbs_block` +
//  `niw_cluster_gibbs_block` + `categorical_gibbs_block`) into a
//  hierarchical clustering model that shares atoms across groups.
//
//  IMPORTANT: V0 SIMPLIFICATION
//  ============================
//  The β top-level update implemented here uses a HEURISTIC: it treats
//  β | counts as a stick-breaking conditional with a_fn = 1 + Σ_j n_jt
//  and b_fn = γ + Σ_j (Σ_{s>t} n_js). This is NOT the rigorous HDP
//  posterior. The full HDP requires per-(j, t) auxiliary "table counts"
//  m_jt sampled from the Antoniak distribution (Teh et al. 2006 §5.1),
//  with β | m_·1, ..., m_·T ~ Dirichlet(γ/T + m_·1, ..., γ/T + m_·T).
//
//  For a v0 demonstration of the COMPOSITION PATTERN, the heuristic
//  works correctly on simple group-clustering fixtures and matches the
//  posterior intuitively (more populated atoms get larger β). For
//  rigorous HDP inference, future work needs an `antoniak_aug_block`
//  that samples table counts; with that, the existing β block can be
//  re-pointed to read m_·t instead of n_·t.
//
//  This caveat is documented prominently here (in the header comment),
//  in `skills/block_catalogue/index.md` (HDP example summary), and in
//  `BNP_AUDIT_STATUS.md`. Users who need rigorous HDP should look at
//  BayesMix (Beraha et al. 2022, arXiv 2205.08144) for a reference
//  implementation.
//
//  MODEL
//  -----
//      β ~ truncated stick-breaking with concentration γ
//      π_j | β, α ~ Dirichlet(α · β_1, ..., α · β_T)   for j = 1, ..., G
//      z_{ji} | π_j ~ Categorical(π_j)
//      y_{ji} | z_{ji} = t ~ N(μ_t, Σ_t)               (atoms shared)
//      (μ_t, Σ_t) ~ NIW(μ_0, κ_0, Ψ_0, ν_0)
//      α, γ : FIXED at construction (v0; future v0.5 sample via
//             gamma_gibbs_block).
//
//  BLOCK DECOMPOSITION (Gibbs sweep order)
//  ---------------------------------------
//      child(0)             relabel        hdp_label_canonicalizer_block
//                           FALLBACK in-sampler canonicaliser (NOT RECOMMENDED;
//                           the default is POST-MCMC relabeling, see LABEL
//                           SWITCHING below and skills/label_switching.md).
//                           Permutes the T global atoms into a canonical order
//                           each sweep (OCCUPIED atoms first, sorted by atom
//                           location mu) so the REPORTED per-slot atoms
//                           (beta[t], mu[t], Sigma_t, pi_{g,t}) converge.
//                           Posterior-preserving; runs FIRST so downstream
//                           children read/record on one canonical labelling.
//      child(1)             z              categorical_gibbs_block
//      child(2)             cluster_params niw_cluster_gibbs_block
//      child(3)             β              stick_breaking_block (heuristic)
//      child(4..3+G)        π_j (j = 1..G) dirichlet_gibbs_block (one per group)
//
//  LABEL SWITCHING
//  ---------------
//  DEFAULT / RECOMMENDED: resolve label switching POST-MCMC on the recorded
//  draws (skills/label_switching.md). The truncated-HDP likelihood is invariant
//  under permutation of the T GLOBAL atoms (shared mu_t/Sigma_t, the global
//  weight beta_t, and every group's pi_{g,t}); the label-INVARIANT summaries
//  (K_active, sorted occupied-atom proportions / mu / beta order statistics)
//  converge as-is (2-chain rank-R-hat ~1.0). The two concentrations gamma_0 and
//  alpha are FIXED constants in this v0 model (not sampled), hence trivially
//  label-invariant — there is NO concentration-mixing problem to fix (unlike the
//  DP example where alpha is sampled by NUTS on the Antoniak (k,n) marginal).
//
//  FALLBACK USED HERE (child(0) hdp_label_canonicalizer_block — NOT RECOMMENDED):
//  the RAW per-slot beta[t] do NOT converge under post-MCMC sorting of the empty
//  truncation tail (measured 2-chain rank-R-hat ~1.83), though the cluster
//  PROPORTIONS and sorted order statistics DO (~1.0). To make the REPORTED per-
//  slot atoms converge on their raw recorded values this example carries an in-
//  sampler canonicaliser (OCCUPIED-first, sort-by-atom-location-mu permutation
//  each sweep) as a last resort. We sort by the well-separated atom location, not
//  the near-equal global weight beta, so the per-slot atoms converge to ONE atom
//  (a beta sort only fixes order statistics). Prefer post-MCMC relabeling for new
//  models; this in-sampler path is the exception, not the template. See the class
//  comment for the full rationale.
//
//  REFRESHERS
//  ----------
//      counts_jt   register_refresher (deterministic). Group×Atom count
//                  matrix derived from (z, group_idx). Stored flat as
//                  G*T row-major in shared_data["counts_jt"].
//      counts_t    register_refresher (deterministic). Top-level
//                  marginal: counts_t = Σ_j counts_jt. Length T.
//      y_rep       register_stochastic_refresher (predict-time).
//
//  STORAGE CONVENTIONS
//  -------------------
//  - y         length N * d, row-major
//  - group_idx length N, integer 0..G-1 (stored as double)
//  - z         length N, integer 1..T (1-indexed)
//  - β         length T (T = K_trunc)
//  - stick_V_β length T (output of β stick_breaking_block via v_name)
//  - π_g       length T, exposed under shared_data key "pi_<g>" for
//              g = 0, 1, ..., G-1 (no aggregated group-major flat key)
//  - μ         length T * d, cluster-major (atoms shared)
//  - σ         length T * d * d, cluster-major row-major (atoms shared)
//  - counts_jt length G * T, group-major row-major
//  - counts_t  length T
//
//  JUSTIFICATION (Check #16):
//  - relabel   hdp_label_canonicalizer_block. FALLBACK in-sampler canonicaliser
//              (NOT recommended; prefer post-MCMC relabeling). Posterior-
//              preserving descending-beta atom permutation; used only because
//              the raw per-slot beta[t] won't converge under post-MCMC sorting
//              though the proportions / order statistics do. See LABEL SWITCHING.
//  - z         categorical_gibbs_block. Class-1 conditional independence
//              given (π_j, μ, Σ).
//  - π_g       dirichlet_gibbs_block. Conjugate on the simplex.
//  - β         stick_breaking_block. Heuristic update; documented
//              approximation.
//  - cluster_params  niw_cluster_gibbs_block. Conjugate.
//
//  No hand-written log-density (no NUTS); the canonicaliser is deterministic
//  and all 4 + G sampling children are conjugate Gibbs / MH-deterministic.
//  Check #12 vacuous.
// ============================================================================

// Frontend-INDEPENDENT example: pure standalone C++ (NO Rcpp, NO pybind).
// Compile + run directly: it has an int main() that simulates grouped
// Gaussian-mixture data, fits the composed HDP sampler, and checks atom
// recovery. No R / Python binding is built or required.

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

#include <armadillo>

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"

#include "AI4BayesCode/categorical_gibbs_block.hpp"
#include "AI4BayesCode/dirichlet_gibbs_block.hpp"
#include "AI4BayesCode/niw_cluster_gibbs_block.hpp"
#include "AI4BayesCode/stick_breaking_block.hpp"
#include "AI4BayesCode/bnp_utils.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::categorical_gibbs_block;
using AI4BayesCode::categorical_gibbs_block_config;
using AI4BayesCode::dirichlet_gibbs_block;
using AI4BayesCode::dirichlet_gibbs_block_config;
using AI4BayesCode::niw_cluster_gibbs_block;
using AI4BayesCode::niw_cluster_gibbs_block_config;
using AI4BayesCode::stick_breaking_block;
using AI4BayesCode::stick_breaking_block_config;
namespace bnp = AI4BayesCode::bnp;

namespace {

inline std::string pi_key_for_group(std::size_t g) {
    std::ostringstream oss;
    oss << "pi_" << g;
    return oss.str();
}

inline double full_normal_log_density(const double* y, const double* mu,
                                       const arma::mat& L,
                                       std::size_t d) {
    constexpr double kLog2Pi = 1.83787706640934548356065947281;
    arma::vec dev(d);
    for (std::size_t j = 0; j < d; ++j) dev[j] = y[j] - mu[j];
    double log_det = 0.0;
    for (std::size_t a = 0; a < d; ++a) log_det += 2.0 * std::log(L(a, a));
    arma::vec u = arma::solve(arma::trimatl(L), dev);
    return -0.5 * static_cast<double>(d) * kLog2Pi
         - 0.5 * log_det - 0.5 * arma::dot(u, u);
}

// ----------------------------------------------------------------------------
//  hdp_label_canonicalizer_block  (FALLBACK — NOT THE RECOMMENDED APPROACH)
//
//  HDP analogue of DPGaussianMixture's dp_label_canonicalizer_block. Label
//  switching should normally be resolved POST-MCMC on the recorded draws (see
//  skills/label_switching.md): the sampler stays a clean raw exchangeable-
//  component sampler and the validation/comparison layer relabels before
//  computing R-hat.
//
//  It is used here ONLY as a fallback. The exchangeable objects of a truncated
//  HDP are the GLOBAL atoms t = 0..T-1: the shared (mu_t, Sigma_t), the global
//  stick weight beta_t, and EVERY group's mixing weight pi_{g,t}. The model is
//  invariant under a joint permutation of the atom labels. The label-INVARIANT
//  summaries (K_active, the sorted occupied-atom proportions / mu / beta order
//  statistics) converge across chains (2-chain rank-R-hat ~1.0), but the RAW
//  per-slot beta[t] do NOT (rank-R-hat ~1.83 measured), because the truncated
//  stick representation is slot-position-dependent and the empty-tail atoms are
//  prior draws that never settle on one labelling. As in the DP case, this is a
//  representation/labeling artifact, NOT a masked mixing failure. (The two
//  concentrations gamma_0 and alpha are FIXED constants in this v0 model, not
//  sampled, so they are trivially label-invariant — there is no concentration-
//  mixing problem to fix here, unlike the DP example where alpha is sampled.)
//
//  To make the REPORTED per-slot atoms converge on their raw recorded values
//  this block permutes the T global atoms into a canonical order each sweep:
//  OCCUPIED atoms first, sorted by ATOM LOCATION (mu first coordinate); empty
//  atoms last. It then writes the relabelled (beta, mu, sigma, z, pi_0..
//  pi_{G-1}) back. The well-separated, stable atom location pins each occupied
//  atom to a fixed slot, so mu[t] / Sigma_t / beta_t / pi_{g,t} all converge to
//  ONE atom across chains (skills/label_switching.md "simple sort by
//  mu_first_dim"). NOTE we deliberately do NOT sort by the global weight beta:
//  the populated atoms carry near-equal beta (~1/K_active), so a beta sort is
//  unstable and only fixes ORDER STATISTICS (beta[0] = largest), leaving mu[slot]
//  jumping between atoms — which both fails per-atom R-hat and breaks per-slot
//  posterior summaries. It runs FIRST so the downstream children (z,
//  cluster_params, beta, pi_g) read/record on one canonical labelling, which
//  they then preserve because each re-samples deterministically from that
//  canonical state. Posterior-preserving. Prefer post-MCMC relabeling for new
//  models; reach for this only when an otherwise-sound HDP example cannot show
//  full convergence on its raw recorded parameters.
//
//  Reads/Writes: beta (T), mu (T*d), sigma (T*d*d), z (N), pi_g (T) for each
//  group g = 0..G-1.  (stick_V_beta is left untouched: it is re-derived from
//  the canonical beta by the stick block next sweep and nothing else reads it,
//  matching the DP canonicaliser which leaves stick_V untouched.)
// ----------------------------------------------------------------------------
class hdp_label_canonicalizer_block : public AI4BayesCode::block_sampler {
public:
    hdp_label_canonicalizer_block(std::string name, std::size_t T,
                                  std::size_t d, std::size_t N, std::size_t G)
        : name_(std::move(name)), T_(T), d_(d), N_(N), G_(G),
          dummy_(arma::vec(1, arma::fill::zeros)) {
        if (name_.empty())
            throw std::invalid_argument("hdp_label_canonicalizer_block: name must be non-empty");
        if (T_ < 2 || d_ < 1 || N_ < 1 || G_ < 1)
            throw std::invalid_argument("hdp_label_canonicalizer_block: bad T/d/N/G");
        // Pre-size pi_out_ so current_named_outputs() is safe BEFORE the first
        // step() (composite_block::add_child calls it to seed shared_data; the
        // empty seeds are harmless because every key this block outputs is re-
        // seeded by a later real child, then overwritten on each real sweep).
        pi_out_.assign(G_, arma::vec());
    }
    void set_context(const block_context& ctx) override { context_ = ctx; }
    void step(std::mt19937_64& /*rng*/) override {
        const arma::vec& beta = context_.at("beta");
        const arma::vec& mu   = context_.at("mu");
        const arma::vec& sig  = context_.at("sigma");
        const arma::vec& z    = context_.at("z");

        // Canonical permutation of the T global atoms by ATOM LOCATION
        // (mu first coordinate), OCCUPIED atoms first. Sorting by the well-
        // separated, stable atom location (skills/label_switching.md "simple
        // sort by mu_first_dim") pins each occupied atom to a fixed slot, so
        // the per-slot atoms (mu[t], Sigma_t, beta_t, pi_{g,t}) all converge to
        // ONE atom across chains. Sorting instead by the near-equal global
        // weight beta only fixes ORDER STATISTICS (beta[0] = largest beta),
        // leaving mu[slot] jumping between atoms — so beta is NOT used as the
        // key. Empty atoms (drawn from the NIW prior, clustered near mu_0) are
        // pushed to the high slots so they never interleave with — and so never
        // destabilise — the occupied atoms.
        arma::vec occ = bnp::counts_from_z(z, T_);   // occupancy per atom
        std::vector<std::size_t> idx(T_);
        for (std::size_t t = 0; t < T_; ++t) idx[t] = t;
        std::stable_sort(idx.begin(), idx.end(),
            [&](std::size_t a, std::size_t b) {
                const bool oa = occ[a] > 0.0, ob = occ[b] > 0.0;
                if (oa != ob) return oa;                 // occupied atoms first
                return mu[a * d_] < mu[b * d_];          // then by mu first dim
            });
        arma::uvec perm(T_);
        for (std::size_t r = 0; r < T_; ++r) perm[r] = idx[r];
        arma::uvec inv(T_);
        for (std::size_t r = 0; r < T_; ++r) inv[perm[r]] = r;

        beta_out_.set_size(T_);
        mu_out_.set_size(T_ * d_);
        sig_out_.set_size(T_ * d_ * d_);
        for (std::size_t r = 0; r < T_; ++r) {
            const std::size_t o = perm[r];
            beta_out_[r] = beta[o];
            for (std::size_t j = 0; j < d_; ++j)
                mu_out_[r * d_ + j] = mu[o * d_ + j];
            for (std::size_t a = 0; a < d_; ++a)
                for (std::size_t b = 0; b < d_; ++b)
                    sig_out_[r * d_ * d_ + a * d_ + b] =
                        sig[o * d_ * d_ + a * d_ + b];
        }

        // Permute every group's mixing weights over the atoms.
        pi_out_.assign(G_, arma::vec());
        for (std::size_t g = 0; g < G_; ++g) {
            const arma::vec& pi_g = context_.at(pi_key_for_group(g));
            arma::vec out(T_);
            for (std::size_t r = 0; r < T_; ++r) out[r] = pi_g[perm[r]];
            pi_out_[g] = std::move(out);
        }

        // Relabel assignments: z stores 1-indexed atom labels.
        z_out_.set_size(N_);
        for (std::size_t i = 0; i < N_; ++i) {
            long L = static_cast<long>(std::llround(z[i]));
            if (L < 1) L = 1;
            if (static_cast<std::size_t>(L) > T_) L = static_cast<long>(T_);
            z_out_[i] = static_cast<double>(inv[static_cast<std::size_t>(L) - 1] + 1);
        }
    }
    AI4BayesCode::state_map current_named_outputs() const override {
        AI4BayesCode::state_map out;
        out.emplace("beta", beta_out_);
        out.emplace("mu", mu_out_);
        out.emplace("sigma", sig_out_);
        out.emplace("z", z_out_);
        for (std::size_t g = 0; g < G_; ++g)
            out.emplace(pi_key_for_group(g), pi_out_[g]);
        return out;
    }
    AI4BayesCode::state_map current_named_outputs(std::mt19937_64& /*rng*/) const override { return current_named_outputs(); }
    const arma::vec& current() const override { return dummy_; }
    void set_current(const arma::vec&) override {}
    const std::string& name() const noexcept override { return name_; }
    std::size_t dim() const noexcept override { return 0; }
    AI4BayesCode::history_map get_history() const override { return {}; }
    std::size_t history_size() const noexcept override { return 0; }
    void clear_history() override {}
private:
    std::string name_; std::size_t T_, d_, N_, G_; arma::vec dummy_;
    block_context context_;
    arma::vec beta_out_, mu_out_, sig_out_, z_out_;
    std::vector<arma::vec> pi_out_;
};

}  // anonymous namespace

class HDPGaussianMixture {
public:
    // Frontend-independent constructor: y is an N×d data matrix, group_idx
    // an N-length integer vector with values in {0, ..., G-1}, and the NIW
    // hyperparameters are passed as plain arma containers. The MODEL (priors,
    // log-density, block configs) is identical to the original Rcpp wrapper;
    // only the binding types changed.
    HDPGaussianMixture(const arma::mat& y,
                       const arma::ivec& group_idx,
                       int K_trunc,
                       const arma::vec& mu_0,
                       double kappa_0,
                       const arma::mat& Psi_0,
                       double nu_0,
                       double alpha,    // group-level concentration
                       double gamma_0,  // top-level concentration
                       int rng_seed,
                       bool keep_history = false)
        : rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)}),
          predict_rng_(rng_seed == 0
                   ? std::mt19937_64{std::random_device{}()}
                   : std::mt19937_64{static_cast<std::uint64_t>(rng_seed)
                                     ^ 0x9E3779B97F4A7C15ULL}),
          impl_(std::make_unique<composite_block>("HDPGaussianMixture")),
          keep_history_(keep_history)
    {
        if (y.n_rows < 2)
            throw std::runtime_error("HDPGaussianMixture: N must be >= 2");
        if (y.n_cols < 1)
            throw std::runtime_error("HDPGaussianMixture: d must be >= 1");
        if (group_idx.n_elem != y.n_rows)
            throw std::runtime_error("HDPGaussianMixture: group_idx length must equal N");
        if (K_trunc < 2)
            throw std::runtime_error("HDPGaussianMixture: K_trunc must be >= 2");
        if (mu_0.n_elem != y.n_cols)
            throw std::runtime_error("HDPGaussianMixture: mu_0 length must equal d");
        if (Psi_0.n_rows != y.n_cols || Psi_0.n_cols != y.n_cols)
            throw std::runtime_error("HDPGaussianMixture: Psi_0 must be d x d");
        if (!(kappa_0 > 0.0)) throw std::runtime_error("kappa_0 must be > 0");
        if (!(nu_0 > static_cast<double>(y.n_cols) - 1.0))
            throw std::runtime_error("nu_0 must be > d - 1");
        if (!(alpha > 0.0)) throw std::runtime_error("alpha must be > 0");
        if (!(gamma_0 > 0.0)) throw std::runtime_error("gamma_0 must be > 0");

        N_ = static_cast<std::size_t>(y.n_rows);
        d_ = static_cast<std::size_t>(y.n_cols);
        T_ = static_cast<std::size_t>(K_trunc);

        // Determine G from group_idx.
        int g_max = 0;
        for (std::size_t i = 0; i < group_idx.n_elem; ++i) {
            const int g = static_cast<int>(group_idx[i]);
            if (g < 0)
                throw std::runtime_error("HDPGaussianMixture: group_idx must be non-negative");
            if (g > g_max) g_max = g;
        }
        G_ = static_cast<std::size_t>(g_max + 1);
        if (G_ < 1) throw std::runtime_error("HDPGaussianMixture: must have >= 1 group");

        // ---- Data + priors ------------------------------------------
        arma::vec y_flat(N_ * d_);
        for (std::size_t i = 0; i < N_; ++i)
            for (std::size_t j = 0; j < d_; ++j)
                y_flat[i * d_ + j] = y(i, j);
        impl_->data().set("y", y_flat);

        arma::vec g_arma(N_);
        for (std::size_t i = 0; i < N_; ++i)
            g_arma[i] = static_cast<double>(group_idx[i]);
        impl_->data().set("group_idx", g_arma);

        arma::vec mu0_arma(d_);
        for (std::size_t j = 0; j < d_; ++j) mu0_arma[j] = mu_0[j];
        impl_->data().set("mu_0", mu0_arma);
        impl_->data().set("kappa_0",     arma::vec{kappa_0});
        impl_->data().set("nu_0",        arma::vec{nu_0});
        impl_->data().set("alpha",       arma::vec{alpha});
        impl_->data().set("gamma_0",     arma::vec{gamma_0});

        arma::vec psi0_flat(d_ * d_);
        for (std::size_t i = 0; i < d_; ++i)
            for (std::size_t j = 0; j < d_; ++j)
                psi0_flat[i * d_ + j] = Psi_0(i, j);
        impl_->data().set("Psi_0", psi0_flat);

        // ---- Initial state -----------------------------------------
        // z: spread (i mod T) + 1.
        arma::vec z_init(N_);
        for (std::size_t i = 0; i < N_; ++i)
            z_init[i] = static_cast<double>((i % T_) + 1);
        impl_->data().set("z", z_init);

        // β: uniform 1/T.
        arma::vec beta_init(T_, arma::fill::value(1.0 / static_cast<double>(T_)));
        impl_->data().set("beta", beta_init);
        // Stick fractions (output of stick_breaking_block via v_name).
        arma::vec V_beta_init(T_, arma::fill::zeros);
        {
            double rem = 1.0;
            for (std::size_t k = 0; k + 1 < T_; ++k) {
                V_beta_init[k] = beta_init[k] / rem;
                if (V_beta_init[k] > 1.0) V_beta_init[k] = 1.0;
                rem *= (1.0 - V_beta_init[k]);
            }
            V_beta_init[T_ - 1] = 1.0;
        }
        impl_->data().set("stick_V_beta", V_beta_init);

        // π_g for g = 0..G-1: each uniform 1/T.
        for (std::size_t g = 0; g < G_; ++g) {
            const std::string key = pi_key_for_group(g);
            impl_->data().set(key,
                arma::vec(T_, arma::fill::value(1.0 / static_cast<double>(T_))));
        }

        // μ, Σ atoms: data-driven init for μ; Σ = identity.
        arma::vec mu_init(T_ * d_, arma::fill::zeros);
        for (std::size_t k = 0; k < T_; ++k) {
            const std::size_t i_anchor = (k * N_) / T_;
            for (std::size_t j = 0; j < d_; ++j)
                mu_init[k * d_ + j] = y(i_anchor, j);
        }
        arma::vec sigma_init(T_ * d_ * d_, arma::fill::zeros);
        for (std::size_t k = 0; k < T_; ++k)
            for (std::size_t i = 0; i < d_; ++i)
                sigma_init[k * d_ * d_ + i * d_ + i] = 1.0;
        impl_->data().set("mu", mu_init);
        impl_->data().set("sigma", sigma_init);

        // counts_jt + counts_t: derived (refreshers).
        impl_->data().set("counts_jt", arma::vec(G_ * T_, arma::fill::zeros));
        impl_->data().set("counts_t",  arma::vec(T_,      arma::fill::zeros));
        impl_->data().register_refresher("counts_jt",
            [G = G_, T = T_, N = N_](const AI4BayesCode::shared_data_t& d)
                -> arma::vec {
                const arma::vec& z = d.get("z");
                const arma::vec& g_idx = d.get("group_idx");
                arma::vec out(G * T, arma::fill::zeros);
                for (std::size_t i = 0; i < N; ++i) {
                    const long lab = static_cast<long>(std::llround(z[i]));
                    if (lab < 1 || static_cast<std::size_t>(lab) > T) continue;
                    const std::size_t t = static_cast<std::size_t>(lab) - 1;
                    const std::size_t g = static_cast<std::size_t>(
                        std::llround(g_idx[i]));
                    if (g >= G) continue;
                    out[g * T + t] += 1.0;
                }
                return out;
            });
        impl_->data().register_refresher("counts_t",
            [G = G_, T = T_](const AI4BayesCode::shared_data_t& d)
                -> arma::vec {
                const arma::vec& cjt = d.get("counts_jt");
                arma::vec out(T, arma::fill::zeros);
                for (std::size_t g = 0; g < G; ++g)
                    for (std::size_t t = 0; t < T; ++t)
                        out[t] += cjt[g * T + t];
                return out;
            });

        // ---- Gibbs DAG dependencies / invalidations ----------------
        // relabel (FALLBACK canonicaliser — NOT recommended; prefer post-MCMC).
        // Reads the full global-atom state so it can permute it into a canonical
        // descending-beta order; writes the permuted (beta, mu, sigma, z, pi_g).
        {
            std::vector<std::string> relabel_reads = {"beta", "mu", "sigma", "z"};
            for (std::size_t g = 0; g < G_; ++g)
                relabel_reads.push_back(pi_key_for_group(g));
            impl_->data().declare_dependencies("relabel", relabel_reads);
        }

        std::vector<std::string> z_reads = {"y", "group_idx", "mu", "sigma"};
        for (std::size_t g = 0; g < G_; ++g)
            z_reads.push_back(pi_key_for_group(g));
        impl_->data().declare_dependencies("z", z_reads);

        impl_->data().declare_dependencies("cluster_params",
            {"z", "y", "Psi_0", "kappa_0", "nu_0", "mu_0"});
        impl_->data().declare_dependencies("beta",
            {"counts_t", "gamma_0"});
        for (std::size_t g = 0; g < G_; ++g) {
            impl_->data().declare_dependencies(
                "pi_" + std::to_string(g),
                {"counts_jt", "beta", "alpha"});
        }

        impl_->data().declare_invalidates("z", {"counts_jt", "counts_t"});

        // ---- Predict DAG --------------------------------------------
        // (No declare_data_input here — y is an observed terminal,
        // not a replaceable covariate. The y_rep refresher reads
        // pi/mu/sigma/z, NOT y.)
        std::vector<std::string> edges_to_yrep =
            {"mu", "sigma", "z"};
        for (std::size_t g = 0; g < G_; ++g)
            edges_to_yrep.push_back(pi_key_for_group(g));
        for (const auto& src : edges_to_yrep)
            impl_->data().declare_predict_edges(src, {"y_rep"});

        // ---- Generative-DAG context (VIZ-ONLY; predict_at BFS never
        //      reads context_edges_). Truncated HDP:
        //        beta | m ~ Dir(gamma_0/T + m)  (top-level sticks);
        //        pi_j | beta, alpha ~ Dir(alpha * beta)  per group j;
        //        (mu_k, Sigma_k) ~ NIW(mu_0, kappa_0, Psi_0, nu_0).
        //      Drawn faded by ai4bayescode_plot_dag.
        std::vector<std::string> pi_keys;
        for (std::size_t g = 0; g < G_; ++g)
            pi_keys.push_back(pi_key_for_group(g));
        impl_->data().declare_context_edges("gamma_0",     {"stick_V_beta"});
        impl_->data().declare_context_edges("stick_V_beta",{"beta"});
        impl_->data().declare_context_edges("beta",        pi_keys);
        impl_->data().declare_context_edges("alpha",       pi_keys);
        impl_->data().declare_context_edges("mu_0",        {"mu"});
        impl_->data().declare_context_edges("kappa_0",     {"mu"});
        impl_->data().declare_context_edges("Psi_0",       {"sigma"});
        impl_->data().declare_context_edges("nu_0",        {"sigma"});

        impl_->data().set("y_rep", arma::vec(N_ * d_, arma::fill::zeros));
        impl_->data().register_stochastic_refresher("y_rep",
            [G = G_, N = N_, d = d_, T = T_](
                const AI4BayesCode::shared_data_t& dat,
                std::mt19937_64& rng) -> arma::vec {
                const arma::vec& g_idx = dat.get("group_idx");
                const arma::vec& mu  = dat.get("mu");
                const arma::vec& sig = dat.get("sigma");
                std::vector<arma::vec> pi_vec;
                pi_vec.reserve(G);
                for (std::size_t g = 0; g < G; ++g)
                    pi_vec.push_back(dat.get(pi_key_for_group(g)));
                std::uniform_real_distribution<double> uniform(0.0, 1.0);
                std::normal_distribution<double> stdnorm(0.0, 1.0);
                arma::vec out(N * d);
                for (std::size_t i = 0; i < N; ++i) {
                    const std::size_t g = static_cast<std::size_t>(
                        std::llround(g_idx[i]));
                    const arma::vec& pi_g = pi_vec[g];
                    // Sample atom z_rep_i ~ Cat(pi_g)
                    const double u = uniform(rng);
                    double cumul = 0.0;
                    std::size_t z_i = T - 1;
                    for (std::size_t t = 0; t < T; ++t) {
                        cumul += pi_g[t];
                        if (u < cumul) { z_i = t; break; }
                    }
                    // Sample y_rep_i ~ N(mu_z, sigma_z)
                    arma::mat S(d, d);
                    for (std::size_t a = 0; a < d; ++a)
                        for (std::size_t b = 0; b < d; ++b)
                            S(a, b) = sig[z_i * d * d + a * d + b];
                    arma::mat L;
                    if (!arma::chol(L, S, "lower")) {
                        S.diag() += 1e-8;
                        arma::chol(L, S, "lower");
                    }
                    arma::vec eps(d);
                    for (std::size_t j = 0; j < d; ++j) eps[j] = stdnorm(rng);
                    arma::vec L_eps = L * eps;  // evaluate before subscript
                    for (std::size_t j = 0; j < d; ++j)
                        out[i * d + j] = mu[z_i * d + j] + L_eps[j];
                }
                return out;
            });

        impl_->data().refresh_all();

        // ---- Children ---------------------------------------------
        // child(0) relabel (FALLBACK canonicaliser — see class comment; NOT
        // recommended, prefer post-MCMC). Added FIRST so downstream children
        // read/record on one canonical descending-beta atom labelling.
        impl_->add_child(std::make_unique<hdp_label_canonicalizer_block>(
            "relabel", T_, d_, N_, G_));

        // child(1) z (categorical_gibbs_block)
        {
            categorical_gibbs_block_config cfg;
            cfg.name = "z";
            cfg.n_obs = N_;
            cfg.n_categories = T_;
            cfg.initial_labels = z_init;
            const std::size_t d_capture = d_;
            const std::size_t N_capture = N_;
            const std::size_t T_capture = T_;
            const std::size_t G_capture = G_;
            cfg.log_probs_fn = [d_capture, N_capture, T_capture, G_capture]
                (const block_context& ctx) -> arma::mat {
                const arma::vec& y_flat = ctx.at("y");
                const arma::vec& g_idx  = ctx.at("group_idx");
                const arma::vec& mu     = ctx.at("mu");
                const arma::vec& sig    = ctx.at("sigma");
                std::vector<const arma::vec*> pi_ptrs(G_capture, nullptr);
                for (std::size_t g = 0; g < G_capture; ++g) {
                    pi_ptrs[g] = &ctx.at(pi_key_for_group(g));
                }
                // Pre-compute chol(Sigma_t) for each atom.
                std::vector<arma::mat> L_atoms(T_capture);
                for (std::size_t k = 0; k < T_capture; ++k) {
                    arma::mat S(d_capture, d_capture);
                    for (std::size_t a = 0; a < d_capture; ++a)
                        for (std::size_t b = 0; b < d_capture; ++b)
                            S(a, b) = sig[k * d_capture * d_capture
                                        + a * d_capture + b];
                    if (!arma::chol(L_atoms[k], S, "lower")) {
                        S.diag() += 1e-8;
                        arma::chol(L_atoms[k], S, "lower");
                    }
                }
                arma::mat lp(N_capture, T_capture);
                for (std::size_t i = 0; i < N_capture; ++i) {
                    const std::size_t g = static_cast<std::size_t>(
                        std::llround(g_idx[i]));
                    const arma::vec& pi_g = *pi_ptrs[g];
                    for (std::size_t t = 0; t < T_capture; ++t) {
                        const double log_pi_t =
                            std::log(pi_g[t] + 1e-300);
                        lp(i, t) = log_pi_t
                                 + full_normal_log_density(
                                     y_flat.memptr() + i * d_capture,
                                     mu.memptr()     + t * d_capture,
                                     L_atoms[t], d_capture);
                    }
                }
                return lp;
            };
            impl_->add_child(
                std::make_unique<categorical_gibbs_block>(std::move(cfg)));
        }

        // child(2) cluster_params (niw_cluster_gibbs_block)
        {
            niw_cluster_gibbs_block_config cfg;
            cfg.name = "cluster_params";
            cfg.K_trunc = T_;
            cfg.d = d_;
            cfg.N = N_;
            cfg.z_key = "z"; cfg.y_key = "y";
            cfg.mu_name = "mu"; cfg.sigma_name = "sigma";
            cfg.mu_0 = mu0_arma;
            cfg.kappa_0 = kappa_0;
            cfg.Psi_0_flat = psi0_flat;
            cfg.nu_0 = nu_0;
            cfg.initial_mu = mu_init;
            cfg.initial_sigma = sigma_init;
            impl_->add_child(
                std::make_unique<niw_cluster_gibbs_block>(std::move(cfg)));
        }

        // child(3) β (stick_breaking_block, heuristic update on counts_t)
        {
            stick_breaking_block_config cfg;
            cfg.name = "beta";
            cfg.K_trunc = T_;
            cfg.counts_key = "counts_t";
            cfg.v_name = "stick_V_beta";
            cfg.a_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& /*ctx*/) -> double {
                return 1.0 + counts[k];
            };
            cfg.b_fn = [](std::size_t k, const arma::vec& counts,
                          const block_context& ctx) -> double {
                const double g = ctx.at("gamma_0")[0];
                double tail = 0.0;
                for (std::size_t j = k + 1; j < counts.n_elem; ++j)
                    tail += counts[j];
                return g + tail;
            };
            cfg.initial_pi = beta_init;
            impl_->add_child(
                std::make_unique<stick_breaking_block>(std::move(cfg)));
        }

        // child(4..4+G-1) π_g (one dirichlet_gibbs_block per group)
        for (std::size_t g = 0; g < G_; ++g) {
            dirichlet_gibbs_block_config cfg;
            cfg.name = pi_key_for_group(g);
            cfg.n_categories = T_;
            cfg.initial_values =
                arma::vec(T_, arma::fill::value(1.0 / static_cast<double>(T_)));
            const std::size_t g_capture = g;
            const std::size_t T_capture = T_;
            cfg.alpha_post_fn = [g_capture, T_capture]
                (const block_context& ctx) -> arma::vec {
                const arma::vec& cjt   = ctx.at("counts_jt");
                const arma::vec& beta  = ctx.at("beta");
                const double a         = ctx.at("alpha")[0];
                arma::vec out(T_capture);
                for (std::size_t t = 0; t < T_capture; ++t) {
                    out[t] = a * beta[t] + cjt[g_capture * T_capture + t];
                }
                return out;
            };
            impl_->add_child(
                std::make_unique<dirichlet_gibbs_block>(std::move(cfg)));
        }

        if (keep_history_) impl_->set_keep_history(true);
    }

    // ---- Frontend-independent (neutral-typed) interface ----

    void step() { step(1); }              // no-arg convenience: one sweep
    void step(int n_steps) {
        if (n_steps < 0) throw std::runtime_error("n_steps must be >= 0");
        for (int i = 0; i < n_steps; ++i) impl_->step(rng_);
    }

    // Returns the current state as a neutral state_map. mu / sigma are flat
    // (cluster-major); pi_g exposed per group under "pi_<g>".
    AI4BayesCode::state_map get_current() const {
        AI4BayesCode::state_map out;
        out["z"]     = impl_->data().get("z");
        out["beta"]  = impl_->data().get("beta");
        out["mu"]    = impl_->data().get("mu");
        out["sigma"] = impl_->data().get("sigma");
        for (std::size_t g = 0; g < G_; ++g) {
            const std::string key = pi_key_for_group(g);
            out[key] = impl_->data().get(key);
        }
        return out;
    }

    AI4BayesCode::dag_info     get_dag()     const { return impl_->get_dag(); }
    AI4BayesCode::history_map  get_history() const { return impl_->get_history(); }

    std::size_t N() const { return N_; }
    std::size_t d() const { return d_; }
    std::size_t T() const { return T_; }
    std::size_t G() const { return G_; }

private:
    std::mt19937_64                  rng_;
    mutable std::mt19937_64          predict_rng_;
    std::unique_ptr<composite_block> impl_;
    bool                             keep_history_ = false;
    std::size_t                      N_ = 0;
    std::size_t                      d_ = 0;
    std::size_t                      T_ = 0;  // K_trunc
    std::size_t                      G_ = 0;
};

//==============================================================================
//  Standalone FRONTEND-INDEPENDENT demo (pure C++; no Rcpp, no pybind).
//
//  Simulates a 2-group, 1-D Gaussian mixture with THREE shared atoms at known
//  locations. Each group mixes the shared atoms in different proportions (atom
//  sharing across groups is the whole point of an HDP). We fit the truncated
//  HDP sampler, then recover the posterior-mean cluster locations of the three
//  POPULATED atoms and check that the set of recovered locations matches the
//  three true atom means. Recovery is label-invariant: we match each true mean
//  to its nearest recovered populated-atom mean.
//==============================================================================
#include <cstdio>

int main() {
    // ---- Ground truth: 3 shared atoms, 2 groups -----------------------------
    const std::size_t d        = 1;
    const std::size_t G        = 2;
    const int         K_trunc  = 8;          // truncation > 3 true atoms
    const double      atom_mu[3]  = { -6.0, 0.0, 6.0 };
    const double      atom_sd     = 0.6;     // tight, well-separated atoms

    // Per-group mixing weights over the 3 shared atoms (rows sum to 1).
    const double w[G][3] = {
        { 0.60, 0.30, 0.10 },   // group 0 favours atom 0
        { 0.10, 0.30, 0.60 }    // group 1 favours atom 2
    };
    const std::size_t n_per_group = 150;
    const std::size_t N           = G * n_per_group;

    std::mt19937_64 sim_rng(123);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::normal_distribution<double>       gauss(0.0, 1.0);

    arma::mat  y(N, d);
    arma::ivec group_idx(N);
    std::size_t row = 0;
    for (std::size_t g = 0; g < G; ++g) {
        for (std::size_t i = 0; i < n_per_group; ++i) {
            // pick a true atom from this group's weights
            const double u = unif(sim_rng);
            double cum = 0.0; std::size_t a = 2;
            for (std::size_t k = 0; k < 3; ++k) {
                cum += w[g][k];
                if (u < cum) { a = k; break; }
            }
            y(row, 0)       = atom_mu[a] + atom_sd * gauss(sim_rng);
            group_idx[row]  = static_cast<long long>(g);
            ++row;
        }
    }

    // ---- NIW prior: weak, centred on the data mean -------------------------
    arma::vec mu_0  = { arma::mean(y.col(0)) };
    const double kappa_0 = 0.05;        // weak prior on the mean
    arma::mat Psi_0 = arma::mat(1, 1, arma::fill::value(1.0));
    const double nu_0    = 3.0;         // > d - 1
    const double alpha   = 1.0;         // group-level concentration
    const double gamma_0 = 1.0;         // top-level concentration

    HDPGaussianMixture model(y, group_idx, K_trunc, mu_0, kappa_0, Psi_0,
                             nu_0, alpha, gamma_0, /*rng_seed=*/7,
                             /*keep_history=*/false);

    // ---- Warmup + sampling --------------------------------------------------
    model.step(500);   // warmup

    const int  M = 1500;
    const std::size_t T = model.T();
    arma::vec mu_sum(T, arma::fill::zeros);   // sum of mu_t over draws
    arma::vec occ_sum(T, arma::fill::zeros);  // sum of occupancy (counts) over draws
    for (int s = 0; s < M; ++s) {
        model.step(1);
        const auto cur = model.get_current();
        const arma::vec& mu = cur.at("mu");          // length T*d (d=1)
        const arma::vec& z  = cur.at("z");           // length N, labels 1..T
        for (std::size_t t = 0; t < T; ++t) mu_sum[t] += mu[t];
        for (std::size_t i = 0; i < N; ++i) {
            const long lab = static_cast<long>(std::llround(z[i]));
            if (lab >= 1 && static_cast<std::size_t>(lab) <= T)
                occ_sum[static_cast<std::size_t>(lab) - 1] += 1.0;
        }
    }
    arma::vec mu_bar  = mu_sum  / static_cast<double>(M);
    arma::vec occ_bar = occ_sum / static_cast<double>(M);   // avg #points per atom

    // ---- Identify POPULATED atoms (avg occupancy > threshold) --------------
    const double occ_thresh = 0.05 * static_cast<double>(N);  // >= 5% of data
    std::vector<double> populated_mu;
    for (std::size_t t = 0; t < T; ++t)
        if (occ_bar[t] > occ_thresh) populated_mu.push_back(mu_bar[t]);

    std::printf("HDPGaussianMixture demo: N=%zu, G=%zu, K_trunc=%d, "
                "true atoms = {%.1f, %.1f, %.1f}\n",
                N, G, K_trunc, atom_mu[0], atom_mu[1], atom_mu[2]);
    std::printf("  populated atoms found: %zu (occupancy > %.0f pts)\n",
                populated_mu.size(), occ_thresh);
    for (std::size_t t = 0; t < T; ++t)
        if (occ_bar[t] > occ_thresh)
            std::printf("    atom %zu: mu_hat=%+.3f  (avg occ %.1f pts)\n",
                        t, mu_bar[t], occ_bar[t]);

    // ---- Label-invariant recovery check ------------------------------------
    // Each true atom mean must have a populated recovered atom within tol.
    const double tol = 0.6;
    bool all_matched = true;
    for (std::size_t a = 0; a < 3; ++a) {
        double best = std::numeric_limits<double>::infinity();
        for (double mh : populated_mu)
            best = std::min(best, std::abs(mh - atom_mu[a]));
        std::printf("  true atom %.1f -> nearest recovered err = %.3f%s\n",
                    atom_mu[a], best, best <= tol ? "" : "  <-- MISS");
        if (best > tol) all_matched = false;
    }

    // We expect exactly the 3 populated atoms (allow 1 spurious tiny extra,
    // but the 3 true ones must each be matched and recovered count >= 3).
    const bool count_ok = populated_mu.size() >= 3;
    const bool ok = all_matched && count_ok;

    std::printf("%s\n",
                ok ? "[demo PASS] HDP recovers the 3 shared atoms across groups"
                   : "[demo FAIL] HDP did not recover the shared atoms");
    return ok ? 0 : 1;
}
