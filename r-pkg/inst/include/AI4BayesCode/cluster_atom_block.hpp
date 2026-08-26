/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *  cluster_atom_block -- per-component parameter update for mixture models,
 *  for priors with NO conjugate closed form.
 *
 *  WHAT IT SAMPLES (geometry, system_design Sec.11.1 "fixed-dim absolutely
 *  continuous"). For each component k = 0 .. K_trunc-1, given the CURRENT
 *  allocations z (owned by another block, fixed during this block's step):
 *
 *      p(theta_k | z, y, .)  proportional to  pi(theta_k) * prod_{i: z_i = k+1} f(y_i | theta_k)
 *
 *  theta_k is that component's "atom": one or more slices, each with its own
 *  support declared as a `joint_constraint` (mu REAL, sigma POSITIVE, ...).
 *  The dimension NEVER changes -- K_trunc is fixed and every component always
 *  carries an atom, so this is NOT the trans-dimensional case of Sec.11.2(a).
 *  The dimension-changing part of a BNP mixture lives in the truncated
 *  stick-breaking stack (`stick_breaking_block` + `categorical_gibbs_block`);
 *  this block is that stack's atom step.
 *
 *  WHY PER COMPONENT, AND WHY THAT IS EXACT
 *  ----------------------------------------
 *  Ishwaran & James (2001), "Gibbs Sampling Methods for Stick-Breaking Priors"
 *  (JASA 96:161-173), blocked Gibbs step (a), Eq. 18: GIVEN the allocations the
 *  atoms are CONDITIONALLY INDEPENDENT across components -- each Z_k has its own
 *  conditional, and the product runs only over the observations assigned to k.
 *  So updating the K components one at a time is the target's own factorisation,
 *  not an approximation, and each component gets its own step size.
 *
 *  EMPTY COMPONENTS NEED NO SPECIAL CASE. When n_k = 0 the product above is
 *  empty and the target reduces to the prior pi(theta_k) EXACTLY -- Ishwaran &
 *  James's "simulate Z_k ~iid H for each unoccupied k" falls out of the same
 *  expression. There is no branch to forget, and no separate prior-sampler
 *  callback to supply.
 *
 *  WHY SLICE AND NOT NUTS
 *  ----------------------
 *  The target moves every sweep: n_k changes as z reassigns observations, so a
 *  component's conditional flips between "prior-wide" (empty) and "data-narrow"
 *  (occupied), a gap of one to two orders of magnitude. NUTS must freeze its
 *  step size after warmup to stay valid (`n_warmup_per_step` MUST stay 0 --
 *  validator Check #20; ongoing per-step adaptation produces silently wrong
 *  posteriors), and a frozen step size cannot track a moving target. Measured
 *  on a truncated-DP Gaussian mixture, K_trunc = 10, 20 datasets, 20k+20k, the
 *  fraction of sweeps in which the mu vector did not move AT ALL:
 *
 *      one joint_nuts_block over all K components   74 %
 *      per-component joint_nuts_block               15 %
 *      per-component slice (this block)              0 %
 *
 *  and the corresponding cross-chain rank R-hat (median over datasets):
 *  1.1018 / 1.0866 / 1.0018. Slice needs no tuning at all: stepping-out and
 *  shrinkage adapt WITHIN a single step and leave the target invariant exactly
 *  (Neal 2003), so a per-sweep change of target shape costs it nothing.
 *
 *  THE KERNEL IS NOT REIMPLEMENTED HERE. Each scalar coordinate is driven by an
 *  internally held `univariate_slice_sampling_block`, which already carries the
 *  Neal (2003) Eq. 5 RANDOM SPLIT of the step-out budget that reversibility
 *  requires, and its own Check #15 parity test. This block is a coordinator:
 *  the K loop, the counts, the constraint routing and the write-back.
 *
 *  WHEN NOT TO USE IT
 *  ------------------
 *  If the per-component prior IS conjugate, prefer the exact blocks --
 *  `normal_gamma_cluster_gibbs_block` (diagonal Normal-Gamma) or
 *  `niw_cluster_gibbs_block` (Normal-Inverse-Wishart). They draw the same
 *  conditional in closed form: independent samples, and cheaper.
 *
 *  CONSTRAINTS (v1 scope)
 *  ----------------------
 *  Per-element kinds only -- REAL, POSITIVE, LOWER_BOUNDED, UPPER_BOUNDED,
 *  INTERVAL -- i.e. exactly the kinds for which one natural element maps to one
 *  unconstrained coordinate. Coupled / dimension-changing kinds (ORDERED,
 *  SIMPLEX, CHOLESKY_*, CORR_MATRIX, COV_MATRIX, ...) THROW from the
 *  constructor: a coordinate-wise sweep over them is well defined but needs its
 *  own ground-truth coverage, which v1 does not carry. Adding a kind is
 *  additive; shipping one untested is not.
 *
 *  THE USER SUPPLIES ONE FUNCTION
 *  ------------------------------
 *      log_density(theta_k_natural, k, ctx) -> double
 *  the component's conditional on the NATURAL scale, up to a constant, i.e.
 *  log pi(theta_k) + sum over the observations with z_i = k+1. The block adds
 *  the log|Jacobian| of the constraint transforms itself -- do NOT include it,
 *  and do NOT hand-write a Jacobian (`jacobian.md` Sec.10.1).
 *================================================================================
 */
#ifndef AI4BAYESCODE_CLUSTER_ATOM_BLOCK_HPP
#define AI4BAYESCODE_CLUSTER_ATOM_BLOCK_HPP

#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "block_sampler.hpp"
#include "constraints.hpp"
#include "joint_nuts_block.hpp"              // joint_constraint, joint_nuts_sub_param
#include "univariate_slice_sampling_block.hpp"

namespace AI4BayesCode {

/// Per-component conditional log-density on the NATURAL scale, up to a
/// constant. `theta_k` is component k's atom (all slices concatenated in
/// config order); `k` is the 0-based component index; `ctx` is the block
/// context (y, z, hyperparameters -- whatever the model declared).
using cluster_atom_density_fn = std::function<
    double(const arma::vec& theta_k_nat, std::size_t k,
           const block_context& ctx)>;

struct cluster_atom_block_config {
    /// Block name. NOT a shared_data key: this block writes one key per atom
    /// slice (see `sub_params`), so `current_named_outputs()` is overridden.
    std::string name;

    /// Number of mixture components (truncation level). Must match the length
    /// of the counts vector under `counts_key`.
    std::size_t K_trunc = 0;

    /// shared_data key holding the per-component counts n_k (length K_trunc).
    /// Typically a deterministic refresher over z.
    std::string counts_key = "cluster_counts";

    /// The atom's slices, PER COMPONENT. Same element type and same role as
    /// `joint_nuts_block_config::sub_params`, deliberately the same field name. `dim` is the per-component dimension
    /// of that slice (e.g. {"mu", d, REAL}, {"sigma", d, POSITIVE}); the block
    /// writes each one as a K_trunc * dim vector, cluster-major.
    std::vector<joint_nuts_sub_param> sub_params;

    /// The one user function. See the header comment.
    cluster_atom_density_fn log_density;

    /// Initial atoms, concatenated (cf. `joint_nuts_block_config::initial_cat`),
    /// cluster-major over components then slice order:
    /// [comp0 slice0 | comp0 slice1 | ... | comp1 slice0 | ...].
    arma::vec initial_cat;

    /// Slice tuning, forwarded verbatim to every internal
    /// univariate_slice_sampling_block.
    double      w                 = 1.0;
    std::size_t max_step_out_iter = 50;
    std::size_t max_shrink_iter   = 100;
};

/// Per-component atom sampler for mixture models -- Ishwaran & James (2001)
/// blocked Gibbs step (a) with a slice kernel. See the file header.
class cluster_atom_block : public block_sampler {
public:
    explicit cluster_atom_block(cluster_atom_block_config cfg)
        : cfg_(std::move(cfg)) {
        if (cfg_.name.empty())
            throw std::runtime_error("cluster_atom_block: cfg.name is empty");
        if (cfg_.K_trunc == 0)
            throw std::runtime_error("cluster_atom_block '" + cfg_.name +
                                     "': K_trunc must be > 0");
        if (cfg_.sub_params.empty())
            throw std::runtime_error("cluster_atom_block '" + cfg_.name +
                                     "': cfg.sub_params is empty");
        if (!cfg_.log_density)
            throw std::runtime_error("cluster_atom_block '" + cfg_.name +
                                     "': cfg.log_density is not set");
        if (cfg_.counts_key.empty())
            throw std::runtime_error("cluster_atom_block '" + cfg_.name +
                                     "': cfg.counts_key is empty");

        // v1 scope gate. A coordinate-wise sweep needs one natural element per
        // unconstrained coordinate; the coupled kinds do not have that, and
        // shipping them without ground-truth coverage is how a block gets
        // silently wrong. Reject loudly instead.
        for (const auto& sp : cfg_.sub_params) {
            if (sp.name.empty())
                throw std::runtime_error("cluster_atom_block '" + cfg_.name +
                                         "': an atom slice has an empty name");
            if (sp.dim == 0)
                throw std::runtime_error("cluster_atom_block '" + cfg_.name +
                                         "': atom slice '" + sp.name +
                                         "' has dim 0");
            if (!per_element_kind_(sp.constraint))
                throw std::runtime_error(
                    "cluster_atom_block '" + cfg_.name + "': atom slice '" +
                    sp.name + "' uses a COUPLED constraint kind. v1 supports "
                    "REAL / POSITIVE / LOWER_BOUNDED / UPPER_BOUNDED / INTERVAL "
                    "only (one natural element per unconstrained coordinate). "
                    "For a coupled kind, sample that slice in its own "
                    "joint_nuts_block, or reparameterise.");
            atom_dim_ += sp.dim;
        }
        total_dim_ = cfg_.K_trunc * atom_dim_;

        if (cfg_.initial_cat.n_elem != total_dim_)
            throw std::runtime_error(
                "cluster_atom_block '" + cfg_.name + "': initial_cat length " +
                std::to_string(cfg_.initial_cat.n_elem) + " != K_trunc * " +
                "sum(atom dims) = " + std::to_string(total_dim_));

        theta_nat_ = cfg_.initial_cat;      // cluster-major, natural scale
        build_sub_blocks_();
    }

    // ---- block_sampler contract (Tier B) --------------------------------

    void set_context(const block_context& ctx) override { context_ = ctx; }

    void step(std::mt19937_64& rng) override {
        const auto it = context_.find(cfg_.counts_key);
        if (it == context_.end())
            throw std::runtime_error("cluster_atom_block '" + cfg_.name +
                                     "': counts_key '" + cfg_.counts_key +
                                     "' not in context");
        if (it->second.n_elem != cfg_.K_trunc)
            throw std::runtime_error(
                "cluster_atom_block '" + cfg_.name + "': counts under '" +
                cfg_.counts_key + "' have length " +
                std::to_string(it->second.n_elem) + " != K_trunc " +
                std::to_string(cfg_.K_trunc));

        // Components are conditionally independent given z (I&J Eq. 18), so
        // this loop is the target's own factorisation. An empty component is
        // NOT special-cased: with n_k = 0 the user's log_density reduces to the
        // prior, which is exactly what step (a) prescribes.
        for (std::size_t k = 0; k < cfg_.K_trunc; ++k) {
            for (std::size_t c = 0; c < atom_dim_; ++c) {
                auto& sub = *subs_[k * atom_dim_ + c];
                sub.set_context(context_);
                sub.set_current(arma::vec{unc_coord_(k, c)});
                sub.step(rng);
                set_from_unc_(k, c, sub.current()[0]);
            }
        }
        record_history_();
    }

    const arma::vec& current() const override { return theta_nat_; }

    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != total_dim_)
            throw std::runtime_error(
                "cluster_atom_block '" + cfg_.name + "': set_current length " +
                std::to_string(theta.n_elem) + " != " +
                std::to_string(total_dim_));
        for (std::size_t k = 0; k < cfg_.K_trunc; ++k) {
            std::size_t off = 0;
            for (const auto& sp : cfg_.sub_params) {
                for (std::size_t j = 0; j < sp.dim; ++j) {
                    const double v = theta[k * atom_dim_ + off + j];
                    if (!support_ok_(sp.constraint, v, sp))
                        throw std::runtime_error(
                            "cluster_atom_block '" + cfg_.name +
                            "': set_current value out of support for slice '" +
                            sp.name + "' (component " + std::to_string(k) + ")");
                }
                off += sp.dim;
            }
        }
        theta_nat_ = theta;
    }

    std::size_t dim() const noexcept override { return total_dim_; }

    const std::string& name() const noexcept override { return cfg_.name; }

    /// One shared_data key per atom slice, each a K_trunc * dim vector in
    /// cluster-major order -- the same layout the conjugate cluster blocks use,
    /// so this block is a drop-in for them.
    state_map current_named_outputs() const override {
        state_map out;
        std::size_t off = 0;
        for (const auto& sp : cfg_.sub_params) {
            arma::vec v(cfg_.K_trunc * sp.dim);
            for (std::size_t k = 0; k < cfg_.K_trunc; ++k)
                for (std::size_t j = 0; j < sp.dim; ++j)
                    v[k * sp.dim + j] = theta_nat_[k * atom_dim_ + off + j];
            out.emplace(sp.name, std::move(v));
            off += sp.dim;
        }
        return out;
    }

    /// REQUIRED: composite_block::step() skips a frozen child's step(), so the
    /// per-sweep history append is skipped with it. Without this hook a frozen
    /// block's history stalls while its siblings grow and predict_at throws
    /// "inconsistent history sizes".
    void record_held_history() override { record_history_(); }

    history_map get_history() const override {
        history_map out;
        std::size_t off = 0;
        for (const auto& sp : cfg_.sub_params) {
            const std::size_t w = cfg_.K_trunc * sp.dim;
            arma::mat m(hist_.empty() ? 1 : hist_.size(), w);
            if (hist_.empty()) {
                for (std::size_t k = 0; k < cfg_.K_trunc; ++k)
                    for (std::size_t j = 0; j < sp.dim; ++j)
                        m(0, k * sp.dim + j) =
                            theta_nat_[k * atom_dim_ + off + j];
            } else {
                for (std::size_t r = 0; r < hist_.size(); ++r)
                    for (std::size_t k = 0; k < cfg_.K_trunc; ++k)
                        for (std::size_t j = 0; j < sp.dim; ++j)
                            m(r, k * sp.dim + j) =
                                hist_[r][k * atom_dim_ + off + j];
            }
            out.emplace(sp.name, std::move(m));
            off += sp.dim;
        }
        return out;
    }

    std::size_t history_size() const noexcept override {
        return hist_.empty() ? 1 : hist_.size();
    }

    void clear_history() override { hist_.clear(); }

private:
    // ---- constraint helpers ---------------------------------------------

    static bool per_element_kind_(joint_constraint c) {
        switch (c) {
            case joint_constraint::REAL:
            case joint_constraint::POSITIVE:
            case joint_constraint::LOWER_BOUNDED:
            case joint_constraint::UPPER_BOUNDED:
            case joint_constraint::INTERVAL:
                return true;
            default:
                return false;
        }
    }

    static bool support_ok_(joint_constraint c, double v,
                            const joint_nuts_sub_param& sp) {
        if (!std::isfinite(v)) return false;
        switch (c) {
            case joint_constraint::POSITIVE:       return v > 0.0;
            case joint_constraint::LOWER_BOUNDED:  return v > sp.lower;
            case joint_constraint::UPPER_BOUNDED:  return v < sp.upper;
            case joint_constraint::INTERVAL:       return v > sp.lower && v < sp.upper;
            default:                               return true;   // REAL
        }
    }

    /// Which atom slice owns flat per-component coordinate c, and its offset.
    const joint_nuts_sub_param& slice_of_(std::size_t c,
                                          std::size_t& within) const {
        std::size_t off = 0;
        for (const auto& sp : cfg_.sub_params) {
            if (c < off + sp.dim) { within = c - off; return sp; }
            off += sp.dim;
        }
        throw std::runtime_error("cluster_atom_block: coordinate out of range");
    }

    double nat_coord_(std::size_t k, std::size_t c) const {
        return theta_nat_[k * atom_dim_ + c];
    }

    double unc_coord_(std::size_t k, std::size_t c) const {
        std::size_t within = 0;
        const auto& sp = slice_of_(c, within);
        const arma::vec one{nat_coord_(k, c)};
        return to_unc_(sp, one)[0];
    }

    void set_from_unc_(std::size_t k, std::size_t c, double x_unc) {
        std::size_t within = 0;
        const auto& sp = slice_of_(c, within);
        const arma::vec one{x_unc};
        theta_nat_[k * atom_dim_ + c] = to_nat_(sp, one)[0];
    }

    static arma::vec to_nat_(const joint_nuts_sub_param& sp,
                             const arma::vec& unc) {
        switch (sp.constraint) {
            case joint_constraint::POSITIVE:
                return constraints::positive::constrain(unc);
            case joint_constraint::LOWER_BOUNDED:
                return constraints::lower_bounded::constrain(unc, sp.lower);
            case joint_constraint::UPPER_BOUNDED:
                return constraints::upper_bounded::constrain(unc, sp.upper);
            case joint_constraint::INTERVAL:
                return constraints::interval::constrain(unc, sp.lower, sp.upper);
            default:
                return unc;   // REAL
        }
    }

    static arma::vec to_unc_(const joint_nuts_sub_param& sp,
                             const arma::vec& nat) {
        switch (sp.constraint) {
            case joint_constraint::POSITIVE:
                return constraints::positive::unconstrain(nat);
            case joint_constraint::LOWER_BOUNDED:
                return constraints::lower_bounded::unconstrain(nat, sp.lower);
            case joint_constraint::UPPER_BOUNDED:
                return constraints::upper_bounded::unconstrain(nat, sp.upper);
            case joint_constraint::INTERVAL:
                return constraints::interval::unconstrain(nat, sp.lower, sp.upper);
            default:
                return nat;   // REAL
        }
    }

    /// Substitute ONE unconstrained coordinate into component k's atom and
    /// evaluate the user's natural-scale conditional. The log|Jacobian| is
    /// supplied by the library transform via  -- never hand-written here
    /// ( Sec.10.1). The gradient slot is nullptr: slice does not
    /// use one, and the inner lambda ignores it.
    double wrapped_lp_(std::size_t k, std::size_t c, const arma::vec& x_unc,
                       const block_context& ctx) const {
        std::size_t within = 0;
        const auto& sp = slice_of_(c, within);
        auto inner = [this, k, c, &ctx](const arma::vec& v_nat,
                                        arma::vec* /*g_nat*/) -> double {
            arma::vec th = atom_of_(k);
            th[c] = v_nat[0];
            return cfg_.log_density(th, k, ctx);
        };
        switch (sp.constraint) {
            case joint_constraint::POSITIVE:
                return constraints::positive::wrap(x_unc, nullptr, inner);
            case joint_constraint::LOWER_BOUNDED:
                return constraints::lower_bounded::wrap(x_unc, nullptr, sp.lower, inner);
            case joint_constraint::UPPER_BOUNDED:
                return constraints::upper_bounded::wrap(x_unc, nullptr, sp.upper, inner);
            case joint_constraint::INTERVAL:
                return constraints::interval::wrap(x_unc, nullptr, sp.lower, sp.upper, inner);
            default:
                return constraints::real::wrap(x_unc, nullptr, inner);
        }
    }

    // ---- internal slice sub-blocks --------------------------------------

    /// One univariate_slice_sampling_block per (component, coordinate). The
    /// kernel -- including the Neal (2003) Eq. 5 random split of the step-out
    /// budget -- is that block's, not ours.
    void build_sub_blocks_() {
        subs_.reserve(total_dim_);
        for (std::size_t k = 0; k < cfg_.K_trunc; ++k) {
            for (std::size_t c = 0; c < atom_dim_; ++c) {
                univariate_slice_sampling_block_config sc;
                sc.name        = cfg_.name + "__k" + std::to_string(k) +
                                 "_c" + std::to_string(c);
                sc.initial_unc = arma::vec{unc_coord_(k, c)};
                // The sub-block tracks ONE unconstrained scalar; the natural
                // value and the Jacobian are handled in the closure, so its own
                // transform is the identity.
                sc.constrain   = [](const arma::vec& v) { return v; };
                sc.unconstrain = [](const arma::vec& v) { return v; };
                sc.w                 = cfg_.w;
                sc.max_step_out_iter = cfg_.max_step_out_iter;
                sc.max_shrink_iter   = cfg_.max_shrink_iter;
                sc.log_density =
                    [this, k, c](const arma::vec& x_unc,
                                 const block_context& ctx) -> double {
                        return wrapped_lp_(k, c, x_unc, ctx);
                    };
                auto blk = std::make_unique<univariate_slice_sampling_block>(
                    std::move(sc));
                blk->set_keep_history(false);
                subs_.push_back(std::move(blk));
            }
        }
    }

    arma::vec atom_of_(std::size_t k) const {
        return theta_nat_.subvec(k * atom_dim_, (k + 1) * atom_dim_ - 1);
    }

    void record_history_() {
        if (keep_history_) hist_.push_back(theta_nat_);
    }

    cluster_atom_block_config cfg_;
    block_context             context_;
    arma::vec                 theta_nat_;     // cluster-major, natural scale
    std::size_t               atom_dim_  = 0; // per-component dimension
    std::size_t               total_dim_ = 0; // K_trunc * atom_dim_
    std::vector<std::unique_ptr<univariate_slice_sampling_block>> subs_;
    std::vector<arma::vec>    hist_;
};

}  // namespace AI4BayesCode
#endif  // AI4BAYESCODE_CLUSTER_ATOM_BLOCK_HPP
