/*================================================================================
 *  AI4BayesCode/genbart_block.hpp  --  block_sampler wrapper around the
 *                                      vendored genbart::genbart_model
 *                                      (generalized BART via RJMCMC,
 *                                      Linero 2022 arXiv:2202.09924).
 *
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under GPL-2.0-or-later (matching the upstream BART lineage).
 *================================================================================
 *
 *  BACKEND-NEUTRAL (Rcpp-free)
 *  ===========================
 *  This header sits on the pure-C++ / Armadillo genBART kernel
 *  (bart_pure_cpp/src/genbart_model.h) and names NO Rcpp:: or R:: symbols.
 *  Error/warning reporting goes through ai4b::stop / ai4b::warning
 *  (backend_neutral.hpp), so it compiles under BOTH the Rcpp (R) backend
 *  and the pybind11 / standalone backend. All data containers are
 *  arma::mat / arma::vec / arma::uvec; tree snapshots are std::string;
 *  the kernel history is a plain genbart::GenbartHistory struct.
 *
 *  WHAT THIS BLOCK DOES
 *  ====================
 *  genbart_block runs ONE generalized-BART RJMCMC tree-ensemble sweep
 *  per call to block_sampler::step(). The RJMCMC scheme (Linero 2022
 *  Algorithm 2) uses Laplace-approximated BIRTH / DEATH / CHANGE
 *  proposals and accepts any plug-in likelihood satisfying the
 *  `genbart::likelihood` three-method contract (log_f, score, obs_info).
 *  genBART supports DART + the 10 shipped likelihoods; this wrapper keeps
 *  that entire surface intact and only swaps container / error types.
 *
 *  HISTORY RECORDING
 *  -----------------
 *  Kernel history (per-step forest snapshot) is OFF by default; turned
 *  ON only when the wrapper opts into keep_tree via set_keep_tree(true).
 *  When ON, every successful step() appends the current forest + sigma +
 *  sigma_mu + s + var_counts to the internal genbart_model history
 *  buffers, which predict_history() iterates over.
 *
 *  TREE SERIALIZATION
 *  ------------------
 *  get_tree() / set_tree(s) wrap the internal genbart_model's
 *  serialize / deserialize API (now std::string in / out, mirroring
 *  bart_block). set_tree() rebuilds the live forest in-place and
 *  refreshes the cached r(X_train) for downstream blocks.
 *
 *  RNG
 *  ---
 *  The pure-C++ genBART kernel draws from its own seedable RNG stream
 *  (bart_rng in bart_pure_cpp/src/r_compat.h, driven through the `arn`
 *  adapter). The `std::mt19937_64&` argument passed to step() is IGNORED.
 *  To reproduce a run, seed the kernel stream before the run starts.
 *================================================================================*/

#ifndef AI4BAYESCODE_GENBART_BLOCK_HPP
#define AI4BAYESCODE_GENBART_BLOCK_HPP

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

// Pull in the vendored pure-C++ genBART engine. genbart_model.h aggregates
// the BART tree infrastructure + RJMCMC core + all 10 shipped likelihoods.
#include "../../bart_pure_cpp/src/genbart_model.h"

#include "block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"   // ai4b::stop / ai4b::warning

#include <cmath>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

// ---------------------------------------------------------------------------
// Configuration bundle for genbart_block.
// ---------------------------------------------------------------------------

struct genbart_block_config {
    /// Unique name for this block within its composite; also the key
    /// under which shared_data_t stores this block's current r(X)
    /// vector.
    std::string name = "r_bart";

    /// Design matrix X (rows = observations, columns = predictors).
    arma::mat x_train;

    /// Initial Y (response). Length must equal x_train.n_rows.
    arma::vec y_init;

    /// Initial offset. Empty = zeros (no offset).
    arma::vec offset_init;

    /// Non-owning likelihood pointer. MUST outlive this block.
    genbart::likelihood* lik = nullptr;

    /// Number of trees.
    std::size_t ntrees = 50;

    /// Tree / leaf / DART hyperparameters.
    genbart::rjmcmc_hypers hypers{};

    /// shared_data key for Y, refreshed at each set_context. Empty =
    /// "Y is fixed training data, never refresh".
    std::string y_key = "";

    /// shared_data key for offset, refreshed at each set_context.
    std::string offset_key = "";

    /// Number of candidate cutpoints per variable.
    int numcut = 100;

    /// If true, use empirical quantiles as cutpoints.
    bool usequants = false;

    /// Start DART (s + theta Gibbs) at construction. Use this for
    /// burn-in-free DART; alternatively call startdart() later from
    /// the wrapper for the Linero (2018) half-burnin convention.
    bool start_dart = false;
};

// ---------------------------------------------------------------------------
// genbart_block: a block_sampler that runs one generalized-BART RJMCMC
// tree-ensemble sweep per step() invocation.
// ---------------------------------------------------------------------------

class genbart_block : public block_sampler {
public:
    explicit genbart_block(genbart_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.lik == nullptr) {
            ai4b::stop("genbart_block '" + cfg_.name +
                       "': likelihood pointer is null");
        }
        if (cfg_.x_train.n_rows == 0 || cfg_.x_train.n_cols == 0) {
            ai4b::stop("genbart_block '" + cfg_.name +
                       "': x_train must have at least one row and one "
                       "column");
        }
        if (cfg_.y_init.n_elem != cfg_.x_train.n_rows) {
            ai4b::stop("genbart_block '" + cfg_.name +
                       "': y_init length must match x_train row count");
        }
        if (cfg_.offset_init.n_elem != 0 &&
            cfg_.offset_init.n_elem != cfg_.x_train.n_rows) {
            ai4b::stop("genbart_block '" + cfg_.name +
                       "': offset_init must be empty or match x_train "
                       "row count");
        }

        impl_ = std::make_unique<genbart::genbart_model>(
            cfg_.x_train, cfg_.y_init, cfg_.offset_init,
            cfg_.lik, cfg_.hypers, cfg_.ntrees,
            cfg_.numcut, cfg_.usequants, /*cont=*/false);

        // Optionally start DART at construction; Tier A wrapper can
        // also call startdart() later.
        if (cfg_.start_dart) impl_->startdart();

        // Kernel history (per-step forest snapshot) is OFF by default;
        // turned ON only when wrapper opts into keep_tree via
        // set_keep_tree(true).  Previously unconditional ON paid the
        // serialise cost on every step.
        impl_->set_history(false);

        // Cache static info (xinfo, p, ntrees) once at construction.
        cache_static_info_();

        // Initial r(X_train) = zeros (no Gibbs sweep yet).
        refresh_current_r_();
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        const arma::uword n = cfg_.x_train.n_rows;

        // Refresh Y if a key is configured and present.
        if (!cfg_.y_key.empty()) {
            auto it = ctx.find(cfg_.y_key);
            if (it != ctx.end()) {
                if (it->second.n_elem != n) {
                    ai4b::stop("genbart_block '" + cfg_.name
                        + "': Y context length "
                        + std::to_string(it->second.n_elem)
                        + " does not match x_train row count "
                        + std::to_string(n));
                }
                impl_->set_Y(it->second);
            }
        }

        // Refresh offset similarly.
        if (!cfg_.offset_key.empty()) {
            auto it = ctx.find(cfg_.offset_key);
            if (it != ctx.end()) {
                if (it->second.n_elem != n) {
                    ai4b::stop("genbart_block '" + cfg_.name
                        + "': offset context length "
                        + std::to_string(it->second.n_elem)
                        + " does not match x_train row count "
                        + std::to_string(n));
                }
                impl_->set_offset(it->second);
            }
        }
    }

    void step(std::mt19937_64& /*rng*/) override {
        // genBART uses its own RNG (genbart::arn), seeded via the kernel
        // RNG stream. The mt19937 argument is ignored.
        impl_->update_step(gen_);
        refresh_current_r_();
        if (keep_history_) {
            history_r_buf_.push_back(current_r_);
        }
    }

    /// r(X_i) at the training points on the linear-predictor scale.
    const arma::vec& current() const override { return current_r_; }

    void set_current(const arma::vec& /*theta*/) override {
        ai4b::stop("genbart_block '" + cfg_.name
            + "': set_current(arma::vec) cannot overwrite the tree "
            "forest. To restore a forest from a serialized snapshot, "
            "use set_tree(s). To update data inputs use set_X / set_Y "
            "/ set_offset / set_data instead.");
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    // Kernel-control freeze BLACKLIST (DESIGN_NOTES Sec.6): same class as
    // bart_block (non-invertible forest + derived-state hazard).
    bool supports_freeze() const noexcept override { return false; }
    std::string freeze_not_supported_reason() const override {
        return "freezing genbart_block not supported "
               "(forest is non-invertible + derived-state hazard); "
               "same rationale as bart_block";
    }

    std::size_t dim() const noexcept override {
        return static_cast<std::size_t>(cfg_.x_train.n_rows);
    }

    // ---- Data-input setters ------------------------------------------

    /// Y-only update. Triggers likelihood->prepare(Y) for per-obs cache.
    void set_Y(const arma::vec& y_new) {
        const arma::uword n = cfg_.x_train.n_rows;
        if (y_new.n_elem != n) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': set_Y length " + std::to_string(y_new.n_elem)
                + " does not match current n = " + std::to_string(n));
        }
        impl_->set_Y(y_new);
        refresh_current_r_();
    }

    /// Offset-only update. Length must match n, or empty (= zeros).
    void set_offset(const arma::vec& off_new) {
        const arma::uword n = cfg_.x_train.n_rows;
        if (off_new.n_elem != 0 && off_new.n_elem != n) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': set_offset length " + std::to_string(off_new.n_elem)
                + " does not match current n = " + std::to_string(n)
                + " (length 0 for zero offset is also accepted)");
        }
        impl_->set_offset(off_new);
        // set_offset does NOT change f_train, so current_r_ stays valid.
    }

    /// X-only update. Trees PRESERVED; obs re-routed through existing
    /// trees and f_per_tree recomputed. Row count must equal current n.
    void set_X(const arma::mat& x_new) {
        if (x_new.n_cols != cfg_.x_train.n_cols) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': set_X col count " + std::to_string(x_new.n_cols)
                + " does not match current p = "
                + std::to_string(cfg_.x_train.n_cols));
        }
        if (x_new.n_rows != cfg_.x_train.n_rows) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': set_X row count " + std::to_string(x_new.n_rows)
                + " does not match current n = "
                + std::to_string(cfg_.x_train.n_rows)
                + " (genBART set_X preserves N; to change N, "
                + "reconstruct the wrapper)");
        }
        impl_->set_X(x_new);
        cfg_.x_train = x_new;
        refresh_current_r_();
    }

    /// Atomic X + Y + offset update. Same N constraint as set_X.
    void set_data(const arma::mat& x_new,
                  const arma::vec& y_new,
                  const arma::vec& off_new) {
        if (x_new.n_cols != cfg_.x_train.n_cols) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': set_data X col count " + std::to_string(x_new.n_cols)
                + " does not match current p = "
                + std::to_string(cfg_.x_train.n_cols));
        }
        if (x_new.n_rows != cfg_.x_train.n_rows) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': set_data X row count " + std::to_string(x_new.n_rows)
                + " does not match current n = "
                + std::to_string(cfg_.x_train.n_rows));
        }
        if (y_new.n_elem != x_new.n_rows) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': set_data Y length " + std::to_string(y_new.n_elem)
                + " does not match X row count "
                + std::to_string(x_new.n_rows));
        }
        if (off_new.n_elem != 0 && off_new.n_elem != x_new.n_rows) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': set_data offset length " + std::to_string(off_new.n_elem)
                + " does not match X row count "
                + std::to_string(x_new.n_rows)
                + " (length 0 for zero offset is also accepted)");
        }
        impl_->set_X(x_new);
        impl_->set_Y(y_new);
        impl_->set_offset(off_new);
        cfg_.x_train = x_new;
        refresh_current_r_();
    }

    // ---- Tree serialization (NEW canonical API) ----------------------

    /// Serialize the current forest to a string.
    /// Format: "T\n<tree1>---\n<tree2>---\n...". Round-trips via set_tree.
    std::string get_tree() const { return impl_->get_tree(); }

    /// Replace the current forest with one parsed from `s` (strict
    /// validation; see genbart_model::set_tree for the checks).
    /// Refreshes cached r(X_train).
    void set_tree(const std::string& s) {
        impl_->set_tree(s);
        refresh_current_r_();
    }

    // ---- genbart_block-specific accessors -----------------------------

    /// Raw pointer to the underlying genBART model.
    genbart::genbart_model* underlying() { return impl_.get(); }
    const genbart::genbart_model* underlying() const { return impl_.get(); }

    /// Current sigma_mu (adaptive half-Cauchy hyperprior).
    double current_sigma_mu() const { return impl_->current_sigma_mu(); }

    /// Current DART concentration theta.
    double current_dart_theta() const { return impl_->current_dart_theta(); }

    /// Current sigma (first nuisance parameter; 0 for likelihoods
    /// without nuisance, e.g. Logistic).
    double current_sigma() const { return impl_->current_sigma(); }

    /// Current split-variable probabilities (length p).
    arma::vec current_var_probs() const {
        return impl_->current_var_probs();
    }

    /// Current split-variable usage counts across all trees (length p).
    arma::uvec current_var_counts() const {
        return const_cast<genbart::genbart_model*>(impl_.get())
                 ->current_var_counts();
    }

    /// Manually activate DART (Linero 2018 §3 half-Cauchy warmup
    /// convention is typical: call after `nburn / 2` iterations).
    void startdart() { impl_->startdart(); }

    /// Predict r(X_new) at new test points using the current live trees.
    /// Does NOT include offset.
    arma::vec predict_r(const arma::mat& X_new) {
        if (X_new.n_cols != impl_->p()) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': predict_r X_new col count "
                + std::to_string(X_new.n_cols)
                + " does not match training p = "
                + std::to_string(impl_->p()));
        }
        return impl_->predict_once(X_new);
    }

    /// Predict r(X_new) at the CURRENT forest. Alias for predict_r;
    /// kept for parity with bart_block's interface.
    arma::vec predict(const arma::mat& X_new) {
        return predict_r(X_new);
    }

    /// Predict at X_new for ALL historical draws. Returns
    /// (n_draws x Nt) matrix on the linear-predictor scale (no offset).
    /// If no history has been recorded yet, returns 1xNt from current
    /// forest.
    arma::mat predict_history(const arma::mat& X_new) {
        genbart::GenbartHistory h = impl_->get_history();
        const std::size_t n_draws = h.tree_history.size();
        const std::size_t Nt = static_cast<std::size_t>(X_new.n_rows);

        if (n_draws == 0) {
            arma::vec v = predict_r(X_new);
            arma::mat m(1, v.n_elem);
            for (arma::uword i = 0; i < v.n_elem; ++i) m(0, i) = v[i];
            return m;
        }

        if (X_new.n_cols != static_cast<arma::uword>(cached_p_)) {
            ai4b::stop("genbart_block '" + cfg_.name
                + "': predict_history X_new col count "
                + std::to_string(X_new.n_cols)
                + " does not match training p = "
                + std::to_string(cached_p_));
        }
        arma::mat result(n_draws, Nt, arma::fill::zeros);
        std::vector<double> x_row(cached_p_);
        for (std::size_t d = 0; d < n_draws; ++d) {
            std::istringstream is(h.tree_history[d]);
            std::size_t T = 0;
            is >> T;
            std::vector<genbart::tree> trees(T);
            for (std::size_t t = 0; t < T; ++t) {
                is >> trees[t];
                std::string sep;
                is >> sep;  // "---"
            }
            for (std::size_t row = 0; row < Nt; ++row) {
                for (int j = 0; j < cached_p_; ++j) x_row[j] = X_new(row, j);
                result(d, row) = genbart::genbart_walk_forest_row_(
                    trees, x_row, cached_xinfo_);
            }
        }
        return result;
    }

    // ---- History overrides --------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(cfg_.name, history_r_buf_, current_r_);
    }

    /// Accessor for the full internal genBART history (tree strings,
    /// sigma, sigma_mu, s, var counts, xinfo, ntrees, p) as a backend-
    /// neutral struct.
    genbart::GenbartHistory get_full_history() const {
        return impl_->get_history();
    }

    /// Tree-string history; one entry per recorded draw, or one entry
    /// (current forest) if no history has been recorded.
    std::vector<std::string> get_tree_history() const {
        genbart::GenbartHistory h = impl_->get_history();
        const std::size_t n = h.tree_history.size();
        std::vector<std::string> out;
        if (n == 0) {
            out.push_back(impl_->get_tree());
        } else {
            out = h.tree_history;
        }
        return out;
    }

    std::size_t history_size() const noexcept override {
        return history_r_buf_.empty() ? 1 : history_r_buf_.size();
    }

    void clear_history() override {
        impl_->clear_history();
        history_r_buf_.clear();
    }

    // keep_history -> numeric per-step buffer (history_r_buf_) only.
    void set_keep_history(bool keep) override {
        block_sampler::set_keep_history(keep);
    }

    // keep_tree -> kernel forest snapshot serialisation. Default OFF.
    // Mirror of bart_block::set_keep_tree.
    void set_keep_tree(bool keep) override {
        block_sampler::set_keep_tree(keep);
        impl_->set_history(keep);
    }

private:
    void cache_static_info_() {
        cached_p_ = static_cast<int>(impl_->p());
        genbart::GenbartHistory h = impl_->get_history();
        cached_xinfo_.resize(cached_p_);
        for (int j = 0; j < cached_p_; ++j) {
            cached_xinfo_[j] = h.xinfo[j];
        }
    }

    void refresh_current_r_() {
        current_r_ = impl_->current_f_train();
    }

    genbart_block_config                    cfg_;
    std::unique_ptr<genbart::genbart_model> impl_;
    genbart::arn                            gen_;
    arma::vec                               current_r_;
    std::vector<arma::vec>                  history_r_buf_;

    // Cached static info (xinfo for posterior walk over history).
    int cached_p_ = 0;
    genbart::xinfo cached_xinfo_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_GENBART_BLOCK_HPP
