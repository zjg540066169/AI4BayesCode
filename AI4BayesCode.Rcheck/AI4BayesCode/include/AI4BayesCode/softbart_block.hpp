/*================================================================================
 *  AI4BayesCode/softbart_block.hpp  --  block_sampler wrapper around the
 *                                       vendored softbart::softbart_model
 *                                       (Linero & Yang 2018 JRSSB, "Soft BART").
 *
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under GPL-2.0-or-later (matching upstream SoftBart R package).
 *================================================================================
 *
 *  BACKEND-NEUTRAL (Rcpp-free)
 *  ===========================
 *  This header sits on the pure-C++ / Armadillo SoftBart kernel
 *  (bart_pure_cpp/src/softbart_model.h) and names NO Rcpp:: or R:: symbols.
 *  Error/warning reporting goes through ai4b::stop / ai4b::warning
 *  (backend_neutral.hpp), so it compiles under BOTH the Rcpp (R) backend
 *  and the pybind11 / standalone backend. All data containers are
 *  arma::mat / arma::vec / arma::uvec; tree snapshots are std::string;
 *  the kernel history is a plain softbart::SoftbartHistory struct.
 *
 *  WHAT THIS BLOCK DOES
 *  ====================
 *  softbart_block runs ONE Soft BART tree-ensemble sweep per call to
 *  block_sampler::step(). Soft BART replaces hard cutpoints with
 *  smooth logistic activation around a learned bandwidth tau, which
 *  yields differentiable predictions and tends to outperform hard BART
 *  for smooth response surfaces.
 *
 *  ARCHITECTURE
 *  ------------
 *    Tier C -- softbart::softbart_model: thin wrapper around the
 *              vendored SoftBart `Forest` class in SOFTBART_VENDOR/.
 *              Exposes the same plug-in API as bart_model and
 *              genbart_model: set_X / set_Y / set_data / set_offset,
 *              update_step, predict, set_history, get_tree / set_tree.
 *
 *    Tier B -- THIS FILE. Wraps Tier C in the block_sampler interface.
 *
 *    Tier A -- user-facing wrapper in examples/SoftBartNoise.cpp (and
 *              future SoftBart* examples). Exposes the six canonical R
 *              methods via RCPP_MODULE.
 *
 *  HISTORY RECORDING
 *  -----------------
 *  The constructor unconditionally calls `impl_->set_history(true)`, so
 *  every successful step() snapshots the current forest + sigma +
 *  sigma_mu + s + var_counts into the internal softbart_model buffers.
 *  predict_history() iterates over those snapshots when called by
 *  wrapper-level predict_at.
 *
 *  TREE SERIALIZATION
 *  ------------------
 *  Soft BART trees serialize differently from BART/genbart (they carry
 *  per-branch tau + per-branch val + per-branch var). See
 *  softbart_model::get_tree() for the exact format. set_tree(s) parses
 *  this format with strict validation and refreshes the cached f(X).
 *
 *  RNG
 *  ---
 *  The pure-C++ SoftBart kernel draws from its own seedable RNG stream
 *  (bart_rng in bart_pure_cpp/src/r_compat.h, reached via the vendored
 *  Forest's bare unif_rand()/norm_rand() calls). The `std::mt19937_64&`
 *  argument passed to step() is IGNORED. To reproduce a run, seed the
 *  kernel stream (bart_rng::set_seed(seed)) before the run starts;
 *  under the Rcpp backend, call set.seed() in R instead.
 *================================================================================*/

#ifndef AI4BAYESCODE_SOFTBART_BLOCK_HPP
#define AI4BAYESCODE_SOFTBART_BLOCK_HPP

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

// Pull in the vendored pure-C++ SoftBart engine + its definitions. The
// unity header drags in soft_bart_impl.h so that Rcpp::sourceCpp() (or a
// single-TU standalone / pybind build) of any downstream wrapper has all
// linker symbols resolved. R-package builds should compile soft_bart_impl.h
// in exactly one .cpp file and opt out of the unity header via
// AI4BAYESCODE_SOFTBART_NO_UNITY.
#include "../../bart_pure_cpp/src/softbart_model.h"
#include "../../bart_pure_cpp/src/softbart_kernel_unity.h"

#include "block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"   // ai4b::stop / ai4b::warning

#include <cmath>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

// ---------------------------------------------------------------------------
// Configuration bundle for softbart_block.
// ---------------------------------------------------------------------------

struct softbart_block_config {
    /// Unique name for this block within its composite; also the key
    /// under which shared_data_t stores this block's current f(X).
    std::string name = "f_softbart";

    /// Design matrix X (rows = observations, columns = predictors).
    arma::mat x_train;

    /// Initial Y (response). Length must equal x_train.n_rows.
    arma::vec y_init;

    /// shared_data key for working response, refreshed each set_context.
    std::string working_response_key = "softbart_target";

    /// shared_data key for sigma. If non-empty and present, the wrapper
    /// forwards the value to update_step(sigma_external).
    std::string sigma_key = "sigma";

    /// SoftBart hyperparameters.
    int    ntrees      = 50;
    double k           = 2.0;
    double sigma_hat   = -1.0;     // <= 0 -> sd(Y)
    double alpha       = 1.0;
    double beta        = 2.0;
    double gamma       = 0.95;
    double width       = 0.1;      // initial branch tau scale
    double shape       = 1.0;
    double tau_rate    = 10.0;
    double alpha_scale = -1.0;     // <= 0 -> p
    double alpha_shape_1 = 0.5;
    double alpha_shape_2 = 1.0;
    bool   dart        = false;
    bool   verbose     = false;

    /// Center Y in set_Y (subtract mean(Y) before feeding the kernel).
    /// Soft BART's vendored API defaults to NOT centering, but block
    /// users typically want centering for parity with bart_block.
    bool   center_Y    = true;
};

// ---------------------------------------------------------------------------
// softbart_block: a block_sampler that runs one SoftBart sweep per step().
// ---------------------------------------------------------------------------

class softbart_block : public block_sampler {
public:
    explicit softbart_block(softbart_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.x_train.n_rows == 0 || cfg_.x_train.n_cols == 0) {
            ai4b::stop("softbart_block '" + cfg_.name +
                       "': x_train must have at least one row and one "
                       "column");
        }
        if (cfg_.y_init.n_elem != cfg_.x_train.n_rows) {
            ai4b::stop("softbart_block '" + cfg_.name +
                       "': y_init length must match x_train row count");
        }

        impl_ = std::make_unique<softbart::softbart_model>(
            cfg_.x_train, cfg_.y_init,
            cfg_.ntrees, cfg_.k, cfg_.sigma_hat,
            cfg_.alpha, cfg_.beta, cfg_.gamma,
            cfg_.width, cfg_.shape, cfg_.tau_rate,
            cfg_.alpha_scale, cfg_.alpha_shape_1, cfg_.alpha_shape_2,
            cfg_.dart, cfg_.verbose);

        // Honour cfg_.center_Y: re-set Y after construction with the
        // requested centering. (softbart_model's constructor copies Y
        // verbatim with no centering.)
        if (cfg_.center_Y) impl_->set_Y(cfg_.y_init, true);

        // Kernel history (per-step forest snapshot serialisation) is OFF
        // by default.  Turned ON only when the wrapper opts into
        // keep_tree (which forwards via set_keep_tree below).  Previously
        // unconditional ON cost ~T*tree_size of serialisation per step.
        impl_->set_history(false);

        // Initial fit (no Gibbs sweep yet): predict at training X.
        refresh_current_fit_();
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        // 1. Working response.
        auto it_y = ctx.find(cfg_.working_response_key);
        if (it_y == ctx.end()) {
            ai4b::stop("softbart_block '" + cfg_.name
                + "': context is missing working-response key '"
                + cfg_.working_response_key + "'");
        }
        const arma::vec& arma_y = it_y->second;
        if (arma_y.n_elem != cfg_.x_train.n_rows) {
            ai4b::stop("softbart_block '" + cfg_.name
                + "': working response length "
                + std::to_string(arma_y.n_elem)
                + " does not match x_train row count "
                + std::to_string(cfg_.x_train.n_rows));
        }
        // Kernel takes arma::vec directly (no Rcpp copy needed).
        impl_->set_Y(arma_y, cfg_.center_Y);

        // 2. Sigma override (sentinel <= 0 means "use internal sigma").
        sigma_override_ = -1.0;
        if (!cfg_.sigma_key.empty()) {
            auto it_s = ctx.find(cfg_.sigma_key);
            if (it_s != ctx.end()) {
                if (it_s->second.n_elem != 1) {
                    ai4b::stop("softbart_block '" + cfg_.name
                        + "': sigma context must be a length-1 vector");
                }
                const double s = it_s->second[0];
                if (!(s > 0.0) || !std::isfinite(s)) {
                    ai4b::stop("softbart_block '" + cfg_.name
                        + "': sigma must be strictly positive and finite");
                }
                sigma_override_ = s;
            }
        }
    }

    void step(std::mt19937_64& /*rng*/) override {
        if (sigma_override_ > 0.0) {
            impl_->update_step(sigma_override_);
        } else {
            impl_->update_step();
        }
        refresh_current_fit_();
    }

    const arma::vec& current() const override { return current_fit_; }

    void set_current(const arma::vec& /*theta*/) override {
        ai4b::stop("softbart_block '" + cfg_.name
            + "': set_current(arma::vec) cannot overwrite the tree "
            "forest. To restore a forest from a serialized snapshot, "
            "use set_tree(s). To update data inputs use set_X / set_Y "
            "/ set_offset / set_data instead.");
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override {
        return static_cast<std::size_t>(cfg_.x_train.n_rows);
    }

    // ---- Data-input setters -------------------------------------------

    /// y-only update. Uses cfg_.center_Y to decide centering.
    void set_Y(const arma::vec& y_new) {
        const long n_cur = static_cast<long>(cfg_.x_train.n_rows);
        if (static_cast<long>(y_new.n_elem) != n_cur) {
            ai4b::stop("softbart_block '" + cfg_.name
                + "': set_Y length "
                + std::to_string(y_new.n_elem)
                + " does not match current n = "
                + std::to_string(n_cur));
        }
        impl_->set_Y(y_new, cfg_.center_Y);
        cfg_.y_init = y_new;
        refresh_current_fit_();
    }

    /// X-only update. Row count and col count must match.
    void set_X(const arma::mat& X_new) {
        const arma::uword n_cur = cfg_.x_train.n_rows;
        const arma::uword p_cur = cfg_.x_train.n_cols;
        if (X_new.n_rows != n_cur) {
            ai4b::stop("softbart_block '" + cfg_.name
                + "': set_X row count "
                + std::to_string(X_new.n_rows)
                + " does not match current n = "
                + std::to_string(n_cur));
        }
        if (X_new.n_cols != p_cur) {
            ai4b::stop("softbart_block '" + cfg_.name
                + "': set_X col count "
                + std::to_string(X_new.n_cols)
                + " does not match current p = "
                + std::to_string(p_cur));
        }
        impl_->set_X(X_new);
        cfg_.x_train = X_new;
        refresh_current_fit_();
    }

    /// Atomic X + y update.
    void set_data(const arma::mat& X_new, const arma::vec& y_new) {
        if (X_new.n_rows != y_new.n_elem) {
            ai4b::stop("softbart_block '" + cfg_.name
                + "': set_data rows (X) / length (y) mismatch: "
                + std::to_string(X_new.n_rows) + " vs "
                + std::to_string(y_new.n_elem));
        }
        impl_->set_data(X_new, y_new, cfg_.center_Y);
        cfg_.x_train = X_new;
        cfg_.y_init  = y_new;
        refresh_current_fit_();
    }

    /// Offset-only update (length 0 -> zeros).
    void set_offset(const arma::vec& off_new) {
        const long n_cur = static_cast<long>(cfg_.x_train.n_rows);
        if (off_new.n_elem != 0 &&
            static_cast<long>(off_new.n_elem) != n_cur) {
            ai4b::stop("softbart_block '" + cfg_.name
                + "': set_offset length " + std::to_string(off_new.n_elem)
                + " does not match current n = " + std::to_string(n_cur)
                + " (length 0 for zero offset is also accepted)");
        }
        impl_->set_offset(off_new);
        // set_offset does not change f_train inside the Forest, so
        // current_fit_ stays valid.
    }

    // ---- Tree serialization (NEW canonical API) ----------------------

    /// Serialize the current forest to a string.
    /// Soft BART format: T\n followed by pre-order (B|L) lines.
    std::string get_tree() const { return impl_->get_tree(); }

    /// Replace the current forest with one parsed from `s` (strict
    /// validation; see softbart_model::set_tree).
    void set_tree(const std::string& s) {
        impl_->set_tree(s);
        refresh_current_fit_();
    }

    // ---- softbart_block-specific accessors ----------------------------

    double current_sigma() const    { return impl_->current_sigma(); }
    double current_sigma_mu() const { return impl_->current_sigma_mu(); }

    arma::vec current_var_probs() const {
        return const_cast<softbart::softbart_model*>(impl_.get())
                 ->current_var_probs();
    }
    arma::uvec current_var_counts() const {
        return const_cast<softbart::softbart_model*>(impl_.get())
                 ->current_var_counts();
    }

    /// DART activation (no-op under the new softbart_model API; DART
    /// state is set at construction via cfg_.dart).
    void startdart() { impl_->startdart(); }

    softbart::softbart_model* underlying() { return impl_.get(); }
    const softbart::softbart_model* underlying() const { return impl_.get(); }

    /// Predict f(X_new) at the CURRENT forest. Returns length-Nt vector.
    arma::vec predict(const arma::mat& X_new) {
        return impl_->predict(X_new);
    }

    /// Predict at X_new for ALL historical draws. Returns
    /// (n_draws x Nt). If no history has been recorded yet, returns
    /// a 1xNt matrix from the current forest.
    arma::mat predict_history(const arma::mat& X_new) {
        softbart::SoftbartHistory h = impl_->get_history();
        const std::size_t n_draws = h.tree_history.size();
        const std::size_t Nt = static_cast<std::size_t>(X_new.n_rows);

        if (n_draws == 0) {
            arma::vec v = predict(X_new);
            arma::mat m(1, v.n_elem);
            for (arma::uword i = 0; i < v.n_elem; ++i) m(0, i) = v[i];
            return m;
        }

        arma::mat result(n_draws, Nt, arma::fill::zeros);
        for (std::size_t d = 0; d < n_draws; ++d) {
            std::vector<::Node*> trees_d =
                softbart::deserialize_forest_from_str(h.tree_history[d]);
            arma::vec f = softbart::predict_softbart_forest_(trees_d, X_new);
            for (arma::uword k = 0; k < Nt; ++k) result(d, k) = f[k];
            softbart::free_forest_(trees_d);
        }
        return result;
    }

    // ---- History overrides --------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(
            cfg_.name, history_fit_buf_, current_fit_);
    }

    /// Accessor for the full internal SoftBart history (tree strings,
    /// sigma, sigma_mu, s, var_counts) as a backend-neutral struct.
    /// Useful for downstream posterior predictive analyses outside the
    /// wrapper.
    softbart::SoftbartHistory get_full_history() const {
        return impl_->get_history();
    }

    /// Tree-string history; per-draw serialized snapshots of the forest.
    /// Returns one entry per recorded draw, or one entry (current forest)
    /// if no history has been recorded.
    std::vector<std::string> get_tree_history() const {
        softbart::SoftbartHistory h = impl_->get_history();
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
        return history_fit_buf_.empty() ? 1 : history_fit_buf_.size();
    }

    void clear_history() override {
        impl_->clear_history();
        history_fit_buf_.clear();
    }

    // keep_history -> only numeric per-step buffer (history_fit_buf_).
    void set_keep_history(bool keep) override {
        block_sampler::set_keep_history(keep);
    }

    // keep_tree -> kernel forest snapshot serialisation, expensive.
    // Default OFF.  Mirror of bart_block::set_keep_tree.
    void set_keep_tree(bool keep) override {
        block_sampler::set_keep_tree(keep);
        impl_->set_history(keep);
    }

private:
    void refresh_current_fit_() {
        // SoftBart's predict() walks the live forest at X_train and
        // returns an arma::vec directly (kernel is pure-arma).
        current_fit_ = impl_->predict(cfg_.x_train);
        if (keep_history_) {
            history_fit_buf_.push_back(current_fit_);
        }
    }

    softbart_block_config                    cfg_;
    std::unique_ptr<softbart::softbart_model> impl_;
    arma::vec                                current_fit_;
    std::vector<arma::vec>                   history_fit_buf_;

    double sigma_override_ = -1.0;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_SOFTBART_BLOCK_HPP
