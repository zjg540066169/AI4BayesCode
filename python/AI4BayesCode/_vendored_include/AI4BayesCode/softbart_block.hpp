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
 *  PREPROCESSING
 *  -------------
 *  As SoftBart (Linero & Yang 2018; R package SoftBart) does, the forest
 *  works on normalised inputs; every input and output of this block is on
 *  the user's scale:
 *    - X: each column is mapped to [0, 1] by the ranks of its distinct
 *      training values (the k-th smallest of m maps to k / (m - 1), SoftBart's
 *      rank transform), linearly between them for new X, 0 below and 1 above
 *      the training range, 0.5 for a constant column; the kernel draws split
 *      points in [0, 1]. set_X / set_data make the map again from the new X.
 *    - Y: the forest fits Z = (Y - m) / y_scale with y_scale fixed at
 *      construction (default max(y_init) - min(y_init)), so the leaf prior
 *      sd 0.5 / (k sqrt(ntrees)) has SoftBart's calibration; m is the mean of
 *      the working response (center_Y) or the midrange of y_init. f, sigma
 *      and predictions are returned on the scale of Y, and a sigma from the
 *      context is divided by y_scale.
 *    - sigma_hat: default sqrt(RSS / n) of the least-squares fit of y_init on
 *      an intercept and X (SoftBart uses a cross-validated lasso fit).
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
 *  With set_keep_tree(true) (off at construction), every successful
 *  step() snapshots the current forest + sigma + sigma_mu + s +
 *  var_counts into the internal softbart_model buffers.
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

#include <algorithm>
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
    /// Scale of the half-Cauchy prior on sigma when the block samples sigma
    /// (sigma_key empty), and its initial value; on the scale of Y. <= 0 ->
    /// sqrt(RSS / n) of the least-squares fit of y_init on an intercept and
    /// x_train (sd(y_init) when n <= p + 1 or the RSS is 0).
    double sigma_hat   = -1.0;
    /// Scale of the response: the forest fits Z = (Y - m) / y_scale (see
    /// center_Y for m), as SoftBart fits Y rescaled to [-0.5, 0.5]. <= 0 ->
    /// max(y_init) - min(y_init) (1 when that is 0). Set it when y_init is a
    /// placeholder (e.g. a latent variable started at 0).
    double y_scale     = -1.0;
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

    /// m in Z = (Y - m) / y_scale: true -> m = mean of the current working
    /// response, and f excludes it (parity with bart_block); false -> m =
    /// the midrange (max + min) / 2 of y_init, and f = m + y_scale f_Z
    /// (SoftBart's normalisation).
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

        if (!cfg_.y_init.is_finite()) {
            ai4b::stop("softbart_block '" + cfg_.name +
                       "': y_init must be finite");
        }

        // SoftBart's preprocessing (see PREPROCESSING above): the forest
        // works on X mapped to [0, 1] and on Z = (Y - m) / y_scale_.
        make_x_map_(cfg_.x_train);
        xn_train_ = map_x_(cfg_.x_train);
        const double y_max = cfg_.y_init.max(), y_min = cfg_.y_init.min();
        y_mid_ = 0.5 * (y_max + y_min);
        if (cfg_.y_scale > 0.0 && std::isfinite(cfg_.y_scale)) {
            y_scale_ = cfg_.y_scale;
        } else {
            const double r = y_max - y_min;
            y_scale_ = (r > 0.0 && std::isfinite(r)) ? r : 1.0;
        }
        const double sigma_hat_y =
            (cfg_.sigma_hat > 0.0) ? cfg_.sigma_hat
                                   : default_sigma_hat_(cfg_.x_train, cfg_.y_init);

        impl_ = std::make_unique<softbart::softbart_model>(
            xn_train_, to_z_(cfg_.y_init),
            cfg_.ntrees, cfg_.k, sigma_hat_y / y_scale_,
            cfg_.alpha, cfg_.beta, cfg_.gamma,
            cfg_.width, cfg_.shape, cfg_.tau_rate,
            cfg_.alpha_scale, cfg_.alpha_shape_1, cfg_.alpha_shape_2,
            cfg_.dart, cfg_.verbose);
        model_x_key_ = impl_->x_key();

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
        impl_->set_Y(to_z_(arma_y), false);

        // 2. Sigma override (sentinel <= 0 means "use internal sigma").
        sigma_override_ = -1.0;
        if (!cfg_.sigma_key.empty()) {
            auto it_s = ctx.find(cfg_.sigma_key);
            if (it_s == ctx.end()) {
                ai4b::stop("softbart_block '" + cfg_.name + "': sigma_key '" + cfg_.sigma_key
                    + "' is configured but not present in the context. Add it to "
                    "declare_dependencies(\"" + cfg_.name + "\", {...}), or leave "
                    "sigma_key empty if this model has no external sigma.");
            }
            {
                if (it_s->second.n_elem != 1) {
                    ai4b::stop("softbart_block '" + cfg_.name
                        + "': sigma context must be a length-1 vector");
                }
                const double s = it_s->second[0];
                if (!(s > 0.0) || !std::isfinite(s)) {
                    ai4b::stop("softbart_block '" + cfg_.name
                        + "': sigma must be strictly positive and finite");
                }
                sigma_override_ = s / y_scale_;   // sigma of Z
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

    // Kernel-control freeze BLACKLIST (families.md): same class as
    // bart_block (non-invertible forest + derived-state hazard).
    bool supports_freeze() const noexcept override { return false; }
    std::string freeze_not_supported_reason() const override {
        return "freezing softbart_block not supported: a fitted tree "
               "ensemble cannot be restored from a stored value. To make "
               "predictions at new data without changing the fitted model, "
               "use predict_at().";
    }

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
        if (!y_new.is_finite()) {
            ai4b::stop("softbart_block '" + cfg_.name + "': set_Y values must be finite");
        }
        impl_->set_Y(to_z_(y_new), false);
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
        // The map to [0, 1] is made again from the new X (as SoftBart
        // ranks the X it is given); nothing changes when this throws.
        std::vector<std::vector<double>> knots = x_knots_;
        make_x_map_(X_new);
        arma::mat xn;
        try { xn = map_x_(X_new); impl_->set_X(xn); }
        catch (...) { x_knots_.swap(knots); throw; }
        xn_train_ = std::move(xn);
        cfg_.x_train = X_new;
        model_x_key_ = impl_->x_key();
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
        if (!y_new.is_finite()) {
            ai4b::stop("softbart_block '" + cfg_.name + "': set_data y values must be finite");
        }
        std::vector<std::vector<double>> knots = x_knots_;
        make_x_map_(X_new);
        arma::mat xn;
        try { xn = map_x_(X_new); impl_->set_data(xn, to_z_(y_new), false); }
        catch (...) { x_knots_.swap(knots); throw; }
        xn_train_ = std::move(xn);
        cfg_.x_train = X_new;
        cfg_.y_init  = y_new;
        model_x_key_ = impl_->x_key();
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
        impl_->set_offset(off_new.n_elem == 0 ? off_new
                                              : arma::vec(off_new / y_scale_));
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

    /// sigma on the scale of Y.
    double current_sigma() const    { return y_scale_ * impl_->current_sigma(); }
    /// Leaf scale on the scale of Z = (Y - m) / y_scale().
    double current_sigma_mu() const { return impl_->current_sigma_mu(); }
    /// The scale of the response (see softbart_block_config::y_scale).
    double y_scale() const { return y_scale_; }
    /// X mapped to [0, 1] with the map of the training X (see PREPROCESSING).
    arma::mat x_to_unit(const arma::mat& X) const { return map_x_(X); }

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
        return to_y_(impl_->predict(map_x_(X_new)));
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

        const arma::mat xn_new = map_x_(X_new);
        arma::mat result(n_draws, Nt, arma::fill::zeros);
        for (std::size_t d = 0; d < n_draws; ++d) {
            std::vector<::Node*> trees_d =
                softbart::deserialize_forest_from_str(h.tree_history[d]);
            arma::vec f = to_y_(softbart::predict_softbart_forest_(trees_d, xn_new));
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
    /// sigma_history is on the scale of Y, sigma_mu_history on the scale of
    /// Z, and the trees act on X mapped to [0, 1].
    softbart::SoftbartHistory get_full_history() const {
        softbart::SoftbartHistory h = impl_->get_history();
        h.sigma_history *= y_scale_;
        return h;
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
        // f(cfg_.x_train) at the live forest. The kernel keeps f at its own
        // X after each sweep; that X equals cfg_.x_train while the kernel's
        // X key is the one recorded when this block last set it.
        const arma::vec* kept = (impl_->x_key() == model_x_key_)
                                    ? impl_->train_fit() : nullptr;
        current_fit_ = to_y_(kept ? *kept : impl_->predict(xn_train_));
        if (keep_history_) {
            history_fit_buf_.push_back(current_fit_);
        }
    }

    // ---- preprocessing (SoftBart's normalisation) ----------------------

    // Per column, the sorted distinct values of X (non-finite values throw).
    void make_x_map_(const arma::mat& X) {
        std::vector<std::vector<double>> knots(X.n_cols);
        for (arma::uword j = 0; j < X.n_cols; ++j) {
            std::vector<double> u(X.colptr(j), X.colptr(j) + X.n_rows);
            for (double v : u) {
                if (!std::isfinite(v)) {
                    ai4b::stop("softbart_block '" + cfg_.name +
                               "': x_train must be finite");
                }
            }
            std::sort(u.begin(), u.end());
            u.erase(std::unique(u.begin(), u.end()), u.end());
            knots[j].swap(u);
        }
        x_knots_.swap(knots);
    }

    // x -> [0, 1]: the k-th smallest distinct training value (k = 0..m-1)
    // maps to k / (m - 1), as SoftBart's rank transform of the training X;
    // linear between them, 0 below the smallest and 1 above the largest;
    // 0.5 for a column with one distinct value.
    arma::mat map_x_(const arma::mat& X) const {
        if (X.n_cols != x_knots_.size()) {
            ai4b::stop("softbart_block '" + cfg_.name + "': X has " +
                       std::to_string(X.n_cols) + " columns, the training X " +
                       std::to_string(x_knots_.size()));
        }
        arma::mat out(X.n_rows, X.n_cols);
        for (arma::uword j = 0; j < X.n_cols; ++j) {
            const std::vector<double>& u = x_knots_[j];
            const std::size_t m = u.size();
            for (arma::uword i = 0; i < X.n_rows; ++i) {
                const double x = X(i, j);
                if (!std::isfinite(x)) {
                    ai4b::stop("softbart_block '" + cfg_.name +
                               "': X must be finite");
                }
                double t;
                if (m == 1) t = 0.5;
                else if (x <= u.front()) t = 0.0;
                else if (x >= u.back()) t = 1.0;
                else {
                    const std::size_t k = static_cast<std::size_t>(
                        std::lower_bound(u.begin(), u.end(), x) - u.begin());
                    t = (u[k] == x)
                        ? static_cast<double>(k) / static_cast<double>(m - 1)
                        : (static_cast<double>(k - 1) + (x - u[k - 1]) / (u[k] - u[k - 1]))
                              / static_cast<double>(m - 1);
                }
                out(i, j) = t;
            }
        }
        return out;
    }

    // Z = (Y - m) / y_scale_, m = mean(Y) (center_Y) or y_mid_.
    arma::vec to_z_(const arma::vec& y) const {
        return cfg_.center_Y ? arma::vec((y - arma::mean(y)) / y_scale_)
                             : arma::vec((y - y_mid_) / y_scale_);
    }
    // f on the scale of Y from f of Z.
    arma::vec to_y_(const arma::vec& fz) const {
        return cfg_.center_Y ? arma::vec(y_scale_ * fz)
                             : arma::vec(y_mid_ + y_scale_ * fz);
    }

    // sqrt(RSS / n) of the least-squares fit of y on an intercept and the
    // columns of X (Householder; a column whose norm after the earlier
    // reflections is below 1e-7 times its initial norm is skipped), or sd(y)
    // when n <= p + 1 or the RSS is not positive.
    static double default_sigma_hat_(const arma::mat& X, const arma::vec& y) {
        const arma::uword n = X.n_rows, k = X.n_cols + 1;
        const double sd = (n > 1) ? arma::stddev(y) : 1.0;
        if (n <= k) return sd;
        arma::mat A(n, k);
        A.col(0).ones();
        A.cols(1, k - 1) = X;
        arma::vec r = y;
        arma::uword row = 0;
        for (arma::uword j = 0; j < k && row < n; ++j) {
            const double nrm0 = arma::norm(A.col(j));
            double* c = A.colptr(j);
            double s = 0.0;
            for (arma::uword i = row; i < n; ++i) s += c[i] * c[i];
            s = std::sqrt(s);
            if (!(nrm0 > 0.0) || !(s >= 1e-7 * nrm0)) continue;
            const double a0 = c[row];
            const double v0 = a0 + (a0 >= 0.0 ? s : -s);
            const double h = s * (s + std::fabs(a0));   // v'v / 2
            c[row] = v0;                                // v = (v0, c[row+1..])
            for (arma::uword l = j + 1; l < k; ++l) {
                double* d = A.colptr(l);
                double dot = 0.0;
                for (arma::uword i = row; i < n; ++i) dot += c[i] * d[i];
                const double f = dot / h;
                for (arma::uword i = row; i < n; ++i) d[i] -= f * c[i];
            }
            double dot = 0.0;
            for (arma::uword i = row; i < n; ++i) dot += c[i] * r[i];
            const double f = dot / h;
            for (arma::uword i = row; i < n; ++i) r[i] -= f * c[i];
            ++row;
        }
        double rss = 0.0;
        for (arma::uword i = row; i < n; ++i) rss += r[i] * r[i];
        if (!(rss > 0.0) || !std::isfinite(rss)) return sd;
        return std::sqrt(rss / static_cast<double>(n));
    }

    softbart_block_config                    cfg_;
    std::unique_ptr<softbart::softbart_model> impl_;
    arma::vec                                current_fit_;
    std::vector<arma::vec>                   history_fit_buf_;
    std::uint64_t                            model_x_key_ = 0;
    std::vector<std::vector<double>>         x_knots_;    // see make_x_map_
    arma::mat                                xn_train_;   // x_train mapped to [0, 1]
    double                                   y_scale_ = 1.0;
    double                                   y_mid_   = 0.0;

    double sigma_override_ = -1.0;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_SOFTBART_BLOCK_HPP
