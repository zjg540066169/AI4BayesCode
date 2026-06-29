/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later) -- transitively required because this header pulls
 *  in the vendored CRAN BART R package kernel under AI4BayesCode/bart/.
 *  See LICENSE / THIRD_PARTY_LICENSES.md and the LICENSE block below
 *  for details.
 *================================================================================
 *
 *  AI4BayesCode/bart_block.hpp  --  block_sampler wrapper around the vendored
 *                                   stdbart::bart_model in AI4BayesCode/bart/.
 *================================================================================
 *
 *  BACKEND-NEUTRAL (Rcpp-free)
 *  ===========================
 *  This header sits on the pure-C++ / Armadillo BART kernel
 *  (bart_pure_cpp/src/bart_model.h) and names NO Rcpp:: or R:: symbols.
 *  Error/warning reporting goes through ai4b::stop / ai4b::warning
 *  (backend_neutral.hpp), so it compiles under BOTH the Rcpp (R) backend
 *  and the pybind11 / standalone backend. All data containers are
 *  arma::mat / arma::vec; tree snapshots are std::string.
 *
 *  ARCHITECTURE
 *  ------------
 *  bart_block owns an internal `stdbart::bart_model` instance. One call
 *  to block_sampler::step() corresponds to EXACTLY ONE tree-ensemble
 *  sweep (one invocation of `bart_model::update_step()` if sigma is
 *  internal, or `bart_model::update_step(sigma_ext)` if the composite
 *  supplies a sigma via the configured sigma_key).
 *
 *  HISTORY RECORDING
 *  -----------------
 *  The constructor unconditionally calls `impl_->set_history(true)`, so
 *  every successful step() appends the current forest + sigma + var
 *  probs/counts to the internal bart_model history buffers. This is
 *  required by predict_history() so that wrapper-level predict_at can
 *  walk every stored draw of the BART forest. clear_history() forwards
 *  to both the internal model buffers and the external fit buffer.
 *
 *  TREE SERIALIZATION
 *  ------------------
 *  get_tree() / set_tree(s) wrap the internal bart_model's serialize /
 *  deserialize API. The wrapper exposes these so that the Tier-A
 *  Rcpp::List dispatcher (set_current(list(tree = s)) etc.) can
 *  round-trip the forest state between R and C++ without touching
 *  internal pointers. set_tree() rebuilds the live forest in-place and
 *  refreshes the cached fit at x_train.
 *
 *  CACHED FIT (current_fit_) -- BOUNDARY DISCIPLINE
 *  -------------------------------------------------
 *  block_sampler::current() must return a consistent value at any time
 *  with no intervening step(). Because the BART block's "state" is a
 *  tree forest (not a vec), current_fit_ caches the forest's predictions
 *  at the stored x_train so that current() is O(N) (a single vec copy).
 *
 *  The cache is REFRESHED IN-MEMORY ONLY -- never via serialize +
 *  deserialize. The kernel (stdbart::bart_model) maintains a live
 *  cache invariant: bm.allfit[i] = sum_t tree_t(stored_x_train_i),
 *  refreshed inside every state-changing routine:
 *
 *    Trigger                Kernel refresh path                   Cost
 *    ---------------------  ------------------------------------  -----
 *    step()                 heterbart::draw incremental update    O(1)
 *    set_X / set_data       bm.setdata() -> predict(allfit)       O(N*T*d)
 *    set_Y                  trees(x) unchanged, fmean updated     O(1)
 *    set_tree(s)            bm.refresh_allfit() in-memory walk    O(N*T*d)
 *
 *  bart_block then reads the cache via impl_->current_fit_train(),
 *  which returns bm.f(i) + fmean -- an O(N) memcpy.  NO get_tree() ->
 *  string -> parse -> re-evaluate round-trip on the hot path.
 *
 *  Boundary discipline (also followed by genbart_block and
 *  softbart_block): serialize / deserialize happen only at the
 *  user-facing input/output boundary -- get_tree() (output snapshot
 *  for the user) and set_tree(s) (input from a saved snapshot, where
 *  the live forest must be rebuilt). The internal refresh path never
 *  serializes, even when set_tree mutates the live forest.
 *
 *  This pattern mirrors:
 *    genbart_model::current_f_train()       <-> bart_model::current_fit_train()
 *    genbart_block::refresh_current_r_      <-> bart_block::refresh_current_fit_
 *    softbart_model::predict(x_train)       <-> bart_model::current_fit_train()
 *
 *  Historical note: the previous bart_block::refresh_current_fit_
 *  walked the cycle
 *      current_fit_ = predict_at_serialized_(impl_->get_tree(),
 *                                            cfg_.x_train);
 *  i.e. serialize live forest to string, parse it back into a
 *  temporary vector<tree>, evaluate on x_train, then throw it away.
 *  Measured ~50x slower than the in-memory accessor on BartNoise
 *  (N=500, ntree=200). The function predict_at_serialized_ is now
 *  reserved strictly for the user-facing predict_at(new_X) path
 *  with an externally-supplied serialized snapshot.
 *
 *  WORKING-RESPONSE SEMANTICS
 *  --------------------------
 *  In a composite model like
 *      y_i = f_bart(x_i) + B_i' * beta + epsilon_i
 *  the BART block's full conditional sees the partial residual
 *      r = y - B * beta
 *  as its working response, not y itself. block_context gives us a
 *  clean mechanism for this: the composite refreshes a derived key
 *  (typically "bart_target" or "residual") whenever any non-BART
 *  block updates, and bart_block reads that key on every set_context
 *  call.
 *
 *  CENTERING
 *  ---------
 *  bart_model::set_Y has an internal centering step (subtract fmean
 *  before feeding heterbart). bart_block ALWAYS calls it with
 *  center=true to match BART::wbart behaviour. fmean is refreshed
 *  whenever the working response or y changes; the cached fmean
 *  used by predict_at_serialized_ is updated alongside.
 *
 *  DART
 *  ----
 *  The new stdbart::bart_model constructor hard-codes dart=false and
 *  does not expose a public setter for it. If cfg_.dart is true, the
 *  wrapper emits a one-time warning recommending genbart_block for
 *  DART support (genbart_block flows DART activation through its
 *  rjmcmc_hypers + startdart()).
 *
 *  RNG
 *  ---
 *  The pure-C++ BART kernel draws from its own seedable std::mt19937_64
 *  stream (bart_rng in bart_pure_cpp/src/BART/r_compat.h), driven through
 *  the `arn` adapter. The `std::mt19937_64&` argument passed to
 *  bart_block::step() is IGNORED. To reproduce a run, seed the kernel
 *  stream (bart_rng::set_seed(seed)) before the run starts.
 *================================================================================*/

#ifndef AI4BAYESCODE_BART_BLOCK_HPP
#define AI4BAYESCODE_BART_BLOCK_HPP

#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif

// Bring in the unified pure-C++ BART kernel. `stdbart::bart_model` lives in
// bart_pure_cpp/src/. The kernel-unity header pulls in all per-translation-
// unit helpers needed when a downstream wrapper .cpp does not separately
// compile the sibling .cpp files under bart_pure_cpp/src/BART/.
#include "../../bart_pure_cpp/src/bart_model.h"
#include "../../bart_pure_cpp/src/bart_kernel_unity.h"

#include "block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"   // ai4b::stop / ai4b::warning

#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

// ---------------------------------------------------------------------------
// Configuration bundle for bart_block.
// ---------------------------------------------------------------------------

struct bart_block_config {
    /// Unique name for this block within its composite; also the key
    /// under which shared_data_t stores this block's current fitted
    /// value (the length-n vector f(x_i) + fmean on the natural
    /// scale of the working response).
    std::string name = "f_bart";

    /// Design matrix X on the user-facing scale. Rows are observations,
    /// columns are predictors.
    arma::mat x_train;

    /// Initial working response. Used by the constructor to build the
    /// internal bart_model. The composite will typically overwrite
    /// this on every subsequent set_context.
    arma::vec y_init;

    /// shared_data key the composite refreshes each sweep to give
    /// bart_block its working response. Default "bart_target".
    std::string working_response_key = "bart_target";

    /// shared_data key for the current sigma. If non-empty and present
    /// in the context, bart_block forwards the scalar to
    /// bart_model::update_step(sigma) rather than letting bart_model
    /// run its own internal Inv-Chi^2 step.
    std::string sigma_key = "sigma";

    /// shared_data key for a PER-OBSERVATION weight vector w (length n).
    /// EMPTY (default) => unweighted: behaviour is exactly as before
    /// (scalar update_step / update_step(sigma)). When non-empty AND the
    /// key is present in the context, bart_block performs a WEIGHTED
    /// (heteroscedastic) Gibbs sweep: it forwards a per-observation noise
    /// sd vector sigma_i = w_i * sigma to
    /// bart_model::update_step_weighted (the same convention as the
    /// kernel's batch "Weighted Update"). Weighted mode REQUIRES a
    /// scalar sigma (sigma_key set and supplied) — the per-obs scale is
    /// w_i * sigma. Typical use: known per-study/observation variances
    /// (w_i = 1/sqrt(v_i)) or a heteroscedastic outer model.
    std::string weights_key = "";

    /// BART hyperparameters, forwarded to stdbart::bart_model's
    /// constructor. Defaults mirror the CRAN BART R package.
    int    ntrees   = 200;
    long   numcut   = 100;
    bool   usequants = false;
    bool   cont     = false;
    bool   rm_const = false;
    double k        = 2.0;
    double power    = 2.0;
    double base     = 0.95;
    double nu       = 3.0;
    /// Reserved for future use; the new bart_model constructor ignores
    /// it (it derives sigest from OLS residuals when p<n, else from
    /// sd(y_init)).
    double sigma_init = 1.0;

    /// Request DART (Linero 2018). The new stdbart::bart_model hardcodes
    /// dart=false with no public activator; the wrapper logs a warning
    /// if this flag is set and recommends genbart_block instead.
    bool dart = false;

    /// Augmented DART (Linero 2018 §4). No-op under the new API; see
    /// `dart` comment.
    bool aug = false;

    /// Binary leaf-prior tau formula: 3/(k*sqrt(ntrees)) instead of
    /// (max(y)-min(y))/(2*k*sqrt(ntrees)). Set to TRUE for probit BART
    /// where the working response is a latent z (range ~6).
    bool binary = false;
};

// ---------------------------------------------------------------------------
// bart_block: a block_sampler that runs one BART tree-ensemble sweep per
// step() invocation, reading its working response and (optionally) sigma
// from the composite's block_context each time.
// ---------------------------------------------------------------------------

class bart_block : public block_sampler {
public:
    explicit bart_block(bart_block_config cfg)
        : cfg_(std::move(cfg))
    {
        if (cfg_.x_train.n_rows == 0 || cfg_.x_train.n_cols == 0) {
            ai4b::stop("bart_block: x_train must have at least one row "
                       "and one column");
        }
        if (cfg_.y_init.n_elem != cfg_.x_train.n_rows) {
            ai4b::stop("bart_block: y_init length must match x_train row "
                       "count");
        }

        impl_ = std::make_unique<stdbart::bart_model>(
            cfg_.x_train, cfg_.y_init,
            cfg_.numcut, cfg_.usequants, cfg_.cont, cfg_.rm_const,
            cfg_.ntrees,
            /*sigmaf=*/ std::numeric_limits<double>::quiet_NaN(),
            cfg_.k, cfg_.power, cfg_.base, cfg_.nu,
            /*binary=*/ cfg_.binary);

        if (cfg_.dart) {
            ai4b::warning(
                "bart_block '%s': DART requested but the new "
                "stdbart::bart_model has no public DART activator. "
                "Use genbart_block for DART support.",
                cfg_.name.c_str());
        }

        // Kernel history recording is OFF by default; turned ON only when
        // the wrapper opts into history via set_keep_history(true).
        // Previously this was UNCONDITIONALLY ON, which made every
        // update_step() call record_history_() — serializing the entire
        // forest into a history string EVERY step, even when the user
        // didn't ask for history. Measured to be ~30x of CRAN BART::pbart
        // wall-clock on probit_BART_model when keep_history=false.
        impl_->set_history(false);

        // Cache static info (xinfo, p, ntrees, fmean) once at
        // construction; refreshed on set_Y / set_data because fmean
        // depends on y.
        cache_static_info_();

        // Initial fit is fmean for every row (empty stumps at
        // construction; bm.allfit[] was initialized by bm.setdata() in
        // the kernel ctor via predict(p, n, x, allfit)). Read it via
        // the same in-memory accessor used by refresh_current_fit_()
        // — never round-trip through serialize on the internal path.
        current_fit_ = impl_->current_fit_train();
    }

    // ---- block_sampler interface --------------------------------------

    void set_context(const block_context& ctx) override {
        // 1. Working response.
        auto it_y = ctx.find(cfg_.working_response_key);
        if (it_y == ctx.end()) {
            ai4b::stop("bart_block '" + cfg_.name
                + "': context is missing working-response key '"
                + cfg_.working_response_key + "'");
        }
        const arma::vec& arma_y = it_y->second;
        if (arma_y.n_elem != cfg_.x_train.n_rows) {
            ai4b::stop("bart_block '" + cfg_.name
                + "': working response length "
                + std::to_string(arma_y.n_elem)
                + " does not match x_train row count "
                + std::to_string(cfg_.x_train.n_rows));
        }
        impl_->set_Y(arma_y, /*center_Y=*/true);
        // fmean may have changed inside set_Y; refresh cached value.
        refresh_cached_fmean_();

        // 2. Sigma override from context (sentinel <= 0 means "use
        // internal sigma, drawn each step via Inv-Chi^2").
        sigma_override_ = -1.0;
        if (!cfg_.sigma_key.empty()) {
            auto it_s = ctx.find(cfg_.sigma_key);
            if (it_s != ctx.end()) {
                if (it_s->second.n_elem != 1) {
                    ai4b::stop("bart_block '" + cfg_.name
                        + "': sigma context must be a length-1 vector");
                }
                const double s = it_s->second[0];
                if (!(s > 0.0) || !std::isfinite(s)) {
                    ai4b::stop("bart_block '" + cfg_.name
                        + "': sigma must be strictly positive and finite");
                }
                sigma_override_ = s;
            }
        }

        // 3. Per-observation weights (weighted / heteroscedastic BART).
        //    EMPTY weights_key => stay unweighted (legacy behaviour).
        weights_.clear();
        if (!cfg_.weights_key.empty()) {
            auto it_w = ctx.find(cfg_.weights_key);
            if (it_w != ctx.end()) {
                const arma::vec& w = it_w->second;
                if (w.n_elem != cfg_.x_train.n_rows) {
                    ai4b::stop("bart_block '" + cfg_.name
                        + "': weights length "
                        + std::to_string(w.n_elem)
                        + " does not match x_train row count "
                        + std::to_string(cfg_.x_train.n_rows));
                }
                weights_.resize(w.n_elem);
                for (std::size_t i = 0; i < w.n_elem; ++i) {
                    if (!(w[i] > 0.0) || !std::isfinite(w[i])) {
                        ai4b::stop("bart_block '" + cfg_.name
                            + "': weights must be strictly positive and "
                              "finite");
                    }
                    weights_[i] = w[i];
                }
            }
        }
    }

    void step(std::mt19937_64& /*rng*/) override {
        // BART uses R's RNG via arn. mt19937 is ignored.
        if (!weights_.empty()) {
            // Weighted (heteroscedastic) sweep: per-observation sd
            // sigma_i = w_i * sigma. Requires an external scalar sigma
            // (same convention as the kernel's batch Weighted Update).
            if (!(sigma_override_ > 0.0)) {
                ai4b::stop("bart_block '" + cfg_.name
                    + "': weighted mode (weights_key set) requires a "
                      "scalar sigma via sigma_key — per-obs sd is "
                      "w_i * sigma.");
            }
            std::vector<double> svec(weights_.size());
            for (std::size_t i = 0; i < weights_.size(); ++i)
                svec[i] = weights_[i] * sigma_override_;
            impl_->update_step_weighted(svec);
        } else if (sigma_override_ > 0.0) {
            impl_->update_step(sigma_override_);
        } else {
            impl_->update_step();
        }
        // update_step() records into bart_model's internal history
        // (set_history(true) in ctor). Refresh our cached fit on
        // x_train + append to history_fit_buf_ if keep_history_.
        refresh_current_fit_();
    }

    const arma::vec& current() const override { return current_fit_; }

    void set_current(const arma::vec& /*theta*/) override {
        ai4b::stop("bart_block '" + cfg_.name
            + "': set_current(arma::vec) cannot overwrite the tree "
            "forest (no unique inverse from fitted values to forest "
            "state). To restore the forest from a serialized snapshot, "
            "use set_tree(s). To update data the BART block sees, use "
            "set_X(X_new), set_Y(y_new), or set_data(X_new, y_new) "
            "instead -- these rebind the kernel's data inputs and "
            "leave the live trees untouched.");
    }

    const std::string& name() const noexcept override { return cfg_.name; }

    std::size_t dim() const noexcept override {
        return static_cast<std::size_t>(cfg_.x_train.n_rows);
    }

    // ---- Data-input setters (for nested MCMC: imputed X, working
    //      response y, etc.) -----------------------------------------------

    /// y-only update. Equivalent to pushing y into working_response_key
    /// in shared_data and letting the next set_context() pick it up.
    /// Forces center=true to match BART::wbart conventions and to
    /// refresh fmean for predict_at_serialized_.
    void set_Y(const arma::vec& y_new) {
        const long n_cur = static_cast<long>(cfg_.x_train.n_rows);
        if (static_cast<long>(y_new.n_elem) != n_cur) {
            ai4b::stop("bart_block '" + cfg_.name
                + "': set_Y length "
                + std::to_string(y_new.n_elem)
                + " does not match current n = "
                + std::to_string(n_cur));
        }
        impl_->set_Y(y_new, /*center_Y=*/true);
        cfg_.y_init = y_new;
        refresh_cached_fmean_();
        refresh_current_fit_();
    }

    /// X-only update. Row count and col count must match the original
    /// x_train dimensions.
    void set_X(const arma::mat& X_new) {
        const arma::uword n_cur = cfg_.x_train.n_rows;
        const arma::uword p_cur = cfg_.x_train.n_cols;
        if (X_new.n_rows != n_cur) {
            ai4b::stop("bart_block '" + cfg_.name
                + "': set_X row count "
                + std::to_string(X_new.n_rows)
                + " does not match current n = "
                + std::to_string(n_cur));
        }
        if (X_new.n_cols != p_cur) {
            ai4b::stop("bart_block '" + cfg_.name
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
            ai4b::stop("bart_block '" + cfg_.name
                + "': set_data rows (X) / length (y) mismatch: "
                + std::to_string(X_new.n_rows) + " vs "
                + std::to_string(y_new.n_elem));
        }
        impl_->set_data(X_new, y_new, /*center_Y=*/true);
        cfg_.x_train = X_new;
        cfg_.y_init  = y_new;
        refresh_cached_fmean_();
        refresh_current_fit_();
    }

    // ---- Tree serialization (NEW canonical API) ----------------------

    /// Serialize the current forest to a string.
    /// Format: "T\n<tree1>---\n<tree2>---\n...". The returned string
    /// can be passed to set_tree(s) to round-trip the forest, or saved
    /// to disk for later restoration.
    std::string get_tree() const { return impl_->get_tree(); }

    /// Replace the current forest with one parsed from `s`. Performs
    /// strict structural validation (see stdbart::bart_model::set_tree
    /// for the full list of 14 checks). On success, refreshes the
    /// cached fit at x_train.
    void set_tree(const std::string& s) {
        impl_->set_tree(s);
        // set_tree mutates the forest but does not append to
        // tree_history_. Refresh current_fit_ so downstream blocks see
        // the new state on the next set_context cycle.
        refresh_current_fit_();
    }

    // ---- bart_block-specific accessors --------------------------------

    /// Current sigma inside the internal bart_model.
    double current_sigma() const { return impl_->current_sigma(); }

    /// Current split-variable probabilities (length p).
    arma::vec current_var_probs() const {
        return const_cast<stdbart::bart_model*>(impl_.get())->current_var_probs();
    }
    /// Current split-variable usage counts across all trees (length p).
    arma::uvec current_var_counts() const {
        return const_cast<stdbart::bart_model*>(impl_.get())->current_var_counts();
    }

    /// Raw pointer to the internal bart_model for advanced operations.
    stdbart::bart_model* underlying() { return impl_.get(); }
    const stdbart::bart_model* underlying() const { return impl_.get(); }

    /// Predict f(X_new) at the CURRENT forest. Returns length-Nt vector
    /// on the natural Y scale (cached fmean is added back).
    arma::vec predict(const arma::mat& X_new) {
        return predict_at_serialized_(impl_->get_tree(), X_new);
    }

    /// Predict at X_new for ALL historical draws stored in the internal
    /// model. Returns (n_draws x Nt) matrix on the natural Y scale.
    /// If no history has been recorded yet, returns a 1xNt matrix from
    /// the current forest.
    arma::mat predict_history(const arma::mat& X_new) {
        stdbart::BartHistory h = impl_->get_history();
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
            arma::vec v = predict_at_serialized_(h.tree_history[d], X_new);
            for (arma::uword k = 0; k < Nt; ++k) result(d, k) = v[k];
        }
        return result;
    }

    // ---- History overrides -----------------------------------------------

    history_map get_history() const override {
        return detail::make_history_map(
            cfg_.name, history_fit_buf_, current_fit_);
    }

    /// Accessor for the full internal BART history (tree strings, sigma,
    /// var probs, var counts, xinfo, ntrees, p, fmean) as a backend-
    /// neutral struct. Useful for downstream posterior predictive
    /// analyses outside the wrapper.
    stdbart::BartHistory get_full_history() const {
        return impl_->get_history();
    }

    /// Tree-string history; per-draw serialized snapshots of the forest.
    /// Returns one entry per recorded draw, or one entry (current
    /// forest) if no history has been recorded.
    std::vector<std::string> get_tree_history() const {
        stdbart::BartHistory h = impl_->get_history();
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

    // keep_history controls ONLY the numeric per-step buffer
    // (history_fit_buf_).  Cheap O(N) per step.
    void set_keep_history(bool keep) override {
        block_sampler::set_keep_history(keep);
    }

    // keep_tree controls the kernel's per-step forest serialization
    // (record_history_ -> tree_history_).  EXPENSIVE
    // O(T * tree_size) per step; needed only when the wrapper plans to
    // call predict_history(X_new) for posterior prediction at new X.
    // Default OFF; turning ON typically dominates wall-clock and brings
    // BartNoise's per-step cost up to ~30x of CRAN BART::pbart, hence
    // the strong default of OFF.
    void set_keep_tree(bool keep) override {
        block_sampler::set_keep_tree(keep);
        impl_->set_history(keep);  // kernel "history" is actually tree-history
    }

private:
    void cache_static_info_() {
        stdbart::BartHistory h = impl_->get_history();
        cached_p_ = static_cast<int>(h.xinfo.size());
        cached_xinfo_.resize(cached_p_);
        for (int j = 0; j < cached_p_; ++j) {
            cached_xinfo_[j] = h.xinfo[j];
        }
        cached_ntrees_ = static_cast<int>(h.ntrees);
        cached_fmean_  = h.fmean;
    }

    void refresh_cached_fmean_() {
        stdbart::BartHistory h = impl_->get_history();
        cached_fmean_ = h.fmean;
    }

    // Mirrors genbart_block::refresh_current_r_ — read the kernel's
    // cached live fit at x_train via current_fit_train() (O(N)).  The
    // kernel maintains the cache invariant: setdata() refreshes allfit
    // after set_X / set_data, heterbart::draw maintains it incrementally
    // during step, bm.refresh_allfit() is called inside set_tree, and
    // set_Y doesn't touch trees(x).  NO serialize/deserialize round-trip
    // (previous predict_at_serialized_(get_tree(), x_train) path was ~50x
    // slower as measured on BartNoise).
    void refresh_current_fit_() {
        current_fit_ = impl_->current_fit_train();
        if (keep_history_) {
            history_fit_buf_.push_back(current_fit_);
        }
    }

    /// Evaluate trees serialized in `s` at rows of X_new, add fmean,
    /// return length-Nt vector. Mirrors bart_predict_posterior's
    /// per-iter inner loop using cached_xinfo_ for cutpoints.
    arma::vec predict_at_serialized_(const std::string& s,
                                     const arma::mat& X_new)
    {
        if (s.empty()) {
            ai4b::stop("bart_block: predict input is an empty serialized "
                       "forest string");
        }
        std::istringstream is(s);
        std::size_t T = 0;
        if (!(is >> T)) {
            ai4b::stop("bart_block: failed to parse tree count in "
                       "serialized forest");
        }
        std::vector<tree> trees(T);
        for (std::size_t t = 0; t < T; ++t) {
            is >> trees[t];
            std::string sep;
            is >> sep;  // "---"
        }
        const std::size_t Nt = static_cast<std::size_t>(X_new.n_rows);
        const std::size_t p_local = static_cast<std::size_t>(X_new.n_cols);
        if (p_local != static_cast<std::size_t>(cached_p_)) {
            ai4b::stop("bart_block::predict: X_new has %d columns, "
                       "expected %d (model was built with p = %d).",
                       (int)p_local, cached_p_, cached_p_);
        }
        arma::vec out(Nt);
        std::vector<double> x_row(p_local);
        for (std::size_t row = 0; row < Nt; ++row) {
            for (std::size_t j = 0; j < p_local; ++j) x_row[j] = X_new(row, j);
            double f = 0.0;
            for (auto& tr : trees) {
                tree* node = &tr;
                while (node->getl() != 0) {
                    const std::size_t v = node->getv();
                    const std::size_t c = node->getc();
                    node = (x_row[v] < cached_xinfo_[v][c])
                             ? node->getl() : node->getr();
                }
                f += node->gettheta();
            }
            out[row] = f + cached_fmean_;
        }
        return out;
    }

    bart_block_config                    cfg_;
    std::unique_ptr<stdbart::bart_model> impl_;
    arma::vec                            current_fit_;
    std::vector<arma::vec>               history_fit_buf_;

    // Cached static info; fmean is refreshed on set_Y / set_data.
    int    cached_p_      = 0;
    int    cached_ntrees_ = 0;
    double cached_fmean_  = 0.0;
    std::vector<std::vector<double> > cached_xinfo_;

    // Sigma override sourced from set_context (negative = "use internal
    // sigma drawn each step via Inv-Chi^2").
    double sigma_override_ = -1.0;

    // Per-observation weights sourced from set_context when
    // cfg_.weights_key is non-empty (empty otherwise => unweighted,
    // identical legacy behaviour). Weighted mode forwards
    // sigma_i = w_i * sigma_override_ to update_step_weighted.
    std::vector<double> weights_;
};

} // namespace AI4BayesCode

#endif // AI4BAYESCODE_BART_BLOCK_HPP
