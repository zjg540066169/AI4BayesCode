/*================================================================================
 *  AI4BayesCode: stateful modular MCMC for composable Gibbs samplers
 *  Copyright (C) 2026 AI4BayesCode.
 *  Licensed under the GNU General Public License v2.0 or later
 *  (GPL-2.0-or-later). See COPYING / LICENSE at the repo root.
 *================================================================================
 *  stick_label_swap_block.hpp -- label-switching Metropolis-Hastings moves for
 *      a TRUNCATED stick-breaking mixture (Dirichlet process or Pitman-Yor).
 *
 *  WHY THIS BLOCK EXISTS
 *  =====================
 *  Under stick-breaking the slot weights are stochastically ordered, so a
 *  cluster's weight depends on which slot holds it. The blocked Gibbs sampler
 *  (categorical z | pi, atoms | z, sticks V | counts) relabels a cluster only
 *  by emptying it one observation at a time, so different chains keep
 *  different slot arrangements and the slot weights pi and sticks V do not
 *  mix across chains even when every label-invariant quantity has converged.
 *  These moves relabel whole clusters in one step.
 *
 *  MODEL (truncated, K = K_trunc slots, 0-based k)
 *  ----------------------------------------------
 *      V_k ~ Beta(1 - d, alpha + (k + 1) d),  k = 0, ..., K - 2,   V_{K-1} = 1
 *      pi_k = V_k prod_{l<k} (1 - V_l)
 *      z_i | pi ~ Categorical(pi),   theta_k iid,   y | z, theta
 *  d = 0 is the Dirichlet process. Labels in z are 1-based.
 *
 *  MOVES (each a Metropolis-Hastings step that leaves the joint posterior of
 *  (z, theta, V) invariant; a deterministic scan over c = 0, ..., K - 3, where
 *  both sticks V_c and V_{c+1} are free, plus the last pair for move (1))
 *  ---------------------------------------------------------------------
 *  (1) Neighbour swap (Papaspiliopoulos & Roberts 2008): exchange the labels
 *      c and c+1 in z and theta and exchange V_c and V_{c+1}. Every pi_l with
 *      l > c+1 is unchanged. Acceptance
 *          min{1, (1 - V_{c+1})^{n_c} (1 - V_c)^{-n_{c+1}}
 *                 ((1 - V_c) / (1 - V_{c+1}))^d }.
 *      The last pair (K-2, K-1) has V_{K-1} = 1 fixed, so there the two
 *      weights are exchanged instead (V_{K-2} -> 1 - V_{K-2}); the likelihood
 *      of z is unchanged and the acceptance is
 *          min{1, (v / (1 - v))^{b - a}},  v = V_{K-2},
 *          a = 1 - d,  b = alpha + (K - 1) d.
 *      Without it a cluster in the last slot could never be relabelled.
 *  (2) Weight-preserving swap (Hastie, Liverani & Richardson 2015, Stat.
 *      Comput. 25, Sec. 3.2, Eqs. 4-10; Dirichlet process only): exchange the
 *      labels c and c+1 in z and theta and re-split psi+ = pi_c + pi_{c+1}
 *      as
 *          u' = (1 - u) R1 / D,   u = pi_c / psi+,   D = (1 - u) R1 + u R2,
 *          R1 = (1 + alpha + n_{c+1} + N) / (alpha + n_{c+1} + N),
 *          R2 = (alpha + n_c + N) / (1 + alpha + n_c + N),   N = sum_{l>c+1} n_l,
 *      which is an involution. The acceptance ratio is derived here for the
 *      truncated representation from the (V_c, V_{c+1}) -> (psi+, u) change
 *      of variables, |d(V_c, V_{c+1}) / d(psi+, u)| = psi+ / (a^2 (1 - V_c)),
 *      a = prod_{l<c} (1 - V_l):
 *          R = R1^{n_{c+1}} R2^{n_c} D^{-(n_c + n_{c+1})}
 *              * (1 - V_c) / (1 - V_c')  *  R1 R2 / D^2.
 *      The last two factors are the Jacobian of the change of variables and
 *      of the u -> u' map; tests/test_stick_label_swap_exact.cpp checks the
 *      stationary distribution against exact enumeration.
 *
 *  STATE
 *  -----
 *  Reads z, every atom key, pi, stick_V and alpha from the context and writes
 *  z, the atom keys, pi and stick_V back (current_named_outputs). Its own
 *  current() is the number of accepted moves of each kind in the last step;
 *  that is also what its history records, one row per step, so the history
 *  stays aligned with the other blocks. Its position in the sweep does not
 *  affect correctness; placed last, the other blocks record the state before
 *  the sweep's relabelling, which is an equally valid draw.
 *================================================================================*/

#ifndef AI4BAYESCODE_STICK_LABEL_SWAP_BLOCK_HPP
#define AI4BAYESCODE_STICK_LABEL_SWAP_BLOCK_HPP

#include "block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"

#include <armadillo>
#include <cmath>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace AI4BayesCode {

struct stick_label_swap_block_config {
    std::string name = "label_swap";
    std::size_t K_trunc = 0;
    std::string z_key = "z";               ///< 1-based labels in {1, ..., K_trunc}
    /// Per-component parameters: (key, width). Component k occupies entries
    /// [k * width, (k + 1) * width) of the key (cluster-major layout).
    std::vector<std::pair<std::string, std::size_t>> atom_keys;
    std::string pi_key = "pi";
    std::string stick_key = "stick_V";
    std::string alpha_key = "alpha";
    double discount = 0.0;                 ///< Pitman-Yor d in [0, 1); 0 = Dirichlet process
    bool neighbour_swap = true;            ///< move (1)
    bool weight_preserving_swap = true;    ///< move (2); used only when discount == 0
};

class stick_label_swap_block : public block_sampler {
public:
    explicit stick_label_swap_block(stick_label_swap_block_config cfg)
        : cfg_(std::move(cfg)), acc_(2, arma::fill::zeros) {
        if (cfg_.name.empty())
            ai4b::stop("stick_label_swap_block: name must be non-empty");
        if (cfg_.K_trunc < 3)
            ai4b::stop("stick_label_swap_block '" + cfg_.name + "': K_trunc must be at least 3");
        if (!(cfg_.discount >= 0.0 && cfg_.discount < 1.0))
            ai4b::stop("stick_label_swap_block '" + cfg_.name + "': discount must be in [0, 1)");
        for (const auto& a : cfg_.atom_keys)
            if (a.first.empty() || a.second == 0)
                ai4b::stop("stick_label_swap_block '" + cfg_.name + "': every atom key needs a name and a positive width");
    }

    void set_context(const block_context& ctx) override { context_ = ctx; }

    void step(std::mt19937_64& rng) override {
        const std::size_t K = cfg_.K_trunc;
        arma::vec z = get_(cfg_.z_key);
        V_ = get_(cfg_.stick_key);
        if (get_(cfg_.pi_key).n_elem != K || V_.n_elem != K)
            ai4b::stop("stick_label_swap_block '" + cfg_.name + "': pi and stick_V must have length K_trunc");
        const double alpha = get_(cfg_.alpha_key)[0];
        const double d = cfg_.discount;
        if (!(alpha > -d) || !std::isfinite(alpha))
            ai4b::stop("stick_label_swap_block '" + cfg_.name + "': concentration out of range");
        atoms_.clear();
        for (const auto& a : cfg_.atom_keys) {
            arma::vec v = get_(a.first);
            if (v.n_elem != K * a.second)
                ai4b::stop("stick_label_swap_block '" + cfg_.name + "': atom key '" + a.first +
                           "' must have length K_trunc * width");
            atoms_.push_back(std::move(v));
        }
        std::vector<double> n(K, 0.0);
        for (std::size_t i = 0; i < z.n_elem; ++i) {
            const long k = std::lround(z[i]) - 1;
            if (k < 0 || k >= static_cast<long>(K))
                ai4b::stop("stick_label_swap_block '" + cfg_.name + "': z must hold labels 1..K_trunc");
            n[static_cast<std::size_t>(k)] += 1.0;
        }
        // slot_of[j]: slot now holding the cluster that started in slot j;
        // owner[s]: the starting slot of the cluster now in slot s.
        std::vector<std::size_t> slot_of(K), owner(K);
        for (std::size_t s = 0; s < K; ++s) slot_of[s] = owner[s] = s;
        std::uniform_real_distribution<double> unif(0.0, 1.0);
        auto swap_slots = [&](std::size_t c) {
            std::swap(n[c], n[c + 1]);
            std::swap(owner[c], owner[c + 1]);
            slot_of[owner[c]] = c;
            slot_of[owner[c + 1]] = c + 1;
            for (std::size_t m = 0; m < atoms_.size(); ++m) {
                const std::size_t w = cfg_.atom_keys[m].second;
                for (std::size_t j = 0; j < w; ++j) std::swap(atoms_[m][c * w + j], atoms_[m][(c + 1) * w + j]);
            }
        };
        acc_.zeros();
        if (cfg_.neighbour_swap) {
            for (std::size_t c = 0; c + 2 < K; ++c) {
                const double l1 = std::log1p(-V_[c]), l2 = std::log1p(-V_[c + 1]);
                const double logR = n[c] * l2 - n[c + 1] * l1 + d * (l1 - l2);
                if (std::log(unif(rng)) < logR) {
                    swap_slots(c);
                    std::swap(V_[c], V_[c + 1]);
                    acc_[0] += 1.0;
                }
            }
            // Last pair (K-2, K-1): V_{K-1} = 1 is fixed, so exchange the two
            // weights instead, V_{K-2} -> 1 - V_{K-2}. The likelihood of z is
            // unchanged and the Jacobian is 1; only the stick prior changes.
            {
                const std::size_t c = K - 2;
                const double v = V_[c];
                const double a_k = 1.0 - d, b_k = alpha + static_cast<double>(K - 1) * d;
                if (v > 0.0 && v < 1.0) {
                    const double logR = (b_k - a_k) * (std::log(v) - std::log1p(-v));
                    if (std::log(unif(rng)) < logR) {
                        swap_slots(c);
                        V_[c] = 1.0 - v;
                        acc_[0] += 1.0;
                    }
                }
            }
        }
        if (cfg_.weight_preserving_swap && d == 0.0) {
            for (std::size_t c = 0; c + 2 < K; ++c) {
                double log_a = 0.0;
                for (std::size_t l = 0; l < c; ++l) log_a += std::log1p(-V_[l]);
                const double a = std::exp(log_a);
                const double pc = a * V_[c], pc1 = a * (1.0 - V_[c]) * V_[c + 1], psi = pc + pc1;
                if (!(a > 0.0) || !(psi > 0.0)) continue;
                double N = 0.0;
                for (std::size_t l = c + 2; l < K; ++l) N += n[l];
                const double R1 = (1.0 + alpha + n[c + 1] + N) / (alpha + n[c + 1] + N);
                const double R2 = (alpha + n[c] + N) / (1.0 + alpha + n[c] + N);
                const double u = pc / psi, D = (1.0 - u) * R1 + u * R2, u_new = (1.0 - u) * R1 / D;
                const double Vc_new = psi * u_new / a;
                const double Vc1_new = psi * (1.0 - u_new) / (a - psi * u_new);
                if (!(Vc_new > 0.0 && Vc_new < 1.0 && Vc1_new > 0.0 && Vc1_new < 1.0)) continue;
                const double logR = n[c + 1] * std::log(R1) + n[c] * std::log(R2) - (n[c] + n[c + 1]) * std::log(D)
                                  + std::log1p(-V_[c]) - std::log1p(-Vc_new)
                                  + std::log(R1) + std::log(R2) - 2.0 * std::log(D);
                if (std::log(unif(rng)) < logR) {
                    swap_slots(c);
                    V_[c] = Vc_new;
                    V_[c + 1] = Vc1_new;
                    acc_[1] += 1.0;
                }
            }
        }
        // Relabel z and rebuild pi from the sticks.
        for (std::size_t i = 0; i < z.n_elem; ++i)
            z[i] = static_cast<double>(slot_of[static_cast<std::size_t>(std::lround(z[i]) - 1)] + 1);
        z_ = std::move(z);
        pi_.set_size(K);
        double rest = 1.0;
        for (std::size_t k = 0; k + 1 < K; ++k) { pi_[k] = V_[k] * rest; rest *= (1.0 - V_[k]); }
        pi_[K - 1] = rest;
        V_[K - 1] = 1.0;
        have_state_ = true;
        if (keep_history_) hist_.push_back(acc_);
    }

    const arma::vec& current() const override { return acc_; }
    void set_current(const arma::vec& theta) override {
        if (theta.n_elem != 2) ai4b::stop("stick_label_swap_block::set_current: length must be 2");
        acc_ = theta;
    }
    std::size_t dim() const noexcept override { return 2; }
    const std::string& name() const noexcept override { return cfg_.name; }

    state_map current_named_outputs() const override {
        state_map out;
        out.emplace(cfg_.name, acc_);
        if (!have_state_) return out;
        out.emplace(cfg_.z_key, z_);
        for (std::size_t m = 0; m < atoms_.size(); ++m) out.emplace(cfg_.atom_keys[m].first, atoms_[m]);
        out.emplace(cfg_.pi_key, pi_);
        out.emplace(cfg_.stick_key, V_);
        return out;
    }

    void record_held_history() override { if (keep_history_) hist_.push_back(acc_); }
    history_map get_history() const override {
        history_map out;
        if (!keep_history_ || hist_.empty()) return block_sampler::get_history();
        arma::mat m(hist_.size(), 2);
        for (std::size_t r = 0; r < hist_.size(); ++r) m.row(r) = hist_[r].t();
        out.emplace(cfg_.name, std::move(m));
        return out;
    }
    std::size_t history_size() const noexcept override { return keep_history_ ? hist_.size() : 1; }
    void clear_history() override { hist_.clear(); }

private:
    const arma::vec& get_(const std::string& key) const {
        auto it = context_.find(key);
        if (it == context_.end())
            ai4b::stop("stick_label_swap_block '" + cfg_.name + "': key '" + key +
                       "' not in context; declare it via declare_dependencies(\"" + cfg_.name + "\", {...}).");
        return it->second;
    }

    stick_label_swap_block_config cfg_;
    block_context context_;
    arma::vec acc_, z_, pi_, V_;
    std::vector<arma::vec> atoms_;
    std::vector<arma::vec> hist_;
    bool have_state_ = false;
};

}  // namespace AI4BayesCode

#endif  // AI4BAYESCODE_STICK_LABEL_SWAP_BLOCK_HPP
