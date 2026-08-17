/*
 * test_vi_publish_and_history.cpp
 *
 * Three invariants a VI block has to hold that no other test covered.
 *
 *  A. structured_categorical_vi_block publishes STATES, not phi.
 *     phi_ is clique-indexed (one joint state per clique); the per-node
 *     argmax has to come from per_node_marginals(). Reading phi_ with the
 *     mean-field stride (cardinalities[i]) silently returns another
 *     clique's joint probabilities as if they were node i's marginal --
 *     and that value is what every sibling block reads before the first
 *     step and after a composite set_current() restore.
 *
 *  B. clear_history() clears the q-draw buffer too. It is documented as
 *     the drop-burn-in operation, and the generated set_current N-change
 *     path calls it; leaving the per-sweep draws behind means the bare-name
 *     series keeps every pre-clear draw and no longer lines up with the
 *     sibling blocks it is meant to be read alongside.
 *
 *  C. A FROZEN VI child still advances its history by one row per outer
 *     sweep (block_sampler::record_held_history). Without it the VI child
 *     alone stalls while its siblings -- including siblings inside the same
 *     frozen composite -- keep going, and predict_at() then trips
 *     "inconsistent history sizes".
 */

#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/structured_categorical_vi_block.hpp"
#include "AI4BayesCode/mean_field_gaussian_vi_block.hpp"

#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

using AI4BayesCode::block_context;
using AI4BayesCode::composite_block;
using AI4BayesCode::mean_field_gaussian_vi_block;
using AI4BayesCode::mean_field_gaussian_vi_block_config;
using AI4BayesCode::structured_categorical_vi_block;
using AI4BayesCode::structured_categorical_vi_block_config;

namespace {

int checks = 0, failures = 0;

void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL  %s\n", what); }
    else       std::printf("  ok    %s\n", what);
}

// ---------------------------------------------------------------------------
// A. structured VI publishes per-node states
// ---------------------------------------------------------------------------
// n = 4 latents, K = {2, 2, 3, 3}, cliques {{0,1}, {2,3}}. phi_ therefore has
// 2*2 + 3*3 = 13 entries while the mean-field stride would read 2+2+3+3 = 10:
// any node past the first clique is read out of the wrong block of phi_.
void test_structured_publishes_states() {
    std::printf("A. structured_categorical_vi_block publishes states\n");

    structured_categorical_vi_block_config cfg;
    cfg.name              = "z";
    cfg.cardinalities     = arma::uvec{2, 2, 3, 3};
    cfg.clique_partition  = {{0, 1}, {2, 3}};
    cfg.exact_enumeration = true;
    // A density with one clearly-preferred state per node, and the preference
    // DIFFERENT per node so a mis-strided read cannot land on it by luck:
    // z = (1, 1, 2, 1).
    const std::vector<std::size_t> want = {1, 1, 2, 1};
    cfg.log_density = [want](const arma::uvec& z, const block_context&) {
        double lp = 0.0;
        for (std::size_t i = 0; i < z.n_elem; ++i)
            lp += (z[i] == want[i]) ? 0.0 : -4.0;
        return lp;
    };

    auto comp = std::make_unique<composite_block>();
    structured_categorical_vi_block* vi = nullptr;
    { auto b = std::make_unique<structured_categorical_vi_block>(cfg);
      vi = b.get(); comp->add_child(std::move(b)); }
    comp->set_keep_history(true);

    std::mt19937_64 rng(20260816);
    for (int i = 0; i < 200; ++i) comp->step(rng);

    // The deterministic publish must equal the per-node marginal argmax.
    const arma::mat marg = vi->per_node_marginals();
    const auto published = vi->current_named_outputs();
    check(published.count("z") == 1, "publishes under the block's own name");
    const arma::vec& z = published.at("z");
    check(z.n_elem == 4, "published vector has one entry per NODE (not per phi entry)");

    bool all_match = true, all_want = true;
    for (std::size_t i = 0; i < 4; ++i) {
        const std::size_t Ki = cfg.cardinalities[i];
        const auto arg = marg.row(i).cols(0, Ki - 1).index_max();
        if (static_cast<std::size_t>(z[i]) != arg)       all_match = false;
        if (static_cast<std::size_t>(z[i]) != want[i])   all_want  = false;
        // Every published value must be a legal state for THAT node.
        check(z[i] >= 0.0 && z[i] < static_cast<double>(Ki),
              "published z_i is in range for its own cardinality");
    }
    check(all_match, "published z == per_node_marginals() argmax");
    check(all_want,  "published z == the state the density prefers");
}

// ---------------------------------------------------------------------------
// B + C. history bookkeeping
// ---------------------------------------------------------------------------
void test_vi_history_bookkeeping() {
    std::printf("\nB/C. VI history: clear_history and frozen holds\n");

    auto make_vi = [](const char* nm) {
        mean_field_gaussian_vi_block_config c;
        c.name             = nm;
        c.initial_unc      = arma::vec(2, arma::fill::zeros);
        c.log_density_grad = [](const arma::vec& t, const block_context&,
                                arma::vec* g) {
            if (g) { g->set_size(2); (*g)[0] = -t[0]; (*g)[1] = -t[1]; }
            return -0.5 * (t[0] * t[0] + t[1] * t[1]);
        };
        return std::make_unique<mean_field_gaussian_vi_block>(std::move(c));
    };

    // ---- B: clear_history drops the q-draws as well as the trajectory ----
    {
        auto comp = std::make_unique<composite_block>();
        mean_field_gaussian_vi_block* vi = nullptr;
        { auto b = make_vi("w"); vi = b.get(); comp->add_child(std::move(b)); }
        comp->set_keep_history(true);

        std::mt19937_64 rng(7);
        for (int i = 0; i < 20; ++i) comp->step(rng);
        check(vi->q_draw_history().size() == 20, "20 sweeps -> 20 q-draws");

        vi->clear_history();
        check(vi->q_draw_history().empty(), "clear_history() empties the q-draw buffer");

        for (int i = 0; i < 10; ++i) comp->step(rng);
        check(vi->q_draw_history().size() == 10,
              "10 more sweeps -> 10 q-draws (not 30)");
    }

    // ---- C: a frozen VI child holds a row per outer sweep ----------------
    // vi_block::supports_freeze() is false, so a VI block is never frozen by
    // name -- it is held when an ANCESTOR composite is frozen, and the
    // composite then calls record_held_history() on it. That is the path
    // this exercises.
    {
        auto inner = std::make_unique<composite_block>("inner");
        mean_field_gaussian_vi_block* vi = nullptr;
        { auto b = make_vi("w"); vi = b.get(); inner->add_child(std::move(b)); }

        auto outer = std::make_unique<composite_block>();
        composite_block* inner_raw = inner.get();
        outer->add_child(std::move(inner));
        { auto b = make_vi("g"); outer->add_child(std::move(b)); }
        outer->set_keep_history(true);

        std::mt19937_64 rng(11);
        for (int i = 0; i < 10; ++i) outer->step(rng);
        const std::size_t before = vi->q_draw_history().size();
        check(before == 10, "10 sweeps unfrozen -> 10 q-draws");

        check(!vi->supports_freeze(),
              "a VI block is not freezable by name -- only held via an ancestor");
        outer->freeze(std::vector<std::string>{"inner"});
        for (int i = 0; i < 10; ++i) outer->step(rng);

        check(vi->q_draw_history().size() == 20,
              "a frozen VI child still advances one row per outer sweep");
        check(vi->vi_history().mu.size() == 20,
              "the vi_history_t trajectory advances too");
        check(vi->vi_history().mu.back().n_elem == 2 &&
              arma::approx_equal(vi->vi_history().mu.back(),
                                 vi->vi_history().mu[9], "absdiff", 1e-15),
              "the held rows repeat the value frozen at, unchanged");
        check(outer->history_size() == 20,
              "the composite's own history_size() is not dragged back");
        (void)inner_raw;
    }
}

} // namespace

int main() {
    std::printf("=== test_vi_publish_and_history ===\n\n");
    test_structured_publishes_states();
    test_vi_history_bookkeeping();
    std::printf("\n%d checks, %d failures\n", checks, failures);
    if (failures) { std::printf("FAILED\n"); return 1; }
    std::printf("PASSED\n");
    return 0;
}
