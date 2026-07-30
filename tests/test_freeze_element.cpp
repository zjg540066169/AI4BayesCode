/*================================================================================
 *  element-level freeze acceptance test (2026-07-27).
 *
 *  freeze("slot[k]") holds ONE natural element (1-based k) of a PER-ELEMENT slot
 *  (REAL / POSITIVE / LOWER_BOUNDED / UPPER_BOUNDED / INTERVAL) fixed while the
 *  rest of the joint block keeps sampling; element e maps 1:1 to unconstrained
 *  coord unc_off+e so it is held exactly. Coupled / dimension-changing slots
 *  (ORDERED, SIMPLEX, CORR_MATRIX, COV_MATRIX, ...) reject an element index --
 *  freeze the whole slot instead.
 *================================================================================*/
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/block_sampler.hpp"
#include "AI4BayesCode/backend_neutral.hpp"
#include "AI4BayesCode/shared_data.hpp"
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <algorithm>

using namespace AI4BayesCode;

// N(0, I) over the whole cat.
static double ld(const arma::vec& x, const block_context&, arma::vec* g){
    if (g) { g->set_size(x.n_elem); for (arma::uword i=0;i<x.n_elem;++i) (*g)[i] = -x[i]; }
    return -0.5 * arma::dot(x, x);
}

static std::unique_ptr<joint_nuts_block> make_beta3(){
    joint_nuts_block_config cfg;
    cfg.name = "j";
    cfg.sub_params.push_back(joint_nuts_sub_param{"beta", 3, joint_constraint::REAL});
    cfg.initial_cat = arma::vec{0.0,0.0,0.0};
    cfg.log_density_grad = &ld;
    cfg.n_warmup_first_call = 20;
    cfg.initial_step_size = 0.1;
    return std::make_unique<joint_nuts_block>(std::move(cfg));
}

int main(){
    bool ok = true;

    // ---- (1) element freeze on a REAL vector slot -----------------------
    {
        auto root = std::make_unique<composite_block>("root");
        root->add_child(make_beta3());
        std::mt19937_64 rng(1);
        for (int i=0;i<10;++i) root->step(rng);       // warm + move
        arma::vec before = root->child(0).current();  // [b0,b1,b2]
        root->freeze(std::vector<std::string>{"beta[2]"});  // 1-based -> element idx 1
        for (int i=0;i<10;++i) root->step(rng);
        arma::vec after = root->child(0).current();

        bool held   = std::abs(after[1]-before[1]) < 1e-14;             // beta[2] frozen
        bool moved0  = std::abs(after[0]-before[0]) > 1e-9;             // beta[1] free
        bool moved2  = std::abs(after[2]-before[2]) > 1e-9;             // beta[3] free
        auto fr = root->get_frozen();
        bool reported = std::find(fr.begin(), fr.end(), std::string("j.beta[2]")) != fr.end()
                      || std::find(fr.begin(), fr.end(), std::string("beta[2]")) != fr.end();
        std::printf("(1) element freeze beta[2]: held=%d free0=%d free2=%d reported=%d  frozen=[",
                    held, moved0, moved2, reported);
        for (auto& f : fr) std::printf("%s ", f.c_str());
        std::printf("]\n");
        std::printf("    before=[%.4f %.4f %.4f] after=[%.4f %.4f %.4f]\n",
                    before[0],before[1],before[2], after[0],after[1],after[2]);
        ok = ok && held && moved0 && moved2 && reported;

        // unfreeze -> element moves again
        root->unfreeze(std::vector<std::string>{"beta[2]"});
        arma::vec u0 = root->child(0).current();
        for (int i=0;i<10;++i) root->step(rng);
        arma::vec u1 = root->child(0).current();
        bool unfroze = std::abs(u1[1]-u0[1]) > 1e-9;
        std::printf("(1b) after unfreeze beta[2] moves again: %d\n", unfroze);
        ok = ok && unfroze;
    }

    // ---- (2) coupled-constraint slot: element freeze must THROW ----------
    {
        joint_nuts_block_config cfg;
        cfg.name = "jc";
        cfg.sub_params.push_back(joint_nuts_sub_param{"w", 3, joint_constraint::ORDERED});
        cfg.initial_cat = arma::vec{0.0, 1.0, 2.0};
        cfg.log_density_grad = &ld;
        cfg.n_warmup_first_call = 5;
        joint_nuts_block jb(std::move(cfg));
        bool threw = false;
        try { jb.freeze_sub("w[1]"); }
        catch (const std::exception& e) {
            threw = true;
            std::printf("(2) coupled slot element freeze correctly threw: %.90s...\n", e.what());
        }
        // whole-slot freeze of the SAME coupled slot must still work
        bool whole_ok = true;
        try { jb.freeze_sub("w"); } catch (...) { whole_ok = false; }
        std::printf("(2b) whole-slot freeze of coupled slot works: %d\n", whole_ok);
        ok = ok && threw && whole_ok;
    }

    // ---- (3) out-of-range element index must THROW -----------------------
    {
        auto jb = make_beta3();
        bool threw = false;
        try { jb->freeze_sub("beta[9]"); } catch (const std::exception&) { threw = true; }
        std::printf("(3) out-of-range beta[9] correctly threw: %d\n", threw);
        ok = ok && threw;
    }

    std::printf("%s\n", ok ? "PASS" : "FAILED");
    return ok ? 0 : 1;
}
