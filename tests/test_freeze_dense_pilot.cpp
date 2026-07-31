// test_freeze_dense_pilot.cpp
// ---------------------------------------------------------------------------
// Regression test for the FROZEN-HOLD bug (2026-07-30): a joint_nuts_block
// frozen sub-parameter must be held BIT-EXACTLY across step(), including when
// the first step() runs the metric-adaptation pilot / 3-phase warmup with the
// freeze already active. Before the fix, adapt_dense_metric_() and
// adapt_three_phase_warmup_() integrated frozen coords as free particles and
// left theta_cat_ at the drifted last pilot draw WITHOUT restoring frozen
// coords; step() then re-snapshotted the drifted value and pinned it forever.
// The pre-fix behavior was value/data-dependent (some cases held by accident
// via the degenerate-pilot escape reset), so this test asserts BIT-EXACT hold
// across the tricky combinations, plus statistical correctness of the free
// coordinate's conditional posterior.
// ---------------------------------------------------------------------------
#ifndef MCMC_ENABLE_ARMA_WRAPPERS
# define MCMC_ENABLE_ARMA_WRAPPERS
#endif
#ifndef ARMA_DONT_USE_WRAPPER
# define ARMA_DONT_USE_WRAPPER
#endif
#include <armadillo>
#include "AI4BayesCode/composite_block.hpp"
#include "AI4BayesCode/joint_nuts_block.hpp"
#include "AI4BayesCode/constraints.hpp"
#include <cstdio>
#include <memory>
#include <random>
#include <string>

using namespace AI4BayesCode;

static const int kP = 3;
static arma::mat gX;   // N x P
static arma::vec gY;   // N

static int g_pass = 0, g_fail = 0;
static void check(bool ok, const std::string& msg, const std::string& extra = "") {
    if (ok) { std::printf("  PASS  %s\n", msg.c_str()); ++g_pass; }
    else    { std::printf("  FAIL  %s%s%s\n", msg.c_str(),
                          extra.empty() ? "" : "  -- ", extra.c_str()); ++g_fail; }
}

// Gaussian regression with Jeffreys p(sigma) prop 1/sigma:
//   lp = -(N+1) log sigma - 0.5 sse/sigma^2  =>  sigma^2 | beta ~ InvGamma(N/2, sse/2).
static double dens(const arma::vec& c, const block_context&, arma::vec* g){
    if (c.n_elem != kP+1) return -std::numeric_limits<double>::infinity();
    const arma::vec b = c.subvec(0,kP-1); const double s=c[kP];
    if(!(s>0.0)||!std::isfinite(s)){ if(g){g->set_size(kP+1);g->zeros();} return -std::numeric_limits<double>::infinity();}
    const double N=(double)gY.n_elem; const arma::vec r=gY-gX*b; const double sse=arma::dot(r,r), s2=s*s;
    const double lp=-(N+1.0)*std::log(s)-0.5*sse/s2;
    if(g){g->set_size(kP+1);g->zeros();g->subvec(0,kP-1)=gX.t()*r/s2;(*g)[kP]=-(N+1.0)/s+sse/(s2*s);}
    return std::isfinite(lp)?lp:-std::numeric_limits<double>::infinity();
}

// metric: 2 = dense, 1 = diagonal, 0 = identity.
static std::unique_ptr<joint_nuts_block> mk(int metric){
    joint_nuts_block_config cfg; cfg.name="beta_sigma_joint";
    cfg.sub_params.push_back(joint_nuts_sub_param{"beta",kP,joint_constraint::REAL});
    cfg.sub_params.push_back(joint_nuts_sub_param{"sigma",1,joint_constraint::POSITIVE});
    cfg.initial_cat=arma::vec{0.0,0.0,0.0,1.0}; cfg.log_density_grad=&dens;
    cfg.n_warmup_first_call=1000; cfg.max_tree_depth=8;
    if (metric==2){ cfg.use_dense_metric=true; cfg.dense_metric_pilot_iters=200; cfg.dense_metric_adapt_iters=500; }
    else if (metric==1){ cfg.use_diagonal_metric=true; }
    return std::make_unique<joint_nuts_block>(std::move(cfg));
}

// Freeze `slot` BEFORE the first step (so the pilot runs frozen), optionally
// set_current AFTER freeze, step, and assert the frozen coords are BIT-EXACT.
static void hold_case(int metric, const std::string& slot, bool do_set,
                      arma::vec setcat, arma::vec expect, arma::uvec fidx,
                      const std::string& label){
    auto root=std::make_unique<composite_block>("r"); root->add_child(mk(metric));
    std::mt19937_64 r(7); auto& jb=dynamic_cast<joint_nuts_block&>(root->child(0));
    root->freeze(std::vector<std::string>{slot});
    if(do_set) jb.set_current(setcat);
    for(int i=0;i<25;++i) root->step(r);
    arma::vec got=jb.current().elem(fidx);
    double drift=arma::max(arma::abs(got-expect));
    check(drift==0.0, label, "drift=" + std::to_string(drift));
}

int main(){
    std::mt19937_64 rng(42); const int N=100; gX.set_size(N,kP);
    arma::vec bt{1.5,0.05,0.30}; const double sigma_true=0.5;
    std::normal_distribution<double> z(0,1);
    for(int i=0;i<N;++i){gX(i,0)=1;gX(i,1)=60+10*z(rng);gX(i,2)=(i%2);}
    gY.set_size(N); for(int i=0;i<N;++i) gY[i]=arma::dot(gX.row(i),bt.t())+sigma_true*z(rng);

    std::printf("=== BIT-EXACT hold across freeze-before-pilot combinations ===\n");
    for (int metric=1; metric<=2; ++metric){
        const char* mn = (metric==2)?"dense":"diagonal";
        hold_case(metric,"sigma",false,{},                arma::vec{1.0},   arma::uvec{3},
                  std::string("freeze sigma, no set_current (")+mn+")");
        hold_case(metric,"sigma",true, arma::vec{0,0,0,2.0},arma::vec{2.0},   arma::uvec{3},
                  std::string("freeze sigma, set_current s=2 (")+mn+")");
        hold_case(metric,"beta", false,{},                arma::vec{0,0,0},  arma::uvec{0,1,2},
                  std::string("freeze beta, no set_current (")+mn+")");
        hold_case(metric,"beta", true, arma::vec{1,2,3,1.0},arma::vec{1,2,3},arma::uvec{0,1,2},
                  std::string("freeze beta, set_current [1,2,3] (")+mn+")");
    }

    std::printf("=== element-level freeze holds bit-exact (dense) ===\n");
    {
        auto root=std::make_unique<composite_block>("r"); root->add_child(mk(2));
        std::mt19937_64 r(7); auto& jb=dynamic_cast<joint_nuts_block&>(root->child(0));
        jb.set_current(arma::vec{1.0,2.0,3.0,0.5});
        root->freeze(std::vector<std::string>{"beta[1]"});   // 1-based -> beta[0]
        arma::vec before=jb.current();
        for(int i=0;i<25;++i) root->step(r);
        arma::vec after=jb.current();
        check(after[0]==1.0, "element beta[1] held bit-exact under dense pilot",
              "beta0=" + std::to_string(after[0]));
        check(std::abs(after[1]-before[1])>1e-9 && std::abs(after[2]-before[2])>1e-9,
              "sibling elements beta[2],beta[3] still move");
    }

    std::printf("=== conditional posterior correctness (frozen beta, free sigma) ===\n");
    for (int metric=1; metric<=2; ++metric){
        const char* mn = (metric==2)?"dense":"diagonal";
        auto root=std::make_unique<composite_block>("r"); root->add_child(mk(metric));
        std::mt19937_64 r(123); auto& jb=dynamic_cast<joint_nuts_block&>(root->child(0));
        jb.set_current(arma::join_cols(bt, arma::vec{sigma_true}));
        root->freeze(std::vector<std::string>{"beta"});
        root->step(r);                                       // pilot + warmup
        const int M=5000; arma::vec s2(M);
        for(int i=0;i<M;++i){ root->step(r); s2[i]=std::pow(jb.current()[kP],2.0); }
        const double sse=arma::dot(gY-gX*bt, gY-gX*bt);
        const double analytic_mean=sse/(N-2);
        const double rel=std::abs(arma::mean(s2)-analytic_mean)/analytic_mean;
        check(arma::approx_equal(jb.current().subvec(0,kP-1), bt, "absdiff", 0.0),
              std::string("beta stays bit-exact at truth through sampling (")+mn+")");
        check(rel<0.05, std::string("E[sigma^2] matches analytic InvGamma within 5% (")+mn+")",
              "sample=" + std::to_string(arma::mean(s2)) + " analytic=" + std::to_string(analytic_mean));
    }

    std::printf("\n=== SUMMARY: %d passed, %d failed ===\n", g_pass, g_fail);
    return g_fail==0 ? 0 : 1;
}
