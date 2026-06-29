---
name: AI4BayesCode-sim3-workflow
description: |
  Governs Setting 3 of the AI4Bayes evaluation — POST-HOC validation of
  cells generated WITHOUT the systematic validation framework during
  generation. Cells were produced by `sim3/scripts/run_sim3.sh` which
  runs ONLY L1 syntax-retry (up to 5 attempts) — the L2 25-check
  semantic audit and L3 R1+R2+R3 runtime protocol were intentionally
  REMOVED from generation. This skill specifies how to retroactively
  apply L2 + L3 + (optional) cross-impl sim2-style comparison to all
  generated cells, and how to aggregate metrics into the headline
  numbers reported in the paper.
---

# sim3 workflow — post-hoc validation of "unguarded" cells

## Purpose

Settings 1 and 2 measure validated AI4BayesCode samplers. Setting 3
ablates **the validation framework itself** to isolate its contribution.
The pipeline:

1. **Generation** (already done by `sim3/scripts/run_sim3.sh`):
   - 10 independent regens × 20 models = 200 cells
   - Each cell: ONE `claude -p` call, only the L1 compile-retry loop
   - No L2 self-audit, no L3 runtime protocol, no rhat/BPV/LOO
     iteration during generation
   - Output: `models<K>/<MODEL>/{cpp, runner.R, STATUS, attempts.txt,
     session.log}`. STATUS contains only `SUCCESS` (compiled within 5
     attempts) or `FAILED_L1`.

2. **Post-hoc validation** (THIS skill):
   - Stage A: per-cell L2 audit (25 checks from `validator.md`)
   - Stage B: per-cell L3 runtime protocol (R1 + R2 + R3)
   - Stage C: for cells passing both, sim2-style cross-impl rhat
     against the same reference used in Setting 2

3. **Reporting** (3 headline numbers):
   - Mean / quantiles of `attempts` (from `attempts.txt`)
   - Semantic-pass rate = #cells passing L2 / 200
   - Runtime-pass rate  = #cells passing L3 / 200
   - For sim2-comparable cells: median rhat distribution

## Directory layout

```
sim3/
├── AI4BayesCode_sim3/              ← library + skills (validator.md
│                                       intentionally removed for the
│                                       generation phase; restore here
│                                       to evaluate post-hoc)
├── scripts/                        ← run_sim3.sh (DONE)
├── models1/  ...  models10/        ← 200 generated cells
│   └── <MODEL>/
│       ├── <MODEL>.cpp
│       ├── <MODEL>_runner.R
│       ├── STATUS                  (line 1: SUCCESS | FAILED_L1)
│       ├── attempts.txt            (single integer 1..5)
│       ├── session.log
│       └── <NEW> validation outputs added by post-hoc:
│           ├── l2_report_posthoc.md     (Stage A)
│           ├── l3_smoke.log             (Stage B-R1)
│           ├── l3_2chain.csv            (Stage B-R2)
│           ├── l3_bpv_loo.csv           (Stage B-R3)
│           ├── l3_status.txt            (PASS_L3 | FAILED_L3 + reason)
│           └── sim2_compare.csv         (Stage C, if L2+L3 PASS)
└── results/
    ├── posthoc_summary.csv         ← aggregate metrics across all cells
    └── posthoc_summary.rds
```

## Stage A — L2 semantic audit (post-hoc)

For each cell where `STATUS == SUCCESS` (compiled cleanly):

1. Spawn a fresh `claude -p` session in the cell directory with
   the FULL skill set (including `validator.md` — restored for
   post-hoc evaluation).
2. Prompt: "Read `<MODEL>.cpp` and `<MODEL>_runner.R`. Apply the
   25 checks from `skills/validator.md` Layer 2. Write a
   single-file report to `l2_report_posthoc.md` with PASS / FAIL /
   N/A per check + brief justification."
3. Pass criterion: 25/25 PASS or N/A. Any FAIL → cell flagged
   `L2_FAIL_<check_id>`.
4. Cells with `STATUS == FAILED_L1` are skipped (no cpp to audit).

This is the SAME L2 audit run inline during sim2 generation. The
difference for sim3 is that the agent never saw it during the
codegen pass — so any L2 failure is a "would-have-been-caught-by-
the-validator" bug that shipped because validation was off.

Implementation: `scripts/run_l2_posthoc.sh` is **NOT YET IMPLEMENTED**
(this script does not exist in the repository — do not attempt to
invoke it; run the L2 audit manually per the protocol above). When
written it would loop over `models[1-9]*/*/STATUS == SUCCESS` cells
and submit one short `claude -p` per cell. Token cost: roughly 1/3 of
generation cost (only the audit step, not the generate-then-audit loop).

## Stage B — L3 runtime protocol (post-hoc)

For each cell where `STATUS == SUCCESS`:

### B-R1: smoke

Run the cell's `<MODEL>_runner.R` (10-step warmup, finiteness check,
predict_at non-mutation). PASS if no crash and final
`get_current()` is finite.

### B-R2: 2-chain R-hat convergence

Run two chains with seeds 101L and 202L:
- Stage 1: 4000 burn + 4000 keep per chain. Compute
  rank-normalized R-hat over all parameters.
- Stage 2 (if Stage 1 max R-hat ≥ 1.05): re-run at 20000 +
  20000. PASS if max R-hat < 1.05 at whichever budget was used.

ESS is diagnostic only; record `min(ESS_bulk)` / `min(ESS_tail)`
but do NOT fail on ESS thresholds (legit slow-mixing models can
have low ESS).

### B-R3: posterior-predictive p-values + PSIS-LOO

Use `hist$y_rep` from a `keep_history = TRUE` run (or fall back
to looped `predict_at()`). For up to 6 summary statistics
(mean, sd, min, max, q25, q75), compute Bayesian p-value against
observed. PASS if all in (0.05, 0.95).

LOO: PASS if ≥ 50% Pareto k < 0.5 AND < 10% k ≥ 0.7.

### Aggregating B

`l3_status.txt` contains one of:
- `PASS_L3` — R1 + R2 + R3 all PASS
- `FAILED_L3_R1` — smoke crash
- `FAILED_L3_R2` — R-hat ≥ 1.05 even at 20k+20k
- `FAILED_L3_R3` — BPV out of range OR LOO Pareto-k bad

Implementation: `scripts/run_l3_posthoc.sh` is **NOT YET IMPLEMENTED**
(this script does not exist in the repository — do not attempt to
invoke it; run the L3 protocol manually as described above). When
written it would loop over `STATUS == SUCCESS` cells and run the
protocol locally.
Heavy models (`hierarchical_gp`, `sir`, `multi_occupancy`) may
need longer wall budget than sim3's 5–15 min/cell — give them
30–120 min headroom, or punt to the server like sim1's
hierarchical_gp / sir.

## Stage C — sim2-style cross-impl comparison (optional)

For the subset of cells where L2 and L3 BOTH pass:

Run the cell's cpp against the canonical reference (Stan or
R-package, same as Setting 2) on 20 simulated datasets. Compute
per-replicate cross-impl rhat, ESS, coverage of truth. This
mirrors the `sim2/results/<MODEL>/sim2_<MODEL>.R` workflow but
ONE cell at a time (not 5 sister cells); the per-cell CSV row
includes `sampler_id = <regen_K>` for grouping.

Output: `models<K>/<MODEL>/sim2_compare.csv`.

When per-model 10 regens have 10 sim2_compare rows, aggregate as
sim2 does (median [min, max] across regens).

## Aggregate metrics (paper headline numbers)

After Stages A + B (+ optional C) finish for all 200 cells, run
`scripts/aggregate_sim3.R` to produce
`results/posthoc_summary.csv` with columns:

| col | meaning |
|---|---|
| `model` | model id |
| `regen_k` | 1..10 |
| `attempts` | from attempts.txt (1..5; lower = better) |
| `l1_status` | SUCCESS \| FAILED_L1 |
| `l2_pass` | TRUE/FALSE (25/25 PASS) |
| `l2_fail_check` | comma-sep list of failed check IDs (empty if pass) |
| `l3_status` | PASS_L3 \| FAILED_L3_R1 \| FAILED_L3_R2 \| FAILED_L3_R3 |
| `l3_rhat_max` | numeric |
| `l3_min_ess_bulk` | numeric |
| `l3_bpv_min`, `l3_bpv_max` | extremes across 6 stats |
| `l3_loo_pareto_k_lt_0.5_pct` | numeric in [0,1] |
| `sim2_rhat_max` | NA if Stage C not run |
| `sim2_cov_AI`, `sim2_cov_REF` | NA if Stage C not run |

Headline numbers reported in paper:

```
Settings 3 results (200 cells = 10 regens × 20 models):

  Compile (L1):     X / 200 cells SUCCESS  (median attempts = M)
  Semantic (L2):    Y / Z (cells with cpp)
  Runtime  (L3):    W / Z (cells with cpp)
  Sim2-comparable:  V / 200 (passed all of L1 + L2 + L3)

  For V comparable cells, cross-impl rhat distribution:
    median = ...   95% CrI = ...
```

## Anti-patterns to avoid

1. **Don't skip cells with STATUS=FAILED_L1.** They count as 0/1 in
   the "compile rate" denominator (200 cells × 1 = 200).
2. **Don't re-generate FAILED_L1 cells.** Generation already used
   its 5 attempts. The denominator stays 200.
3. **Don't run L2 in the SAME `claude -p` as generation.** That
   would defeat the ablation. The post-hoc L2 is a SEPARATE,
   later session.
4. **Don't fail cells on ESS alone.** Slow-mixing-but-correct
   samplers (large GPs, hierarchical, ICAR) legitimately have
   low ESS. R-hat is the convergence gate.
5. **Don't spend Claude API on Stage A for FAILED_L1 cells.** No
   cpp to audit — record `l2_pass = NA` and move on.

## Relationship to other skills

- `validator.md` — defines the 25 checks of Stage A and the R1/R2/R3
  protocol of Stage B. THIS skill orchestrates them; the check
  logic itself stays in validator.md.
- `sim2_workflow.md` — defines the cross-impl comparison Stage C
  reuses (per-cell → per-regen → aggregate). Stage C is sim2 with
  the 5 cells potentially ALL failing some upstream gate (vs sim2
  where all 5 are pre-validated).
- `sim1_workflow.md` — defines the per-replicate rhat / coverage
  computation Stage C calls into.
- `codegen.md`, `codegen_cpp.md`, `codegen_priors.md`,
  `block_catalogue.md`, `hierarchical_re.md`, etc. — the
  domain-knowledge skills the codegen agent USED during sim3
  generation. These were intentionally LEFT IN (per
  `scripts/README.md`); only `validator.md` and the L2/L3 retry
  loop were removed for the generation phase. The post-hoc Stage
  A restores `validator.md` for evaluation.

## Effort estimate

| Stage | Per-cell cost | Total (200 cells) |
|---|---|---|
| A — L2 audit | 1 `claude -p` call, ~3-5 min, ~$0.02-0.05 token | ~10–15 hr API + a few $ |
| B — L3 protocol | local R, ~5–60 min depending on model | ~30 hr serial / 4 hr 8-core |
| C — sim2 compare | local R, ~5–60 min × 20 reps | ~100 hr serial — heavy; consider only cells passing A+B |

Stage A is API-bound (subject to 5-hour quota). Stage B is local
CPU. Stage C is the most expensive and best done after A+B narrow
down the candidate set.

A reasonable schedule:
- Day 1: Stage A in batches around 5-hour quota windows
- Day 2: Stage B in parallel on local cores
- Day 3+: Stage C only on V cells that passed both
