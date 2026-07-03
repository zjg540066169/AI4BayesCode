"""Post-run per-sweep performance hint.

Python analogue of the R-side ``ai4bayescode_perf_hint()``: given the total
wall-clock time and the number of MCMC sweeps, print the per-sweep time and,
when it is slow, a hint to consider ``joint_nuts_block`` over tightly-coupled
continuous parameters.
"""
from __future__ import annotations

__all__ = ["perf_hint"]


def perf_hint(wall_sec: float,
              n_sweeps_total: int,
              uses_joint_nuts: bool = False,
              slow_sweep_sec: float = 0.5,
              thresholds: dict | None = None) -> None:
    """Print a per-sweep timing summary (and a joint_nuts_block hint if slow).

    Parameters mirror the R ``ai4bayescode_perf_hint()``. ``thresholds`` accepts
    a dict (signature parity with R's ``thresholds = list(slow_sweep_sec = 0.5)``);
    when given, its ``slow_sweep_sec`` key overrides the flat ``slow_sweep_sec``
    kwarg (which is kept for back-compat).
    """
    if thresholds is not None:
        slow_sweep_sec = thresholds.get("slow_sweep_sec", slow_sweep_sec)
    if n_sweeps_total <= 0:
        return
    per = wall_sec / n_sweeps_total
    print(f"[AI4BayesCode perf] total {wall_sec:.1f}s across {n_sweeps_total} "
          f"sweeps ({per:.3f}s / sweep)")
    if per <= slow_sweep_sec:
        print("[AI4BayesCode perf] per-sweep time looks OK.")
        return
    if uses_joint_nuts:
        print(
            "[AI4BayesCode perf] per-sweep time is high even with joint_nuts_block.\n"
            "  Possible causes: (a) N * J grad eval is genuinely expensive,\n"
            "  (b) NUTS tree depth maxing out -> try raising n_warmup_first_call\n"
            "      or seeding cfg.initial_step_size,\n"
            "  (c) mass-matrix adaptation not yet converged -> longer warmup.\n"
            "  Validator reminder: joint_nuts_block usage requires Check #11.")
        return
    print(
        "[AI4BayesCode perf] per-sweep time is high.\n"
        "  If this sampler has tightly-coupled continuous parameters in the\n"
        "  likelihood (e.g. additive linear mean, shift invariance,\n"
        "  fixed+random effects sharing mean), consider regenerating with\n"
        "  joint_nuts_block over the coupled parameters:\n"
        "    -> see skills/codegen.md Section 4a (Coupling analysis)\n"
        "    -> see examples/IRT1PL_joint.cpp for a reference\n"
        "  WARNING: joint_nuts_block has a higher semantic-bug surface than\n"
        "  modular NUTS (concatenate-and-slice code is easier to get\n"
        "  subtly wrong). Any joint sampler you generate must pass\n"
        "  validator Check #11 (grad slicing, prior completeness, scale\n"
        "  consistency, Jacobian, write-back offsets, dim asserts).")
