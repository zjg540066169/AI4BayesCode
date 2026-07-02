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
              slow_sweep_sec: float = 0.5) -> None:
    """Print a per-sweep timing summary (and a joint_nuts_block hint if slow)."""
    if n_sweeps_total <= 0:
        return
    per = wall_sec / n_sweeps_total
    print(f"[AI4BayesCode perf] total {wall_sec:.1f}s across {n_sweeps_total} "
          f"sweeps ({per:.3f}s / sweep)")
    if per <= slow_sweep_sec:
        print("[AI4BayesCode perf] per-sweep time looks OK.")
        return
    if uses_joint_nuts:
        print("[AI4BayesCode perf] per-sweep time is high even with "
              "joint_nuts_block; check N*J gradient cost, NUTS tree depth "
              "maxing out, or mass-matrix warmup length.")
        return
    print("[AI4BayesCode perf] per-sweep time is high. If the sampler has "
          "tightly-coupled continuous parameters (additive linear mean, shift "
          "invariance, fixed+random effects), consider joint_nuts_block over "
          "them (see skills/codegen.md Section 4a).")
