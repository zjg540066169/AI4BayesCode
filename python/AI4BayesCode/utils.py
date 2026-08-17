"""Convergence diagnostics (R-hat, ESS, posterior summary).

Python equivalents of the R `posterior` package functions used in the
R helper. `rhat` implements the rank-normalized split-R-hat of Vehtari
et al. (2021); `ess_bulk` / `ess_tail` implement the same paper's split-ESS
with Geyer's initial positive / monotone sequence. All three are checked
against `posterior` on 28 draw sets covering odd draw counts, ties, a single
chain, antithetic draws, collapsed scales and a total mixing failure; the
worst relative difference measured there is 6e-5, and every NA / Inf verdict
matches (see `python/tests/test_diagnostics_vs_posterior.py`).

Inputs are numpy arrays shaped (n_draws, n_chains) or a single 1-D
array which is treated as a single chain.

Heavy-duty workflows should prefer ArviZ (`az.rhat`, `az.ess`) for the
full suite of ranked diagnostics; these helpers cover the common
cases with no external-package dependency beyond numpy.
"""

from __future__ import annotations

import numpy as np


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError(
            f"expected 1-D or 2-D array (n_draws,) or (n_draws, n_chains); "
            f"got shape {x.shape}"
        )
    return x


def _is_constant(x: np.ndarray) -> bool:
    """`posterior`'s is_constant(): |x - x[0]| < .Machine$double.eps, an
    ABSOLUTE tolerance, not exact equality.

    Only the quantile path uses it (ess_tail), because only
    `.ess_quantile` calls should_return_NA(). Draws of order 1e-300 -- a
    collapsed variance -- are constant under this test, and posterior returns
    NA for their tail ESS while still reporting a normal rhat and ess_bulk.
    """
    return bool(np.all(np.abs(x - x.flat[0]) < np.finfo(float).eps))


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Split each chain in half -- doubles chain count, halves draw count.

    With an ODD draw count the two halves cannot both be taken from the front:
    `posterior` drops the MIDDLE draw and keeps the last `half`, so the second
    half ends at the final draw. Taking `x[half:2*half]` instead drops the LAST
    draw and shifts the second half one position earlier, which is a real
    numerical difference -- measured on 101x4 draws, ess_bulk 447.4 (drop-last)
    vs 436.4 (drop-middle) against posterior's 436.2.
    """
    n_draws, n_chains = x.shape
    half = n_draws // 2
    if half == 0:
        raise ValueError("need at least 2 draws to split-R-hat")
    first = x[:half]
    second = x[n_draws - half:]
    return np.concatenate([first, second], axis=1)


def _rank_normalize(x: np.ndarray) -> np.ndarray:
    """Rank-normalize draws: average ranks over all chains, then the
    inverse normal CDF of the Blom-transformed ranks (Vehtari et al. 2021,
    Sec. 3.2). Makes R-hat invariant to monotone reparameterization and
    finite for heavy-tailed or infinite-variance targets."""
    from scipy.special import ndtri          # noqa: PLC0415 (optional dep)

    flat = x.reshape(-1)
    # Average ranks, 1-based, ties averaged.
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=float)
    ranks[order] = np.arange(1, flat.size + 1, dtype=float)
    # Average the ranks within each group of tied values.
    srt = flat[order]
    start = 0
    for i in range(1, srt.size + 1):
        if i == srt.size or srt[i] != srt[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    z = ndtri((ranks - 3.0 / 8.0) / (flat.size + 0.25))   # Blom
    return z.reshape(x.shape)


def _rank_normalize_numpy(x: np.ndarray) -> np.ndarray:
    """_rank_normalize without scipy: same Blom ranks, inverse normal CDF
    via a rational approximation (Acklam), max abs error ~1.15e-9."""
    flat = x.reshape(-1)
    order = np.argsort(flat, kind="stable")
    ranks = np.empty(flat.size, dtype=float)
    ranks[order] = np.arange(1, flat.size + 1, dtype=float)
    srt = flat[order]
    start = 0
    for i in range(1, srt.size + 1):
        if i == srt.size or srt[i] != srt[start]:
            if i - start > 1:
                ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    p = (ranks - 3.0 / 8.0) / (flat.size + 0.25)
    return _ndtri_acklam(p).reshape(x.shape)


def _ndtri_acklam(p: np.ndarray) -> np.ndarray:
    """Inverse standard-normal CDF (Peter Acklam's rational approximation)."""
    a = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
         1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
    b = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
         6.680131188771972e01, -1.328068155288572e01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
         -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
         3.754408661907416e00)
    p = np.asarray(p, dtype=float)
    out = np.empty_like(p)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)

    q = np.sqrt(-2 * np.log(p[lo]))
    out[lo] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
              ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    out[hi] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p[mid] - 0.5
    r = q * q
    out[mid] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5])*q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)
    return out


def _rank_normalize_dispatch(x: np.ndarray) -> np.ndarray:
    """Rank-normalize with scipy when available, numpy-only otherwise."""
    try:
        return _rank_normalize(x)
    except Exception:
        return _rank_normalize_numpy(x)


def _plain_split_rhat(x: np.ndarray, *, already_split: bool = False) -> float:
    """Classic split-R-hat on the values as given (no rank transform).

    `already_split` is for the rank-normalized callers: `posterior` splits
    BEFORE it rank-normalizes, so they hand in an array that is already split.
    """
    if not already_split:
        x = _split_chains(x)
    n, _ = x.shape
    chain_means = x.mean(axis=0)
    chain_vars = x.var(axis=0, ddof=1)
    B = n * np.var(chain_means, ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        # Every chain constant. If they are constant at DIFFERENT values --
        # four chains each stuck on a different mixture-component count, the
        # textbook total-mixing failure -- posterior returns Inf, and it has
        # to: NaN > 1.05 is False, so NaN sails through any convergence gate
        # while the same draws fail on the R side.
        return float("inf") if np.var(chain_means, ddof=1) > 0 else float("nan")
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def rhat(samples: np.ndarray) -> float:
    """Rank-normalized split-R-hat (Vehtari et al. 2021).

    Computed as `max(bulk, tail)`:
      * bulk -- split-R-hat of the rank-normalized draws, which detects
        disagreement in the centre of the distribution;
      * tail -- split-R-hat of the rank-normalized FOLDED draws
        `|x - median(x)|`, which detects disagreement in scale that the
        bulk statistic can miss.

    This matches `posterior::rhat()`, so the R and Python frontends report
    the same number for the same draws and the 1.05 gate means the same
    thing on both sides.

    Parameters
    ----------
    samples : array shape (n_draws,) or (n_draws, n_chains)

    Returns
    -------
    float : R-hat. Values near 1.0 indicate convergence; > 1.01 is
    traditionally flagged as a concern.
    """
    x = _as_2d(samples)
    if x.shape[0] < 4:            # too few draws for a meaningful split-R-hat
        return float("nan")
    if not np.all(np.isfinite(x)):
        return float("nan")
    if np.all(x == x.flat[0]):    # constant: no variance to compare
        return float("nan")
    # Every chain constant at a DIFFERENT value -- the loudest possible mixing
    # failure, for which posterior returns Inf. Decided here on the RAW draws,
    # because deciding it downstream from W == 0 makes the verdict ride on
    # floating point: rank-normalizing first leaves the per-chain values equal
    # only to within an ULP on some platforms, so W lands at ~1e-32 instead of
    # exactly 0 and R-hat reads 6e15 rather than Inf. Same conclusion, but the
    # test for it has to be exact.
    if x.shape[1] > 1 and np.all(x == x[0]) and np.ptp(x[0]) > 0:
        return float("inf")

    # ORDER MATTERS: posterior::rhat is .rhat(z_scale(.split_chains(x))) --
    # split first, rank-normalize second. Reversing it is identical for an
    # even draw count (same multiset) but not for an odd one, where posterior
    # drops the middle draw BEFORE ranking and the reversed order drops it
    # after. Measured on 101x4 draws: R-hat differs by 5.3e-5, ess_bulk by
    # 2.3e-3 -- enough to flip a verdict either side of the 1.05 gate.
    bulk = _plain_split_rhat(_rank_normalize_dispatch(_split_chains(x)),
                             already_split=True)
    # The fold is over the UNSPLIT draws (median of everything), matching
    # posterior: sims_folded <- abs(x - median(x)), then split, then z_scale.
    folded = np.abs(x - np.median(x))
    tail = _plain_split_rhat(_rank_normalize_dispatch(_split_chains(folded)),
                             already_split=True)

    # NaN PROPAGATES, matching posterior: if either half is undefined (a
    # constant folded series, say -- antithetic draws fold to a constant), the
    # combined statistic is undefined too. Dropping the NaN and returning the
    # other half would report a converged-looking number for draws posterior
    # declines to summarize.
    if np.isnan(bulk) or np.isnan(tail):
        return float("nan")
    return float(max(bulk, tail))


def _autocov(chain: np.ndarray) -> np.ndarray:
    """Autocovariance via FFT for a single chain, normalized by n.

    The 1/n (biased) normalization is the one `posterior::autocovariance` uses
    and the one the ESS estimator below is calibrated against. Switching to the
    1/(n - lag) unbiased form inflates the large-lag terms, which is invisible
    for a well-mixing chain (the sum stops early) but overstates ESS by tens of
    percent exactly where it matters -- a poorly-mixing chain, whose initial
    sequence runs out to large lags.
    """
    x = chain - chain.mean()
    n = x.size
    fft_size = 1 << (2 * n - 1).bit_length()
    f = np.fft.fft(x, fft_size)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return ac / n


def _ess_raw(x: np.ndarray, *, already_split: bool = False) -> float:
    """Split-ESS of the values as given (Vehtari et al. 2021, Sec. 4).

    The combined autocorrelation is

        rho_t = 1 - (W - mean_c s_c^2 rho_{c,t}) / var_hat_plus

    which is what makes ESS see BETWEEN-chain disagreement: when one chain is
    parked away from the others, W stays small while var_hat_plus is large, so
    every rho_t stays near 1, tau blows up, and ESS collapses. Normalizing the
    averaged autocovariance by var_hat_plus alone (without the `1 - (W - ...)`
    wrapper) gives rho_t ~ 0 for exactly that case and reports a full ESS.
    """
    if not already_split:
        if x.shape[0] < 4:        # too few draws for a meaningful split-ESS
            return float("nan")
        x = _split_chains(x)  # split-ESS (Vehtari 2021); also makes a single
                              # chain well-defined (m becomes 2, not 1)
    n, m = x.shape
    chain_means = x.mean(axis=0)
    chain_vars = x.var(axis=0, ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        return float("nan")
    var_hat = ((n - 1) / n) * W + np.var(chain_means, ddof=1)
    if var_hat <= 0:
        return float("nan")

    # mean_c s_c^2 rho_{c,t}: the per-chain autocovariance, averaged. _autocov
    # already returns a covariance (not a correlation), so this IS s_c^2 rho_c.
    acov = np.zeros(n)
    for c in range(m):
        acov += _autocov(x[:, c])
    acov /= m

    rho = np.zeros(n)
    rho_all = 1.0 - (W - acov) / var_hat
    rho[0] = 1.0

    # Geyer's initial POSITIVE sequence: walk forward in pairs while the pair
    # sum is positive, then the initial MONOTONE sequence: force the pair sums
    # to be non-increasing. (Vehtari et al. 2021 Sec. 4; same procedure as
    # posterior::ess_mean / Stan.)
    t = 0
    rho_even = 1.0
    rho[0] = rho_even
    rho_odd = rho_all[1] if n > 1 else 0.0
    if n > 1:
        rho[1] = rho_odd
    while t < n - 5 and not np.isnan(rho_even + rho_odd) and \
            (rho_even + rho_odd) > 0:
        t += 2
        rho_even = rho_all[t]
        rho_odd = rho_all[t + 1]
        if (rho_even + rho_odd) >= 0:
            rho[t] = rho_even
            rho[t + 1] = rho_odd
    max_t = t
    if rho_even > 0:                 # used by the improved tau estimate below
        rho[max_t] = rho_even

    t = 0
    while t <= max_t - 4:
        t += 2
        if rho[t] + rho[t + 1] > rho[t - 2] + rho[t - 1]:
            rho[t] = (rho[t - 2] + rho[t - 1]) / 2.0
            rho[t + 1] = rho[t]

    n_total = n * m
    if max_t == 0:
        # The Geyer loop never advanced: rho[:0] is empty, so the general
        # formula gives tau = -1 + 0 + rho[0] = 0, which the floor below then
        # turns into 1/log10(N) -- an ESS several times the draw count (measured
        # 31224 for 8000 antithetic draws, where posterior gives 4000). posterior
        # evaluates sum(rho_hat_t[1:max_t]) as rho[1] = 1 in this case, i.e.
        # tau = 2. Too few post-split rows to estimate anything is NaN, matching
        # posterior's NA.
        if n < 3:
            return float("nan")
        tau = 2.0
    else:
        tau = -1.0 + 2.0 * float(np.sum(rho[:max_t])) + float(rho[max_t])
    tau = max(tau, 1.0 / np.log10(n_total))
    return float(n_total / tau)


def ess_bulk(samples: np.ndarray) -> float:
    """Bulk effective sample size (Vehtari et al. 2021).

    Split-ESS of the RANK-NORMALIZED draws, matching `posterior::ess_bulk`.
    Rank normalization is what makes the estimate finite and comparable for
    heavy-tailed targets; the between-chain term in the combined
    autocorrelation is what makes it collapse when a chain is stuck.

    Parameters
    ----------
    samples : array shape (n_draws,) or (n_draws, n_chains)
    """
    x = _as_2d(samples)
    if x.shape[0] < 4:
        return float("nan")
    if not np.all(np.isfinite(x)):
        return float("nan")
    if np.all(x == x.flat[0]):
        return float("nan")
    # posterior::ess_bulk is .ess(z_scale(.split_chains(x))) -- split first.
    # See the note in rhat() for what the reversed order costs on odd n.
    return _ess_raw(_rank_normalize_dispatch(_split_chains(x)),
                    already_split=True)


def ess_tail(samples: np.ndarray, quantile_lo: float = 0.05,
             quantile_hi: float = 0.95) -> float:
    """Tail ESS -- min of the ESS at the 5% and 95% quantile indicators,
    matching `posterior::ess_tail`."""
    x = _as_2d(samples)
    if x.shape[0] < 4:
        return float("nan")
    if not np.all(np.isfinite(x)):
        return float("nan")
    if _is_constant(x):
        return float("nan")
    lo = np.quantile(x, quantile_lo)
    hi = np.quantile(x, quantile_hi)
    # BOTH indicators are "<=", as posterior does (ess_mean(x <= q)). Using
    # `x >= hi` for the upper one is not the complement when the draws have
    # TIES -- and ties are the norm for the parameters this matters most for:
    # inclusion indicators, counts, cluster sizes.
    ind_lo = (x <= lo).astype(float)
    ind_hi = (x <= hi).astype(float)
    # A constant indicator carries no information; posterior returns NA.
    out = []
    for ind in (ind_lo, ind_hi):
        if np.all(ind == ind.flat[0]):
            return float("nan")
        out.append(_ess_raw(ind))
    return float(min(out))


def posterior_summary(samples: np.ndarray, prob: float = 0.90,
                      *, chains: bool = False) -> dict:
    """Compact posterior summary: mean, median, sd, mad, CI bounds, R-hat, ESS.

    The credible-interval level defaults to ``prob=0.90`` to match R's
    ``posterior::summarise_draws`` (q5/q95) -- the same 90% interval the shipped
    diagnostics advertise. ``median`` and ``mad`` (normal-consistent, x1.4826)
    are included for parity with the R summary. Works for a scalar or a single
    vector component (call per-component for a vector).
    """
    if not (0.0 < float(prob) < 1.0):
        # prob = -1 produced ci_lower > ci_upper -- an inverted interval
        # reported as a normal result.
        raise ValueError(
            f"posterior_summary: prob must be strictly between 0 and 1; "
            f"got {prob!r}")
    a = np.asarray(samples, dtype=float)
    if a.ndim == 2 and a.shape[1] > 1 and not chains:
        raise ValueError(
            f"posterior_summary got a {a.shape[0]}x{a.shape[1]} array. A 2-D "
            "array is read as (n_draws, n_chains), so passing a VECTOR "
            "parameter's draws here would summarize its components as if they "
            "were chains -- the mean would be the mean over components. "
            "Summarize one component at a time "
            "(`posterior_summary(x[:, j])`), or pass chains=True if the "
            "columns really are chains of one scalar.")
    x = _as_2d(samples)
    alpha = (1.0 - prob) / 2.0
    flat = x.reshape(-1)
    med = float(np.median(flat))
    return {
        "mean": float(flat.mean()),
        "median": med,
        "sd": float(flat.std(ddof=1)),
        "mad": float(np.median(np.abs(flat - med)) * 1.4826),
        "ci_lower": float(np.quantile(flat, alpha)),
        "ci_upper": float(np.quantile(flat, 1 - alpha)),
        "rhat": rhat(samples),
        "ess_bulk": ess_bulk(samples),
        "ess_tail": ess_tail(samples),
    }
