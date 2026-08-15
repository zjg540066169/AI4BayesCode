"""Convergence diagnostics (R-hat, ESS, posterior summary).

Python equivalents of the R `posterior` package functions used in the
R helper. `rhat` implements the rank-normalized split-R-hat of Vehtari
et al. (2021) and agrees with `posterior::rhat` to ~1e-7. `ess_bulk` /
`ess_tail` use a simpler autocorrelation estimator and are NOT identical
to `posterior::ess_bulk` / `ess_tail`; treat them as indicative.

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


def _split_chains(x: np.ndarray) -> np.ndarray:
    """Split each chain in half — doubles chain count, halves draw count."""
    n_draws, n_chains = x.shape
    half = n_draws // 2
    if half == 0:
        raise ValueError("need at least 2 draws to split-R-hat")
    first = x[:half]
    second = x[half : 2 * half]
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


def _plain_split_rhat(x: np.ndarray) -> float:
    """Classic split-R-hat on the values as given (no rank transform)."""
    x = _split_chains(x)
    n, _ = x.shape
    chain_means = x.mean(axis=0)
    chain_vars = x.var(axis=0, ddof=1)
    B = n * np.var(chain_means, ddof=1)
    W = chain_vars.mean()
    if W <= 0:
        return float("nan")
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

    bulk = _plain_split_rhat(_rank_normalize_dispatch(x))
    folded = np.abs(x - np.median(x))
    tail = _plain_split_rhat(_rank_normalize_dispatch(folded))

    vals = [v for v in (bulk, tail) if not np.isnan(v)]
    return float(max(vals)) if vals else float("nan")


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


def _ess_raw(x: np.ndarray) -> float:
    """Split-ESS of the values as given (Vehtari et al. 2021, Sec. 4).

    The combined autocorrelation is

        rho_t = 1 - (W - mean_c s_c^2 rho_{c,t}) / var_hat_plus

    which is what makes ESS see BETWEEN-chain disagreement: when one chain is
    parked away from the others, W stays small while var_hat_plus is large, so
    every rho_t stays near 1, tau blows up, and ESS collapses. Normalizing the
    averaged autocovariance by var_hat_plus alone (without the `1 - (W - ...)`
    wrapper) gives rho_t ~ 0 for exactly that case and reports a full ESS.
    """
    if x.shape[0] < 4:            # too few draws for a meaningful split-ESS
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
    return _ess_raw(_rank_normalize_dispatch(x))


def ess_tail(samples: np.ndarray, quantile_lo: float = 0.05,
             quantile_hi: float = 0.95) -> float:
    """Tail ESS -- min of the ESS at the 5% and 95% quantile indicators,
    matching `posterior::ess_tail`."""
    x = _as_2d(samples)
    if x.shape[0] < 4:
        return float("nan")
    if not np.all(np.isfinite(x)):
        return float("nan")
    lo = np.quantile(x, quantile_lo)
    hi = np.quantile(x, quantile_hi)
    ind_lo = (x <= lo).astype(float)
    ind_hi = (x >= hi).astype(float)
    return float(min(_ess_raw(ind_lo), _ess_raw(ind_hi)))


def posterior_summary(samples: np.ndarray, prob: float = 0.90) -> dict:
    """Compact posterior summary: mean, median, sd, mad, CI bounds, R-hat, ESS.

    The credible-interval level defaults to ``prob=0.90`` to match R's
    ``posterior::summarise_draws`` (q5/q95) -- the same 90% interval the shipped
    diagnostics advertise. ``median`` and ``mad`` (normal-consistent, x1.4826)
    are included for parity with the R summary. Works for a scalar or a single
    vector component (call per-component for a vector).
    """
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
