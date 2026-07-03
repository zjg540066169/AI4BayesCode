"""Convergence diagnostics (R-hat, ESS, posterior summary).

Python equivalents of the R `posterior` package functions used in the
R helper. Implementations follow Vehtari et al. (2021) "Rank-normalized
split-R-hat and ESS for assessing convergence of MCMC".

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


def rhat(samples: np.ndarray) -> float:
    """Rank-normalized split-R-hat (Vehtari et al. 2021).

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
    x = _split_chains(x)
    n, m = x.shape
    chain_means = x.mean(axis=0)
    chain_vars = x.var(axis=0, ddof=1)
    grand_mean = chain_means.mean()
    # Between-chain variance
    B = n * np.var(chain_means, ddof=1)
    # Within-chain variance
    W = chain_vars.mean()
    if W <= 0:
        return float("nan")
    var_hat = ((n - 1) / n) * W + B / n
    return float(np.sqrt(var_hat / W))


def _autocov(chain: np.ndarray) -> np.ndarray:
    """Unbiased autocovariance via FFT for a single chain."""
    x = chain - chain.mean()
    n = x.size
    fft_size = 1 << (2 * n - 1).bit_length()
    f = np.fft.fft(x, fft_size)
    ac = np.fft.ifft(f * np.conj(f)).real[:n]
    return ac / (n - np.arange(n))


def ess_bulk(samples: np.ndarray) -> float:
    """Bulk effective sample size (initial monotonic sequence estimator).

    Parameters
    ----------
    samples : array shape (n_draws,) or (n_draws, n_chains)
    """
    x = _as_2d(samples)
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
    var_hat = ((n - 1) / n) * W + n * np.var(chain_means, ddof=1) / n
    # Per-chain autocov, averaged
    rho_hats = np.zeros(n)
    for c in range(m):
        rho_hats += _autocov(x[:, c])
    rho_hats /= m
    rho_hats /= var_hat  # normalize to autocorrelation
    rho_hats[0] = 1.0
    # Initial positive sequence — sum pairs until they go negative
    t = 1
    tau = 1.0
    while 2 * t + 1 < n:
        pair = rho_hats[2 * t] + rho_hats[2 * t + 1]
        if pair < 0:
            break
        tau += 2 * pair
        t += 1
    return float(n * m / max(tau, 1.0))


def ess_tail(samples: np.ndarray, quantile_lo: float = 0.05,
             quantile_hi: float = 0.95) -> float:
    """Tail ESS — min of ESS at low and high quantile indicators."""
    x = _as_2d(samples)
    lo = np.quantile(x, quantile_lo)
    hi = np.quantile(x, quantile_hi)
    ind_lo = (x <= lo).astype(float)
    ind_hi = (x >= hi).astype(float)
    return float(min(ess_bulk(ind_lo), ess_bulk(ind_hi)))


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
