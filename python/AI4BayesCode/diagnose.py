"""Model-independent posterior diagnostics for an AI4BayesCode chain.

Ships the diagnostics + plot used by generated
``run_chain_<Model>(diagnosis=True)`` runners, so each runner calls one
library function instead of re-emitting the code by hand. Mirrors the R
``ai4b_diagnose()`` in the AI4BayesCode R package.

Per parameter it reports split-R-hat, bulk / tail ESS, and posterior
mean / sd / 90% CI (all numpy, via utils.py), plus a combined trace +
autocorrelation + density plot (matplotlib). No PSIS-LOO -- that needs a
model-specific pointwise log-likelihood.

Diagnostics are computed with the package's own numpy helpers, NOT ArviZ:
a generated runner produces ONE chain, and split-R-hat / initial-sequence
ESS are the right single-chain diagnostics (ArviZ's ``az.summary`` reports
R-hat = NaN for a single chain, and a partially-broken ArviZ install can
import yet fail at plot time). matplotlib is the only plotting dependency,
and it is optional -- without it the summary table is still returned.
"""

from __future__ import annotations

import numpy as np

from .utils import posterior_summary


def _flatten(hist, n_burn=0):
    """Flatten a named draw dict into ``{col_name: 1-D ndarray}``.

    Scalars -> ``"name"``; vector params -> ``"name[0]"``, ``"name[1]"``,
    ... The first ``n_burn`` draws are dropped (``get_history()`` returns
    burn-in + keep; pass ``n_burn=0`` when the draws are already stripped).
    """
    cols = {}
    for nm, x in hist.items():
        a = np.asarray(x)[n_burn:]
        if a.ndim == 1:
            cols[nm] = a
        else:
            a2 = a.reshape(a.shape[0], -1)
            for j in range(a2.shape[1]):
                cols[f"{nm}[{j}]"] = a2[:, j]
    return cols


def _acf(v, nlags):
    """Normalized autocorrelation of a 1-D series up to ``nlags`` lags."""
    vc = v - v.mean()
    ac = np.correlate(vc, vc, mode="full")[len(vc) - 1:]
    if ac[0] == 0:
        return np.zeros(nlags + 1)
    ac = ac / ac[0]
    return ac[:nlags + 1]


def ai4b_diagnose(hist, n_burn=0, plot=True):
    """Compute draws-only diagnostics and a trace / ACF / density plot.

    Parameters
    ----------
    hist : dict[str, numpy.ndarray]
        Named posterior draws (a runner's ``out["hist"]`` or
        ``model.get_history()``). Scalars are ``(n_draws,)``; vector
        parameters are ``(n_draws, dim)``.
    n_burn : int
        Number of leading draws to drop. ``get_history()`` includes
        burn-in, so pass the burn-in length there; use ``0`` when ``hist``
        is already burn-in-stripped.
    plot : bool
        If ``True`` (default) also build the diagnostic plot.

    Returns
    -------
    (summary, plot_fn) : tuple
        ``summary`` is a pandas DataFrame (one row per parameter:
        mean, sd, ci_lower, ci_upper, rhat, ess_bulk, ess_tail), or a dict
        of dicts when pandas is unavailable. ``plot_fn`` is a zero-argument
        callable that draws trace + autocorrelation + density panels with
        matplotlib and returns the Figure, or ``None`` when ``plot=False``
        or matplotlib is not installed.
    """
    cols = _flatten(hist, n_burn)

    # Per-parameter numpy diagnostics. posterior_summary() uses split-R-hat
    # and an initial-sequence ESS, so it is correct for a single chain.
    rows = {nm: posterior_summary(v) for nm, v in cols.items()}
    try:
        import pandas as pd
        summary = pd.DataFrame(rows).T
    except Exception:
        summary = rows  # dict of dicts when pandas is unavailable

    plot_fn = None
    if plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            print("Install 'matplotlib' for diagnostic plots "
                  "(pip install matplotlib); returning the summary table only.")
        else:
            def plot_fn():
                names = list(cols)
                k = len(names)
                fig, axes = plt.subplots(k, 3, figsize=(11, 2.4 * k),
                                         squeeze=False)
                for i, nm in enumerate(names):
                    v = cols[nm]
                    axes[i, 0].plot(v, lw=0.6)
                    axes[i, 0].set_title(f"trace: {nm}")
                    lag = min(40, len(v) - 1)
                    ac = _acf(v, lag)
                    axes[i, 1].bar(range(len(ac)), ac, width=0.4)
                    axes[i, 1].set_title(f"ACF: {nm}")
                    axes[i, 2].hist(v, bins=40, density=True)
                    axes[i, 2].set_title(f"density: {nm}")
                fig.tight_layout()
                return fig

    return summary, plot_fn
