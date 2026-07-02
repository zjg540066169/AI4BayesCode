"""Model-independent posterior diagnostics for an AI4BayesCode chain.

Ships the diagnostics + plot used by generated
``run_chain_<Model>(diagnosis=True)`` runners, so each runner calls one
library function instead of re-emitting the code by hand. Mirrors the R
``ai4bayescode_diagnose()`` in the AI4BayesCode R package.

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


def _label_switch_scan(hist, hi=1.05, drop=0.1):
    """Flag matrix keys whose max split-R-hat is high but collapses after ordering
    each draw's components (order statistics are invariant to relabelling)."""
    from .utils import rhat
    out = {}
    for nm, x in hist.items():
        a = np.asarray(x)
        if a.ndim < 2 or a.shape[1] < 2 or a.shape[0] < 8:
            continue
        raw = max(rhat(a[:, j]) for j in range(a.shape[1]))
        xs = np.sort(a, axis=1)
        ordd = max(rhat(xs[:, j]) for j in range(a.shape[1]))
        if np.isfinite(raw) and np.isfinite(ordd) and raw > hi and (raw - ordd) > drop:
            out[nm] = {"raw": float(raw), "ordered": float(ordd)}
    return out


class _SummaryDict(dict):
    """A dict that also carries an ``.attrs`` mapping (parity with
    ``pandas.DataFrame.attrs``), so ``diagnose`` can expose ``label_switch``
    even when pandas is unavailable and the summary is a plain dict of dicts."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs = {}


def diagnose(hist, n_burn=0, plot=True, order_components=False):
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
    hb = {nm: np.asarray(x)[n_burn:] for nm, x in hist.items()}
    _lens = [np.asarray(v).shape[0] for v in hb.values()]
    if _lens and min(_lens) == 0:
        raise ValueError(
            f"diagnose: n_burn={n_burn} leaves no post-burn-in draws "
            f"(the shortest history key has <= {n_burn} draws). Reduce n_burn.")
    # Detect label switching on the RAW draws (before any ordering).
    label_switch = _label_switch_scan(hb)
    if label_switch and not order_components:
        nm0 = next(iter(label_switch)); ex = label_switch[nm0]
        print(f"diagnose: possible LABEL SWITCHING in {', '.join(label_switch)} -- the "
              f"high R-hat is a labelling artefact, NOT non-convergence (e.g. {nm0}: "
              f"{ex['raw']:.2f} -> {ex['ordered']:.2f} after ordering components within each "
              f"draw). Pass order_components=True for a label-invariant summary, or "
              f"canonicalize the labels in the sampler.")
    if order_components:
        hb = {nm: (np.sort(a, axis=1) if (a := np.asarray(x)).ndim >= 2 and a.shape[1] >= 2
                   else np.asarray(x)) for nm, x in hb.items()}
    cols = _flatten(hb, 0)   # burn-in already stripped above

    # Per-parameter numpy diagnostics. posterior_summary() uses split-R-hat
    # and an initial-sequence ESS, so it is correct for a single chain.
    rows = {nm: posterior_summary(v) for nm, v in cols.items()}
    try:
        import pandas as pd
        summary = pd.DataFrame(rows).T          # DataFrame already carries .attrs
    except Exception:
        summary = _SummaryDict(rows)            # dict-of-dicts + an .attrs mapping
    # Expose label switching the SAME way with or without pandas (parity with the
    # R ai4bayescode_diagnose()$label_switch): attach it to summary.attrs so callers can
    # read it even when pandas is absent (a bare dict would otherwise drop it).
    summary.attrs["label_switch"] = label_switch

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

