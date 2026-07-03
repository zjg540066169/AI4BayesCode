"""Cross-chain R-hat / ESS summary over a ``run_chains()`` result.

Python analogue of R's ``ai4bayescode_rhat_summary()``. Takes the list of
per-chain dicts returned by :func:`run_chains` (each with a ``"hist"`` key) -- or
a bare list of history dicts -- and reports, per parameter, the rank-normalized
split-R-hat and bulk-ESS pooled across chains. With a single chain this is the
within-chain split-R-hat. Matrix parameters are flagged for LABEL SWITCHING
conservatively (same rule as :func:`diagnose`): only when ordering the components
within each draw brings the max R-hat below a converged level (< 1.05).
"""
from __future__ import annotations

import numpy as np

from .utils import ess_bulk, rhat

__all__ = ["rhat_summary"]


def _hist_list(chains):
    """Accept the run_chains() output (list of dicts with 'hist') OR a list of
    raw history dicts; drop failed chains (hist is None)."""
    out = []
    for c in chains:
        h = c["hist"] if isinstance(c, dict) and "hist" in c else c
        if h is not None:
            out.append(h)
    return out


def _stack(hists, key, j, drop_burn, order):
    """Column j of `key`, stacked across chains -> (n_draws, n_chains)."""
    cols = []
    for h in hists:
        a = np.asarray(h[key], dtype=float)
        a = a.reshape(a.shape[0], -1) if a.ndim > 1 else a.reshape(-1, 1)
        if drop_burn and drop_burn < a.shape[0]:
            a = a[drop_burn:]
        if order:
            a = np.sort(a, axis=1)
        cols.append(a[:, j])
    n = min(c.shape[0] for c in cols)
    return np.column_stack([c[:n] for c in cols])


def rhat_summary(chains, keys=None, drop_burn=0, order_components=False):
    """Per-parameter cross-chain split-R-hat + bulk-ESS.

    Parameters
    ----------
    chains : list
        A ``run_chains()`` result (list of ``{"seed","wall","hist"}`` dicts) or a
        list of history dicts.
    keys : list of str, optional
        Restrict to these parameter names (default: all keys of the first chain).
    drop_burn : int
        Drop this many leading draws from each chain before summarising.
    order_components : bool
        If ``True``, sort each draw's components within every matrix key (a
        label-invariant summary for exchangeable/mixture parameters).

    Returns
    -------
    dict
        ``{param: {"rhat", "ess_bulk", "max_rhat", "min_ess"}}``. A
        ``"_label_switch"`` entry (``{param: {"raw","ordered"}}``) is added for
        matrix keys where ordering resolves an otherwise-high R-hat.
    """
    hists = _hist_list(chains)
    if not hists:
        raise ValueError("no chains with a history to summarise")
    if len(hists) == 1:
        print("rhat_summary: only 1 chain -- reporting within-chain split-R-hat "
              "(the one chain split in half); use >= 2 chains for the standard "
              "between-chain R-hat.")
    all_keys = list(hists[0].keys())
    if keys is None:
        keys = all_keys

    out = {}
    label_switch = {}
    for k in keys:
        if k not in all_keys:
            continue
        a0 = np.asarray(hists[0][k])
        dim = 1 if a0.ndim == 1 else a0.reshape(a0.shape[0], -1).shape[1]
        rh = np.array([rhat(_stack(hists, k, j, drop_burn, order_components))
                       for j in range(dim)])
        eb = np.array([ess_bulk(_stack(hists, k, j, drop_burn, order_components))
                       for j in range(dim)])

        # Label switching (matrix keys only): compare raw vs ordered max R-hat;
        # flag ONLY when ordering brings it below a converged level (< 1.05).
        if dim >= 2 and not order_components:
            rh_raw = np.array([rhat(_stack(hists, k, j, drop_burn, False))
                               for j in range(dim)])
            rh_ord = np.array([rhat(_stack(hists, k, j, drop_burn, True))
                               for j in range(dim)])
            mr, mo = float(np.nanmax(rh_raw)), float(np.nanmax(rh_ord))
            if np.isfinite(mr) and np.isfinite(mo) and mr > 1.05 and mo < 1.05:
                label_switch[k] = {"raw": mr, "ordered": mo}

        # R parity: scalar keys report only {rhat, ess_bulk}; matrix keys also
        # report {max_rhat, min_ess} across components.
        if dim == 1:
            out[k] = {"rhat": float(rh[0]), "ess_bulk": float(eb[0])}
        else:
            out[k] = {"rhat": rh, "ess_bulk": eb,
                      "max_rhat": float(np.nanmax(rh)), "min_ess": float(np.nanmin(eb))}

    if label_switch:
        out["_label_switch"] = label_switch
        if not order_components:
            nm0 = next(iter(label_switch))
            ex = label_switch[nm0]
            print(f"rhat_summary: {', '.join(label_switch)} MIGHT have LABEL SWITCHING "
                  f"-- ordering components within each draw brings R-hat to a converged "
                  f"level (e.g. {nm0}: {ex['raw']:.2f} -> {ex['ordered']:.2f}), so the "
                  f"high raw R-hat MAY be a labelling artefact rather than "
                  f"non-convergence. Pass order_components=True for a label-invariant "
                  f"summary, or canonicalize the labels in the sampler.")
    return out
