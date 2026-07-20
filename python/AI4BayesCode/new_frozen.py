"""Ctor helper: construct wrapper + set + freeze in one call.

DESIGN_NOTES_FREEZE_UNFREEZE_2026-07-19.md Sec.10.e.

Equivalent to the two-step form:
    m = module_class(*args, **kwargs)
    m.set_current(fixed)
    m.freeze(list(fixed.keys()), quiet=quiet_freeze)

Scope: `fixed` keys must be FLAT (top-level child block names or
joint_nuts_block slot names -- whatever set_current() accepts).
Dot-path names (nested composite descent, rjmcmc sub-key) are REJECTED
because set_current does not route dot-path keys -- freezing a value
that was never actually set would silently produce wrong posteriors.
For dot-path freezes, use the two-step post-construction form with
set_current called at the correct composite level.

RESERVED KWARGS: `fixed`, `quiet_freeze`. If the wrapper's own __init__
takes a kwarg with either of these names, use the two-step form
(module_class(...) then m.set_current(...) then m.freeze(...)) instead.
"""


def new_frozen(module_class, *args, fixed=None, quiet_freeze=True, **kwargs):
    """Construct a wrapper and immediately fix + freeze specified sub-kernels.

    Args:
        module_class: pybind11 wrapper class (e.g. mod.MyGaussianReg).
        *args: forwarded to module_class(...).
        fixed: mapping (dict-like) of {name: value} to set + freeze.
               Must be flat names (no dot-paths).
        quiet_freeze: passed as `quiet` to freeze() -- default True since
                      no prior state exists to warn about at ctor time.
        **kwargs: forwarded to module_class(...).

    Returns:
        The constructed wrapper instance.
    """
    if fixed is None:
        fixed = {}
    if fixed:
        for k in fixed:
            if not isinstance(k, str):
                raise TypeError(
                    "AI4BayesCode.new_frozen(): `fixed` keys must be strings "
                    f"(got {type(k).__name__})")
        bad = [k for k in fixed if "." in k]
        if bad:
            raise ValueError(
                "AI4BayesCode.new_frozen(): dot-path names not allowed in "
                f"`fixed` ({bad}). Use post-construction m.set_current(...) "
                "at the correct composite level + m.freeze([<dot.path>]) "
                "instead.")
    m = module_class(*args, **kwargs)
    if fixed:
        m.set_current(dict(fixed))
        m.freeze(list(fixed.keys()), quiet=bool(quiet_freeze))
    return m
