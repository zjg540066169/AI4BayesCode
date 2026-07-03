"""ai4bayescode.doc(): a usage card for a sampler compiled by
source().

Mirrors the R-side ai4bayescode_doc(). pybind11 already gives named/defaulted
constructor args via ``help(mod.MyModel)``, but the model description and the
method semantics live in the ``.cpp`` -- this parses them into one card.

Accepts the loaded module, the loaded class, the class name, or a ``.cpp``
path. ``source()`` registers the class -> source mapping so
``AI4BayesCode.doc(mod)`` just works.
"""

from __future__ import annotations

import os
import re
from types import ModuleType

__all__ = ["doc"]

# class/module name -> source .cpp path (populated by source()).
_DOC_REGISTRY: dict[str, str] = {}


def _register_source(cpp_path: str) -> list[str]:
    try:
        src = open(cpp_path, "r").read()
    except OSError:
        return []
    names = _exposed_names(src)
    for nm in names:
        _DOC_REGISTRY[nm] = os.path.abspath(cpp_path)
    return names


def _class_names(src: str) -> list[str]:
    """The actual exposed CLASS names from `class_<X>(m, "Name")` -- this is
    what `mod.<Name>(...)` uses (NOT the PYBIND11_MODULE name, which may carry
    a `_module` suffix)."""
    return list(dict.fromkeys(
        re.findall(r'class_<[^>]+>\s*\(\s*(?:m\s*,\s*)?"([^"]+)"', src)))


def _exposed_names(src: str) -> list[str]:
    """All names that should resolve back to this source: the class names AND
    the PYBIND11_MODULE name (so doc(mod) works when they differ)."""
    names = list(_class_names(src))
    for m in re.finditer(r"PYBIND11_MODULE\(\s*([A-Za-z_]\w*)", src):
        names.append(m.group(1))
    return list(dict.fromkeys(names))


def _type_hint(t: str) -> str:
    t = re.sub(r"const|&|\s+", " ", t).strip()
    t = re.sub(r"\s+", " ", t)
    has = lambda p: re.search(p, t) is not None
    if has(r"arma::mat|Mat<"):                                   return "2-D numpy array"
    if has(r"arma::ivec|arma::uvec|Col<arma::(sword|uword)"):    return "1-D int array"
    if has(r"arma::vec|Col<double|NumericVector"):               return "1-D numpy array"
    if has(r"NumericMatrix|IntegerMatrix"):                      return "2-D numpy array"
    if has(r"IntegerVector"):                                    return "1-D int array"
    if has(r"\bbool\b"):                                         return "bool"
    if has(r"\bint\b|size_t|uint"):                              return "int"
    if has(r"double|float"):                                    return "float"
    if has(r"std::string|String"):                               return "str"
    return t


def _args_after(src: str, open_idx: int):
    """Balance-match the ')' for the '(' at open_idx; parse the param list."""
    depth = 0
    end = None
    for i in range(open_idx, len(src)):
        c = src[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        return None
    inside = re.sub(r"\s+", " ", src[open_idx + 1:end].strip())
    if not inside:
        return []
    parts, buf, ang = [], "", 0
    for ch in inside:
        if ch == "<":
            ang += 1
        elif ch == ">":
            ang -= 1
        if ch == "," and ang == 0:
            parts.append(buf); buf = ""
        else:
            buf += ch
    parts.append(buf)
    args = []
    for p in parts:
        p = p.strip()
        default = None
        if "=" in p:
            sp = p.split("=")
            default = "=".join(sp[1:]).strip()
            p = sp[0].strip()
        toks = p.split()
        if not toks:
            continue
        name = re.sub(r"[&*]", "", toks[-1])
        typ = p[:p.rfind(name)].strip() if name and name in p else p
        args.append({"name": name, "type": typ, "hint": _type_hint(typ), "default": default})
    return args


def _parse_constructor(src: str, class_name: str):
    if not class_name:
        return None
    m = re.search(r"(?m)^[ \t]*" + re.escape(class_name) + r"[ \t]*\(", src)
    if not m:
        return None
    return _args_after(src, m.end() - 1)


def _parse_methods(src: str) -> list[str]:
    out: list[str] = []
    for m in re.finditer(r'\.(?:def|method)\(\s*"([^"]+)"', src):
        if m.group(1) not in out:
            out.append(m.group(1))
    return out


def _parse_header(src: str) -> list[str]:
    out: list[str] = []
    for ln in src.split("\n"):
        s = ln.strip()
        if s.startswith("#") or s.startswith("// [[") or re.match(r"//\s*@example:", s):
            break
        if s.startswith("//"):
            out.append(re.sub(r"^//\s?", "", s))
        elif s:
            break
    out = [d for d in out if not re.search(
        r"Copyright|Licensed under|GPL|See COPYING|^=+$|^-+$", d)]
    while out and not out[0].strip():
        out.pop(0)
    return out[:18]


def _parse_example(src: str, lang: str = "python"):
    """Extract a language-tagged runnable example from the header doc-block:
    ``// @example:R ... // @example:python ... // @example:end``. Returns the
    chosen language's lines (// prefix stripped), or None if absent."""
    lines = src.split("\n")
    open_re = re.compile(r"^\s*//\s*@example:" + re.escape(lang) + r"\b", re.I)
    tag_re = re.compile(r"^\s*//\s*@example:")
    cmt_re = re.compile(r"^\s*//")
    st = next((i for i, ln in enumerate(lines) if open_re.search(ln)), None)
    if st is None:
        return None
    out: list[str] = []
    for ln in lines[st + 1:]:
        if tag_re.search(ln):       # next @example:* tag or :end
            break
        if not cmt_re.search(ln):   # left the comment block
            break
        out.append(re.sub(r"^\s*//\s?", "", ln))
    while out and not out[-1].strip():
        out.pop()
    return out or None


_CANONICAL = {
    "step":         "step(n_steps)            advance the sampler n_steps iterations",
    "get_current":  "get_current()            current draw of each parameter -> dict",
    "set_current":  "set_current(params)      set the current draw (dict)",
    "predict_at":   "predict_at(new_data)     posterior prediction at new_data",
    "get_dag":      "get_dag()                model DAG (feed to plot_dag)",
    "get_history":  "get_history()            all kept draws (needs keep_history=True)",
    "readapt_NUTS": "readapt_NUTS(n, adapt)   re-run NUTS warm-up (online / sequential use)",
}


def doc(x) -> None:
    """Print a usage card for an AI4BayesCode sampler.

    `x` may be the loaded module, the loaded class, the class name (str), or a
    path to the `.cpp` source.
    """
    src_path = None
    class_name = None
    if isinstance(x, str) and x.endswith(".cpp") and os.path.isfile(x):
        src_path = os.path.abspath(x)
    elif isinstance(x, str):
        class_name = x
    elif isinstance(x, ModuleType):
        class_name = getattr(x, "__name__", None)
    else:
        class_name = getattr(x, "__name__", None) or type(x).__name__

    if class_name and src_path is None:
        src_path = _DOC_REGISTRY.get(class_name)

    # Bundled-example fallback: resolve <name>.cpp from the vendored examples,
    # so doc("BartNoise") shows the full card for any shipped example -- not just
    # classes that have been compiled this session.
    if class_name and (not src_path or not os.path.isfile(src_path)):
        import importlib.resources
        from pathlib import Path as _Path
        cand = _Path(str(importlib.resources.files("AI4BayesCode"))) / "_examples" / f"{class_name}.cpp"
        if cand.is_file():
            src_path = str(cand)

    if not src_path or not os.path.isfile(src_path):
        print(f"  {class_name or x}: source not found — pass the .cpp path for the full "
              f"card, or use help(mod.{class_name or 'YourModel'}) for the pybind "
              f"signature.")
        return None

    src = open(src_path, "r").read()
    # Use the actual exposed CLASS name for the card (the passed name may be the
    # PYBIND11_MODULE name, e.g. "<Class>_module").
    classes = _class_names(src)
    if class_name not in classes:
        class_name = classes[0] if classes else class_name

    ctor = _parse_constructor(src, class_name)
    methods = _parse_methods(src)
    desc = _parse_header(src)
    ex_block = _parse_example(src, "python")
    bar = "─" * 64

    print(bar)
    print(f"  {class_name}")
    if desc:
        print(bar)
        for d in desc:
            print(f"  {d}")
    print(bar)

    if ctor:
        print("Construct:")
        print(f"  m = mod.{class_name}(")
        for i, a in enumerate(ctor):
            tail = f"  [optional, default {a['default']}]" if a["default"] is not None else ""
            comma = "," if i < len(ctor) - 1 else ")"
            print(f"      {a['name']:<14}{comma}   # {a['hint']}{tail}")
    else:
        print(f"Construct:  mod.{class_name}(...)   "
              f"(could not parse args — try help(mod.{class_name}))")

    print("\nMethods:")
    shown = []
    for nm, line in _CANONICAL.items():
        if methods and nm not in methods:
            continue
        print(f"  .{line}")
        shown.append(nm)
    for nm in methods:
        if nm not in shown:
            print(f"  .{nm}()")

    if ex_block:
        # Author-supplied runnable DGP from the .cpp header (@example:python).
        print("\nExample:")
        for ln in ex_block:
            print(ln)
    elif ctor:
        # Fallback: auto-generated mod.<Class>(...) stub (placeholder args).
        req = [a["name"] for a in ctor if a["default"] is None]
        print("\nExample:")
        print(f"  m = mod.{class_name}({', '.join(req)})")
        print("  m.step(2000); print(m.get_current())")
    print(bar)

    # doc() is a DISPLAY function -- the formatted card was just printed above.
    # Return None so an interactive `doc(x)` in IPython/Jupyter does NOT echo the
    # raw dict after the card (mirrors R's ai4bayescode_doc() -> invisible(list())).
    return None
