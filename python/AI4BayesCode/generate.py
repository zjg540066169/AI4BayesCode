"""NL -> validated sampler code (Python front doors), mirroring the R side.

    AI4BayesCode.prompt(model_description, ...)    -- pure prompt builder
    AI4BayesCode.generate(model_description, ...)  -- console-interactive NL->code
    AI4BayesCode.models()                          -- supported LLM registry

Design (2026-06-21): LLM-agnostic (the skill corpus works with any LLM); `LLM`
resolves through models() to a (provider, model id); Claude/Anthropic is the
implemented illustration, others are an extension point. `effort` is a separate
knob (default "high"; low -> low-quality samplers). Interactive-if-missing: any
None/empty argument is asked in the console; PRIORS are elicited interactively by
the model via an `ask_user` tool. The online API path uses the ``anthropic``
SDK (a core dependency, installed automatically); with no key the prompt
is written to ``output_path/PROMPT.txt``.

Validation is MANDATORY and runs a validate -> repair-to-convergence loop: each
generation attempt is compiled (``AI4BayesCode.source``) and its emitted Python
runner is executed; the runner MUST print ``AI4BAYES_VALIDATE: PASS`` as its final
line (max rank-normalized R-hat < 1.01 across two chains). On any failure the
compile / run output is fed back to the model for repair, up to ``max_attempts``
times.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

__all__ = ["prompt", "generate", "models", "skills_path", "set_key", "key_status"]


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------
def skills_path(name: str | None = None) -> str:
    """Absolute path to the bundled skills directory, or to a named skill file.

    Python analogue of R's ``ai4bayescode_skills_path()``. Mirrors R's
    ``system.file()`` contract: returns ``""`` (empty string) when the resolved
    directory (``name is None``) or file does not exist. Returns a ``str``.
    """
    import importlib.resources
    pkg = Path(str(importlib.resources.files("AI4BayesCode")))
    if name == "start.md":   # entry-point doc at the package root, not in _skills/
        p = pkg / "start.md"
    else:
        base = pkg / "_skills"
        p = base if name is None else base / name
    return str(p) if p.exists() else ""


def _skill(name: str) -> str:
    p = skills_path(name)
    if not p:
        return ""
    try:
        return Path(p).read_text()
    except OSError:
        return ""


def _default_skill_set(backend: str) -> list[str]:
    base = ["start.md", "codegen.md", "codegen_cpp.md", "block_catalogue/index.md",
            "constraints.md", "codegen_priors.md", "rcpp_api.md", "validator.md"]
    runner = {"R": ["codegen_r_runner.md"], "Python": ["codegen_python_runner.md"],
              "both": ["codegen_r_runner.md", "codegen_python_runner.md"]}[backend]
    out: list[str] = []
    for s in base + runner:
        if s not in out:
            out.append(s)
    return out


_CLASSNAME_FILLER = {
    "need", "needs", "want", "wants", "would", "like", "please", "fit", "fits", "fitting",
    "sample", "sampler", "model", "models", "the", "this", "that", "using", "use", "with",
    "for", "data", "have", "has", "give", "make", "build", "create", "code", "generate",
    "implement", "and", "from", "our", "your", "some", "just",
}


def _derive_class_name(description: str) -> str:
    d = description or ""
    # A filesystem path (a slash, no spaces) is NOT a model description -- deriving
    # from its components yields garbage like "UsersJz3138Documents".
    if re.search(r"[/\\]", d) and not re.search(r"\s", d.strip()):
        return "GeneratedModel"
    # Use only the PROSE before the first formula delimiter (`:` `~` `=` `(`), and
    # drop filler words, so "I need a linear regression: y ~ ..." -> "LinearRegression".
    d0 = re.split(r"[:~=(]", d, maxsplit=1)[0]
    words = [w for w in re.findall(r"[A-Za-z][A-Za-z0-9]*", d0)
             if len(w) >= 3 and w.lower() not in _CLASSNAME_FILLER][:3]
    if not words:
        return "GeneratedModel"
    nm = "".join(w[:1].upper() + w[1:] for w in words)
    if not re.match(r"[A-Za-z_]", nm):
        nm = "M_" + nm
    return nm[:60]


def _prior_block(priors) -> str:
    if priors == "interactive":
        return (
"  - Priors: FIRST use every prior the model description ALREADY specifies\n"
"    -- e.g. `p(beta) propto 1`, `p(sigma) propto 1/sigma`, `beta ~ N(0, 10)`,\n"
"    `sigma ~ Half-Normal(2.5)` -- EXACTLY, and do NOT ask about them (a written\n"
"    prior IS the user's decision). ELICIT (via the `ask_user` tool, ONE\n"
"    parameter at a time: non-informative (default) / weakly-informative /\n"
"    literature-informed / fixed value / custom) ONLY the priors that are\n"
"    MISSING or genuinely AMBIGUOUS (e.g. Gamma(2,3) rate-vs-scale). If the\n"
"    description already specifies EVERY parameter's prior, ask NOTHING and go\n"
"    straight to code (codegen.md §2: 'Spec already COMPLETE -> skip the entire\n"
"    elicitation'). (See codegen_priors.md.)")
    if isinstance(priors, dict):
        lines = "\n".join(f"    - {k}: {v}" for k, v in priors.items())
        return ("  - Priors: use these EXACTLY where given; otherwise a strictly\n"
                "    NON-INFORMATIVE prior.\n" + lines)
    if priors == "weakly":
        return ("  - Priors: weakly-informative defaults are acceptable where the model\n"
                "    description does not specify a prior.")
    return (
"  - Priors: where the model description specifies a prior, use it EXACTLY; for any\n"
"    parameter whose prior is NOT specified, use a strictly NON-INFORMATIVE prior\n"
"    (per-support defaults -- flat improper on real coefs, Jeffreys 1/sigma on\n"
"    positive scales, Beta(1,1) / Dirichlet(1..1) / LKJ(1) on probability / simplex /\n"
"    correlation, Jeffreys on count-rate & concentration -- with the full rationale\n"
"    in codegen_priors.md). Apply them SILENTLY; do not stop to ask.")


def _build_user(description, backend, output_path, classname, priors, max_attempts,
                confirm_model=False):
    runner = ""
    if backend in ("R", "both"):
        runner += f"  - {output_path}/{classname}_runner.R             (R runner / Layer-3 harness)\n"
    if backend in ("Python", "both"):
        runner += f"  - {output_path}/{classname}_runner.py            (Python runner)\n"
    if backend == "both":
        runner += ("  - backend=both -> the ONE .cpp above must contain BOTH an RCPP_MODULE\n"
                   "    AND a PYBIND11_MODULE, each under its `#ifdef AI4BAYESCODE_RCPP_MODULE`\n"
                   "    / `#ifdef AI4BAYESCODE_PYBIND_MODULE` guard, so the single file compiles\n"
                   "    from R AND Python (dual-module -- see codegen_cpp.md / examples/ODE_SIR.cpp).\n")
    if confirm_model:
        confirm_block = (
"  - PRE-GENERATION MODEL CONFIRMATION -- do NOT skip it (codegen.md §3). After\n"
"    eliciting the priors and BEFORE writing ANY code, present the model AS YOU\n"
"    UNDERSTOOD IT for the user to verify, via the `ask_user` tool. ALWAYS include both:\n"
"      (a) the FULL model as display-math formulas (codegen.md §0b: `$$ ... $$`, never\n"
"          inline `$...$`) -- the likelihood AND every prior; and\n"
"      (b) a parameter summary table -- name / role / support / prior.\n"
"    Do NOT render a DAG.\n"
"    Ask the user to sign off, offering exactly: 'Correct -- generate the sampler' /\n"
"    'Not quite -- I will correct it'. If they do NOT confirm, ask what to change, apply\n"
"    it, and re-confirm. Only AFTER sign-off do you write any code; then emit the\n"
"    COMPLETE files in that next message (do not defer).\n")
    else:
        confirm_block = (
"  - SKIP the model-confirmation step (no LaTeX summary / DAG image). After any prior\n"
"    elicitation, emit the COMPLETE files immediately -- do not pause to summarize.\n")
    return (
"You are deploying the AI4BayesCode library to GENERATE A SAMPLER for the model\n"
"described at the end of this message.\n\n"
"TOOLS -- you have `read_file`, `grep`, and `glob` over the INSTALLED AI4BayesCode\n"
"package (examples/, skills/, include/). The skills reference worked reference\n"
"implementations ('see examples/GaussianLocationScale.cpp') and headers: READ the\n"
"relevant example with `read_file` BEFORE writing the .cpp (its class shape,\n"
"set_current, predict_at, get_current, the block config are all there), and read\n"
"the on-demand skills (skills/system_design.md, skills/joint_nuts_failure.md,\n"
"skills/hierarchical_re.md, skills/label_switching.md) when the model calls for\n"
"them. Do NOT invent an API you can read -- `grep` the headers/examples first.\n\n"
"Settings:\n"
f"  - Runtime backend: {backend}\n"
f"  - Output folder:   {output_path}/\n"
f"  - Class name:      {classname} (valid C++ identifier). Use this name; but if it is\n"
"    generic (e.g. GeneratedModel) or does not describe the model, REPLACE it with a\n"
"    short descriptive PascalCase name (e.g. BayesianLinearRegression, PoissonGLMM). Use\n"
"    the SAME name in the class, the module (RCPP_MODULE / PYBIND), and the `// path:` files.\n"
f"{_prior_block(priors)}\n"
f"{confirm_block}"
f"  - Up to {max_attempts} total generation attempts; iterate on failures.\n"
"\nDeliverables (emit as fenced code blocks labelled `// path: <out>/<file>`):\n"
f"  - {output_path}/{classname}.cpp                  (the sampler)\n"
f"{runner}"
"  - Register a y_rep stochastic refresher for the observation likelihood\n"
"    (MANDATORY -- Layer-3 R3 Bayesian p-values need posterior-predictive draws).\n"
"  - Code comments in English only.\n"
"  (Compile pitfalls -- set_current takes a concatenated arma::vec not a map;\n"
"   Rcpp/pybind11 do not fill C++ default args -- and every other API detail are in\n"
"   codegen_cpp.md + the reference examples; READ them, do not guess.)\n"
"\nVALIDATION PROTOCOL (how the generator detects success -- keep this EXACT):\n"
"  - Emit a self-contained runner whose FULL structure is in codegen_python_runner.md\n"
"    / codegen_r_runner.md (FOLLOW it): simulate data at known values; run TWO\n"
"    over-dispersed chains, each doing step -> readapt_NUTS -> step for a\n"
"    NUTS/joint_nuts model; rank-normalized R-hat across the chains. Its VERY LAST\n"
"    printed line MUST be EXACTLY one of:\n"
"        AI4BAYES_VALIDATE: PASS                    (max rank-R-hat < 1.01)\n"
"        AI4BAYES_VALIDATE: FAIL maxRhat=<value>    (otherwise)\n"
"  - The generator greps stdout for `AI4BAYES_VALIDATE: PASS`; on ANY miss (compile\n"
"    error, runner error, or non-convergence) it feeds the output back and asks you\n"
"    to FIX and RE-EMIT the FULL .cpp and runner.\n"
"\n---\nModel description:\n" + description + "\n")


# ---------------------------------------------------------------------------
# prompt()
# ---------------------------------------------------------------------------
def prompt(model_description: str, *, backend: str = "both", output_path: str = "./generated",
           classname: str | None = None, priors="noninformative",
           max_attempts: int = 5,
           include_skills: bool = False, skills: Iterable[str] | None = None,
           confirm_model: bool = False) -> dict:
    """Assemble the codegen system + user prompt (pure, no network)."""
    backend = {"r": "R", "python": "Python", "both": "both"}.get(str(backend).strip().lower(), backend)
    if backend not in ("R", "Python", "both"):
        raise ValueError("backend must be 'R', 'Python', or 'both'")
    if isinstance(model_description, str) and os.path.isfile(model_description) and \
            model_description.endswith((".txt", ".md")):
        model_description = Path(model_description).read_text()
    if not isinstance(model_description, str) or not model_description.strip():
        raise ValueError("model_description must be a non-empty model description")
    if classname is None:
        classname = _derive_class_name(model_description)

    framing = (
"You are an expert Bayesian statistician + C++ engineer using the AI4BayesCode\n"
"header-only library to generate a stateful, modular MCMC sampler. Follow the\n"
"AI4BayesCode workflow exactly.\n\n")

    if include_skills:
        used = list(skills) if skills is not None else _default_skill_set(backend)
        blocks = "".join(f"\n\n===== SKILL: {s} =====\n{_skill(s)}" for s in used if _skill(s))
        system = framing + "Skill corpus (read in order):" + blocks
    else:
        used = ["start.md"]
        system = (framing + "Entry point (read FIRST; load further skills on demand):\n\n"
                  "===== SKILL: start.md =====\n" + _skill("start.md"))

    return {"system": system,
            "user": _build_user(model_description, backend, output_path, classname,
                                priors, max_attempts, confirm_model),
            "classname": classname, "backend": backend, "skills": used}


# ---------------------------------------------------------------------------
# Offline emit + code extraction
# ---------------------------------------------------------------------------
def _offline_emit(p: dict, output_path: str, verbose: bool) -> dict:
    out = Path(output_path); out.mkdir(parents=True, exist_ok=True)
    pf = out / "PROMPT.txt"
    pf.write_text("===== SYSTEM =====\n" + p["system"] + "\n\n===== USER =====\n" + p["user"])
    rf = out / "README_generate.txt"
    rf.write_text(
        "No API key was available, so generate() emitted the prompt.\n"
        "  (A) Claude Code: \"Read AI4BayesCode/start.md first, then <model description>.\"\n"
        "  (B) Online: set a key once with AI4BayesCode.set_key(\"sk-YOUR-KEY-HERE\", \"anthropic\")\n"
        "      (or pass API_key= / LLM=), then re-run AI4BayesCode.generate(...).\n"
        "      (The Messages API path uses the 'anthropic' SDK, a core dependency.)\n"
        f"Target class: {p['classname']}   backend: {p['backend']}\n")
    if verbose:
        print(f"No API key -- wrote prompt to:\n  {pf}\n  {rf}")
    return {"prompt_path": str(pf), "readme_path": str(rf), "prompt": p, "called_api": False}


def _extract_code(text: str) -> list[dict]:
    out = []
    for m in re.finditer(r"```[A-Za-z]*\n(.*?)```", text, re.DOTALL):
        body = m.group(1)
        lines = body.splitlines()
        path = None
        marker_idx = None
        for i, hl in enumerate(lines[:3]):
            pm = re.search(r"path:\s*(\S+)", hl)
            if pm:
                path = pm.group(1)
                # Strip the routing-marker line so it never leaks into the written
                # file -- ONLY when it is a comment-style directive (`// path:` or
                # `# path:`), never a genuine code line. A `// path:` header is a
                # harmless C++ comment but an R/Python SYNTAX ERROR, so writing it
                # verbatim crashes any .R/.py runner whose marker used `//`.
                if re.match(r"^\s*(//+|#+)\s*path:", hl):
                    marker_idx = i
                break
        if marker_idx is not None:
            lines = lines[:marker_idx] + lines[marker_idx + 1:]
        out.append({"path": path, "code": "\n".join(lines)})
    return out


# ---------------------------------------------------------------------------
# Validation: compile + run the emitted runner, detect convergence
# ---------------------------------------------------------------------------
def _validate(cpp_path, runner_path, classname, verbose: bool = False) -> dict:
    """Compile the emitted .cpp, run its Python runner, grep for the sentinel.

    Returns ``{ok, stage, detail}`` where ``stage`` is one of "compile",
    "no_runner", "convergence", or "converged" (mirror of the R ``.ai4b_validate``).
    """
    # ---- 1. compile ----
    try:
        from .source import source
        source(cpp_path, rebuild=True, quiet=True)
    except Exception as e:  # noqa: BLE001
        if verbose:
            print("  [validate] compile failed")
        return {"ok": False, "stage": "compile", "detail": str(e)}

    # ---- 2. runner present? ----
    if not runner_path or not Path(runner_path).is_file():
        if verbose:
            print("  [validate] no runner emitted")
        return {"ok": False, "stage": "no_runner", "detail": "no runner emitted"}

    # ---- 3. run the runner; capture stdout+stderr ----
    # Run it FROM the output dir: the runner compiles its .cpp via a RELATIVE
    # `source("<Class>.cpp")` (the start.md convention), so it must execute with
    # the .cpp in its working dir or it errors "no such file" (which the sentinel
    # logic below would otherwise mislabel a "runtime crash").
    try:
        proc = subprocess.run([sys.executable, str(runner_path)],
                              cwd=str(Path(runner_path).parent),
                              capture_output=True, text=True)
        out_txt = (proc.stdout or "") + (proc.stderr or "")
    except Exception as e:  # noqa: BLE001
        out_txt = f"runner error: {e}"

    # ---- 4. convergence sentinel ----
    lines = out_txt.splitlines()
    if re.search(r"AI4BAYES_VALIDATE:\s*PASS", out_txt):
        if verbose:
            print("  [validate] converged")
        return {"ok": True, "stage": "converged", "detail": "\n".join(lines[-15:])}
    # No PASS sentinel -> distinguish WHY, rather than always blaming "convergence".
    detail = "\n".join(lines[-40:])
    if re.search(r"AI4BAYES_VALIDATE:\s*FAIL", out_txt):       # ran to the end; R-hat too high
        if verbose:
            print("  [validate] ran but did not converge")
        return {"ok": False, "stage": "convergence", "detail": detail}
    if re.search(r"Traceback \(most recent|^Error:|Segmentation fault|terminate called|Abort|runner error",
                 out_txt, re.M):                               # compiled but crashed at RUNTIME
        if verbose:
            print("  [validate] runner crashed at runtime")
        return {"ok": False, "stage": "runtime", "detail": detail}
    if verbose:
        print("  [validate] runner did not reach the validate line")
    return {"ok": False, "stage": "incomplete", "detail": detail}


# Default validator, referenced inside generate() where the `_validate` parameter
# shadows the module-level name.
_DEFAULT_VALIDATE = _validate


def _repair_msg(result: dict) -> str:
    """Build the repair user-message fed back to the model after a failed attempt."""
    if result.get("stage") == "no_code":
        return (
"No sampler code was emitted. " + result.get("detail", "") + "\n\n"
"Output the COMPLETE files NOW, and NOTHING else:\n"
"  - the full `.cpp` sampler, and\n"
"  - the full runner.\n"
"Emit EACH as its own fenced code block whose FIRST line is exactly\n"
"`// path: <out>/<file>` (e.g. `// path: ./generated/Model.cpp`). Do NOT write\n"
"any prose, plan, or confirmation before, between, or after the blocks -- output\n"
"ONLY the fenced code blocks. Keep the code complete but concise so it is not cut\n"
"off. The runner MUST print `AI4BAYES_VALIDATE: PASS` (or\n"
"`AI4BAYES_VALIDATE: FAIL maxRhat=<v>`) as its very last line.\n")
    hint = {
        "compile": "The C++ did NOT COMPILE. Fix the compile error below.",
        "runtime": "The sampler COMPILED but the runner CRASHED at RUNTIME -- this is NOT a convergence problem; fix the C++/runner bug below.",
        "incomplete": (
            "The runner file WAS written and executed, but its output did not contain "
            "the `AI4BAYES_VALIDATE` line. This is NOT a file-path / `// path:` / fence "
            "problem (the runner was found and run) -- do NOT change the markers. Find "
            "why the runner did not run to its final "
            "`print(\"AI4BAYES_VALIDATE: ...\")`: an earlier crash, a hang, or (if the "
            "detail below is EMPTY) the runner subprocess could not start in this "
            "environment."),
        "convergence": "The runner ran to completion but rank-R-hat was too high (true non-convergence) -- fix the sampler/model.",
    }.get(result.get("stage"), "The generated sampler did not pass automated validation.")
    return (
f"Validation FAILED at stage `{result.get('stage')}`. {hint}\n"
"Details below:\n\n"
f"-----\n{result.get('detail', '')}\n-----\n\n"
"Please FIX the problem and RE-EMIT the FULL corrected files (do NOT send a diff\n"
"or only the changed lines). Emit the complete `.cpp` sampler AND the complete\n"
"Python runner, each as a fenced code block whose first line is\n"
"`// path: <out>/<file>` (the same paths as before). The runner MUST still print\n"
"`AI4BAYES_VALIDATE: PASS` (or `AI4BAYES_VALIDATE: FAIL maxRhat=<v>`) as its very\n"
"last line so the generator can re-check convergence.\n")


# ---------------------------------------------------------------------------
# LLM registry + resolution (LLM-agnostic)
# ---------------------------------------------------------------------------
def models() -> list[dict]:
    """Supported LLMs: name, provider, model_id, implemented, effort_levels.

    `effort_levels` are valid PER MODEL (xhigh is Opus 4.7+; max not on Haiku;
    Haiku 4.5 has no effort knob -> empty; OpenAI uses minimal..high)."""
    return [
        {"name": "claude-opus-4-8", "provider": "anthropic", "model_id": "claude-opus-4-8", "implemented": True, "effort_levels": ["low", "medium", "high", "xhigh", "max"]},
        {"name": "claude-opus-4-7", "provider": "anthropic", "model_id": "claude-opus-4-7", "implemented": True, "effort_levels": ["low", "medium", "high", "xhigh", "max"]},
        {"name": "claude-sonnet-4-6", "provider": "anthropic", "model_id": "claude-sonnet-4-6", "implemented": True, "effort_levels": ["low", "medium", "high", "max"]},
        {"name": "claude-haiku-4-5", "provider": "anthropic", "model_id": "claude-haiku-4-5", "implemented": True, "effort_levels": []},
        # OpenAI: only the code-specialized model -- generate() is a pure codegen
        # task, so the general gpt-5.5 is never the right pick and is omitted.
        {"name": "gpt-5.5-codex", "provider": "openai", "model_id": "gpt-5.5-codex", "implemented": True, "effort_levels": ["minimal", "low", "medium", "high"]},
    ]


def _model_effort_levels(model: str):
    """None = unknown model; [] = no effort knob; else the valid levels."""
    for m in models():
        if m["model_id"] == model or m["name"] == model:
            return m["effort_levels"]
    return None


def _resolve_llm(LLM: str) -> dict:
    key = LLM.strip().lower()
    key = re.sub(r"^claude[ -]", "claude-", key)
    key = re.sub(r"\s+", "-", key)
    key = re.sub(r"-(max|xhigh|high|medium|low|fast)$", "", key)
    if key == "codex":
        key = "gpt-5.5-codex"                       # back-compat alias
    cands = [key, key.replace(".", "-")]
    for m in models():
        if m["name"].lower() in cands or m["model_id"].lower() in cands:
            return {"provider": m["provider"], "model": m["model_id"], "implemented": m["implemented"]}
    if re.match(r"(gpt|codex|openai)", key):
        prov = "openai"
    elif re.match(r"(gemini|google)", key):
        prov = "google"
    else:
        prov = "anthropic"
    return {"provider": prov, "model": LLM, "implemented": prov in ("anthropic", "openai")}


def _provider_key(provider: str) -> str:
    if provider == "anthropic":
        # API key (pay-per-token) OR a Claude subscription OAuth token (Bearer):
        # ANTHROPIC_API_KEY first, else ANTHROPIC_AUTH_TOKEN (from `claude
        # setup-token` -- billed to your subscription, not per-token).
        return os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN", "")
    env = {"openai": "OPENAI_API_KEY", "google": "GOOGLE_API_KEY"}.get(provider, "ANTHROPIC_API_KEY")
    return os.environ.get(env, "")


# Map a provider to the environment variable set_key() writes (mirrors the R
# `ai4bayescode_set_key` mapping; note this is the WRITE target -- the read path
# in _provider_key additionally falls back to ANTHROPIC_AUTH_TOKEN).
_PROVIDER_ENV = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY",
                 "google": "GOOGLE_API_KEY"}

# The provider the user last chose via set_key(). When set, the interactive model
# menu shows ONLY that provider's models -- so after set_key(..., "anthropic") the
# menu drops the OpenAI models (a stray OPENAI_API_KEY in the env no longer
# surfaces them). None -> show all (no explicit choice made).
_SESSION_PROVIDER: str | None = None


def _mask_key(key: str) -> str:
    """Mask a key for display: a short prefix + the last 4 chars, never the full
    secret. Used by the key helpers and the verbose billing line (mirror of the
    R ``.ai4b_mask_key``)."""
    if not key:
        return "<empty>"
    n = len(key)
    if n <= 10:
        return key[:2] + "..."
    return key[:6] + "..." + key[n - 4:]


def stream_check(LLM: str = "claude-opus-4-8", API_key: str | None = None,
                 effort: str | None = None, progress: bool = True) -> dict:
    """Quick streaming self-check.

    Sends a tiny one-sentence prompt with streaming ON and prints the reply
    token-by-token, so you can confirm the streaming transport works before a
    full :func:`generate` run. Minimal token cost. Returns the parsed
    ``{content, stop_reason}``. Anthropic provider only.
    """
    import time
    llm = _resolve_llm(LLM)
    if llm["provider"] != "anthropic":
        raise ValueError("stream_check verifies the Anthropic streaming path; "
                         f"'{llm['provider']}' is not supported here.")
    key = API_key if API_key else _provider_key(llm["provider"])
    if not key:
        raise ValueError("No API key. Pass API_key= or call set_key().")
    if progress:
        print(f"Streaming check -> {llm['model']}. Reply should appear token-by-token:")
    t0 = time.time()
    parsed = _anthropic_request(
        "You are a connectivity test. Answer in ONE short sentence.",
        [{"role": "user", "content": "In one short sentence, confirm you are streaming this reply."}],
        llm["model"], effort, None, key, 128, 60, stream=True, progress=progress, live=True)
    txt = "".join(b["text"] for b in parsed["content"] if b.get("type") == "text")
    if progress:
        print(f"[streaming OK] {len(txt)} chars, stop={parsed.get('stop_reason')}, {time.time() - t0:.1f}s")
    return parsed


def _rate_limit_hint(api_key, err) -> str:
    """Human-readable guidance appended to a 429 / rate-limit error (empty string
    for any other error). A Claude *subscription* OAuth token (sk-ant-oat...) gets
    a specific note, because Anthropic restricts those for direct API use."""
    m = str(err).lower()
    if not ("429" in m or "rate_limit" in m or "too many requests" in m):
        return ""
    if str(api_key or "").startswith("sk-ant-oat"):
        return (
            "\n  NOTE: this looks like a Claude SUBSCRIPTION key (from `claude setup-token`). "
            "This 429 may be because subscription keys can be rate-limited for API use -- if "
            "it keeps happening, try a regular API key instead "
            "(sk-ant-api03...; see https://console.anthropic.com/settings/keys).")
    return "\n  NOTE: rate limit hit -- wait a bit before retrying."


def set_key(key: str, provider: str = "anthropic", check: bool = True) -> str:
    """Set an LLM provider API key for this session.

    Stores ``key`` in the provider's environment variable for the CURRENT
    process only (via ``os.environ``). It is **NOT written to disk** and does
    not persist across sessions; the key is never printed in full. After this,
    :func:`generate` picks the key up automatically whenever its ``API_key``
    argument is left ``None`` -- you no longer have to pass it on every call
    (you still can, to override per call).

    Args:
        key: Non-empty API-key string (e.g. ``"sk-ant-api..."``, ``"sk-YOUR-KEY-HERE"``).
            A Claude subscription key (``"sk-ant-oat..."`` from
            ``claude setup-token``) is detected from the ``sk-ant-oat`` prefix,
            but subscription keys may be rate-limited for API use (a 429) -- if
            that happens, try a regular API key (``"sk-ant-api03..."`` from
            https://console.anthropic.com/settings/keys).
        provider: One of ``"anthropic"``, ``"openai"``, ``"google"``.

    Returns:
        The provider name.
    """
    if provider not in _PROVIDER_ENV:
        raise ValueError("provider must be 'anthropic', 'openai', or 'google'")
    if not isinstance(key, str) or not key:
        raise ValueError("key must be a single non-empty string")
    import re
    if re.search(r"\.\.\.|[<>]|YOUR.?KEY", key, re.IGNORECASE):
        where = {
            "anthropic": " -- it starts with 'sk-ant-'; get one at https://console.anthropic.com/settings/keys",
            "openai": " -- get one at https://platform.openai.com/api-keys",
            "google": " -- get one at https://aistudio.google.com/apikey",
        }.get(provider, "")
        raise ValueError(
            f"'{key}' is a PLACEHOLDER from the examples, not a real key.\n"
            f"  Replace it with YOUR {provider} key{where}.")
    os.environ[_PROVIDER_ENV[provider]] = key
    global _SESSION_PROVIDER
    _SESSION_PROVIDER = provider     # the menu will then show only this provider's models
    print(f"Set {provider} key for this session ({_mask_key(key)}). Not saved to disk.")
    if check and provider == "anthropic":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            print("  (streaming self-check skipped: could not import 'anthropic' (a "
                  "required dependency -- run  pip install anthropic  or reinstall "
                  "AI4BayesCode); the key IS set for this session.)")
        else:
            try:
                stream_check(API_key=key, progress=True)
            except Exception as e:  # noqa: BLE001
                print(f"[warning] streaming self-check failed: {e}"
                      + _rate_limit_hint(key, e))
    return provider


def key_status() -> dict:
    """Report which LLM provider keys are set for this session.

    Prints, for each known provider, whether a key is currently visible to
    :func:`generate` -- from :func:`set_key` or a pre-existing environment
    variable -- shown masked. The full key is never printed.

    Returns:
        A dict ``{provider: bool}`` (key present per provider).
    """
    present: dict = {}
    for p in ("anthropic", "openai", "google"):
        k = _provider_key(p)
        present[p] = bool(k)
        shown = f"set ({_mask_key(k)})" if k else "not set"
        print(f"  {p:<10} {shown}")
    return present


# ---------------------------------------------------------------------------
# Console prompting + ask_user tool/loop
# ---------------------------------------------------------------------------
# --- LaTeX -> readable plain text (a bare console can't render $...$ math) ---
_LATEX_MAP = {
    "\\alpha": "α", "\\beta": "β", "\\gamma": "γ", "\\delta": "δ",
    "\\epsilon": "ε", "\\varepsilon": "ε", "\\zeta": "ζ", "\\eta": "η",
    "\\theta": "θ", "\\vartheta": "θ", "\\iota": "ι", "\\kappa": "κ",
    "\\lambda": "λ", "\\mu": "μ", "\\nu": "ν", "\\xi": "ξ", "\\pi": "π",
    "\\rho": "ρ", "\\sigma": "σ", "\\tau": "τ", "\\upsilon": "υ",
    "\\phi": "φ", "\\varphi": "φ", "\\chi": "χ", "\\psi": "ψ", "\\omega": "ω",
    "\\Gamma": "Γ", "\\Delta": "Δ", "\\Theta": "Θ", "\\Lambda": "Λ",
    "\\Xi": "Ξ", "\\Pi": "Π", "\\Sigma": "Σ", "\\Phi": "Φ",
    "\\Psi": "Ψ", "\\Omega": "Ω",
    "\\sim": "~", "\\mid": "|", "\\in": "∈", "\\notin": "∉",
    "\\leq": "≤", "\\le": "≤", "\\geq": "≥", "\\ge": "≥",
    "\\neq": "≠", "\\ne": "≠", "\\approx": "≈", "\\equiv": "≡",
    "\\propto": "∝", "\\times": "×", "\\cdot": "·", "\\pm": "±",
    "\\to": "→", "\\rightarrow": "→", "\\Rightarrow": "⇒", "\\mapsto": "↦",
    "\\infty": "∞", "\\partial": "∂", "\\nabla": "∇",
    "\\sum": "Σ", "\\prod": "∏", "\\int": "∫",
    "\\forall": "∀", "\\exists": "∃", "\\cup": "∪", "\\cap": "∩",
    "\\subseteq": "⊆", "\\subset": "⊂", "\\langle": "⟨", "\\rangle": "⟩",
    "\\ldots": "...", "\\cdots": "...", "\\dotsc": "...", "\\dots": "...",
    "\\lVert": "‖", "\\rVert": "‖", "\\Vert": "‖", "\\lvert": "|", "\\rvert": "|",
    "\\quad": "  ", "\\qquad": "    ",
}
_LATEX_SUP = {"^{\\top}": "ᵀ", "^\\top": "ᵀ", "^{-1}": "⁻¹",
              "^{2}": "²", "^{3}": "³", "^{T}": "ᵀ",
              "^2": "²", "^3": "³", "^T": "ᵀ"}


def _strip_code_fences(text: str) -> str:
    """Replace fenced ```...``` code blocks with a one-line placeholder so the
    model's PROSE (reasoning, model confirmation) shows in the console while long
    C++/R source doesn't flood it -- the full code is in the output files +
    transcript."""
    return re.sub(r"```.*?```",
                  "\n  [... code omitted -- written to the output files ...]\n",
                  text, flags=re.S)


def _latex_to_console(s: str) -> str:
    """Render the model's LaTeX (display math) as plain text a bare R/Python
    console can show. Not a LaTeX engine -- just the stats notation the prior
    elicitation prompts use. Non-LaTeX text passes through unchanged."""
    if not s or ("\\" not in s and "$" not in s):
        return s
    t = s
    t = re.sub(r"\\(?:begin|end)\{[^}]*\}", "", t)          # environments
    t = re.sub(r"\\\\\[[^][]*\]", "\n", t)                  # \\[2pt] line break + spacing
    t = t.replace("$$", "\n").replace("\\[", "\n").replace("\\]", "\n")
    t = t.replace("\\(", "").replace("\\)", "").replace("$", "")
    t = t.replace("\\\\", "\n").replace("&", "")            # line breaks / align
    t = re.sub(r"\\(?:left|right|bigg|Bigg|big|Big)\b", "", t)
    t = re.sub(r"\\(?:text|mathrm|mathbf|mathsf|mathtt|operatorname|hat|bar"
               r"|tilde|widehat|widetilde|vec|boldsymbol|overline|underline)\{([^{}]*)\}", r"\1", t)
    t = (t.replace("\\mathbb{R}", "ℝ").replace("\\mathbb{N}", "ℕ")
           .replace("\\mathbb{Z}", "ℤ").replace("\\mathbb{E}", "E"))
    t = re.sub(r"\\math(?:cal|bb|frak|scr)\{([^{}]*)\}", r"\1", t)
    t = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", t)
    t = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", t)
    for k in sorted(_LATEX_SUP, key=len, reverse=True):     # superscripts
        t = t.replace(k, _LATEX_SUP[k])
    t = t.replace("\\top", "ᵀ")
    for k in sorted(_LATEX_MAP, key=len, reverse=True):     # greek + symbols
        t = t.replace(k, _LATEX_MAP[k])
    t = re.sub(r"\\[,;:! ]", " ", t)                        # thin/medium spaces
    t = re.sub(r"\\([A-Za-z]+)", r"\1", t)                  # residual \cmd -> word
    t = t.replace("\\", "").replace("{", "").replace("}", "")
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in t.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip("\n")


def _console_ask(prompt_text: str, options: list | None = None, default: str | None = None) -> str:
    prompt_text = _latex_to_console(prompt_text)
    if options:
        print(prompt_text)
        for i, o in enumerate(options, 1):
            print(f"  {i}: {o}")
        raw = input("Choose [number]: ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(options):
            return options[int(raw) - 1]
        return default if default is not None else options[0]
    dtxt = f" [{default}]" if default else ""
    ans = input(f"{prompt_text}{dtxt}: ").strip()
    return default if (not ans and default is not None) else ans


def _ask_user_tool() -> list[dict]:
    return [{
        "name": "ask_user",
        "description": ("Ask the user a clarifying question in their console, especially to "
                        "ELICIT the PRIOR for each model parameter. Provide `question`; "
                        "optionally `options` (allowed answers, shown as a menu)."),
        "input_schema": {"type": "object",
                         "properties": {"question": {"type": "string"},
                                        "options": {"type": "array", "items": {"type": "string"}}},
                         "required": ["question"]},
    }]


# Read-only file tools -- give the model Claude-Code-style Read/Grep/Glob over the
# installed AI4BayesCode package so it reads the canonical reference examples/headers
# instead of guessing the API (root cause of the recurring compile bugs). The model
# uses LOGICAL paths ('examples/...','skills/...','include/...'); map to the python
# package's physical dirs (_examples / _skills / _vendored_include).
_PKG_DIRMAP = {"examples": "_examples", "skills": "_skills",
               "include": "_vendored_include", "blocks_local": "blocks_local",
               "blocks_download": "blocks_download"}


def _agent_tools() -> list[dict]:
    return _ask_user_tool() + [
        {"name": "read_file",
         "description": ("Read a file from the installed AI4BayesCode package -- its reference "
                         "`examples/*.cpp` (worked, compiling samplers), `skills/*.md`, and "
                         "`include/AI4BayesCode/*.hpp` headers. READ the reference example a skill "
                         "points to (e.g. 'examples/GaussianLocationScale.cpp') BEFORE writing code; "
                         "do not guess an API you can read."),
         "input_schema": {"type": "object",
                          "properties": {"path": {"type": "string",
                              "description": "package-relative, e.g. 'examples/GaussianLocationScale.cpp'"}},
                          "required": ["path"]}},
        {"name": "grep",
         "description": ("Regex-search file CONTENTS across the installed AI4BayesCode package. "
                         "Returns `path:line: text`. Use to find which example/header uses a "
                         "symbol or block (e.g. pattern='joint_nuts_block', glob='examples/*.cpp')."),
         "input_schema": {"type": "object",
                          "properties": {"pattern": {"type": "string"},
                                         "glob": {"type": "string",
                              "description": "restrict search, default 'examples/*.cpp'"}},
                          "required": ["pattern"]}},
        {"name": "glob",
         "description": ("List files in the installed AI4BayesCode package matching a glob, e.g. "
                         "'examples/*.cpp', 'skills/*.md', 'include/AI4BayesCode/*.hpp'."),
         "input_schema": {"type": "object",
                          "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]}},
    ]


def _pkg_root() -> str:
    import os
    return os.path.dirname(os.path.realpath(__file__))


def _tool_base(top):
    # Base dir for a whitelisted prefix: project CWD for blocks_local, the
    # user-global store's parent for blocks_download, else the package root.
    import os
    if top == "blocks_local":
        return os.getcwd()
    if top == "blocks_download":
        from .install_block import _blocks_dir
        return os.path.dirname(_blocks_dir())
    return _pkg_root()


def _resolve_logical(rel):
    # (base_dir, physical_relative_path) for a whitelisted logical path, or None.
    import re
    rel = re.sub(r"^[./]+", "", rel or "")
    if ".." in rel or not rel:
        return None
    top = rel.split("/", 1)[0]
    if top not in _PKG_DIRMAP:
        return None
    base = _tool_base(top)
    # examples/skills/include map to physical package subdirs; block dirs keep name.
    phys = rel if top in ("blocks_local", "blocks_download") \
        else _PKG_DIRMAP[top] + rel[len(top):]
    return (base, phys)


def _phys_to_logical(p, root=None):
    import os
    rp = os.path.realpath(p)
    for logical in _PKG_DIRMAP:
        r = _resolve_logical(logical)
        if r is None:
            continue
        prefix = os.path.realpath(os.path.join(*r))
        if rp == prefix or rp.startswith(prefix + os.sep):
            return (logical + rp[len(prefix):]).replace(os.sep, "/")
    return os.path.basename(p)


def _glob_pkg(pattern, root=None):
    import glob as _g, os
    r = _resolve_logical(pattern)
    if r is None:
        return []
    base, phys = r
    return sorted(_g.glob(os.path.join(base, phys), recursive=True))


def _exec_readonly_tool(name, inp, root):
    import os, re
    if name == "read_file":
        r = _resolve_logical(inp.get("path", ""))
        p = os.path.realpath(os.path.join(*r)) if r else None
        if not r or not p.startswith(os.path.realpath(r[0])) or not os.path.isfile(p):
            return (f"read_file: '{inp.get('path','')}' not found or not allowed "
                    "(only examples/, skills/, include/, blocks_local/, blocks_download/).")
        try:
            lines = open(p, encoding="utf-8", errors="replace").read().splitlines()
        except Exception as e:
            return f"read_file: could not read '{inp.get('path')}': {e}"
        if len(lines) > 1400:
            lines = lines[:1400] + [f"... [truncated; {len(lines)} lines total -- grep for a part]"]
        return "\n".join(lines)
    if name == "grep":
        pat, g = inp.get("pattern", ""), inp.get("glob", "examples/*.cpp")
        try:
            rx = re.compile(pat)
        except re.error as e:
            return f"grep: bad regex: {e}"
        hits = []
        for f in _glob_pkg(g, root):
            if not os.path.isfile(f):
                continue
            try:
                for i, ln in enumerate(open(f, encoding="utf-8", errors="replace"), 1):
                    if rx.search(ln):
                        hits.append(f"{_phys_to_logical(f, root)}:{i}: {ln.rstrip()}")
                        if len(hits) >= 200:
                            break
            except Exception:
                continue
            if len(hits) >= 200:
                break
        return "\n".join(hits) if hits else f"grep: no match for /{pat}/ in {g}"
    if name == "glob":
        fs = [_phys_to_logical(f, root) for f in _glob_pkg(inp.get("pattern", ""), root)]
        return "\n".join(fs) if fs else f"glob: no files match '{inp.get('pattern','')}'"
    return f"(unknown tool '{name}')"


def _anthropic_request(system, messages, model, effort, tools, api_key, max_tokens, timeout,
                       stream=True, progress=True, live=False):
    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "The 'anthropic' package (a required dependency of AI4BayesCode) is not "
            "importable. Reinstall it with:  pip install anthropic  "
            "(the OpenAI backend uses only the standard library and needs no SDK).") from e
    # Streaming keeps the socket alive (token + ping events), so a generous cap
    # instead of the buffered "0 bytes for `timeout`s" death on long generations.
    to = max(timeout, 1800) if stream else timeout
    # OAuth token (`sk-ant-oat...`, from `claude setup-token`) -> Bearer + oauth beta
    # (billed to the subscription); API key (`sk-ant-api...`) -> x-api-key (pay-per-token).
    if api_key.startswith("sk-ant-oat"):
        client = anthropic.Anthropic(auth_token=api_key, timeout=to)
        extra_headers = {"anthropic-beta": "oauth-2025-04-20"}
    else:
        client = anthropic.Anthropic(api_key=api_key, timeout=to)
        extra_headers = {}
    # Anthropic REQUIRES max_tokens. None = "no cap I chose" -> send the model's
    # maximum (64000 for Claude 4.x = effectively unlimited).
    max_tokens_out = 64000 if max_tokens is None else int(max_tokens)
    kwargs = dict(model=model, max_tokens=max_tokens_out, thinking={"type": "adaptive"},
                  system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                  messages=messages, extra_headers=extra_headers)
    if effort:
        kwargs["output_config"] = {"effort": effort}   # omitted for no-effort models
    if tools:
        kwargs["tools"] = tools

    def _from_msg(msg):
        content = []
        for b in msg.content:
            t = getattr(b, "type", None)
            if t == "text":
                content.append({"type": "text", "text": b.text})
            elif t == "tool_use":
                content.append({"type": "tool_use", "id": b.id, "name": b.name, "input": dict(b.input)})
            elif t == "thinking":
                content.append({"type": "thinking", "thinking": b.thinking,
                                "signature": getattr(b, "signature", None)})
            elif t == "redacted_thinking":
                content.append({"type": "redacted_thinking", "data": getattr(b, "data", None)})
        return {"content": content, "stop_reason": msg.stop_reason}

    if not stream:
        return _from_msg(client.messages.create(**kwargs))
    try:
        think_chars, next_dot = 0, 0
        text_chars, next_text_dot = 0, 0
        cur = {"type": None, "text": ""}

        # Wait hint: the model can sit silent for many seconds before the first
        # token (thinking/generating server-side, longer at higher effort). Fill
        # that gap so the console does not look frozen. Skipped for the live
        # stream-check (which prints its own banner).
        if progress and not live:
            eff = f" (effort={effort})" if effort else ""
            print(f"\n  [contacting {model}{eff} -- the model is working; "
                  f"this can take a while, please wait ...]")

        def _flush():
            # Per-block render (live=False): prose / model-confirmations go through
            # the LaTeX->console map so `$$ ... $$` math is readable; code-bearing
            # blocks print verbatim so C++/R source is never mangled. No-op when
            # live (already streamed raw). Idempotent (resets `cur`).
            if progress and not live and cur["type"] == "text" and cur["text"].strip():
                # Show the model's PROSE (reasoning, model confirmation -- $$..$$ math
                # rendered readably); replace ONLY the fenced CODE blocks with a short
                # placeholder (the full .cpp / runner is written to the output files +
                # transcript). Progress / ask_user come from other paths.
                shown = _latex_to_console(_strip_code_fences(cur["text"])).rstrip()
                if shown.strip():
                    print("\n" + shown)
            cur["type"], cur["text"] = None, ""

        with client.messages.stream(**kwargs) as s:
            for ev in s:
                et = getattr(ev, "type", None)
                if et == "content_block_start":
                    _flush()                                     # flush prior text block
                    cbt = getattr(getattr(ev, "content_block", None), "type", None)
                    cur["type"], cur["text"] = cbt, ""
                    if progress and cbt == "thinking":
                        print("\n  [thinking] ", end="", flush=True)
                    elif progress and cbt == "tool_use":
                        print("\n  [preparing a question] ", end="", flush=True)
                elif et == "content_block_delta":
                    d = getattr(ev, "delta", None)
                    dt = getattr(d, "type", None)
                    if dt == "text_delta":
                        cur["text"] += getattr(d, "text", "") or ""
                        if progress and live:
                            print(getattr(d, "text", ""), end="", flush=True)   # raw stream
                        elif progress:
                            text_chars += len(getattr(d, "text", "") or "")
                            if text_chars >= next_text_dot:
                                print(".", end="", flush=True)                  # heartbeat
                                next_text_dot = text_chars + 600
                    elif dt == "thinking_delta":
                        think_chars += len(getattr(d, "thinking", "") or "")
                        if progress and think_chars >= next_dot:
                            print(".", end="", flush=True)                      # heartbeat
                            next_dot = think_chars + 400
                elif et == "content_block_stop":
                    _flush()
            msg = s.get_final_message()
        _flush()                                                 # final safety flush
        if progress:
            print()
        return _from_msg(msg)
    except Exception as e:                                                   # noqa: BLE001
        if progress:
            print(f"\n[stream interrupted: {e} -- retrying without streaming]")
        return _from_msg(client.messages.create(**kwargs))


# ---------------------------------------------------------------------------
# OpenAI provider: Anthropic<->OpenAI shape converters + one request
# ---------------------------------------------------------------------------
def _to_openai_msgs(messages, system) -> list:
    """Convert an Anthropic-shaped conversation (the loop's internal
    representation) to the OpenAI Chat Completions ``messages`` array. The system
    prompt is prepended as a ``{role:"system"}`` message. Plain user/assistant
    strings pass through. An assistant turn carrying a tool_use block becomes
    ``{role:"assistant", tool_calls}`` and a user turn carrying a tool_result
    block becomes ``{role:"tool"}`` (mirror of the R ``.ai4b_to_openai_msgs``).
    """
    import json
    out = [{"role": "system", "content": system}]
    for m in messages:
        ct = m["content"]
        if isinstance(ct, str):                       # plain string passes through
            out.append({"role": m["role"], "content": ct})
            continue
        # content is a list of typed blocks. Detect tool_use / tool_result.
        tu = [b for b in ct if b.get("type") == "tool_use"]
        tr = [b for b in ct if b.get("type") == "tool_result"]
        if tr:                                         # user turn -> role:"tool" message(s)
            for b in tr:
                txt = b.get("content")
                if not isinstance(txt, str):
                    txt = "\n".join(
                        x.get("text", "") if isinstance(x, dict) else str(x)
                        for x in (txt or []))
                out.append({"role": "tool", "tool_call_id": b.get("tool_use_id"),
                            "content": txt})
            continue
        # gather any text blocks into a single content string
        text = "".join(b.get("text", "") for b in ct if b.get("type") == "text")
        if tu:                                         # assistant turn with tool_calls
            tool_calls = [{"id": b.get("id"), "type": "function",
                           "function": {"name": b.get("name"),
                                        "arguments": json.dumps(b.get("input", {}))}}
                          for b in tu]
            out.append({"role": m["role"], "content": text,  # text or "" alongside calls
                        "tool_calls": tool_calls})
        else:                                          # plain typed-text turn
            out.append({"role": m["role"], "content": text})
    return out


def _from_openai_response(resp) -> dict:
    """Convert an OpenAI Chat Completions response (parsed JSON) back to the
    Anthropic-shaped ``{stop_reason, content}`` the agentic loop expects. A
    non-empty assistant text -> a ``{type:"text"}`` block; each tool_call -> a
    ``{type:"tool_use"}`` block; stop_reason is "tool_use" iff there are
    tool_calls (mirror of the R ``.ai4b_from_openai_response``).
    """
    import json
    msg = resp["choices"][0]["message"]
    content = []
    text = msg.get("content")
    if isinstance(text, str) and text:
        content.append({"type": "text", "text": text})
    tcs = msg.get("tool_calls")
    if tcs:
        for tc in tcs:
            args = tc["function"].get("arguments")
            inp = json.loads(args) if isinstance(args, str) and args else {}
            content.append({"type": "tool_use", "id": tc.get("id"),
                            "name": tc["function"].get("name"), "input": inp})
    stop_reason = "tool_use" if tcs else "end_turn"
    return {"stop_reason": stop_reason, "content": content}


def _openai_body(system, messages, model, effort, tools, max_tokens) -> dict:
    """Build the OpenAI Chat Completions request body dict (no network).

    Factored out of :func:`_openai_request` so the max_completion_tokens policy is
    unit-testable: ``max_tokens=None`` -> OMIT ``max_completion_tokens`` entirely so
    the model uses its full output budget; a number -> include ``int(max_tokens)``
    (mirror of the R ``.ai4b_openai_request`` body construction).
    """
    import json  # noqa: F401  (kept for arguments JSON encoding by _to_openai_msgs)
    openai_tools = ([{"type": "function",
                      "function": {"name": t["name"], "description": t["description"],
                                   "parameters": t["input_schema"]}}
                     for t in tools] if tools else None)
    body = {"model": model,
            "messages": _to_openai_msgs(messages, system)}
    # None max_tokens -> omit the cap so the model uses its full output budget.
    if max_tokens is not None:
        body["max_completion_tokens"] = int(max_tokens)
    if openai_tools is not None:
        body["tools"] = openai_tools
        body["tool_choice"] = "auto"
    if effort:                                         # non-empty/non-None string only
        body["reasoning_effort"] = effort
    return body


def _openai_request(system, messages, model, effort, tools, api_key, max_tokens, timeout):
    """One request to the OpenAI Chat Completions API. The Anthropic-shaped
    ``ask_user`` tool is converted to the OpenAI ``{type:"function", function:{...}}``
    shape; ``reasoning_effort`` is sent for reasoning models when ``effort`` is a
    non-empty string. Uses stdlib ``urllib.request`` (no ``openai`` SDK hard dep,
    mirroring how the R side uses raw httr2). Returns the Anthropic-shaped parsed
    response (mirror of the R ``.ai4b_openai_request``).
    """
    import json
    import urllib.request

    body = _openai_body(system, messages, model, effort, tools, max_tokens)
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}",
                 "Content-Type": "application/json"},
        method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        parsed = json.loads(resp.read().decode("utf-8"))
    return _from_openai_response(parsed)


def _claude_p_call(system, msgs, model):
    """`claude -p` CLI transport: render system + conversation as a single prompt,
    shell out to the `claude` CLI, wrap stdout as a parsed Messages-shaped response.

    Used when no API key is available but the CLI is on PATH (or when the caller
    forces it via ``use_cli=True``). The CLI does not support the ask_user tool
    round-trip, so the returned response is always a terminal end_turn.
    """
    rendered = []
    for m in msgs:
        ct = m["content"]
        if not isinstance(ct, str):
            ct = "\n".join(b.get("text", "") if isinstance(b, dict) else ""
                           for b in ct)
        rendered.append(f"{m['role']}: {ct}")
    prompt_text = system + "\n\n" + "\n\n".join(rendered)
    args = ["claude", "-p"] + (["--model", model] if model else [])
    proc = subprocess.run(args, input=prompt_text, capture_output=True, text=True)
    return {"stop_reason": "end_turn",
            "content": [{"type": "text", "text": (proc.stdout or "") + (proc.stderr or "")}]}


def _anthropic_turn(system, msgs, *, call, ask, verbose=False, max_subturns=30):
    """One agentic turn: drive the ask_user sub-loop from ``msgs`` until the model
    emits terminal text (no tool_use). ``call`` is the transport
    ``function(messages) -> parsed``. Returns ``(text, msgs, truncated)`` -- the
    final assistant text, the messages incl. this turn's assistant/tool exchanges,
    and whether the reply was cut off at the token limit (``stop_reason ==
    "max_tokens"``) -- so the caller can append a repair message and call again for
    the next attempt.
    """
    for _ in range(max_subturns):
        parsed = call(msgs)
        content = parsed["content"]
        tus = [b for b in content if b.get("type") == "tool_use"]
        if not tus:
            txt = "".join(b.get("text", "") for b in content if b.get("type") == "text")
            return txt, msgs, parsed.get("stop_reason") == "max_tokens"
        root = _pkg_root()
        results = []
        for t in tus:
            if t.get("name") == "ask_user":
                inp = t.get("input") or {}
                q = inp.get("question", "")
                opts = inp.get("options")
                if verbose:
                    # The console ask() already echoes the question, so printing it
                    # here too would double the prompt -- just announce the ask.
                    print("[model asks]")
                results.append({"type": "tool_result", "tool_use_id": t["id"],
                                "content": ask(q, options=opts)})
            else:
                if verbose:
                    print(f"  [model {t.get('name')}: "
                          f"{t['input'].get('path') or t['input'].get('pattern','')}]")
                results.append({"type": "tool_result", "tool_use_id": t["id"],
                                "content": _exec_readonly_tool(t.get("name"), t["input"], root)})
        msgs = msgs + [{"role": "assistant", "content": content},
                       {"role": "user", "content": results}]
    raise RuntimeError(f"generate: exceeded {max_subturns} conversation turns")


def _write_emitted(txt, output_path, classname):
    """Extract the emitted fenced blocks to files and locate the runner.

    Returns ``(files, cpp_path, runner_path)``; ``runner_path`` = first emitted
    file matching ``*_runner.py``, else ``*.py``, else None (mirror of the R
    ``.ai4b_write_emitted``).
    """
    out = Path(output_path); out.mkdir(parents=True, exist_ok=True)
    files: list[str] = []
    cpp_path = None
    for blk in _extract_code(txt):
        path = blk["path"]
        if path is None:
            # no explicit filename: derive one from the code content
            if any(k in blk["code"] for k in ("PYBIND11_MODULE", "RCPP_MODULE", "#include")):
                pp = out / f"{classname}.cpp"
            else:
                continue
        else:
            # model-provided filename: confine to output_path via .name --
            # never honor an absolute path or a ".." that would escape the dir.
            pp = out / Path(path).name
        pp.parent.mkdir(parents=True, exist_ok=True)
        pp.write_text(blk["code"])
        files.append(str(pp))
        if str(pp).endswith(".cpp"):
            cpp_path = str(pp)
    runners = [f for f in files if f.endswith("_runner.py")]
    if not runners:
        runners = [f for f in files if f.endswith(".py")]
    runner_path = runners[0] if runners else None
    return files, cpp_path, runner_path


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------
def generate(model_description: str | None = None, *, classname: str | None = None,
             LLM: str | None = None, effort: str | None = None,
             output_path: str = "./generated", backend: str | None = None,
             API_key: str | None = None, interactive: bool | None = None,
             use_cli: bool = False, priors=None, confirm_model: bool | None = None,
             max_tokens: int | None = None,
             timeout: float = 600.0, max_attempts: int = 5, verbose: bool = True,
             verify_stream: bool = True,
             _responder=None, _ask=None, _validate=None) -> dict:
    """Console-interactive, LLM-agnostic NL -> validated sampler code.

    Missing (None/empty) arguments are asked in the console when interactive;
    PRIORS are elicited interactively by the model via an ask_user tool. With a
    provider key it drives the LLM (Claude today); with none it writes PROMPT.txt.
    ``confirm_model`` (``None`` -> follows ``interactive``): when ``True`` the model
    must state its understanding (likelihood, parameters, priors) via ``ask_user``
    and get your approval BEFORE it writes any code; ``False`` goes straight to code.
    Each generation attempt then runs a MANDATORY validate -> repair-to-convergence
    loop: the emitted ``.cpp`` is compiled (via ``AI4BayesCode.source``) and its
    Python runner executed; the runner must print ``AI4BAYES_VALIDATE: PASS`` (max
    rank-normalized R-hat < 1.01). On any failure the compile / run output is fed
    back to the model for repair, up to ``max_attempts`` times.

    ``max_tokens`` (``None`` default) imposes NO cap you chose -- the model produces
    up to its maximum (OpenAI: ``max_completion_tokens`` is omitted entirely;
    Anthropic requires a number, so the model's max -- 64000 for Claude 4.x -- is
    sent). Pass an int to cap it. Samplers can be long and hard to size up front, so
    the default avoids an arbitrary limit; a truncated reply is re-asked for FREE
    (no attempt spent).

    The key is read from the provider env var, sent only to the provider, never logged.
    The advanced hooks ``_responder`` (injectable transport ``function(messages)``),
    ``_ask`` (console-prompt override), and ``_validate`` (injectable validator
    ``function(cpp_path, runner_path, classname, verbose)`` returning
    ``{ok, stage, detail}``) exercise the loop offline.

    Returns ``{cpp_path, files, prompt, called_api, transcript, validated, attempts,
    validation}``.
    """
    if interactive is None:
        # Match R's interactive(): ask questions in a REPL. sys.stdin.isatty() is
        # False under IPython/Jupyter even though input() works there, so also
        # treat a running IPython shell as interactive.
        interactive = sys.stdin.isatty()
        if not interactive:
            try:
                from IPython import get_ipython
                interactive = get_ipython() is not None
            except Exception:
                pass
    ask = _ask or _console_ask
    llm = _resolve_llm(LLM or "claude-opus-4-8")

    # interactive-if-missing: ONLY ask for a runtime setting the caller did NOT pass.
    if interactive:
        if not model_description:
            model_description = ask("Model description (text, or path to a .txt)")
        # Pick the LLM model, then its thinking/effort level -- only this model's
        # valid levels are offered, consistent with the per-model effort check below.
        # Offer the FULL registry, defaulting to the flagship Claude model: the
        # menu is about CHOOSING a model, not about which key happens to be in the
        # environment. (Filtering by env key used to HIDE every Claude model when a
        # stray OPENAI_API_KEY was present -- confusing. If the chosen provider has
        # no key, the request fails later with a clear set_key() message.)
        if LLM is None:                       # ask the model ONLY if not provided
            # If the user picked a provider via set_key(), show only its models
            # (so an Anthropic key does not surface gpt-5.5-codex, and vice versa).
            _m = [m for m in models() if _SESSION_PROVIDER in (None, m["provider"])]
            _choices = [m["name"] for m in (_m or models())]
            LLM = ask("LLM model?", options=_choices, default=_choices[0])
        llm = _resolve_llm(LLM)
        lv = _model_effort_levels(llm["model"])
        if effort is None and lv:             # ask effort ONLY if not provided
            dflt = "high" if "high" in lv else lv[-1]
            effort = ask("Thinking / effort level?", options=lv, default=dflt)
        if backend is None:
            backend = ask("Backend? (both = ONE .cpp usable from BOTH R and Python)",
                          options=["both", "Python", "R"], default="both")
        if not output_path:
            output_path = ask("Output folder", default="./generated")
        if not classname:
            classname = ask("Class name", default=_derive_class_name(model_description or "GeneratedModel"))
    else:
        if not model_description:
            raise ValueError("model_description is required when interactive=False")
        backend = backend or "both"   # default: ONE .cpp usable from BOTH R and Python
        output_path = output_path or "./generated"
        classname = classname or _derive_class_name(model_description)
    backend = {"r": "R", "python": "Python", "both": "both"}.get(str(backend).strip().lower(), backend)
    if backend not in ("R", "Python", "both"):
        raise ValueError("backend must be 'R', 'Python', or 'both'")
    if priors is None:
        priors = "interactive" if interactive else "noninformative"
    if confirm_model is None:
        confirm_model = bool(interactive)

    # effort: match the chosen model's valid levels; ask if no match.
    lv = _model_effort_levels(llm["model"])
    if lv is None:
        pass                                   # unknown model -> pass through
    elif not lv:
        effort = None                          # model has no effort knob
    elif not effort or effort not in lv:
        if interactive:
            if effort:
                print(f"effort '{effort}' is not a valid level for {llm['model']}.")
            dflt = "high" if "high" in lv else lv[-1]
            effort = ask(f"Effort / reasoning level for {llm['model']}?", options=lv, default=dflt)
        elif effort:                           # provided but invalid -> error
            raise ValueError(
                f"effort '{effort}' is not valid for {llm['model']}; valid: {', '.join(lv)}.")
        else:                                  # not provided -> default to the model's 'high'
            effort = "high" if "high" in lv else lv[-1]

    if API_key is None:
        API_key = _provider_key(llm["provider"])
    if not API_key and interactive and _responder is None:
        API_key = ask(f"{llm['provider']} API key (Enter to skip -> writes offline PROMPT.txt; "
                      f"or set once via AI4BayesCode.set_key(); input is echoed)", default="")
    # The CLI (claude -p) is OPT-IN only, via use_cli=True. A blank key no longer
    # silently routes to the local CLI -- it lands on the predictable offline
    # PROMPT.txt path. Use set_key() or API_key= to generate online.
    import warnings
    cli_available = bool(use_cli) and shutil.which("claude") is not None
    if use_cli and shutil.which("claude") is None:
        warnings.warn("use_cli=True but `claude` is not on the PATH; "
                      "using the API key / offline path instead.", stacklevel=2)
    online = bool(API_key) or _responder is not None or cli_available

    # Online path inlines ONLY start.md (lazy load): start.md is the entry point +
    # phase-by-phase load schedule and EXPLICITLY says "Do NOT pre-load all skills".
    # The model reads each other skill on demand via the read_file/grep tools, so the
    # system prompt is ~6.5k tokens instead of ~150k. Offline falls back to the full set.
    p = prompt(model_description, backend=backend, output_path=output_path,
               classname=classname, priors=priors, include_skills=online,
               skills=["start.md"] if online else None,
               confirm_model=confirm_model)

    if not online:
        return _offline_emit(p, output_path, verbose)

    # provider dispatch (LLM-agnostic: anthropic + openai implemented)
    if llm["provider"] not in ("anthropic", "openai"):
        raise NotImplementedError(
            f"LLM provider '{llm['provider']}' is not yet implemented. Anthropic (Claude) "
            "and OpenAI are implemented; see AI4BayesCode.models(). The design is "
            "LLM-agnostic -- add a provider by implementing its request function.")

    # transport `call`: function(messages) -> parsed response.
    # `_responder` overrides the transport for BOTH providers (offline testing).
    # The CLI transport is Anthropic-only (Claude CLI); OpenAI uses the API only.
    use_cli_transport = (_responder is None and cli_available
                         and llm["provider"] == "anthropic")
    if _responder is not None:
        call = _responder
    elif use_cli_transport:
        call = lambda messages: _claude_p_call(p["system"], messages, llm["model"])  # noqa: E731
    elif llm["provider"] == "openai":
        tools = _agent_tools()
        call = lambda messages: _openai_request(  # noqa: E731
            p["system"], messages, llm["model"], effort, tools, API_key, max_tokens, timeout)
    else:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            if verbose:
                print("'anthropic' SDK not importable (pip install anthropic); "
                      "emitting prompt offline instead.")
            return _offline_emit(p, output_path, verbose)
        tools = _agent_tools()
        call = lambda messages: _anthropic_request(  # noqa: E731
            p["system"], messages, llm["model"], effort, tools, API_key, max_tokens, timeout)
    validate_fn = _validate or _DEFAULT_VALIDATE

    if verbose:
        bill = ("via the local `claude` CLI" if use_cli_transport else
                "billed pay-per-token (OpenAI API key)" if llm["provider"] == "openai" else
                "via your Claude subscription (OAuth token; no per-token charge)"
                if API_key.startswith("sk-ant-oat") else "billed pay-per-token (API key)")
        print(f"Generating via {llm['model']} (effort={effort or 'n/a'}) -- {bill}. "
              "Priors are elicited interactively.")

    # ---- streaming pre-flight: fail fast if the live stream is broken --------
    if (verify_stream and llm["provider"] == "anthropic"
            and not use_cli_transport and _responder is None):
        try:
            stream_check(LLM=llm["model"], API_key=API_key, progress=verbose)
        except Exception as e:  # noqa: BLE001
            # NON-FATAL: the streaming self-check is only a fast-feedback nicety, and the
            # generation ALREADY falls back to non-streaming if the streaming transport is
            # unavailable. A 429 here is a rate-limit on the streaming path, NOT a bad key --
            # aborting would needlessly block a generation that would otherwise succeed
            # non-streamed. Warn and continue.
            warnings.warn(
                f"streaming pre-flight check failed ({e}); continuing -- the generation "
                f"will fall back to non-streaming (pass verify_stream=False to skip this "
                f"check entirely)." + _rate_limit_hint(API_key, e))

    # ---- validate -> repair-to-convergence loop ----
    # `attempt` counts ONLY replies that produced code and went to the validator;
    # a reply with no code block (the model deferred, or was truncated) is re-asked
    # on a SEPARATE budget (`MAX_CODE_RETRIES`) so it never burns a validation
    # attempt -- max_attempts is reserved for real compile/convergence failures.
    max_attempts = max(1, int(max_attempts))
    MAX_CODE_RETRIES = 6
    out = Path(output_path); out.mkdir(parents=True, exist_ok=True)
    msgs = [{"role": "user", "content": p["user"]}]
    files: list[str] = []
    cpp_path = None
    txt = ""
    result_v = None
    attempt = 0
    code_retries = 0
    while True:
        try:
            txt, msgs, truncated = _anthropic_turn(p["system"], msgs, call=call, ask=ask,
                                                   verbose=verbose)
        except Exception as e:  # noqa: BLE001  -- surface a clear hint on rate-limit / 429
            hint = _rate_limit_hint(API_key, e)
            if hint:
                raise RuntimeError(f"{e}{hint}") from e
            raise
        files, cpp_path, runner_path = _write_emitted(txt, output_path, classname)
        (out / f"{classname}_transcript.md").write_text(txt)

        if not files:
            # No usable code block -- the model deferred the code or was truncated.
            # Re-ask for it WITHOUT spending a validation attempt.
            code_retries += 1
            result_v = {"ok": False, "stage": "no_code", "detail": (
                "Your response contained no fenced code block, so no file was written"
                + (" (your output was TRUNCATED at the token limit -- be more concise, "
                   "or raise max_tokens)" if truncated else "")
                + ".")}
            if verbose:
                print(f"No code in the reply{' (truncated)' if truncated else ''} -- "
                      f"re-asking (free retry {code_retries}/{MAX_CODE_RETRIES}, does "
                      "not use a validation attempt).")
            if code_retries > MAX_CODE_RETRIES:
                if verbose:
                    print(f"Gave up after {MAX_CODE_RETRIES} code-less replies "
                          "(no validation attempt was spent).")
                break
            msgs = msgs + [{"role": "assistant", "content": txt},
                           {"role": "user", "content": _repair_msg(result_v)}]
            continue

        # Code was emitted -> this is a real validation attempt.
        attempt += 1
        if verbose:
            print(f"Attempt {attempt}: wrote {len(files)} file(s) to {output_path}; "
                  "validating ...")
        result_v = validate_fn(cpp_path, runner_path, classname, verbose)
        if result_v.get("ok"):
            if verbose:
                print(f"Attempt {attempt}: validation PASSED.")
            break
        if attempt >= max_attempts:
            if verbose:
                print(f"Attempt {attempt}: validation FAILED at stage "
                      f"`{result_v.get('stage', '?')}`; out of attempts.")
            break
        if verbose:
            print(f"Attempt {attempt}: validation FAILED at stage "
                  f"`{result_v.get('stage', '?')}`; repairing.")
        msgs = msgs + [{"role": "assistant", "content": txt},
                       {"role": "user", "content": _repair_msg(result_v)}]

    result = {"cpp_path": cpp_path, "files": files, "prompt": p, "called_api": True,
              "transcript": txt, "validated": bool(result_v and result_v.get("ok")),
              "attempts": attempt, "validation": result_v}
    return result
