"""Offline tests for the NL->code front doors (no network / no API key).

The agentic ask_user loop is exercised with an injected mock transport
(``_responder``) and a stub console (``_ask``); the validate->repair loop is
exercised with an injected ``_validate`` stub, so no real compilation / runner
subprocess happens here. Mirrors the R-side tests/testthat/test-generate.R.
"""
import importlib
import sys

import pytest

gen = importlib.import_module("AI4BayesCode.generate")


def test_validate_launches_runner_without_doubling_folder(tmp_path, monkeypatch):
    # Regression: with a RELATIVE output_path (the default "./generated"), _validate
    # must launch the runner by its ABSOLUTE path. The old code passed the relative
    # runner_path as the script arg WHILE setting cwd to its parent, so Python
    # resolved it against that cwd -> "generated/generated/X_runner.py: can't open
    # file" and EVERY attempt failed as `incomplete` (the bug the user hit).
    # Patch the compile step. `AI4BayesCode.source` the ATTR is the re-exported
    # function (shadows the submodule), so reach the real submodule via importlib
    # and patch its `source` -- what `_validate`'s `from .source import source` binds.
    _src_mod = importlib.import_module("AI4BayesCode.source")
    monkeypatch.setattr(_src_mod, "source", lambda *a, **k: None)
    gen_dir = tmp_path / "generated"
    gen_dir.mkdir()
    (gen_dir / "X.cpp").write_text("// dummy\n")
    (gen_dir / "X_runner.py").write_text("print('AI4BAYES_VALIDATE: PASS')\n")
    monkeypatch.chdir(tmp_path)                                  # cwd has ./generated as a subdir
    res = gen._validate("generated/X.cpp", "generated/X_runner.py", "X", verbose=False)
    assert res["ok"] and res["stage"] == "converged", res


# ---------------------------------------------------------------------------
# prompt() — pure builder
# ---------------------------------------------------------------------------
def test_prompt_is_start_md_anchored():
    p = gen.prompt("Logistic regression: y ~ Bernoulli(sigmoid(X beta)).",
                   backend="R", output_path="./gen")
    assert "START HERE" in p["system"] or "start.md" in p["system"]
    assert "NON-INFORMATIVE" in p["user"]
    assert p["classname"]
    assert p["skills"] == ["start.md"]


def test_prompt_mandates_validate_sentinel():
    p = gen.prompt("Poisson GLM", backend="Python")
    assert "AI4BAYES_VALIDATE: PASS" in p["user"]
    assert "AI4BAYES_VALIDATE: FAIL maxRhat=" in p["user"]
    assert "rank-normalized R-hat" in p["user"]


def test_prompt_interactive_priors_instruct_ask_user():
    p = gen.prompt("Poisson GLM", priors="interactive")
    assert "ask_user" in p["user"]
    # interactive mode instructs elicitation of the (missing/ambiguous) priors via
    # ask_user -- the prior-no-re-ask policy; wording is "ELICIT ... ONLY the ...".
    assert "ELICIT" in p["user"]


def test_prompt_has_no_gate_param():
    import inspect
    sig = inspect.signature(gen.prompt)
    assert "gate" not in sig.parameters


# ---------------------------------------------------------------------------
# confirm_model step (mirror of the R-side tests)
# ---------------------------------------------------------------------------
def test_prompt_confirm_model_true_instructs_ask_user_confirmation():
    p = gen.prompt("Linear regression.", confirm_model=True)
    user = p["user"]
    assert "MODEL CONFIRMATION" in user
    assert "ask_user" in user
    assert "display-math" in user
    assert "parameter summary table" in user


def test_prompt_defers_design_flow_to_start_md():
    # The thin _build_user must NOT script its own narrow flow -- it defers the
    # design questions (block / MCMC-vs-VI / harness deliverable) to start.md so
    # editing the skills auto-syncs R + Python. It only pins harness settings +
    # the AI4BAYES_VALIDATE grep contract.
    user = gen.prompt("Linear regression.", confirm_model=True)["user"]
    assert "FOLLOW start.md" in user
    assert "Pop EVERY design question" in user
    assert "block preference" in user and "MCMC vs VI" in user
    assert "validation-harness deliverable" in user
    assert "AI4BAYES_VALIDATE: PASS" in user            # grep contract kept
    assert "do NOT render a DAG" in user                # user does not want a DAG


def test_prompt_confirm_model_false_skips_confirmation():
    p = gen.prompt("Linear regression.", confirm_model=False)
    assert "SKIP the model-confirmation" in p["user"]
    assert "MODEL CONFIRMATION" not in p["user"]


def test_generate_confirm_model_defaults_off_when_non_interactive(tmp_path):
    # interactive=False (default) -> confirm_model resolves to False -> SKIP.
    def responder(messages):
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text":
            "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{};\n```"}]}

    td = tmp_path / "confirm_off"
    res = gen.generate("Linear regression.", classname="LR", API_key="sk-x",
                       output_path=str(td), backend="Python", interactive=False,
                       verbose=False, max_attempts=1, _responder=responder,
                       _ask=lambda *a, **k: "", _validate=lambda *a, **k: {"ok": True})
    assert "SKIP the model-confirmation" in res["prompt"]["user"]
    assert "MODEL CONFIRMATION" not in res["prompt"]["user"]


def test_generate_confirm_model_true_forces_confirmation_block(tmp_path):
    # interactive=False but confirm_model=True forced on -> CONFIRM block present.
    def responder(messages):
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text":
            "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{};\n```"}]}

    td = tmp_path / "confirm_on"
    res = gen.generate("Linear regression.", classname="LR", API_key="sk-x",
                       output_path=str(td), backend="Python", interactive=False,
                       confirm_model=True, verbose=False, max_attempts=1,
                       _responder=responder, _ask=lambda *a, **k: "",
                       _validate=lambda *a, **k: {"ok": True})
    assert "MODEL CONFIRMATION" in res["prompt"]["user"]
    assert "ask_user" in res["prompt"]["user"]


def test_generate_confirm_model_end_to_end_confirmation_reaches_ask(tmp_path):
    # turn 1: model calls ask_user with the model-confirmation question;
    # turn 2: after the user confirms, it emits the code.
    state = {"turn": 0}
    asked = []

    def responder(messages):
        state["turn"] += 1
        if state["turn"] == 1:
            return {"stop_reason": "tool_use", "content": [
                {"type": "text", "text": "Confirming my understanding."},
                {"type": "tool_use", "id": "c1", "name": "ask_user",
                 "input": {"question": "Likelihood y~N(Xb,sigma^2), flat priors. Correct?",
                           "options": ["Correct -- generate the sampler",
                                       "Not quite -- I will correct it"]}}]}
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text":
            "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{};\n```"}]}

    def ask_stub(q, options=None, default=None):
        asked.append(q)
        return "Correct -- generate"

    td = tmp_path / "confirm_e2e"
    res = gen.generate("Linear regression.", classname="LR", API_key="sk-x",
                       output_path=str(td), backend="Python", interactive=False,
                       confirm_model=True, verbose=False, max_attempts=1,
                       _responder=responder, _ask=ask_stub,
                       _validate=lambda *a, **k: {"ok": True})
    assert res["validated"] is True
    assert any("Correct?" in q for q in asked)   # the confirmation question reached the ask stub
    assert state["turn"] == 2


# ---------------------------------------------------------------------------
# models() + LLM resolution
# ---------------------------------------------------------------------------
def test_models_and_llm_resolution():
    ms = gen.models()
    keys = set(ms[0].keys())
    assert {"name", "provider", "model_id", "implemented", "effort_levels"} <= keys
    assert gen._resolve_llm("claude-opus-4-8")["provider"] == "anthropic"
    r = gen._resolve_llm("Claude Opus 4.8 Max")
    assert r["model"] == "claude-opus-4-8"
    assert r["implemented"]
    g = gen._resolve_llm("gpt-5.5")
    assert g["provider"] == "openai"
    assert g["implemented"]                       # openai is now implemented


def test_effort_levels_are_per_model():
    assert "max" in gen._model_effort_levels("claude-opus-4-8")
    assert "xhigh" not in gen._model_effort_levels("claude-sonnet-4-6")
    assert gen._model_effort_levels("claude-haiku-4-5") == []
    assert gen._model_effort_levels("totally-unknown-model") is None


def test_default_llm_is_bare_model_id():
    import inspect
    # LLM/effort now default to None (asked ONLY if not provided); the effective
    # model when unset is the bare model id claude-opus-4-8.
    assert inspect.signature(gen.generate).parameters["LLM"].default is None
    assert inspect.signature(gen.generate).parameters["effort"].default is None
    assert gen._resolve_llm("claude-opus-4-8")["model"] == "claude-opus-4-8"


# ---------------------------------------------------------------------------
# generate() — offline emit (no key, no CLI)
# ---------------------------------------------------------------------------
def test_generate_offline_writes_prompt_txt(tmp_path, monkeypatch):
    # The offline branch fires only when there is no key AND no `claude` CLI on the
    # PATH. Blank PATH so a real CLI on this machine cannot be discovered.
    monkeypatch.setenv("PATH", "")
    td = tmp_path / "off"
    res = gen.generate("Gaussian location-scale.", classname="GLS",
                       output_path=str(td), backend="Python", interactive=False,
                       API_key="", verbose=False)
    assert res["called_api"] is False
    assert (td / "PROMPT.txt").is_file()


# ---------------------------------------------------------------------------
# generate() — agentic ask_user loop
# ---------------------------------------------------------------------------
def test_ask_user_loop_elicits_and_extracts_code(tmp_path):
    asked = []
    state = {"turn": 0}

    def responder(messages):
        state["turn"] += 1
        if state["turn"] == 1:
            return {"stop_reason": "tool_use", "content": [
                {"type": "text", "text": "Eliciting priors."},
                {"type": "tool_use", "id": "t1", "name": "ask_user",
                 "input": {"question": "Prior for beta?",
                           "options": ["non-informative", "weakly"]}}]}
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text":
            "```cpp\n// path: GenFoo.cpp\n#include <pybind11/pybind11.h>\nclass GenFoo {};\n```"}]}

    def ask_stub(prompt_text, options=None, default=None):
        asked.append(prompt_text)
        return "non-informative"

    td = tmp_path / "loop"
    res = gen.generate("Logistic regression with coefficient beta.",
                       classname="GenFoo", output_path=str(td), backend="Python",
                       interactive=False, verbose=False,
                       _responder=responder, _ask=ask_stub,
                       _validate=lambda *a, **k: {"ok": True})
    assert res["called_api"]
    assert res["cpp_path"].endswith("GenFoo.cpp")
    assert (td / "GenFoo.cpp").is_file()
    assert "Prior for beta" in asked[0]          # the model's question reached the console
    assert state["turn"] == 2                     # exactly two transport calls


# ---------------------------------------------------------------------------
# generate() — validate -> repair-to-convergence loop  (the two REQUIRED tests)
# ---------------------------------------------------------------------------
def test_validate_repair_repairs_on_attempt_2(tmp_path):
    # transport: each turn emits a .cpp + a runner (terminal end_turn).
    def responder(messages):
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text": (
            "```cpp\n// path: Foo.cpp\n#include <pybind11/pybind11.h>\nclass Foo {};\n```\n"
            "```python\n# path: Foo_runner.py\nprint('AI4BAYES_VALIDATE: PASS')\n```")}]}

    # injected _validate: FAIL (compile) on its first call, PASS on its second.
    state = {"n": 0}

    def validate_stub(cpp_path, runner_path, classname, verbose=False):
        state["n"] += 1
        if state["n"] == 1:
            return {"ok": False, "stage": "compile", "detail": "boom"}
        return {"ok": True, "stage": "converged", "detail": "ok"}

    td = tmp_path / "repair"
    res = gen.generate("Model with parameter mu.", classname="Foo",
                       output_path=str(td), backend="Python", interactive=False,
                       verbose=False, _responder=responder, _ask=lambda *a, **k: "",
                       _validate=validate_stub)
    assert res["attempts"] == 2
    assert res["validated"] is True
    assert state["n"] == 2


def test_validates_on_attempt_1(tmp_path):
    def responder(messages):
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text":
            "```cpp\n// path: Bar.cpp\n#include <x>\nclass Bar{};\n```\n"
            "```python\n# path: Bar_runner.py\nprint('AI4BAYES_VALIDATE: PASS')\n```"}]}

    td = tmp_path / "pass1"
    res = gen.generate("Model with parameter mu.", classname="Bar",
                       output_path=str(td), backend="Python", interactive=False,
                       verbose=False, _responder=responder, _ask=lambda *a, **k: "",
                       _validate=lambda *a, **k: {"ok": True})
    assert res["attempts"] == 1
    assert res["validated"] is True


def test_repair_loop_exhausts_attempts(tmp_path):
    # always-FAIL validator -> loop runs the full max_attempts then stops.
    def responder(messages):
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text":
            "```cpp\n// path: Baz.cpp\n#include <x>\nclass Baz{};\n```\n"
            "```python\n# path: Baz_runner.py\nprint('nope')\n```"}]}

    td = tmp_path / "exhaust"
    res = gen.generate("Model with parameter mu.", classname="Baz",
                       output_path=str(td), backend="Python", interactive=False,
                       verbose=False, max_attempts=3, _responder=responder,
                       _ask=lambda *a, **k: "",
                       _validate=lambda *a, **k: {"ok": False, "stage": "convergence",
                                                  "detail": "maxRhat=1.4"})
    assert res["attempts"] == 3
    assert res["validated"] is False
    assert res["validation"]["stage"] == "convergence"


# ---------------------------------------------------------------------------
# "stuck after prior" bug: model emits NO code block -> must detect + recover
# ---------------------------------------------------------------------------
def test_repair_msg_for_no_code_forcefully_demands_the_code():
    m = gen._repair_msg({"stage": "no_code", "detail": "x"})
    assert "No sampler code was emitted" in m
    assert "ONLY the fenced code" in m
    assert "// path:" in m


def test_zero_file_attempt_is_no_code_repaired_and_loop_recovers(tmp_path):
    # turn 1: model asks for a prior; turn 2 (the BUG): defers, emits NO code
    # block; turn 3: finally emits the code. The 0-file attempt must be flagged
    # no_code and re-asked for FREE -- it must NOT consume a validation attempt
    # (so even with max_attempts=1 the loop still reaches the real code reply).
    state = {"turn": 0}

    def responder(messages):
        state["turn"] += 1
        if state["turn"] == 1:
            return {"stop_reason": "tool_use", "content": [
                {"type": "text", "text": "Eliciting."},
                {"type": "tool_use", "id": "t1", "name": "ask_user",
                 "input": {"question": "Prior for sigma?",
                           "options": ["Jeffreys", "Half-Normal"]}}]}
        if state["turn"] == 2:                       # BUG: defers, emits NO code block
            return {"stop_reason": "end_turn", "content": [
                {"type": "text", "text": "Great, I'll generate the sampler now."}]}
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text":  # then code
            "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{};\n```"}]}

    val_cpp = []

    def validate_stub(cpp_path, runner_path, classname, verbose=False):
        val_cpp.append(cpp_path)
        return {"ok": False, "stage": "compile", "detail": "x"} if cpp_path is None \
            else {"ok": True}

    td = tmp_path / "nocode"
    res = gen.generate("Linear regression.", classname="LR", API_key="sk-x",
                       output_path=str(td), backend="Python", interactive=False,
                       verbose=False, max_attempts=1, _responder=responder,
                       _ask=lambda *a, **k: "Jeffreys", _validate=validate_stub)
    assert res["validated"] is True              # recovered even with max_attempts=1 ...
    assert res["attempts"] == 1                   # ... because the 0-code reply was a FREE retry
    assert len(val_cpp) == 1                       # validate ran only for the real code attempt
    assert val_cpp[0].endswith("LR.cpp")          # only the real code attempt reached validate


def test_truncated_reply_is_flagged_as_truncated_no_code(tmp_path):
    # opens a ```cpp fence but never closes it, with stop_reason=max_tokens.
    def responder(messages):
        return {"stop_reason": "max_tokens", "content": [{"type": "text", "text":
            "```cpp\n// path: LR.cpp\n#include <x>\nclass LR{ // ...cut off"}]}

    def validate_stub(*a, **k):
        raise AssertionError("validate must not run when 0 files were written")

    td = tmp_path / "trunc"
    res = gen.generate("Linear regression.", classname="LR", API_key="sk-x",
                       output_path=str(td), backend="Python", interactive=False,
                       verbose=False, max_attempts=1, _responder=responder,
                       _ask=lambda *a, **k: "", _validate=validate_stub)
    assert res["validated"] is False
    assert res["validation"]["stage"] == "no_code"
    assert "TRUNCATED" in res["validation"]["detail"]


# ---------------------------------------------------------------------------
# _write_emitted — runner selection
# ---------------------------------------------------------------------------
def test_write_emitted_prefers_runner_py(tmp_path):
    txt = ("```cpp\n// path: M.cpp\n#include <x>\nclass M{};\n```\n"
           "```python\n# path: helper.py\nx = 1\n```\n"
           "```python\n# path: M_runner.py\nprint('AI4BAYES_VALIDATE: PASS')\n```")
    files, cpp_path, runner_path = gen._write_emitted(txt, str(tmp_path), "M")
    assert cpp_path.endswith("M.cpp")
    assert runner_path.endswith("M_runner.py")     # _runner.py preferred over plain .py
    assert len(files) == 3


def test_write_emitted_falls_back_to_plain_py(tmp_path):
    txt = ("```cpp\n// path: N.cpp\n#include <x>\nclass N{};\n```\n"
           "```python\n# path: driver.py\nprint('hi')\n```")
    _, cpp_path, runner_path = gen._write_emitted(txt, str(tmp_path), "N")
    assert cpp_path.endswith("N.cpp")
    assert runner_path.endswith("driver.py")


def test_write_emitted_no_runner_is_none(tmp_path):
    txt = "```cpp\n// path: O.cpp\n#include <x>\nclass O{};\n```"
    _, cpp_path, runner_path = gen._write_emitted(txt, str(tmp_path), "O")
    assert cpp_path.endswith("O.cpp")
    assert runner_path is None


# ---------------------------------------------------------------------------
# _validate — stages without a live compile
# ---------------------------------------------------------------------------
# `_validate` does `from .source import source` on each call, so it picks up the
# `source` attribute of the *submodule* (not the function re-exported on the
# package). Patch that attribute to avoid a real compile.
_srcmod = importlib.import_module("AI4BayesCode.source")


def test_validate_no_runner_stage(tmp_path, monkeypatch):
    # compile step is patched to succeed -> we land on the no_runner stage.
    monkeypatch.setattr(_srcmod, "source", lambda *a, **k: None)
    cpp = tmp_path / "X.cpp"
    cpp.write_text("// stub")
    r = gen._validate(str(cpp), None, "X")
    assert r["ok"] is False
    assert r["stage"] == "no_runner"


def test_validate_compile_stage(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("compile boom")

    monkeypatch.setattr(_srcmod, "source", boom)
    cpp = tmp_path / "Y.cpp"
    cpp.write_text("// stub")
    r = gen._validate(str(cpp), None, "Y")
    assert r["ok"] is False
    assert r["stage"] == "compile"
    assert "compile boom" in r["detail"]


def test_validate_converged_and_convergence_via_runner(tmp_path, monkeypatch):
    monkeypatch.setattr(_srcmod, "source", lambda *a, **k: None)
    cpp = tmp_path / "Z.cpp"
    cpp.write_text("// stub")
    ok_runner = tmp_path / "Z_runner.py"
    ok_runner.write_text("print('AI4BAYES_VALIDATE: PASS')\n")
    r = gen._validate(str(cpp), str(ok_runner), "Z")
    assert r["ok"] is True
    assert r["stage"] == "converged"

    bad_runner = tmp_path / "Z_bad_runner.py"
    bad_runner.write_text("print('AI4BAYES_VALIDATE: FAIL maxRhat=1.42')\n")
    r2 = gen._validate(str(cpp), str(bad_runner), "Z")
    assert r2["ok"] is False
    assert r2["stage"] == "convergence"
    assert "1.42" in r2["detail"]


# ---------------------------------------------------------------------------
# _claude_p_call — CLI transport shape (mock subprocess, no real `claude`)
# ---------------------------------------------------------------------------
def test_claude_p_call_wraps_stdout(monkeypatch):
    class _Proc:
        stdout = "```cpp\n// path: C.cpp\nclass C{};\n```"
        stderr = ""

    captured = {}

    def fake_run(args, input=None, capture_output=None, text=None):
        captured["args"] = args
        captured["input"] = input
        return _Proc()

    monkeypatch.setattr(gen.subprocess, "run", fake_run)
    parsed = gen._claude_p_call("SYS", [{"role": "user", "content": "hello"}],
                                "claude-opus-4-8")
    assert parsed["stop_reason"] == "end_turn"
    assert parsed["content"][0]["text"].startswith("```cpp")
    assert captured["args"][:2] == ["claude", "-p"]
    assert "--model" in captured["args"]
    assert "SYS" in captured["input"] and "user: hello" in captured["input"]


# ---------------------------------------------------------------------------
# OpenAI provider — dispatch + Anthropic<->OpenAI converters (offline, no network)
# ---------------------------------------------------------------------------
def test_openai_gpt_is_dispatched_no_longer_errors(tmp_path):
    # gpt-5.5 (openai) must dispatch through the agentic loop exactly like the
    # Anthropic path; the injected _responder stands in for the OpenAI transport,
    # so NO live OpenAI call happens.
    def responder(messages):
        return {"stop_reason": "end_turn", "content": [{"type": "text", "text":
            "```cpp\n// path: LR.cpp\n#include <pybind11/pybind11.h>\nclass LR {};\n```"}]}

    td = tmp_path / "openai"
    res = gen.generate("Linear regression.", classname="LR", LLM="gpt-5.5",
                       API_key="sk-test", effort="high", backend="Python",
                       output_path=str(td), interactive=False, verbose=False,
                       _responder=responder, _ask=lambda *a, **k: "",
                       _validate=lambda *a, **k: {"ok": True})
    assert res["called_api"]
    assert res["cpp_path"].endswith("LR.cpp")
    assert (td / "LR.cpp").is_file()


def test_to_openai_msgs_converts_tool_use_and_tool_result():
    import json
    messages = [
        {"role": "user", "content": "plain user text"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "asking"},
            {"type": "tool_use", "id": "t1", "name": "ask_user",
             "input": {"question": "Prior for beta?", "options": ["a", "b"]}}]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "non-informative"}]},
    ]
    out = gen._to_openai_msgs(messages, "SYS")
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1] == {"role": "user", "content": "plain user text"}
    # assistant tool_use -> tool_calls with JSON-encoded arguments
    asst = out[2]
    assert asst["role"] == "assistant"
    assert asst["content"] == "asking"
    tc = asst["tool_calls"][0]
    assert tc["type"] == "function"
    assert tc["id"] == "t1"
    assert tc["function"]["name"] == "ask_user"
    assert json.loads(tc["function"]["arguments"]) == {"question": "Prior for beta?",
                                                       "options": ["a", "b"]}
    # tool_result -> role:"tool" with the tool_call_id
    tool_msg = out[3]
    assert tool_msg == {"role": "tool", "tool_call_id": "t1", "content": "non-informative"}


def test_from_openai_response_tool_calls_and_plain_text():
    import json
    # tool_calls -> tool_use block + stop_reason tool_use
    resp_tc = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "call_1", "type": "function",
         "function": {"name": "ask_user",
                      "arguments": json.dumps({"question": "Q?"})}}]}}]}
    parsed = gen._from_openai_response(resp_tc)
    assert parsed["stop_reason"] == "tool_use"
    tu = [b for b in parsed["content"] if b["type"] == "tool_use"][0]
    assert tu["id"] == "call_1"
    assert tu["name"] == "ask_user"
    assert tu["input"] == {"question": "Q?"}

    # plain text -> text block + end_turn
    resp_txt = {"choices": [{"message": {"content": "all done", "tool_calls": None}}]}
    parsed2 = gen._from_openai_response(resp_txt)
    assert parsed2["stop_reason"] == "end_turn"
    assert parsed2["content"] == [{"type": "text", "text": "all done"}]


# ---------------------------------------------------------------------------
# max_tokens: None (default) = no chosen cap / model maximum
#   - generate() default is None
#   - OpenAI body OMITS max_completion_tokens on None, includes int on a number
#   - Anthropic builder sends 64000 (the model max) on None
# ---------------------------------------------------------------------------
def test_max_tokens_default_is_none():
    import inspect
    assert inspect.signature(gen.generate).parameters["max_tokens"].default is None


def test_openai_body_omits_max_completion_tokens_when_none():
    tools = gen._ask_user_tool()
    body = gen._openai_body("SYS", [{"role": "user", "content": "hi"}],
                            "gpt-5.5", "high", tools, None)
    assert "max_completion_tokens" not in body          # omitted -> model uses full budget


def test_openai_body_includes_max_completion_tokens_when_int():
    tools = gen._ask_user_tool()
    body = gen._openai_body("SYS", [{"role": "user", "content": "hi"}],
                            "gpt-5.5", "high", tools, 777)
    assert body["max_completion_tokens"] == 777


def test_openai_request_omits_cap_on_none_real_http_path(monkeypatch):
    # Exercise the REAL _openai_request code path: capture the urllib Request that
    # _openai_request builds, then raise a sentinel to stop before any network call.
    import urllib.request

    captured = {}

    class _Sentinel(Exception):
        pass

    def fake_urlopen(req, timeout=None):
        import json
        captured["body"] = json.loads(req.data.decode("utf-8"))
        raise _Sentinel()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    tools = gen._ask_user_tool()
    # None -> omitted
    with pytest.raises(_Sentinel):
        gen._openai_request("SYS", [{"role": "user", "content": "hi"}], "gpt-5.5",
                            "high", tools, "sk-test", None, 600.0)
    assert "max_completion_tokens" not in captured["body"]
    # 777 -> present
    with pytest.raises(_Sentinel):
        gen._openai_request("SYS", [{"role": "user", "content": "hi"}], "gpt-5.5",
                            "high", tools, "sk-test", 777, 600.0)
    assert captured["body"]["max_completion_tokens"] == 777


def test_anthropic_request_sends_model_max_on_none(monkeypatch):
    # Capture the kwargs passed to anthropic.Anthropic(...).messages.create(...) and
    # assert max_tokens defaults to the model max (64000) when None, the int otherwise.
    import types

    captured = {}

    class _Msg:
        content = []
        stop_reason = "end_turn"

    class _Messages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _Msg()

    class _Client:
        def __init__(self, **_kw):
            self.messages = _Messages()

    fake_anthropic = types.SimpleNamespace(Anthropic=_Client)
    monkeypatch.setitem(sys.modules, "anthropic", fake_anthropic)

    # None -> 64000 (model max)
    gen._anthropic_request("SYS", [{"role": "user", "content": "hi"}], "claude-opus-4-8",
                           "high", None, "sk-ant-api03-x", None, 600.0)
    assert captured["max_tokens"] == 64000
    # explicit int -> passed through
    gen._anthropic_request("SYS", [{"role": "user", "content": "hi"}], "claude-opus-4-8",
                           "high", None, "sk-ant-api03-x", 12345, 600.0)
    assert captured["max_tokens"] == 12345


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------
def test_unimplemented_provider_errors():
    # openai is now implemented, so a genuinely unimplemented provider (google /
    # gemini) is what must still error.
    with pytest.raises(NotImplementedError, match="not yet implemented"):
        gen.generate("X", classname="X", LLM="gemini-2.5-pro", interactive=False,
                     _responder=lambda m: None, _ask=lambda *a, **k: "")


def test_gpt_resolves_to_implemented():
    g = gen._resolve_llm("gpt-5.5")
    assert g["provider"] == "openai"
    assert g["implemented"] is True


def test_non_interactive_missing_description_errors():
    with pytest.raises(ValueError, match="required"):
        gen.generate(interactive=False, API_key="")


def test_effort_invalid_for_model_errors_non_interactive():
    with pytest.raises(ValueError, match="not valid for"):
        gen.generate("X", classname="X", LLM="claude-sonnet-4-6", effort="xhigh",
                     interactive=False, _responder=lambda m: None,
                     _ask=lambda *a, **k: "")


def test_interactive_model_and_effort_menus_shown(tmp_path):
    # interactive: LLM is PROVIDED here, so the model menu is NOT shown (it is
    # asked only when unset). effort=xhigh is INVALID for sonnet, so the flow
    # re-asks with an effort menu drawn from THAT model's levels (excludes xhigh).
    asked = []
    eff_opts = {}
    def ask(prompt, options=None, default=None):
        asked.append(prompt)
        if "effort" in prompt.lower():
            eff_opts["opts"] = options
            return "high"
        if "LLM model" in prompt:
            return default                       # accept the default model (sonnet)
        return default if default is not None else "high"
    responder = lambda m: {"stop_reason": "end_turn", "content": [
        {"type": "text",
         "text": "```cpp\n// path: E.cpp\n#include <pybind11/pybind11.h>\nclass E {};\n```"}]}
    res = gen.generate("Model with sigma.", classname="E", LLM="claude-sonnet-4-6",
                       effort="xhigh", output_path=str(tmp_path / "eff"), backend="Python",
                       interactive=True, verbose=False,
                       _responder=responder, _ask=ask,
                       _validate=lambda *a, **k: {"ok": True})
    assert not any("LLM model" in p for p in asked)    # LLM provided -> model menu NOT shown
    assert any("effort" in p.lower() for p in asked)   # invalid effort -> effort re-ask menu
    assert "xhigh" not in eff_opts["opts"]             # sonnet's levels exclude xhigh
    assert "high" in eff_opts["opts"]
    assert res["called_api"]


def test_provider_key_falls_back_to_auth_token(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-oat01-zzz")
    assert gen._provider_key("anthropic") == "sk-ant-oat01-zzz"


# ---------------------------------------------------------------------------
# key helpers: _mask_key / set_key / key_status  (session-only, never full key)
# ---------------------------------------------------------------------------
def test_mask_key_never_reveals_full_key():
    assert gen._mask_key("") == "<empty>"
    assert gen._mask_key(None) == "<empty>"          # falsy -> <empty>
    # short key (<=10 chars): 2-char prefix + "...", no last-4 leak
    assert gen._mask_key("sk-12345") == "sk..."
    assert "12345" not in gen._mask_key("sk-12345")
    # long key: first6 + "..." + last4, never the middle
    long_key = "sk-ant-api03-SECRETMIDDLEPART-tail"
    masked = gen._mask_key(long_key)
    assert masked == "sk-ant...tail"
    assert "SECRETMIDDLE" not in masked
    assert long_key not in masked


def test_set_key_sets_env_masks_output_and_is_readable(monkeypatch, capsys):
    # isolate: no pre-existing anthropic key/token leaks in.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    full = "sk-ant-api03-SUPERSECRETKEYVALUE-1234"
    ret = gen.set_key(full, "anthropic")
    assert ret == "anthropic"
    # the env var the read path uses is now set, and _provider_key returns it
    import os
    assert os.environ["ANTHROPIC_API_KEY"] == full
    assert gen._provider_key("anthropic") == full
    out = capsys.readouterr().out
    assert full not in out                            # full key NEVER printed
    assert "sk-ant" in out                            # masked prefix IS shown
    assert "Not saved to disk" in out


def test_set_key_openai_env_mapping(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    gen.set_key("sk-openai-abcdEFGH1234", "openai")
    import os
    assert os.environ["OPENAI_API_KEY"] == "sk-openai-abcdEFGH1234"
    assert gen._provider_key("openai") == "sk-openai-abcdEFGH1234"


def test_set_key_rejects_empty_and_unknown_provider():
    with pytest.raises(ValueError, match="non-empty"):
        gen.set_key("", "anthropic")
    with pytest.raises(ValueError, match="non-empty"):
        gen.set_key(None, "anthropic")               # type: ignore[arg-type]
    with pytest.raises(ValueError, match="provider"):
        gen.set_key("sk-whatever-key", "azure")


def test_key_status_returns_presence_dict_and_masks(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-PRESENTKEY-9999")
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    status = gen.key_status()
    assert status == {"anthropic": True, "openai": False, "google": False}
    out = capsys.readouterr().out
    assert "sk-ant-api03-PRESENTKEY-9999" not in out  # full key NEVER printed
    assert "sk-ant" in out                            # masked prefix shown
    assert "not set" in out                           # openai/google not set


# ---------------------------------------------------------------------------
# CLI auto-fallback is now OPT-IN: a blank key with use_cli=False -> offline path
# (even if a real `claude` happens to be on PATH).
# ---------------------------------------------------------------------------
def test_blank_key_no_cli_lands_offline(tmp_path, monkeypatch):
    # No key env vars, use_cli defaults to False -> must hit the offline PROMPT.txt
    # path and NOT shell out to any `claude` CLI that may be installed locally.
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # guard: if generate() ever tried the CLI transport, this would blow up.
    monkeypatch.setattr(gen, "_claude_p_call",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("CLI must not be used when use_cli=False")))
    td = tmp_path / "offline_optin"
    res = gen.generate("Linear regression.", classname="LR", API_key="",
                       output_path=str(td), backend="Python", interactive=False,
                       verbose=False)
    assert res["called_api"] is False
    assert (td / "PROMPT.txt").is_file()


def test_use_cli_true_without_claude_warns_and_falls_back(tmp_path, monkeypatch):
    # use_cli=True but no `claude` on PATH -> warn, then offline path (blank key).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(gen.shutil, "which", lambda _name: None)
    td = tmp_path / "cli_warn"
    with pytest.warns(UserWarning, match="not on the PATH"):
        res = gen.generate("Linear regression.", classname="LR", API_key="",
                           output_path=str(td), backend="Python", interactive=False,
                           use_cli=True, verbose=False)
    assert res["called_api"] is False
    assert (td / "PROMPT.txt").is_file()
