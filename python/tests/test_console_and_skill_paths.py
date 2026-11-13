"""Text output that must survive a narrow console code page, and the skill
installer's destination logic. Both matter mostly on Windows but are tested
on every OS."""
from __future__ import annotations

import contextlib
import importlib
import io
from pathlib import Path

import AI4BayesCode
from AI4BayesCode._console import glyph, safe_print

# The package re-exports install_skill() under the submodule's name.
inst = importlib.import_module("AI4BayesCode.install_skill")


def _cp1252_stdout():
    # What a piped or redirected stdout is on a Western-European Windows system.
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", errors="strict")


def test_safe_print_degrades_instead_of_raising():
    out = _cp1252_stdout()
    safe_print("✓ loaded σ ∝ 1/σ", file=out)
    out.flush()
    text = out.buffer.getvalue().decode("cp1252")
    assert "loaded" in text and "?" in text


def test_glyph_falls_back_on_a_narrow_stdout():
    with contextlib.redirect_stdout(_cp1252_stdout()):
        assert glyph("✓", "OK") == "OK"
    assert glyph("x", "OK") == "x"


def test_doc_reads_utf8_with_bom_and_prints_on_cp1252(tmp_path):
    cpp = tmp_path / "Bom.cpp"
    cpp.write_bytes("﻿// Model: p(sigma) ∝ 1/sigma\n"
                    "class Bom {};\n".encode("utf-8"))
    out = _cp1252_stdout()
    with contextlib.redirect_stdout(out):
        AI4BayesCode.doc(str(cpp))           # must not raise
    out.flush()
    assert out.buffer.getvalue()


def test_skill_destinations(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    assert inst._skill_dest("claude") == tmp_path / ".claude" / "skills" / "AI4BayesCode"
    assert inst._skill_dest("codex") == tmp_path / ".agents" / "skills" / "AI4BayesCode"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "cfg"))
    assert inst._skill_dest("claude") == tmp_path / "cfg" / "skills" / "AI4BayesCode"


def test_only_a_skill_folder_may_be_replaced(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert inst._is_skill_dir(empty)
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("---\nname: AI4BayesCode\n---\n", encoding="utf-8")
    assert inst._is_skill_dir(skill)
    other = tmp_path / "other"
    other.mkdir()
    (other / "notes.txt").write_text("keep", encoding="utf-8")
    assert not inst._is_skill_dir(other)


def test_install_skill_rejects_unknown_agents():
    try:
        inst.install_skill("cursor")
    except ValueError as e:
        assert "claude" in str(e)
    else:
        raise AssertionError("expected ValueError")
