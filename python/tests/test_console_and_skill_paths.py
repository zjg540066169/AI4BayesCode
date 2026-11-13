"""Text output that must survive a narrow console code page, and the skill
installer's destination logic. Both matter mostly on Windows but are tested
on every OS."""
from __future__ import annotations

import contextlib
import importlib
import io
import os
import zipfile
from pathlib import Path

import pytest

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


# ---- install_skill() against a local archive in place of the GitHub download --

def _fake_download(monkeypatch, tmp_path, version="new"):
    """Serve install_skill() a zip laid out like GitHub's archive of the repo."""
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("AI4BayesCode-main/SKILL.md", "---\nname: AI4BayesCode\n---\n")
        zf.writestr("AI4BayesCode-main/VERSION.txt", version)
    monkeypatch.setattr(inst.urllib.request, "urlopen",
                        lambda url, timeout=None: open(archive, "rb"))
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    return home / ".claude" / "skills" / "AI4BayesCode"


def _old_skill(d: Path) -> None:
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text("---\nname: AI4BayesCode\n---\n", encoding="utf-8")
    (d / "VERSION.txt").write_text("old", encoding="utf-8")


def _leftovers(d: Path) -> list:
    """Staging folders left beside or inside the skills folder."""
    return sorted(p.name for w in (d.parent.parent, d.parent) for p in w.glob(".AI4BayesCode-*"))


def test_install_skill_replaces_an_old_copy(monkeypatch, tmp_path):
    d = _fake_download(monkeypatch, tmp_path)
    _old_skill(d)
    inst.install_skill("claude", quiet=True)
    assert (d / "VERSION.txt").read_text(encoding="utf-8") == "new"
    assert _leftovers(d) == []


def test_install_skill_leaves_a_git_checkout_alone(monkeypatch, tmp_path):
    d = _fake_download(monkeypatch, tmp_path)
    _old_skill(d)
    (d / ".git").mkdir()
    with pytest.raises(FileExistsError, match="git checkout"):
        inst.install_skill("claude", quiet=True)
    assert (d / ".git").is_dir() and (d / "VERSION.txt").read_text(encoding="utf-8") == "old"
    inst.install_skill("claude", quiet=True, force=True)
    assert not (d / ".git").exists()
    assert (d / "VERSION.txt").read_text(encoding="utf-8") == "new"


def test_install_skill_keeps_the_old_copy_when_it_cannot_be_moved(monkeypatch, tmp_path):
    d = _fake_download(monkeypatch, tmp_path)
    _old_skill(d)
    real_replace = os.replace

    def locked(src, dst):   # what Windows does while a file in the folder is open
        if Path(src) == d:
            raise PermissionError(13, "The process cannot access the file", str(src))
        return real_replace(src, dst)
    monkeypatch.setattr(inst.os, "replace", locked)
    with pytest.raises(RuntimeError, match="unchanged"):
        inst.install_skill("claude", quiet=True)
    assert (d / "SKILL.md").is_file() and (d / "VERSION.txt").read_text(encoding="utf-8") == "old"
    assert _leftovers(d) == []


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need extra rights on Windows")
def test_install_skill_updates_the_folder_a_link_points_to(monkeypatch, tmp_path):
    d = _fake_download(monkeypatch, tmp_path)
    target = tmp_path / "shared copy" / "AI4BayesCode"
    _old_skill(target)
    d.parent.mkdir(parents=True)
    d.symlink_to(target, target_is_directory=True)
    inst.install_skill("claude", quiet=True)
    assert d.is_symlink() and (d / "VERSION.txt").read_text(encoding="utf-8") == "new"
    assert (target / "VERSION.txt").read_text(encoding="utf-8") == "new"
    assert _leftovers(target) == []


@pytest.mark.skipif(os.name == "nt", reason="symbolic links need extra rights on Windows")
def test_install_skill_leaves_a_linked_git_checkout_alone(monkeypatch, tmp_path):
    d = _fake_download(monkeypatch, tmp_path)
    target = tmp_path / "dev clone"
    _old_skill(target)
    (target / ".git").mkdir()
    d.parent.mkdir(parents=True)
    d.symlink_to(target, target_is_directory=True)
    with pytest.raises(FileExistsError, match="git checkout"):
        inst.install_skill("claude", quiet=True)
    assert (target / ".git").is_dir() and d.is_symlink()


@pytest.mark.skipif(os.name == "nt" or (hasattr(os, "geteuid") and os.geteuid() == 0),
                    reason="needs POSIX permissions that apply to this user")
def test_install_skill_stages_inside_the_skills_folder_when_beside_it_fails(monkeypatch, tmp_path):
    # A skills folder that is its own mount, or whose parent cannot be
    # written: the staging folders go inside the skills folder instead.
    d = _fake_download(monkeypatch, tmp_path)
    _old_skill(d)
    os.chmod(d.parent.parent, 0o555)
    try:
        inst.install_skill("claude", quiet=True)
    finally:
        os.chmod(d.parent.parent, 0o755)
    assert (d / "VERSION.txt").read_text(encoding="utf-8") == "new"
    assert _leftovers(d) == []
