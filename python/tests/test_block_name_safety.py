"""`remove_block` must never delete anything but one block directory.

`blocks_path(name)` is `os.path.join(library_root, name)`, and os.path.join
has two behaviours that turn a bad name into a destructive path:

  * `os.path.join(root, "")` returns the library ROOT. It passes `isdir()`,
    so `remove_block("")` used to rmtree every installed block.
  * `os.path.join(root, "/abs/path")` DISCARDS the root entirely, so an
    absolute name reached straight out of the library.

`"../x"` escapes by the ordinary route. Run:

    python -m pytest python/tests/test_block_name_safety.py
"""

from __future__ import annotations

import os

import pytest

from AI4BayesCode.install_block import (_check_block_name, blocks_path,
                                        installed_blocks, remove_block)


@pytest.fixture()
def library(tmp_path, monkeypatch):
    """A block library with two blocks, plus a sibling directory that no block
    operation is ever allowed to touch."""
    root = tmp_path / "data_home"
    for b in ("blockA", "blockB"):
        (root / "blocks_download" / b).mkdir(parents=True)
        (root / "blocks_download" / b / f"{b}.hpp").write_text("// x\n")
    (root / "PRECIOUS_USER_DATA").mkdir(parents=True)
    (root / "PRECIOUS_USER_DATA" / "keepme.txt").write_text("do not delete\n")
    monkeypatch.setenv("AI4BAYESCODE_DATA_HOME", str(root))
    # installed_blocks() reports BOTH tiers, and the local tier is relative to
    # the working directory -- so the cwd has to be isolated too, or the repo's
    # own ./blocks_local/ leaks into every assertion here.
    monkeypatch.chdir(tmp_path)
    return root


# Every one of these resolved to a real directory outside (or above) the single
# block it names.
BAD_NAMES = [
    "",                     # -> the library root itself
    ".",
    "..",                   # -> the data home, i.e. the library's parent
    "../PRECIOUS_USER_DATA",
    "a/b",
    "a\\b",
    None,
    42,
    ["blockA"],
]


@pytest.mark.parametrize("name", BAD_NAMES)
def test_remove_block_rejects_non_names(library, name):
    with pytest.raises((ValueError, TypeError)):
        remove_block(name)
    assert sorted(installed_blocks()) == ["blockA", "blockB"]
    assert (library / "PRECIOUS_USER_DATA" / "keepme.txt").exists()


def test_remove_block_rejects_an_absolute_path(library):
    """os.path.join(root, "/abs") == "/abs" -- the library prefix vanishes, so
    without the guard this deleted a directory with no relation to the library."""
    victim = str(library / "PRECIOUS_USER_DATA")
    with pytest.raises(ValueError):
        remove_block(victim)
    assert os.path.isdir(victim)


# None is NOT a bad name for blocks_path -- it is the documented "give me the
# library root" form. It IS rejected by remove_block, which needs a block.
@pytest.mark.parametrize("name",
                         [n for n in BAD_NAMES if n is not None] + ["/abs/path"])
def test_blocks_path_rejects_non_names(library, name):
    """The guard sits in blocks_path, so install_block and any future caller
    inherit it rather than each needing their own check."""
    with pytest.raises((ValueError, TypeError)):
        blocks_path(name)


def test_blocks_path_with_no_name_is_still_the_library_root(library):
    assert blocks_path() == str(library / "blocks_download")


def test_a_real_block_still_removes(library):
    assert remove_block("blockA") is True
    assert sorted(installed_blocks()) == ["blockB"]
    assert remove_block("blockA") is False        # already gone, not an error


def test_check_block_name_accepts_ordinary_names():
    for ok in ("particle_gibbs_block", "nngp_gaussian_gibbs_block", "b1", "A-B"):
        assert _check_block_name(ok) == ok


# ---------------------------------------------------------------------------
# A registry that cannot be reached is not the same as a block that is absent
# ---------------------------------------------------------------------------
def test_unreachable_registry_is_not_reported_as_a_missing_block(monkeypatch):
    """install_block used to answer "Block 'x' is not in the registry" for ANY
    failure of the manifest fetch. Offline, behind a proxy, or during a GitHub
    outage the message is identical to a typo, so the user goes looking for a
    misspelling that is not there. The registry INDEX settles which it is."""
    import importlib
    m = importlib.import_module("AI4BayesCode.install_block")
    dead = "https://127.0.0.1:9/unreachable"
    monkeypatch.setattr(m, "_raw_url", lambda p: dead)
    monkeypatch.setattr(m, "_tree_url", lambda *a, **k: dead)

    with pytest.raises(ConnectionError) as ei:
        m.install_block("beta_gibbs_block")
    msg = str(ei.value)
    assert "Cannot reach" in msg
    assert "is unknown" in msg
    assert "not in the registry" not in msg


def test_a_reachable_registry_still_reports_a_genuinely_absent_block(monkeypatch):
    """The other direction: with the index reachable, an absent block must
    still say so -- otherwise the fix would hide real typos behind a network
    complaint."""
    import importlib
    m = importlib.import_module("AI4BayesCode.install_block")
    monkeypatch.setattr(m, "_raw_url", lambda p: "https://127.0.0.1:9/unreachable")
    monkeypatch.setattr(m, "available_blocks", lambda: ["beta_gibbs_block"])

    with pytest.raises(ValueError) as ei:
        m.install_block("no_such_block_xyz")
    msg = str(ei.value)
    assert "not in the registry" in msg
    assert "beta_gibbs_block" in msg


# ---------------------------------------------------------------------------
# A project-local block is installed; it is just not the download cache's
# ---------------------------------------------------------------------------
def test_removing_a_project_local_block_says_where_it_is(tmp_path, monkeypatch, capsys):
    """installed_blocks() lists both tiers and the local tier reaches the
    compiler, so answering "not installed" here flatly contradicts what the user
    was just shown, and leaves them with no idea how to get rid of it."""
    import importlib
    m = importlib.import_module("AI4BayesCode.install_block")
    local = tmp_path / "blocks_local" / "toy_block"
    local.mkdir(parents=True)
    (local / "toy_block.hpp").write_text("// x\n")
    dl = tmp_path / "dh" / "blocks_download" / "other_block"
    dl.mkdir(parents=True)
    (dl / "other_block.hpp").write_text("// x\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI4BAYESCODE_DATA_HOME", str(tmp_path / "dh"))

    assert "toy_block" in m.installed_blocks()
    assert m.remove_block("toy_block") is False
    assert "project-local block" in capsys.readouterr().out
    assert local.is_dir()                       # nothing was deleted
    assert m.installed_blocks("download") == ["other_block"]


def test_a_name_in_neither_tier_still_reports_not_installed(tmp_path, monkeypatch, capsys):
    import importlib
    m = importlib.import_module("AI4BayesCode.install_block")
    (tmp_path / "dh" / "blocks_download").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI4BAYESCODE_DATA_HOME", str(tmp_path / "dh"))
    assert m.remove_block("really_not_here") is False
    assert "is not installed" in capsys.readouterr().out
