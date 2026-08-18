"""A changed contributed block must invalidate the compile cache.

`_signature()` used to hash only the model source, the compile flags, and the
compiler id. Reinstalling a block -- `install_block(force=True)` fetching a new
version, or a hand-edit under `./blocks_local/` -- leaves the `-I` paths
byte-identical, so the signature did not move and `source()` handed back the
`.so` built from the PREVIOUS header. A user who updated a block and re-ran kept
executing the old sampler: a silent wrong posterior, which is the exact failure
the contributed-block design exists to prevent.

These tests exercise `_signature` / `_contributed_block_fingerprint` directly
rather than compiling, so they are fast and need no toolchain. The end-to-end
compile path is covered by hand in the same way (edit a header, re-run in a
fresh process, observe the new value) -- see the commit that added this file.

Run: python -m pytest python/tests/test_block_cache_invalidation.py
"""

from __future__ import annotations

import os

import pytest

from AI4BayesCode.source import _contributed_block_fingerprint, _signature

FLAGS_FIXED = ["-I/opt/core/include", "-std=c++17", "-DMCMC_ENABLE_ARMA_WRAPPERS"]


@pytest.fixture()
def block(tmp_path, monkeypatch):
    """A project-local block bundle, plus the -I flags that would reach it."""
    root = tmp_path / "blocks_local" / "toy_block"
    root.mkdir(parents=True)
    hdr = root / "toy_block.hpp"
    hdr.write_text("namespace toy { inline double magic() { return 1.0; } }\n")
    (root / "manifest.dcf").write_text("Block: toy_block\nVersion: 0.1.0\n")
    monkeypatch.chdir(tmp_path)
    return hdr, FLAGS_FIXED + [f"-I{root}"]


def _bump_mtime(path, seconds=2):
    """Advance mtime deterministically -- a same-second rewrite can otherwise
    leave mtime_ns unchanged on a coarse filesystem."""
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 1_000_000_000))


def test_editing_a_block_header_moves_the_signature(block):
    hdr, flags = block
    text = "// the model source, unchanged throughout\n"
    before = _signature(text, flags, [], "clang-19")

    hdr.write_text("namespace toy { inline double magic() { return 42.0; } }\n")
    _bump_mtime(hdr)
    after = _signature(text, flags, [], "clang-19")

    assert before != after, (
        "the model source and the -I flags are identical, so only the block's "
        "CONTENT distinguishes these two builds"
    )


def test_an_untouched_block_keeps_the_signature(block):
    """The cache still has to work -- otherwise every call recompiles."""
    _, flags = block
    text = "// model\n"
    assert _signature(text, flags, [], "clang-19") == \
           _signature(text, flags, [], "clang-19")


def test_a_new_file_in_the_bundle_moves_the_signature(block):
    """A block that gains a vendored header is a different block."""
    hdr, flags = block
    text = "// model\n"
    before = _signature(text, flags, [], "clang-19")
    (hdr.parent / "toy_detail.hpp").write_text("// new\n")
    assert _signature(text, flags, [], "clang-19") != before


def test_removing_a_file_from_the_bundle_moves_the_signature(block):
    hdr, flags = block
    text = "// model\n"
    extra = hdr.parent / "toy_detail.hpp"
    extra.write_text("// new\n")
    before = _signature(text, flags, [], "clang-19")
    extra.unlink()
    assert _signature(text, flags, [], "clang-19") != before


def test_core_include_dirs_are_not_fingerprinted(tmp_path, monkeypatch):
    """Only the contributed-block tiers are walked. The core and vendored trees
    ship with the package and move only when the package is reinstalled, so
    stat-ing all of Eigen on every call would be pure cost."""
    core = tmp_path / "vendored" / "eigen"
    core.mkdir(parents=True)
    (core / "Dense.hpp").write_text("// a\n")
    flags = [f"-I{core}"]
    monkeypatch.chdir(tmp_path)

    before = _contributed_block_fingerprint(flags)
    (core / "Dense.hpp").write_text("// b -- much longer content here\n")
    _bump_mtime(core / "Dense.hpp")
    assert _contributed_block_fingerprint(flags) == before


def test_fingerprint_survives_a_vanished_directory(tmp_path, monkeypatch):
    """remove_block() can delete a directory that is still on a stale flag
    list; fingerprinting must not raise."""
    monkeypatch.chdir(tmp_path)
    gone = tmp_path / "blocks_local" / "not_there"
    assert _contributed_block_fingerprint([f"-I{gone}"]) is not None


# ---------------------------------------------------------------------------
# Bundle validation and tier listing
# ---------------------------------------------------------------------------
def _write_bundle(root, name, *, manifest=True, header=True, block_name=None,
                  drop_fields=()):
    d = root / name
    d.mkdir(parents=True)
    if manifest:
        from AI4BayesCode.install_block import _REQUIRED_MANIFEST_FIELDS
        fields = {f: "x" for f in _REQUIRED_MANIFEST_FIELDS}
        fields["Block"] = block_name or name
        for f in drop_fields:
            fields.pop(f, None)
        (d / "manifest.dcf").write_text(
            "".join(f"{k}: {v}\n" for k, v in fields.items()))
    if header:
        (d / f"{name}.hpp").write_text("// x\n")
    return d


def test_a_well_formed_bundle_validates(tmp_path):
    from AI4BayesCode.install_block import validate_bundle
    d = _write_bundle(tmp_path, "toy_block")
    assert validate_bundle(str(d)) == []


def test_missing_manifest_header_and_fields_are_all_reported(tmp_path):
    """A bundle with none of these used to go on the compile -I path in
    silence, and installed_blocks() reported it as a block."""
    from AI4BayesCode.install_block import validate_bundle
    empty = tmp_path / "empty_block"; empty.mkdir()
    probs = validate_bundle(str(empty))
    assert any("manifest.dcf is missing" in p for p in probs)
    assert any("empty_block.hpp is missing" in p for p in probs)

    thin = _write_bundle(tmp_path, "thin_block",
                         drop_fields=("Version", "License", "SelectWhen"))
    probs = validate_bundle(str(thin))
    assert len(probs) == 1 and "missing required field(s)" in probs[0]
    for f in ("Version", "License", "SelectWhen"):
        assert f in probs[0]


def test_a_manifest_naming_a_different_block_is_reported(tmp_path):
    from AI4BayesCode.install_block import validate_bundle
    d = _write_bundle(tmp_path, "toy_block", block_name="something_else")
    probs = validate_bundle(str(d))
    assert any("directory is 'toy_block'" in p for p in probs)


def test_invalid_bundles_are_skipped_on_the_include_path(tmp_path, monkeypatch):
    import warnings
    from AI4BayesCode.install_block import _block_include_flags
    local = tmp_path / "blocks_local"
    _write_bundle(local, "good_block")
    (local / "junk_block").mkdir(parents=True)          # no manifest, no header
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI4BAYESCODE_DATA_HOME", str(tmp_path / "dh"))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        flags = [f for f in _block_include_flags() if "blocks_local" in f]
    assert any(f.endswith("good_block") for f in flags)
    assert not any("junk_block" in f for f in flags)
    assert any("junk_block" in str(x.message) for x in w)


def test_installed_blocks_sees_the_local_tier(tmp_path, monkeypatch):
    """The local tier reaches the compiler, so a user must be able to see it.
    installed_blocks() used to read the download cache only."""
    from AI4BayesCode.install_block import installed_blocks
    _write_bundle(tmp_path / "blocks_local", "local_only_block")
    dl = tmp_path / "dh" / "blocks_download"
    _write_bundle(dl, "downloaded_block")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AI4BAYESCODE_DATA_HOME", str(tmp_path / "dh"))
    assert installed_blocks() == ["downloaded_block", "local_only_block"]
    assert installed_blocks("local") == ["local_only_block"]
    assert installed_blocks("download") == ["downloaded_block"]
    with pytest.raises(ValueError):
        installed_blocks("bogus")
