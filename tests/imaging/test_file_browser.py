"""Tests for the fsspec-based FileBrowser."""

from pathlib import Path

import fsspec
import pytest

from mtpy.imaging.bokeh_plots.file_browser import FileBrowser, is_remote


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "a.edi").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    sub = tmp_path / "line1" / "deep"
    sub.mkdir(parents=True)
    (tmp_path / "line1" / "c.edi").write_text("c")
    (sub / "d.edi").write_text("d")
    return tmp_path


def _names(browser):
    return sorted(browser._listing["name"])


def test_is_remote():
    assert is_remote("s3://bucket/x")
    assert is_remote("https://host/x")
    assert not is_remote("C:\\data")
    assert not is_remote("/home/user")
    assert not is_remote("file:///tmp")


def test_lists_current_folder_with_pattern(tree):
    fb = FileBrowser(directory=str(tree), file_pattern="*.edi")
    assert _names(fb) == ["a.edi"]
    fb.file_pattern = "*.*"
    assert _names(fb) == ["a.edi", "b.txt"]
    assert "line1" in fb._subdirs.options


def test_recursive_search_and_depth(tree):
    fb = FileBrowser(directory=str(tree), file_pattern="*.edi", recursive=True)
    assert _names(fb) == ["a.edi", "c.edi", "d.edi"]
    fb.max_depth = 1
    assert _names(fb) == ["a.edi"]
    fb.max_depth = 5
    fb.search = "c"
    assert _names(fb) == ["c.edi"]
    fb.search = "d*"
    assert _names(fb) == ["d.edi"]


def test_navigation(tree):
    fb = FileBrowser(directory=str(tree))
    fb._subdirs.value = "line1"
    assert Path(fb.directory) == tree / "line1"
    fb._up.clicks += 1
    assert Path(fb.directory) == tree


def test_selection_persists_across_folders(tree):
    fb = FileBrowser(directory=str(tree), file_pattern="*.edi")
    fb._table.selection = [0]
    first = list(fb.value)
    assert Path(first[0]).name == "a.edi"
    fb.directory = str(tree / "line1")
    assert fb.value == first
    fb._table.selection = [0]
    assert len(fb.value) == 2
    fb.directory = str(tree)
    assert fb._table.selection == [0]
    fb.value = []
    assert fb._table.selection == []


def test_bad_directory_reports_error(tree):
    fb = FileBrowser(directory=str(tree))
    fb.directory = str(tree / "missing")
    assert "⚠️" in fb._message.object
    assert fb._listing.empty


def test_remote_listing_and_download():
    fs = fsspec.filesystem("memory")
    fs.pipe("/mt/x1.edi", b"x1")
    fs.pipe("/mt/sub/x2.edi", b"x2")
    fb = FileBrowser(directory="memory://mt", file_pattern="*.edi", recursive=True)
    assert _names(fb) == ["x1.edi", "x2.edi"]
    paths = list(fb._listing["path"])
    assert all(p.startswith("memory://") for p in paths)
    local = fb.resolve_local(paths)
    assert [Path(p).read_bytes() for p in local] == [b"x1", b"x2"]
    assert fb.parent_of("memory://mt/sub") == "memory://mt"
    fs.rm("/mt", recursive=True)
