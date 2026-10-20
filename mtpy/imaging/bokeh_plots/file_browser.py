"""Directory-tree aware file browser for Panel apps (local and remote).

:class:`FileBrowser` is built on :mod:`fsspec`, so the *location* can be a
local folder or any URL fsspec understands (``s3://bucket/prefix``,
``gs://…``, ``https://…``, ``ftp://…``, ``sftp://…``, ``az://…`` — the matching
backend package, e.g. ``s3fs``, must be installed).

Features
--------
* Type or paste a location, go *Up*, or jump into a sub-folder.
* Search the whole tree below the current location (name substring or glob,
  combined with the file-type filter) with a depth limit and a result cap.
* Multi-select files with check boxes; the selection is kept while moving
  between folders or changing the search.
* :meth:`FileBrowser.resolve_local` downloads remote selections to a local
  cache so existing file readers keep working on plain paths.

The public state mirrors :class:`panel.widgets.FileSelector`
(``value``, ``directory``, ``file_pattern``) so it can replace it directly.
"""

from __future__ import annotations

import fnmatch
import hashlib
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import panel as pn
import param


_COLUMNS = ["name", "folder", "size", "modified", "path"]


def is_remote(location: str) -> bool:
    """Return True if *location* is a non-local URL (``s3://``, ``https://``…)."""
    proto, sep, _ = str(location).partition("://")
    return bool(sep) and proto.lower() not in ("file", "local") and len(proto) > 1


def _human_size(nbytes: Any) -> str:
    try:
        size = float(nbytes)
    except (TypeError, ValueError):
        return ""
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return ""


def _modified_str(info: dict) -> str:
    for key in ("mtime", "LastModified", "last_modified", "updated", "modified"):
        val = info.get(key)
        if val is None:
            continue
        try:
            if isinstance(val, (int, float)):
                return datetime.fromtimestamp(val).strftime("%Y-%m-%d %H:%M")
            if hasattr(val, "strftime"):
                return val.strftime("%Y-%m-%d %H:%M")
            return str(val)[:16]
        except (OverflowError, OSError, ValueError):
            return ""
    return ""


class FileBrowser(pn.viewable.Viewer):
    """Searchable, multi-select file browser over local or remote filesystems.

    Parameters
    ----------
    directory : str
        Current location (local path or fsspec URL).
    file_pattern : str
        Glob applied to file names (e.g. ``"*.edi"``; ``"*.*"`` shows all).
    value : list of str
        Selected files (absolute paths, or full URLs for remote files).
    recursive : bool
        Search below *directory* instead of listing only its content.
    search : str
        Name filter: a substring, or a glob if it contains ``*``/``?``.
    max_depth : int
        Depth limit for recursive search.
    max_results : int
        Cap on listed files, so huge trees stay responsive.
    storage_options : dict
        Extra arguments forwarded to :func:`fsspec.core.url_to_fs`
        (credentials, ``anon=True`` for public S3, …).
    """

    value = param.List(default=[], doc="Selected file paths / URLs.")
    directory = param.String(default="", doc="Current location.")
    file_pattern = param.String(default="*.*")
    recursive = param.Boolean(default=False)
    search = param.String(default="")
    max_depth = param.Integer(default=5, bounds=(1, 20))
    max_results = param.Integer(default=2000, bounds=(10, 100000))
    storage_options = param.Dict(default={})

    def __init__(self, **params: Any) -> None:
        params.setdefault("directory", str(Path.home()))
        super().__init__(**params)
        self._listing = pd.DataFrame(columns=_COLUMNS)
        self._syncing = False
        self._build_widgets()
        self.param.watch(self._on_value_changed, "value")
        self.param.watch(
            self.refresh,
            ["directory", "file_pattern", "recursive", "search", "max_depth"],
        )
        self.refresh()

    # ── Filesystem helpers ────────────────────────────────────────────────

    def _fs_and_path(self, location: str | None = None) -> tuple[Any, str]:
        import fsspec.core

        loc = location if location is not None else self.directory
        if is_remote(loc):
            return fsspec.core.url_to_fs(loc, **self.storage_options)
        return fsspec.core.url_to_fs(str(Path(loc).expanduser()))

    @staticmethod
    def _display_path(fs: Any, path: str, remote: bool) -> str:
        if remote:
            return fs.unstrip_protocol(path)
        return str(Path(path))

    def parent_of(self, location: str) -> str:
        """Return the parent location, or *location* itself at the root."""
        if is_remote(location):
            proto, _, rest = location.partition("://")
            rest = rest.rstrip("/")
            if "/" not in rest:
                return location
            return f"{proto}://{rest.rsplit('/', 1)[0]}"
        path = Path(location).expanduser()
        return str(path.parent)

    def _matches(self, name: str) -> bool:
        pattern = (self.file_pattern or "*.*").lower()
        if not fnmatch.fnmatch(name.lower(), pattern):
            return False
        text = (self.search or "").strip().lower()
        if not text:
            return True
        if any(ch in text for ch in "*?["):
            return fnmatch.fnmatch(name.lower(), text)
        return text in name.lower()

    def _scan(self) -> tuple[list[dict], list[str], bool]:
        """Return (file rows, sub-folder names, truncated) for the location."""
        remote = is_remote(self.directory)
        fs, root = self._fs_and_path()
        root = root.rstrip("/") or root
        if not fs.isdir(root):
            raise FileNotFoundError(f"Not a directory: {self.directory}")

        dirs: list[str] = []
        entries: dict[str, dict] = {}
        for info in fs.ls(root, detail=True):
            if info.get("type") == "directory":
                dirs.append(str(info["name"]).rstrip("/").rsplit("/", 1)[-1])
        if self.recursive:
            entries = fs.find(root, maxdepth=self.max_depth, detail=True)
        else:
            entries = {
                i["name"]: i for i in fs.ls(root, detail=True) if i["type"] == "file"
            }

        rows: list[dict] = []
        truncated = False
        for name, info in entries.items():
            if info.get("type") == "directory":
                continue
            base = str(name).rstrip("/").rsplit("/", 1)[-1]
            if not self._matches(base):
                continue
            if len(rows) >= self.max_results:
                truncated = True
                break
            rel = str(name)[len(root) :].strip("/\\")
            folder = rel.rsplit("/", 1)[0] if "/" in rel else ""
            rows.append(
                {
                    "name": base,
                    "folder": folder,
                    "size": _human_size(info.get("size")),
                    "modified": _modified_str(info),
                    "path": self._display_path(fs, str(name), remote),
                }
            )
        rows.sort(key=lambda r: (r["folder"].lower(), r["name"].lower()))
        return rows, sorted(dirs, key=str.lower), truncated

    # ── Widgets ───────────────────────────────────────────────────────────

    def _build_widgets(self) -> None:
        self._location = pn.widgets.TextInput(
            name="Location (folder path or URL such as s3://bucket/prefix)",
            value=self.directory,
            sizing_mode="stretch_width",
        )
        self._location.param.watch(self._on_location_entered, "value")
        self._up = pn.widgets.Button(name="⬆ Up", width=70, align="end")
        self._up.on_click(lambda e: self._go(self.parent_of(self.directory)))
        self._subdirs = pn.widgets.Select(
            name="Open sub-folder", options=[""], sizing_mode="stretch_width"
        )
        self._subdirs.param.watch(self._on_subdir_selected, "value")
        self._search_input = pn.widgets.TextInput(
            name="Search names",
            placeholder="e.g. gv10  or  *_2020*",
            sizing_mode="stretch_width",
        )
        self._search_input.param.watch(
            lambda e: setattr(self, "search", e.new), "value_input"
        )
        self._search_input.param.watch(
            lambda e: setattr(self, "search", e.new), "value"
        )
        self._recursive_box = pn.widgets.Checkbox(
            name="Include subfolders", value=self.recursive
        )
        self._recursive_box.link(self, value="recursive")
        self._depth = pn.widgets.IntInput(
            name="Max depth", value=self.max_depth, start=1, end=20, width=90
        )
        self._depth.link(self, value="max_depth")
        self._table = pn.widgets.Tabulator(
            self._listing,
            disabled=True,
            show_index=False,
            selectable="checkbox",
            hidden_columns=["path"],
            layout="fit_columns",
            height=300,
            sizing_mode="stretch_width",
            pagination=None,
        )
        self._table.param.watch(self._on_table_selection, "selection")
        self._select_all = pn.widgets.Button(name="Select all shown", width=120)
        self._select_all.on_click(self._on_select_all)
        self._clear = pn.widgets.Button(name="Clear", width=70)
        self._clear.on_click(lambda e: setattr(self, "value", []))
        self._message = pn.pane.Markdown(
            "", styles={"color": "#777", "font-size": "0.85em"}
        )

    def __panel__(self) -> pn.viewable.Viewable:
        return pn.Column(
            pn.Row(self._location, self._up),
            self._subdirs,
            pn.Row(self._search_input, self._recursive_box, self._depth),
            self._table,
            pn.Row(self._select_all, self._clear),
            self._message,
            sizing_mode="stretch_width",
        )

    # ── Callbacks ─────────────────────────────────────────────────────────

    def _go(self, location: str) -> None:
        self.directory = location

    def _on_location_entered(self, event: param.parameterized.Event) -> None:
        loc = event.new.strip()
        if loc and loc != self.directory:
            self.directory = loc

    def _on_subdir_selected(self, event: param.parameterized.Event) -> None:
        if not event.new:
            return
        base = self.directory.rstrip("/\\")
        if is_remote(base):
            self.directory = f"{base}/{event.new}"
        else:
            self.directory = str(Path(self.directory) / event.new)

    def refresh(self, *_events: Any) -> None:
        """Re-read the current location and rebuild the listing."""
        self._table.loading = True
        try:
            rows, dirs, truncated = self._scan()
        except Exception as exc:
            rows, dirs, truncated = [], [], False
            self._message.object = f"⚠️ `{type(exc).__name__}`: {exc}"
        else:
            note = f"{len(rows)} file(s) shown"
            if truncated:
                note += f" (limited to {self.max_results}; refine the search)"
            if self.recursive:
                note += f" · searched {self.max_depth} level(s) deep"
            self._message.object = note
        finally:
            self._table.loading = False

        self._listing = pd.DataFrame(rows, columns=_COLUMNS)
        if self._location.value != self.directory:
            self._location.value = self.directory
        self._subdirs.options = [""] + dirs
        self._subdirs.value = ""
        self._syncing = True
        try:
            self._table.value = self._listing
            self._push_selection_to_table()
        finally:
            self._syncing = False

    def _push_selection_to_table(self) -> None:
        chosen = set(self.value)
        self._table.selection = [
            i for i, p in enumerate(self._listing["path"]) if p in chosen
        ]

    def _on_table_selection(self, event: param.parameterized.Event) -> None:
        """Merge the table's picks with picks made in other folders/searches."""
        if self._syncing:
            return
        shown = set(self._listing["path"])
        kept = [p for p in self.value if p not in shown]
        picked = [
            self._listing["path"].iloc[i] for i in event.new if i < len(self._listing)
        ]
        self._syncing = True
        try:
            self.value = kept + picked
        finally:
            self._syncing = False

    def _on_value_changed(self, event: param.parameterized.Event) -> None:
        if self._syncing:
            return
        self._syncing = True
        try:
            self._push_selection_to_table()
        finally:
            self._syncing = False

    def _on_select_all(self, event: Any) -> None:
        self._table.selection = list(range(len(self._listing)))

    # ── Remote support ────────────────────────────────────────────────────

    def resolve_local(self, paths: list[str] | None = None) -> list[str]:
        """Return local paths for *paths*, downloading remote ones to a cache.

        Local paths pass through unchanged. Remote files are copied into a
        per-URL folder under the system temp directory (re-used on repeat
        loads) so file readers that need real paths keep working.
        """
        paths = list(self.value if paths is None else paths)
        cache_root = Path(tempfile.gettempdir()) / "mtpy_remote_cache"
        out: list[str] = []
        for path in paths:
            if not is_remote(path):
                out.append(path)
                continue
            fs, remote_path = self._fs_and_path(path)
            digest = hashlib.sha1(path.encode()).hexdigest()[:10]
            local_dir = cache_root / digest
            local_dir.mkdir(parents=True, exist_ok=True)
            local = local_dir / remote_path.rstrip("/").rsplit("/", 1)[-1]
            if not local.exists():
                fs.get(remote_path, str(local))
            out.append(str(local))
        return out
