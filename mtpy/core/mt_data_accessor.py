"""xarray DataTree accessor for MTData station trees."""

from __future__ import annotations

from typing import Callable

import xarray as xr


@xr.register_datatree_accessor("mt")
class MTDataTreeAccessor:
    """Minimal convenience accessor for MT station trees.

    Notes
    -----
    This accessor intentionally stays structural and lightweight:
    path resolution, station dataset read/write, and small tree transforms.
    Higher-level workflows (metadata policy, indexing, lazy execution) remain
    in :class:`mtpy.core.mt_data.MTData`.
    """

    SURVEYS_NODE = "surveys"
    STATIONS_NODE = "stations"

    def __init__(self, datatree_obj: xr.DataTree) -> None:
        self._obj = datatree_obj

    def _station_path(self, survey: str, station: str) -> str:
        return f"{self.SURVEYS_NODE}/{survey}/{self.STATIONS_NODE}/{station}"

    def _iter_station_paths(self) -> list[str]:
        station_paths: list[str] = []

        try:
            surveys_node = self._obj[self.SURVEYS_NODE]
        except KeyError:
            return station_paths

        for survey_name, survey_node in surveys_node.children.items():
            stations_node = survey_node.children.get(self.STATIONS_NODE)
            if stations_node is None:
                continue
            for station_name in stations_node.children:
                station_paths.append(self._station_path(survey_name, station_name))

        return station_paths

    def _resolve_station_path(self, station_key: str) -> str:
        if not isinstance(station_key, str) or not station_key.strip():
            raise TypeError("station_key must be a non-empty string")

        key = station_key.strip()
        if key.startswith(f"{self.SURVEYS_NODE}/"):
            return key

        if "/" in key:
            survey, station = key.split("/", 1)
            return self._station_path(survey.strip(), station.strip())

        if "." in key:
            survey, station = key.split(".", 1)
            return self._station_path(survey.strip(), station.strip())

        matches = [
            path
            for path in self._iter_station_paths()
            if path.rsplit("/", 1)[-1] == key
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ValueError(
                f"Station key '{key}' is ambiguous. Provide survey/station."
            )

        raise KeyError(f"Station key '{station_key}' not found")

    @property
    def survey_names(self) -> list[str]:
        """Return survey names under the canonical surveys node."""
        try:
            return list(self._obj[self.SURVEYS_NODE].children.keys())
        except KeyError:
            return []

    @property
    def station_paths(self) -> list[str]:
        """Return canonical station tree paths."""
        return self._iter_station_paths()

    @property
    def short_station_paths(self) -> list[str]:
        """Return compact station keys in survey/station form."""
        return [
            f"{path.split('/')[1]}/{path.split('/')[3]}" for path in self.station_paths
        ]

    def has_station(self, station_key: str) -> bool:
        """Return ``True`` when station key resolves to an existing node."""
        try:
            station_path = self._resolve_station_path(station_key)
            self._obj[station_path]
            return True
        except (KeyError, TypeError, ValueError):
            return False

    def get_station_dataset(
        self,
        station_key: str,
        copy: bool = False,
        deep: bool = False,
    ) -> xr.Dataset:
        """Return one station dataset by canonical, short, or dotted key."""
        station_path = self._resolve_station_path(station_key)
        ds = self._obj[station_path].ds
        return ds.copy(deep=deep) if copy else ds

    def _ensure_parent_nodes(self, tree_obj: xr.DataTree, station_path: str) -> None:
        parent_path, _child_name = station_path.rsplit("/", 1)
        _surveys, survey_name, _stations = parent_path.split("/", 2)
        survey_path = f"{self.SURVEYS_NODE}/{survey_name}"

        if self.SURVEYS_NODE not in tree_obj.children:
            tree_obj[self.SURVEYS_NODE] = xr.DataTree(
                name=self.SURVEYS_NODE,
                dataset=xr.Dataset(),
            )

        try:
            tree_obj[survey_path]
        except KeyError:
            tree_obj[survey_path] = xr.DataTree(name=survey_name, dataset=xr.Dataset())

        try:
            tree_obj[parent_path]
        except KeyError:
            tree_obj[parent_path] = xr.DataTree(
                name=self.STATIONS_NODE,
                dataset=xr.Dataset(),
            )

    def set_station_dataset(
        self,
        station_key: str,
        station_ds: xr.Dataset,
        inplace: bool = True,
        create_parents: bool = True,
    ) -> xr.DataTree | None:
        """Set one station dataset and optionally return a modified copy."""
        station_path = self._resolve_station_path(station_key)
        tree_obj = self._obj if inplace else self._obj.copy(deep=True)

        if create_parents:
            self._ensure_parent_nodes(tree_obj, station_path)

        tree_obj[station_path] = xr.DataTree(
            name=station_path.rsplit("/", 1)[-1],
            dataset=station_ds,
        )

        if inplace:
            return None
        return tree_obj

    def map_stations(
        self,
        transform: Callable[[xr.Dataset], xr.Dataset],
        station_paths: list[str] | None = None,
        inplace: bool = False,
    ) -> xr.DataTree | None:
        """Apply a dataset transform to selected stations."""
        selected_paths = (
            self.station_paths
            if station_paths is None
            else [self._resolve_station_path(path) for path in station_paths]
        )

        source_tree = self._obj
        target_tree = self._obj if inplace else self._obj.copy(deep=True)
        source_accessor = MTDataTreeAccessor(source_tree)
        target_accessor = MTDataTreeAccessor(target_tree)

        for station_path in selected_paths:
            current_ds = source_accessor.get_station_dataset(station_path)
            updated_ds = transform(current_ds)
            target_accessor.set_station_dataset(station_path, updated_ds, inplace=True)

        if inplace:
            return None
        return target_tree

    def select_stations(
        self,
        station_paths: list[str],
        deep: bool = False,
    ) -> xr.DataTree:
        """Return a new tree containing only selected station nodes."""
        selected_paths = [self._resolve_station_path(path) for path in station_paths]

        out_tree = xr.DataTree(name=self._obj.name, dataset=self._obj.ds)
        out_tree.attrs.update(dict(self._obj.attrs))
        out_accessor = MTDataTreeAccessor(out_tree)

        for station_path in selected_paths:
            station_ds = self.get_station_dataset(station_path, copy=True, deep=deep)
            out_accessor._ensure_parent_nodes(out_tree, station_path)
            out_accessor.set_station_dataset(station_path, station_ds, inplace=True)

        return out_tree
