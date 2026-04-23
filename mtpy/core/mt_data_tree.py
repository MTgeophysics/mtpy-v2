# -*- coding: utf-8 -*-
"""
Scaffold for a tree-backed MT data container.

This class is an outline for migrating from OrderedDict-based MTData to an
Xarray tree representation for better scalability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import xarray as xr


if TYPE_CHECKING:
    from .mt import MT
    from .mt_data import MTData


class MTDataTree:
    """
    Tree-backed container for MT collection data.

    Notes
    -----
    Composition is intentionally used instead of inheriting from xarray's tree
    type. This keeps the public MT API independent from xarray internals and
    allows controlled migration from MTData.
    """

    ROOT_NAME = "root"
    SURVEYS_NODE = "surveys"
    STATIONS_NODE = "stations"

    def __init__(self, tree: Any | None = None, **attrs: Any) -> None:
        if hasattr(xr, "DataTree"):
            self._xarray_tree_cls = xr.DataTree
        elif hasattr(xr, "Tree"):
            self._xarray_tree_cls = xr.Tree
        else:
            raise ImportError(
                "xarray tree support is not available. Install a version of xarray "
                "that provides DataTree or Tree."
            )

        self.tree = (
            tree
            if tree is not None
            else self._xarray_tree_cls(name=self.ROOT_NAME, dataset=xr.Dataset())
        )

        # Keep root metadata lightweight and schema-focused at initialization.
        self.tree.attrs.setdefault("schema_name", "mtpy.mt_data_tree")
        self.tree.attrs.setdefault("schema_version", "0.1.0")
        self.tree.attrs.update(attrs)
        self.attrs = self.tree.attrs

        # Initialize a predictable top-level path for survey grouping.
        if self.SURVEYS_NODE not in self.tree.children:
            self.tree[self.SURVEYS_NODE] = self._xarray_tree_cls(
                name=self.SURVEYS_NODE, dataset=xr.Dataset()
            )

    @classmethod
    def from_mt_data(cls, mt_data: "MTData", **attrs: Any) -> "MTDataTree":
        """
        Build an MTDataTree from an MTData instance.

        TODO:
            - Map each survey/station key to a stable tree path.
            - Convert MT transfer function arrays into xarray variables.
            - Preserve metadata and coordinate reference frame conventions.
        """
        _ = mt_data
        return cls(**attrs)

    def to_mt_data(self) -> "MTData":
        """
        Convert tree content back into MTData.

        TODO:
            - Reconstruct MT objects from node datasets.
            - Restore collection-level metadata and modeling settings.
        """
        raise NotImplementedError("MTDataTree.to_mt_data is not implemented yet.")

    @staticmethod
    def _metadata_to_dict(metadata: Any) -> dict[str, Any]:
        """Safely convert mt_metadata objects to dictionaries."""
        if metadata is None:
            return {}
        if hasattr(metadata, "to_dict"):
            for kwargs in ({"single": True}, {}):
                try:
                    out = metadata.to_dict(**kwargs)
                    if isinstance(out, dict):
                        return out
                except TypeError:
                    continue
        return {}

    @staticmethod
    def _clean_name(value: Any, fallback: str) -> str:
        """Normalize path segment names for tree paths."""
        name = str(value).strip() if value is not None else ""
        if not name:
            return fallback
        return name.replace("/", "_")

    def _station_path(self, survey: str, station: str) -> str:
        """Build canonical station path under /surveys."""
        return f"{self.SURVEYS_NODE}/{survey}/{self.STATIONS_NODE}/{station}"

    @staticmethod
    def _coerce_mt_object(mt_obj: "MT | str | Path") -> "MT":
        """Convert supported inputs to an MT instance."""
        from .mt import MT

        if isinstance(mt_obj, MT):
            return mt_obj
        if isinstance(mt_obj, (str, Path)):
            m = MT(mt_obj)
            m.read()
            return m
        raise TypeError(
            "mt_obj must be an MT instance, filename string, or pathlib.Path"
        )

    def _path_exists(self, node_path: str) -> bool:
        """Check if a tree node path exists."""
        try:
            _ = self.tree[node_path]
            return True
        except KeyError:
            return False

    def add_station(
        self,
        mt_obj: "MT | str | Path | list[MT | str | Path]",
        overwrite: bool = True,
    ) -> str | list[str]:
        """
        Add an MT object as a station node in the tree.

        Node path pattern:
            /surveys/{survey_id}/stations/{station_id}

        Parameters
        ----------
        mt_obj : mtpy.core.MT, str, Path, or list
            MT object, filename, pathlib.Path, or a list of mixed supported
            input types.
        overwrite : bool, optional
            If False, raise if station path already exists.

        Returns
        -------
        str or list[str]
            Station node path for scalar inputs or list of paths for list
            inputs.
        """
        if mt_obj is None:
            raise TypeError("mt_obj cannot be None")

        if isinstance(mt_obj, list):
            return [self.add_station(item, overwrite=overwrite) for item in mt_obj]

        mt_obj = self._coerce_mt_object(mt_obj)

        survey = self._clean_name(
            getattr(mt_obj, "survey", None)
            or getattr(getattr(mt_obj, "survey_metadata", None), "id", None),
            "default",
        )
        station = self._clean_name(
            getattr(mt_obj, "station", None)
            or getattr(getattr(mt_obj, "station_metadata", None), "id", None),
            "unknown_station",
        )

        tf_obj = getattr(mt_obj, "_transfer_function", None)
        if tf_obj is None:
            raise TypeError("MT object is missing _transfer_function")

        if isinstance(tf_obj, xr.Dataset):
            station_ds = tf_obj.copy()
        elif hasattr(tf_obj, "to_xarray"):
            station_ds = tf_obj.to_xarray().copy()
        elif hasattr(tf_obj, "_dataset") and isinstance(tf_obj._dataset, xr.Dataset):
            station_ds = tf_obj._dataset.copy()
        else:
            raise TypeError("Could not extract xarray.Dataset from MT object")

        station_ds.attrs.update(
            {
                "survey": survey,
                "station": station,
                "tf_id": getattr(mt_obj, "tf_id", station),
                "coordinate_reference_frame": getattr(
                    mt_obj, "coordinate_reference_frame", None
                ),
                "impedance_units": getattr(mt_obj, "impedance_units", None),
                "survey_metadata": self._metadata_to_dict(
                    getattr(mt_obj, "survey_metadata", None)
                ),
                "station_metadata": self._metadata_to_dict(
                    getattr(mt_obj, "station_metadata", None)
                ),
            }
        )

        station_path = self._station_path(survey, station)
        if self._path_exists(station_path) and not overwrite:
            raise KeyError(f"Station path already exists: {station_path}")

        self.tree[station_path] = self._xarray_tree_cls(
            name=station, dataset=station_ds
        )
        return station_path

    def get_station(self, station_key: str) -> xr.Dataset:
        """Return a station dataset by tree path."""
        return self.tree[station_key].ds

    def remove_station(self, station_key: str) -> None:
        """Remove a station node from the tree."""
        if "/" not in station_key:
            del self.tree[station_key]
            return

        parent_path, child_name = station_key.rsplit("/", 1)
        del self.tree[parent_path][child_name]

    def keys(self) -> list[str]:
        """Return immediate child node keys for quick inspection."""
        return list(self.tree.children.keys())
