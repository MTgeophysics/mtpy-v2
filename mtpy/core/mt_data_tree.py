# -*- coding: utf-8 -*-
"""
Scaffold for a tree-backed MT data container.

This class is an outline for migrating from OrderedDict-based MTData to an
Xarray tree representation for better scalability.
"""

from __future__ import annotations

import importlib
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from mtpy.core.transfer_function import IMPEDANCE_UNITS
from mtpy.modeling.errors import ModelErrors

from .mt_data_tree_index import MTDataTreeIndexStore
from .mt_dataframe import MTDataFrame


COORDINATE_REFERENCE_FRAME_OPTIONS = {
    "+": "ned",
    "-": "enu",
    "z+": "ned",
    "z-": "enu",
    "nez+": "ned",
    "enz-": "enu",
    "ned": "ned",
    "enu": "enu",
    "exp(+ i\\omega t)": "ned",
    "exp(+i\\omega t)": "ned",
    "exp(- i\\omega t)": "enu",
    "exp(-i\\omega t)": "enu",
    None: "ned",
}


if TYPE_CHECKING:
    from .mt import MT
    from .mt_stations import MTStations


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
    METADATA_STORAGE_MODES = {"dict", "summary", "cache"}
    DATASET_COPY_MODES = {"deep", "shallow", "none"}
    COORDINATE_REFERENCE_FRAME = COORDINATE_REFERENCE_FRAME_OPTIONS
    IMPEDANCE_UNITS = IMPEDANCE_UNITS

    def __init__(
        self,
        tree: Any | None = None,
        metadata_storage: str = "cache",
        dataset_copy_mode: str = "shallow",
        use_index: bool = False,
        index_db_path: str = ":memory:",
        **attrs: Any,
    ) -> None:
        storage_mode = str(metadata_storage).strip().lower()
        if storage_mode not in self.METADATA_STORAGE_MODES:
            raise ValueError(
                "metadata_storage must be one of "
                f"{sorted(self.METADATA_STORAGE_MODES)}"
            )
        self.metadata_storage = storage_mode

        copy_mode = str(dataset_copy_mode).strip().lower()
        if copy_mode not in self.DATASET_COPY_MODES:
            raise ValueError(
                "dataset_copy_mode must be one of " f"{sorted(self.DATASET_COPY_MODES)}"
            )
        self.dataset_copy_mode = copy_mode

        # Optional in-memory metadata cache keyed by station tree path.
        self._metadata_cache: dict[str, dict[str, Any]] = {
            "survey": {},
            "station": {},
        }

        # Optional SQLite-backed index for fast geographic / period queries.
        self._index: MTDataTreeIndexStore | None = (
            MTDataTreeIndexStore(index_db_path) if use_index else None
        )
        self._index_db_path = index_db_path
        self._lazy_use_index = use_index

        # Deferred station-level transforms keyed by station path.
        self._lazy_station_transforms: dict[str, Callable[[], xr.Dataset]] = {}

        self.tree = (
            tree
            if tree is not None
            else xr.DataTree(name=self.ROOT_NAME, dataset=xr.Dataset())
        )

        # Keep root metadata lightweight and schema-focused at initialization.
        self.tree.attrs.setdefault("schema_name", "mtpy.mt_data_tree")
        self.tree.attrs.setdefault("schema_version", "0.1.0")
        self.tree.attrs.update(attrs)
        self.attrs = self.tree.attrs

        self._coordinate_reference_frame_options = dict(self.COORDINATE_REFERENCE_FRAME)
        self._coordinate_reference_frame = "+"

        self._impedance_unit_factors = dict(self.IMPEDANCE_UNITS)
        self._impedance_units = "mt"
        self.data_rotation_angle = 0
        self.model_parameters: dict[str, Any] = {}
        self._center_lat = None
        self._center_lon = None
        self._center_elev = 0.0

        self.z_model_error = ModelErrors(
            error_value=5,
            error_type="geometric_mean",
            floor=True,
            mode="impedance",
        )
        self.t_model_error = ModelErrors(
            error_value=0.02,
            error_type="absolute",
            floor=True,
            mode="tipper",
        )

        # Initialize a predictable top-level path for survey grouping.
        if self.SURVEYS_NODE not in self.tree.children:
            self.tree[self.SURVEYS_NODE] = xr.DataTree(
                name=self.SURVEYS_NODE, dataset=xr.Dataset()
            )

        self.coordinate_reference_frame = self.attrs.get(
            "coordinate_reference_frame", "ned"
        )
        self.impedance_units = self.attrs.get("impedance_units", "mt")

    def __deepcopy__(self, memo: dict) -> "MTDataTree":
        """Create a deep copy of MTDataTree object."""
        copied_tree = self.tree.copy(deep=True)
        copied = self.__class__(
            tree=copied_tree,
            metadata_storage=self.metadata_storage,
            dataset_copy_mode=self.dataset_copy_mode,
            use_index=False,
            index_db_path=self._index_db_path,
            **dict(self.attrs),
        )
        memo[id(self)] = copied

        copied._metadata_cache = deepcopy(self._metadata_cache, memo)
        copied._lazy_station_transforms = dict(self._lazy_station_transforms)
        copied._lazy_use_index = self._lazy_use_index

        if self._index is not None and not copied.is_lazy:
            copied.rebuild_index(index_db_path=self._index_db_path)

        return copied

    def copy(self) -> "MTDataTree":
        """Create a deep copy of MTDataTree object."""
        return deepcopy(self)

    def clone_empty(self) -> "MTDataTree":
        """Create a copy of MTDataTree excluding all station datasets."""
        return self.__class__(
            metadata_storage=self.metadata_storage,
            dataset_copy_mode=self.dataset_copy_mode,
            use_index=self._index is not None or self._lazy_use_index,
            index_db_path=self._index_db_path,
            **dict(self.attrs),
        )

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
    def _metadata_to_summary(metadata: Any) -> dict[str, Any]:
        """Build a lightweight metadata summary for fast per-station attrs."""
        if metadata is None:
            return {}
        md_id = getattr(metadata, "id", None)
        if md_id in [None, "", "None", "none", "null"]:
            return {}
        return {"id": str(md_id)}

    def _serialize_metadata(self, metadata: Any) -> dict[str, Any]:
        """Serialize metadata according to configured storage mode."""
        if self.metadata_storage == "dict":
            return self._metadata_to_dict(metadata)
        return self._metadata_to_summary(metadata)

    def _metadata_ref(self, station_path: str, metadata: Any) -> str | None:
        """Return metadata reference key for cache mode without mutating cache."""
        if self.metadata_storage != "cache" or metadata is None:
            return None
        return station_path

    def _commit_cached_metadata(
        self,
        station_path: str,
        survey_metadata: Any,
        station_metadata: Any,
    ) -> None:
        """Persist metadata objects in the in-memory cache after successful insert."""
        if self.metadata_storage != "cache":
            return
        if survey_metadata is not None:
            self._metadata_cache["survey"][station_path] = survey_metadata
        if station_metadata is not None:
            self._metadata_cache["station"][station_path] = station_metadata

    def _clear_cached_metadata(self, node_path: str) -> None:
        """Remove cached metadata for one node path or an entire subtree prefix."""
        for metadata_kind in ["survey", "station"]:
            keys_to_remove = [
                key
                for key in self._metadata_cache[metadata_kind]
                if key == node_path or key.startswith(f"{node_path}/")
            ]
            for key in keys_to_remove:
                self._metadata_cache[metadata_kind].pop(key, None)

    def _resolve_dataset_copy_mode(self, dataset_copy_mode: str | None) -> str:
        """Resolve copy mode from call-level override or instance default."""
        mode = (
            self.dataset_copy_mode if dataset_copy_mode is None else dataset_copy_mode
        )
        mode = str(mode).strip().lower()
        if mode not in self.DATASET_COPY_MODES:
            raise ValueError(
                "dataset_copy_mode must be one of " f"{sorted(self.DATASET_COPY_MODES)}"
            )
        return mode

    @staticmethod
    def _copy_station_dataset(station_ds: xr.Dataset, mode: str) -> xr.Dataset:
        """Copy station dataset according to selected copy mode."""
        if mode == "none":
            return station_ds
        if mode == "deep":
            return station_ds.copy(deep=True)
        return station_ds.copy(deep=False)

    def _extract_station_dataset(
        self, mt_obj: "MT", dataset_copy_mode: str | None = None
    ) -> xr.Dataset:
        """Extract an xarray.Dataset from MT object transfer function."""
        tf_obj = getattr(mt_obj, "_transfer_function", None)
        if tf_obj is None:
            raise TypeError("MT object is missing _transfer_function")

        if isinstance(tf_obj, xr.Dataset):
            source_ds = tf_obj
        elif hasattr(tf_obj, "to_xarray"):
            source_ds = tf_obj.to_xarray()
        elif hasattr(tf_obj, "_dataset") and isinstance(tf_obj._dataset, xr.Dataset):
            source_ds = tf_obj._dataset
        else:
            raise TypeError("Could not extract xarray.Dataset from MT object")

        copy_mode = self._resolve_dataset_copy_mode(dataset_copy_mode)
        return self._copy_station_dataset(source_ds, copy_mode)

    def _build_station_attrs(
        self,
        mt_obj: "MT",
        survey: str,
        station: str,
        survey_metadata: Any,
        station_metadata: Any,
        survey_metadata_ref: str | None,
        station_metadata_ref: str | None,
    ) -> dict[str, Any]:
        """Build default station attrs payload for one MT object."""
        return {
            "survey": survey,
            "station": station,
            "tf_id": getattr(mt_obj, "tf_id", station),
            "latitude": getattr(mt_obj, "latitude", None),
            "longitude": getattr(mt_obj, "longitude", None),
            "elevation": getattr(mt_obj, "elevation", None),
            "datum_crs": getattr(mt_obj, "datum_crs", None),
            "utm_crs": self._get_utm_crs(mt_obj),
            "easting": getattr(mt_obj, "east", None),
            "northing": getattr(mt_obj, "north", None),
            "model_east": getattr(mt_obj, "model_east", 0.0),
            "model_north": getattr(mt_obj, "model_north", 0.0),
            "model_elevation": getattr(mt_obj, "model_elevation", 0.0),
            "profile_offset": getattr(mt_obj, "profile_offset", 0.0),
            "coordinate_reference_frame": getattr(
                mt_obj, "coordinate_reference_frame", None
            ),
            "impedance_units": getattr(mt_obj, "impedance_units", None),
            "survey_metadata": self._serialize_metadata(survey_metadata),
            "station_metadata": self._serialize_metadata(station_metadata),
            "survey_metadata_ref": survey_metadata_ref,
            "station_metadata_ref": station_metadata_ref,
        }

    def _build_station_attrs_from_precomputed(
        self,
        mt_obj: "MT",
        survey: str,
        station: str,
        survey_metadata: Any,
        station_metadata: Any,
        survey_metadata_ref: str | None,
        station_metadata_ref: str | None,
        precomputed_attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """Build attrs from precomputed payload plus required canonical keys."""
        station_attrs = dict(precomputed_attrs)
        station_attrs["survey"] = survey
        station_attrs["station"] = station
        station_attrs.setdefault("tf_id", getattr(mt_obj, "tf_id", station))
        station_attrs.setdefault(
            "survey_metadata", self._serialize_metadata(survey_metadata)
        )
        station_attrs.setdefault(
            "station_metadata", self._serialize_metadata(station_metadata)
        )
        station_attrs["survey_metadata_ref"] = survey_metadata_ref
        station_attrs["station_metadata_ref"] = station_metadata_ref
        return station_attrs

    def _coerce_and_prepare_station(
        self,
        mt_obj: "MT | str | Path",
        dataset_copy_mode: str | None = None,
        precomputed_attrs: dict[str, Any] | None = None,
    ) -> tuple[str, str, xr.Dataset, dict[str, Any]]:
        """Coerce station input and build station path/dataset payload."""
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

        station_path = self._station_path(survey, station)
        station_ds = self._extract_station_dataset(
            mt_obj, dataset_copy_mode=dataset_copy_mode
        )

        survey_metadata_obj = getattr(mt_obj, "survey_metadata", None)
        station_metadata_obj = getattr(mt_obj, "station_metadata", None)
        survey_metadata_ref = self._metadata_ref(station_path, survey_metadata_obj)
        station_metadata_ref = self._metadata_ref(station_path, station_metadata_obj)

        if precomputed_attrs is None:
            station_attrs = self._build_station_attrs(
                mt_obj,
                survey,
                station,
                survey_metadata_obj,
                station_metadata_obj,
                survey_metadata_ref,
                station_metadata_ref,
            )
        else:
            station_attrs = self._build_station_attrs_from_precomputed(
                mt_obj,
                survey,
                station,
                survey_metadata_obj,
                station_metadata_obj,
                survey_metadata_ref,
                station_metadata_ref,
                precomputed_attrs,
            )

        station_ds.attrs.update(station_attrs)
        return (
            station_path,
            station,
            station_ds,
            {
                "survey": survey_metadata_obj,
                "station": station_metadata_obj,
            },
        )

    def _cache_metadata(
        self, station_path: str, metadata_kind: str, metadata: Any
    ) -> str | None:
        """Cache full metadata object in-memory and return reference key."""
        if self.metadata_storage != "cache" or metadata is None:
            return None
        self._metadata_cache[metadata_kind][station_path] = metadata
        return station_path

    @property
    def metadata_cache(self) -> dict[str, dict[str, Any]]:
        """In-memory metadata map keyed by station path for cache mode."""
        return self._metadata_cache

    @property
    def is_lazy(self) -> bool:
        """True when one or more deferred station transforms are pending."""
        return bool(self._lazy_station_transforms)

    @property
    def lazy_station_count(self) -> int:
        """Number of stations with pending deferred transforms."""
        return len(self._lazy_station_transforms)

    @property
    def coordinate_reference_frame(self) -> str:
        """
        Coordinate reference frame.

        Returns
        -------
        str
            Reference frame identifier ('NED' or 'ENU')

        """
        return self._coordinate_reference_frame_options[
            self._coordinate_reference_frame
        ].upper()

    @coordinate_reference_frame.setter
    def coordinate_reference_frame(self, value: str) -> None:
        """
        Set coordinate reference frame.

        Parameters
        ----------
        value : str
            Reference frame identifier. Options:

            - 'NED': x=North, y=East, z=+down
            - 'ENU': x=East, y=North, z=+up

        Raises
        ------
        ValueError
            If value is not a recognized reference frame

        Notes
        -----
        Updates coordinate reference frame for all MT objects in collection

        """

        if not isinstance(value, str):
            raise TypeError("Coordinate reference frame input must be a string.")

        normalized = value.strip().lower()
        if normalized not in self._coordinate_reference_frame_options:
            raise ValueError(
                f"{value} is not understood as a reference frame. "
                f"Options are {self._coordinate_reference_frame_options}"
            )
        if normalized in ["ned", "+"]:
            normalized = "+"
        elif normalized in ["enu", "-"]:
            normalized = "-"

        self._coordinate_reference_frame = normalized
        self.attrs["coordinate_reference_frame"] = self.coordinate_reference_frame

        for station_path in self._iter_station_paths():
            station_ds = self.get_station(station_path)
            station_ds.attrs[
                "coordinate_reference_frame"
            ] = self.coordinate_reference_frame

    @property
    def impedance_units(self) -> str:
        """
        Impedance units.

        Returns
        -------
        str
            Impedance units ('mt' or 'ohm')

        """
        return self._impedance_units

    @impedance_units.setter
    def impedance_units(self, value: str) -> None:
        """
        Set impedance units.

        Parameters
        ----------
        value : str
            Impedance units. Options: 'mt' [mV/km/nT] or 'ohm' [Ohms]

        Raises
        ------
        TypeError
            If value is not a string
        ValueError
            If value is not 'mt' or 'ohm'

        Notes
        -----
        Updates impedance units for all MT objects in collection

        """
        if not isinstance(value, str):
            raise TypeError("Units input must be a string.")
        if value.lower() not in self._impedance_unit_factors.keys():
            raise ValueError(f"{value} is not an acceptable unit for impedance.")

        self._impedance_units = value.lower()
        self.attrs["impedance_units"] = self._impedance_units

        for station_path in self._iter_station_paths():
            station_ds = self.get_station(station_path)
            station_ds.attrs["impedance_units"] = self._impedance_units

    def _realize_station(self, station_path: str) -> str | None:
        """Materialize one deferred station transform if present."""
        transform = self._lazy_station_transforms.pop(station_path, None)
        if transform is None:
            return None

        station_ds = transform()
        self._set_station_dataset(station_path, station_ds)

        if self._index is not None:
            station_row, period_row = MTDataTreeIndexStore._extract_rows(
                station_path,
                station_ds,
            )
            if period_row is None:
                self._index.delete_station_by_tree_path(station_path)
            self._index.upsert_station(station_row)
            if period_row is not None:
                self._index.replace_station_period_rows(period_row)
            return station_row.survey_name
        return None

    @staticmethod
    def _is_dask_delayed(obj: Any) -> bool:
        """Return True when *obj* is a dask delayed object."""
        return obj.__class__.__name__ == "Delayed" and hasattr(obj, "dask")

    def compute(
        self,
        station_paths: list[str] | None = None,
        scheduler: str | None = None,
    ) -> "MTDataTree":
        """Materialize deferred station transforms and update index state."""
        if not self._lazy_station_transforms:
            return self

        if self._index is None and self._lazy_use_index:
            self._index = MTDataTreeIndexStore(self._index_db_path)

        pending_paths = list(self._lazy_station_transforms.keys())
        if station_paths is None:
            paths_to_realize = pending_paths
        else:
            requested = set(station_paths)
            paths_to_realize = [path for path in pending_paths if path in requested]

        realized_datasets: dict[str, xr.Dataset] = {}
        delayed_paths: list[str] = []
        delayed_objs: list[Any] = []

        for station_path in paths_to_realize:
            transform = self._lazy_station_transforms.pop(station_path, None)
            if transform is None:
                continue
            out = transform()
            if self._is_dask_delayed(out):
                delayed_paths.append(station_path)
                delayed_objs.append(out)
            else:
                realized_datasets[station_path] = out

        if delayed_objs:
            try:
                dask = importlib.import_module("dask")
            except ImportError as exc:
                raise RuntimeError(
                    "Dask delayed transforms are pending but dask is not installed."
                ) from exc
            computed = dask.compute(*delayed_objs, scheduler=scheduler)
            for station_path, station_ds in zip(delayed_paths, computed):
                realized_datasets[station_path] = station_ds

        updated_surveys: set[str] = set()
        for station_path in paths_to_realize:
            station_ds = realized_datasets.get(station_path)
            if station_ds is None:
                continue
            if not isinstance(station_ds, xr.Dataset):
                raise TypeError(
                    "Deferred transform must return xr.Dataset, "
                    f"got {type(station_ds)!r}"
                )

            self._set_station_dataset(station_path, station_ds)

            if self._index is not None:
                station_row, period_row = MTDataTreeIndexStore._extract_rows(
                    station_path,
                    station_ds,
                )
                if period_row is None:
                    self._index.delete_station_by_tree_path(station_path)
                self._index.upsert_station(station_row)
                if period_row is not None:
                    self._index.replace_station_period_rows(period_row)
                updated_surveys.add(station_row.survey_name)

        if self._index is not None:
            for survey_name in updated_surveys:
                self._index.refresh_survey_aggregates(survey_name)

        return self

    def persist(
        self,
        station_paths: list[str] | None = None,
        scheduler: str | None = None,
    ) -> "MTDataTree":
        """Alias of compute for deferred station transforms."""
        return self.compute(station_paths=station_paths, scheduler=scheduler)

    def as_dask(
        self,
        chunks: dict[str, int] | str | None,
        station_paths: list[str] | None = None,
        variables: list[str] | None = None,
        inplace: bool = False,
    ) -> "MTDataTree":
        """Convert station datasets to dask-backed arrays using chunking."""
        try:
            importlib.import_module("dask.array")
        except ImportError as exc:
            raise RuntimeError("Dask is required for as_dask()") from exc

        tree_obj = self if inplace else self.get_subset(self._iter_station_paths())
        target_paths = tree_obj._iter_station_paths()
        if station_paths is not None:
            requested = set(station_paths)
            target_paths = [path for path in target_paths if path in requested]

        for station_path in target_paths:
            station_ds = tree_obj.get_station(station_path)
            if variables is not None:
                missing = [
                    name for name in variables if name not in station_ds.data_vars
                ]
                if missing:
                    raise KeyError(f"Variables not found for chunking: {missing}")
                chunked_ds = station_ds.copy(deep=False)
                for var_name in variables:
                    chunked_ds[var_name] = station_ds[var_name].chunk(chunks)
            else:
                chunked_ds = station_ds.chunk(chunks)
            tree_obj._set_station_dataset(station_path, chunked_ds)
        return tree_obj

    def rechunk(
        self,
        chunks: dict[str, int] | str | None,
        station_paths: list[str] | None = None,
        variables: list[str] | None = None,
        inplace: bool = True,
    ) -> "MTDataTree":
        """Rechunk dask-backed station arrays."""
        return self.as_dask(
            chunks=chunks,
            station_paths=station_paths,
            variables=variables,
            inplace=inplace,
        )

    def is_dask_backed(self, station_paths: list[str] | None = None) -> bool:
        """Return True when selected stations have dask-backed data variables."""
        self.compute(station_paths=station_paths)
        target_paths = self._iter_station_paths()
        if station_paths is not None:
            requested = set(station_paths)
            target_paths = [path for path in target_paths if path in requested]
        if not target_paths:
            return False

        for station_path in target_paths:
            station_ds = self.get_station(station_path)
            for da in station_ds.data_vars.values():
                if getattr(da.data, "chunks", None) is None:
                    return False
        return True

    def chunk_plan(
        self,
        station_paths: list[str] | None = None,
    ) -> dict[str, dict[str, tuple[tuple[int, ...], ...] | None]]:
        """Return chunk layout per station/data variable."""
        self.compute(station_paths=station_paths)
        target_paths = self._iter_station_paths()
        if station_paths is not None:
            requested = set(station_paths)
            target_paths = [path for path in target_paths if path in requested]

        plan: dict[str, dict[str, tuple[tuple[int, ...], ...] | None]] = {}
        for station_path in target_paths:
            station_ds = self.get_station(station_path)
            plan[station_path] = {
                var_name: da.chunks for var_name, da in station_ds.data_vars.items()
            }
        return plan

    def map_stations(
        self,
        transform: Callable[[xr.Dataset], xr.Dataset],
        station_paths: list[str] | None = None,
        lazy: bool = True,
        inplace: bool = False,
    ) -> "MTDataTree":
        """Apply a dataset transform across selected stations."""
        tree_obj = self if inplace else self.get_subset(self._iter_station_paths())
        tree_obj.compute()

        target_paths = tree_obj._iter_station_paths()
        if station_paths is not None:
            requested = set(station_paths)
            target_paths = [path for path in target_paths if path in requested]

        for station_path in target_paths:
            station_ds = tree_obj.get_station(station_path).copy(deep=False)
            if lazy:
                tree_obj._lazy_station_transforms[
                    station_path
                ] = lambda ds=station_ds, op=transform: op(ds)
            else:
                out_ds = transform(station_ds)
                if not isinstance(out_ds, xr.Dataset):
                    raise TypeError(
                        "map_stations transform must return xr.Dataset, "
                        f"got {type(out_ds)!r}"
                    )
                tree_obj._set_station_dataset(station_path, out_ds)
        return tree_obj

    def interpolate_dask(
        self,
        new_periods: np.ndarray,
        f_type: str = "period",
        bounds_error: bool = True,
        chunks: dict[str, int] | str | None = None,
        scheduler: str | None = None,
        compute: bool = True,
        inplace: bool = False,
        **kwargs: Any,
    ) -> "MTDataTree":
        """Run interpolation using dask-delayed station tasks."""
        try:
            dask = importlib.import_module("dask")
            delayed = getattr(dask, "delayed")
        except ImportError as exc:
            raise RuntimeError("Dask is required for interpolate_dask()") from exc

        base_tree = self if inplace else self.get_subset(self._iter_station_paths())
        if chunks is not None:
            base_tree.as_dask(chunks=chunks, inplace=True)

        lazy_tree = base_tree.interpolate_lazy(
            new_periods,
            f_type=f_type,
            inplace=True,
            bounds_error=bounds_error,
            **kwargs,
        )

        for station_path, transform in list(lazy_tree._lazy_station_transforms.items()):
            lazy_tree._lazy_station_transforms[
                station_path
            ] = lambda fn=transform: delayed(fn)()

        if compute:
            lazy_tree.compute(scheduler=scheduler)
        elif scheduler is not None:
            dask.config.set(scheduler=scheduler)
        return lazy_tree

    def rotate_dask(
        self,
        rotation_angle: float | np.ndarray,
        chunks: dict[str, int] | str | None = None,
        scheduler: str | None = None,
        compute: bool = True,
        inplace: bool = False,
    ) -> "MTDataTree":
        """Run rotation using dask-delayed station tasks."""
        try:
            dask = importlib.import_module("dask")
            delayed = getattr(dask, "delayed")
        except ImportError as exc:
            raise RuntimeError("Dask is required for rotate_dask()") from exc

        base_tree = self if inplace else self.get_subset(self._iter_station_paths())
        if chunks is not None:
            base_tree.as_dask(chunks=chunks, inplace=True)

        def _rotate_transform(ds: xr.Dataset) -> xr.Dataset:
            crf = ds.attrs.get(
                "coordinate_reference_frame",
                self.attrs.get("coordinate_reference_frame", "ned"),
            )
            return MTDataTree._rotate_station_dataset(
                ds,
                rotation_angle,
                coordinate_reference_frame=crf,
            )

        lazy_tree = base_tree.map_stations(
            _rotate_transform,
            lazy=True,
            inplace=True,
        )

        for station_path, transform in list(lazy_tree._lazy_station_transforms.items()):
            lazy_tree._lazy_station_transforms[
                station_path
            ] = lambda fn=transform: delayed(fn)()

        if compute:
            lazy_tree.compute(scheduler=scheduler)
        elif scheduler is not None:
            dask.config.set(scheduler=scheduler)
        return lazy_tree

    def finalize_index(self) -> None:
        """Ensure index reflects the current tree contents."""
        self.compute()
        self.rebuild_index(index_db_path=self._index_db_path)

    def get_metadata(
        self, station_key: str, metadata_kind: str = "station"
    ) -> Any | dict[str, Any] | None:
        """Return cached metadata object when available, else dataset attrs copy."""
        if metadata_kind not in self._metadata_cache:
            raise KeyError("metadata_kind must be 'survey' or 'station'")

        cached = self._metadata_cache[metadata_kind].get(station_key)
        if cached is not None:
            return cached

        ds = self.get_station(station_key)
        return dict(ds.attrs.get(f"{metadata_kind}_metadata", {}))

    def _hydrate_metadata_from_cache(
        self, mt_obj: "MT", station_ds: xr.Dataset
    ) -> None:
        """Populate MT metadata objects from in-memory cache when references exist."""
        attrs = station_ds.attrs
        for metadata_kind in ["survey", "station"]:
            ref_key = attrs.get(f"{metadata_kind}_metadata_ref")
            if not isinstance(ref_key, str):
                continue
            cached_md = self._metadata_cache[metadata_kind].get(ref_key)
            if cached_md is None:
                continue

            target_md = getattr(mt_obj, f"{metadata_kind}_metadata", None)
            if target_md is None or not hasattr(target_md, "from_dict"):
                continue

            cached_dict = self._metadata_to_dict(cached_md)
            if cached_dict:
                target_md.from_dict(cached_dict)

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

    def _resolve_station_path(self, station_key: str) -> str:
        """Resolve a station key in dot or tree-path form to canonical tree path."""
        if not isinstance(station_key, str) or not station_key.strip():
            raise KeyError("station_key must be a non-empty string")

        key = station_key.strip()
        if key.startswith(f"{self.SURVEYS_NODE}/") and self._path_exists(key):
            return key

        if "." in key:
            survey, station = key.split(".", 1)
            candidate = self._station_path(
                self._clean_name(survey, "default"),
                self._clean_name(station, "unknown_station"),
            )
            if self._path_exists(candidate):
                return candidate

        raise KeyError(f"Station key not found: {station_key}")

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

    def _iter_station_paths(self) -> list[str]:
        """Return all station node paths under /surveys."""
        if self._index is not None:
            return self._index.all_station_paths()

        station_paths: list[str] = []

        def _walk(node: Any, node_path: str = "") -> None:
            ds = getattr(node, "ds", None)
            if isinstance(ds, xr.Dataset) and node_path.count("/") >= 3:
                station_paths.append(node_path)
            for child_name, child in getattr(node, "children", {}).items():
                child_path = f"{node_path}/{child_name}" if node_path else child_name
                _walk(child, child_path)

        _walk(self.tree)
        return station_paths

    @staticmethod
    def _crs_to_epsg(value: Any) -> Any:
        """Convert a CRS-like value to an EPSG code when possible."""
        if value in [None, "", "None", "none", "null"]:
            return None
        if hasattr(value, "to_epsg"):
            return value.to_epsg()
        return value

    @staticmethod
    def _station_locations_columns() -> list[str]:
        """Column order matching MTStations.station_locations."""
        return [
            "survey",
            "station",
            "latitude",
            "longitude",
            "elevation",
            "datum_epsg",
            "east",
            "north",
            "utm_epsg",
            "model_east",
            "model_north",
            "model_elevation",
            "profile_offset",
        ]

    def _station_location_record(self, station_path: str) -> dict[str, Any]:
        """Build one station-location record directly from dataset attrs."""
        attrs = self.get_station(station_path).attrs
        return {
            "survey": attrs.get("survey"),
            "station": attrs.get("station"),
            "latitude": attrs.get("latitude"),
            "longitude": attrs.get("longitude"),
            "elevation": attrs.get("elevation"),
            "datum_epsg": self._crs_to_epsg(attrs.get("datum_crs")),
            "east": attrs.get("easting"),
            "north": attrs.get("northing"),
            "utm_epsg": self._crs_to_epsg(attrs.get("utm_crs")),
            "model_east": attrs.get("model_east", 0.0),
            "model_north": attrs.get("model_north", 0.0),
            "model_elevation": attrs.get("model_elevation", 0.0),
            "profile_offset": attrs.get("profile_offset", 0.0),
        }

    def _station_path_to_location_mt(self, station_path: str) -> "MT":
        """Build a lightweight MT object containing only location metadata."""
        from .mt import MT

        attrs = self.get_station(station_path).attrs
        mt_obj = MT()
        if attrs.get("survey") is not None:
            mt_obj.survey = attrs["survey"]
        if attrs.get("station") is not None:
            mt_obj.station = attrs["station"]
        if attrs.get("datum_crs") is not None:
            mt_obj.datum_crs = attrs["datum_crs"]
        if attrs.get("utm_crs") is not None:
            mt_obj.utm_crs = attrs["utm_crs"]
        for attr_name, attr_value in [
            ("latitude", attrs.get("latitude")),
            ("longitude", attrs.get("longitude")),
            ("elevation", attrs.get("elevation")),
            ("east", attrs.get("easting")),
            ("north", attrs.get("northing")),
            ("model_east", attrs.get("model_east", 0.0)),
            ("model_north", attrs.get("model_north", 0.0)),
            ("model_elevation", attrs.get("model_elevation", 0.0)),
            ("profile_offset", attrs.get("profile_offset", 0.0)),
        ]:
            if attr_value is not None:
                setattr(mt_obj, attr_name, attr_value)
        return mt_obj

    @staticmethod
    def _get_utm_crs(mt_obj: "MT") -> Any:
        """Get UTM CRS information from MT object when available."""
        crs = getattr(mt_obj, "utm_crs", None)
        if crs is not None:
            return crs
        return getattr(mt_obj, "utm_epsg", None)

    @staticmethod
    def _dataset_to_mt(station_ds: xr.Dataset) -> "MT":
        """Build an MT object from a stored station dataset and attrs."""
        from .mt import MT

        mt_obj = MT()
        mt_obj._transfer_function = station_ds.copy()

        attrs = station_ds.attrs
        if attrs.get("survey") is not None:
            mt_obj.survey = attrs["survey"]
        if attrs.get("station") is not None:
            mt_obj.station = attrs["station"]
        if attrs.get("coordinate_reference_frame") is not None:
            crf = attrs["coordinate_reference_frame"]
            if isinstance(crf, str):
                crf_key = crf.upper()
                if crf_key == "NED":
                    crf = "+"
                elif crf_key == "ENU":
                    crf = "-"
            mt_obj.coordinate_reference_frame = crf
        if attrs.get("impedance_units") is not None:
            mt_obj.impedance_units = attrs["impedance_units"]
        if attrs.get("datum_crs") is not None:
            mt_obj.datum_crs = attrs["datum_crs"]
        if attrs.get("utm_crs") is not None:
            mt_obj.utm_crs = attrs["utm_crs"]
        if attrs.get("latitude") is not None:
            mt_obj.latitude = attrs["latitude"]
        if attrs.get("longitude") is not None:
            mt_obj.longitude = attrs["longitude"]
        if attrs.get("elevation") is not None:
            mt_obj.elevation = attrs["elevation"]
        if attrs.get("easting") is not None:
            mt_obj.east = attrs["easting"]
        if attrs.get("northing") is not None:
            mt_obj.north = attrs["northing"]

        survey_md = attrs.get("survey_metadata", {})
        if isinstance(survey_md, dict) and survey_md:
            if hasattr(mt_obj.survey_metadata, "from_dict"):
                mt_obj.survey_metadata.from_dict(survey_md)

        station_md = attrs.get("station_metadata", {})
        if isinstance(station_md, dict) and station_md:
            if hasattr(mt_obj.station_metadata, "from_dict"):
                mt_obj.station_metadata.from_dict(station_md)

        return mt_obj

    @staticmethod
    def _pick_channel_labels(
        available: list[Any], candidates: list[str], required: int
    ) -> list[Any] | None:
        """Pick channel labels from available coordinates using preferred names."""
        channel_map = {str(label).lower(): label for label in available}
        selected: list[Any] = []
        for candidate in candidates:
            key = candidate.lower()
            if key in channel_map and channel_map[key] not in selected:
                selected.append(channel_map[key])
            if len(selected) == required:
                return selected
        return None

    @staticmethod
    def _coerce_epsg_value(value: Any) -> str | None:
        """Normalize CRS/EPSG values to a dataframe-compatible EPSG string."""
        if value is None:
            return None

        if isinstance(value, (int, np.integer)):
            return str(int(value))

        try:
            from pyproj import CRS

            epsg = CRS.from_user_input(value).to_epsg()
            if epsg is not None:
                return str(int(epsg))
        except Exception:
            pass

        value_str = str(value).strip()
        if not value_str:
            return None
        if value_str.isdigit():
            return str(int(value_str))
        return value_str

    def _station_dataset_to_dataframe(
        self,
        station_ds: xr.Dataset,
        utm_crs: Any | None = None,
        cols: list[str] | None = None,
        impedance_units: str = "mt",
    ) -> pd.DataFrame:
        """Convert one station dataset directly into dataframe rows."""
        from .transfer_function import Tipper, Z

        period = np.asarray(station_ds.coords["period"].values, dtype=float)
        n_entries = period.size
        station_df = MTDataFrame(n_entries=n_entries)

        attrs = station_ds.attrs
        station_df.survey = attrs.get("survey", "")
        station_df.station = attrs.get("station", "")
        station_df.latitude = attrs.get("latitude", 0.0)
        station_df.longitude = attrs.get("longitude", 0.0)
        station_df.elevation = attrs.get("elevation", 0.0)
        station_df.datum_epsg = self._coerce_epsg_value(attrs.get("datum_crs"))
        station_df.east = attrs.get("easting", 0.0)
        station_df.north = attrs.get("northing", 0.0)
        station_df.utm_epsg = self._coerce_epsg_value(
            utm_crs if utm_crs is not None else attrs.get("utm_crs")
        )
        station_df.model_east = attrs.get("model_east", 0.0)
        station_df.model_north = attrs.get("model_north", 0.0)
        station_df.model_elevation = attrs.get("model_elevation", 0.0)
        station_df.profile_offset = attrs.get("profile_offset", 0.0)

        station_df.dataframe.loc[:, "period"] = period

        if "output" in station_ds.coords and "input" in station_ds.coords:
            output_labels = list(station_ds.coords["output"].values)
            input_labels = list(station_ds.coords["input"].values)

            z_outputs = self._pick_channel_labels(
                output_labels, ["ex", "ey", "x", "y"], 2
            )
            z_inputs = self._pick_channel_labels(
                input_labels, ["hx", "hy", "x", "y"], 2
            )
            if z_outputs is not None and z_inputs is not None:
                tf = (
                    station_ds["transfer_function"]
                    .sel(output=z_outputs, input=z_inputs)
                    .values
                )
                tf_error = (
                    station_ds["transfer_function_error"]
                    .sel(output=z_outputs, input=z_inputs)
                    .values
                )
                tf_model_error = (
                    station_ds["transfer_function_model_error"]
                    .sel(output=z_outputs, input=z_inputs)
                    .values
                )
                z_object = Z(
                    z=tf,
                    z_error=tf_error,
                    frequency=1.0 / period,
                    z_model_error=tf_model_error,
                    units=impedance_units,
                )
                station_df.from_z_object(z_object)

            t_output = self._pick_channel_labels(output_labels, ["hz", "z"], 1)
            t_inputs = self._pick_channel_labels(
                input_labels, ["hx", "hy", "x", "y"], 2
            )
            if t_output is not None and t_inputs is not None:
                tipper = (
                    station_ds["transfer_function"]
                    .sel(output=t_output, input=t_inputs)
                    .values
                )
                tipper_error = (
                    station_ds["transfer_function_error"]
                    .sel(output=t_output, input=t_inputs)
                    .values
                )
                tipper_model_error = (
                    station_ds["transfer_function_model_error"]
                    .sel(output=t_output, input=t_inputs)
                    .values
                )
                tipper_object = Tipper(
                    tipper=tipper,
                    tipper_error=tipper_error,
                    frequency=1.0 / period,
                    tipper_model_error=tipper_model_error,
                )
                station_df.from_t_object(tipper_object)

        if cols is None:
            return station_df.dataframe
        return station_df.dataframe.loc[:, cols]

    def to_mt_stations(self) -> "MTStations":
        """Build an MTStations view from station datasets in the tree."""
        from .mt_stations import MTStations

        return MTStations(None, station_locations=self.station_locations)

    @property
    def center_point(self) -> Any:
        """Return the geographic center point of the station collection.

        If explicit center coordinates have been stored (e.g. after reading a
        ModEM data file), those values are returned directly.  Otherwise the
        center is derived on-the-fly from all station locations via
        :meth:`to_mt_stations`.

        Returns
        -------
        MTLocation
            Center location with ``latitude``, ``longitude``, ``elevation``,
            ``east``, ``north``, and ``utm_epsg`` attributes populated.

        Examples
        --------
        >>> from mtpy.core import MTDataTree
        >>> tree = MTDataTree()
        >>> # (add stations first)
        >>> cp = tree.center_point
        >>> print(cp.latitude, cp.longitude)
        """
        from .mt_location import MTLocation

        if self._center_lat is not None and self._center_lon is not None:
            center_location = MTLocation()
            center_location.latitude = self._center_lat
            center_location.longitude = self._center_lon
            center_location.elevation = self._center_elev

            utm_epsg = self.attrs.get("utm_epsg")
            if utm_epsg not in [None, "", "None", "none", "null"]:
                center_location.utm_epsg = utm_epsg

            datum_crs = self.attrs.get("datum_crs")
            if datum_crs not in [None, "", "None", "none", "null"]:
                center_location.datum_crs = datum_crs

            center_location.model_east = center_location.east
            center_location.model_north = center_location.north
            center_location.model_elevation = self._center_elev
            return center_location

        return self.to_mt_stations().center_point

    def _dataframe_with_relative_locations(
        self,
        utm_crs: Any | None = None,
        impedance_units: str = "mt",
    ) -> pd.DataFrame:
        """Return a station dataframe with model-relative coordinates populated.

        Calls :meth:`to_dataframe` and, if ``model_east``/``model_north`` are
        all zero but absolute ``east``/``north`` values are available, computes
        the model coordinates relative to :attr:`center_point`.

        Parameters
        ----------
        utm_crs : pyproj CRS or int, optional
            Override UTM CRS passed to :meth:`to_dataframe`, by default
            ``None`` (use the tree's stored CRS).
        impedance_units : str, optional
            Units for the impedance tensor, e.g. ``'mt'`` or ``'ohm'``,
            by default ``'mt'``.

        Returns
        -------
        pandas.DataFrame
            Station dataframe with ``model_east``, ``model_north``, and
            ``model_elevation`` columns filled.

        Raises
        ------
        ValueError
            If ``model_east``/``model_north`` are zero, absolute UTM
            coordinates are available, but no UTM EPSG is set on
            :attr:`center_point`.
        """
        df = self.to_dataframe(utm_crs=utm_crs, impedance_units=impedance_units).copy()
        if df.empty:
            return df

        model_east = pd.to_numeric(df.get("model_east"), errors="coerce").fillna(0.0)
        model_north = pd.to_numeric(df.get("model_north"), errors="coerce").fillna(0.0)
        if not (np.allclose(model_east, 0.0) and np.allclose(model_north, 0.0)):
            return df

        east = pd.to_numeric(df.get("east"), errors="coerce").fillna(0.0)
        north = pd.to_numeric(df.get("north"), errors="coerce").fillna(0.0)
        if np.allclose(east, 0.0) or np.allclose(north, 0.0):
            return df

        center = self.center_point
        if center.utm_epsg is None:
            raise ValueError(
                "Need to input data UTM EPSG or CRS to compute relative station locations"
            )

        df.loc[:, "model_east"] = east - center.east
        df.loc[:, "model_north"] = north - center.north
        df.loc[:, "model_elevation"] = (
            pd.to_numeric(df.get("elevation"), errors="coerce").fillna(0.0)
            - center.elevation
        )
        return df

    def to_geo_df(
        self,
        model_locations: bool = False,
        data_type: str = "station_locations",
    ) -> Any:
        """Create a GeoDataFrame for GIS operations."""
        try:
            import geopandas as gpd
        except ImportError as exc:
            raise ImportError(
                "geopandas is required for to_geo_df but is not installed"
            ) from exc

        if data_type in ["station_locations", "stations"]:
            df = self.station_locations
        elif data_type in ["phase_tensor", "pt"]:
            df = self.to_mt_dataframe().phase_tensor
        elif data_type in ["tipper", "t"]:
            df = self.to_mt_dataframe().tipper
        elif data_type in ["both", "shapefiles"]:
            df = self.to_mt_dataframe().for_shapefiles
        else:
            raise ValueError(f"Option for 'data_type' {data_type} is unsupported.")

        if model_locations:
            return gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df.model_east, df.model_north),
                crs=None,
            )

        crs_value = None
        if "datum_epsg" in df.columns:
            for value in df["datum_epsg"].tolist():
                epsg_value = self._coerce_epsg_value(value)
                if epsg_value is None:
                    continue
                if str(epsg_value).isdigit():
                    crs_value = f"EPSG:{epsg_value}"
                else:
                    crs_value = epsg_value
                break

        return gpd.GeoDataFrame(
            df,
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs=crs_value,
        )

    @property
    def station_locations(self) -> pd.DataFrame:
        """Station-location table built directly from tree dataset attrs."""
        columns = self._station_locations_columns()

        if self._index is not None:
            records = self._index.all_station_records()
            if not records:
                return pd.DataFrame(columns=columns)
            return pd.DataFrame(
                [
                    {
                        "survey": r.survey_name,
                        "station": r.name,
                        "latitude": r.latitude,
                        "longitude": r.longitude,
                        "elevation": r.elevation,
                        "datum_epsg": r.datum_epsg,
                        "east": r.east,
                        "north": r.north,
                        "utm_epsg": r.utm_epsg,
                        "model_east": r.model_east,
                        "model_north": r.model_north,
                        "model_elevation": r.model_elevation,
                        "profile_offset": r.profile_offset,
                    }
                    for r in records
                ],
                columns=columns,
            )

        station_paths = self._iter_station_paths()
        if not station_paths:
            return pd.DataFrame(columns=columns)

        return pd.DataFrame(
            [self._station_location_record(path) for path in station_paths],
            columns=columns,
        )

    @property
    def mt_stations(self) -> "MTStations":
        """Convenience accessor for station locations represented by the tree."""
        return self.to_mt_stations()

    def get_nearby_stations(
        self,
        station_key: str,
        radius: float,
        radius_units: str = "m",
    ) -> list[str]:
        """Find stations near a given station."""
        self.compute()

        station_path = self._resolve_station_path(station_key)
        local_attrs = self.get_station(station_path).attrs

        sdf = self.station_locations.copy()
        if sdf.empty:
            return []

        if radius_units in ["m", "meters", "metres"]:
            if "utm_epsg" not in sdf.columns or (
                sdf["utm_epsg"].replace("", np.nan).dropna().empty
            ):
                raise ValueError(
                    "Cannot estimate distances in meters without a UTM CRS. Set 'utm_crs' first."
                )
            sdf["radius"] = np.sqrt(
                (
                    float(local_attrs.get("easting", 0.0))
                    - pd.to_numeric(sdf.east, errors="coerce").fillna(0.0)
                )
                ** 2
                + (
                    float(local_attrs.get("northing", 0.0))
                    - pd.to_numeric(sdf.north, errors="coerce").fillna(0.0)
                )
                ** 2
            )
        elif radius_units in ["deg", "degrees"]:
            sdf["radius"] = np.sqrt(
                (
                    float(local_attrs.get("longitude", 0.0))
                    - pd.to_numeric(sdf.longitude, errors="coerce").fillna(0.0)
                )
                ** 2
                + (
                    float(local_attrs.get("latitude", 0.0))
                    - pd.to_numeric(sdf.latitude, errors="coerce").fillna(0.0)
                )
                ** 2
            )
        else:
            raise ValueError(
                "radius_units must be one of: m, meters, metres, deg, degrees"
            )

        return [
            f"{row.survey}.{row.station}"
            for row in sdf.loc[(sdf.radius <= radius) & (sdf.radius > 0)].itertuples()
        ]

    def get_profile(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        radius: float,
    ) -> "MTDataTree":
        """Get stations along a profile line."""
        self.compute()

        profile_stations = self.to_mt_stations()._extract_profile(
            x1,
            y1,
            x2,
            y2,
            radius,
        )

        if profile_stations.empty:
            return self.clone_empty()

        key_to_path: dict[tuple[str, str], str] = {}
        for station_path in self._iter_station_paths():
            attrs = self.get_station(station_path).attrs
            key = (str(attrs.get("survey", "")), str(attrs.get("station", "")))
            key_to_path[key] = station_path

        selected_paths: list[str] = []
        for row in profile_stations.itertuples(index=False):
            key = (str(getattr(row, "survey")), str(getattr(row, "station")))
            station_path = key_to_path.get(key)
            if station_path is not None:
                selected_paths.append(station_path)

        profile_tree = self.get_subset(selected_paths)

        for row in profile_stations.itertuples(index=False):
            survey = self._clean_name(getattr(row, "survey", None), "default")
            station = self._clean_name(
                getattr(row, "station", None),
                "unknown_station",
            )
            station_path = profile_tree._station_path(survey, station)
            if not profile_tree._path_exists(station_path):
                continue
            if hasattr(row, "profile_offset"):
                profile_tree.get_station(station_path).attrs["profile_offset"] = float(
                    getattr(row, "profile_offset")
                )

        return profile_tree

    def compute_model_errors(
        self,
        z_error_value: float | None = None,
        z_error_type: str | None = None,
        z_floor: bool | None = None,
        t_error_value: float | None = None,
        t_error_type: str | None = None,
        t_floor: bool | None = None,
    ) -> None:
        """Compute model errors for all stations in the tree."""
        self.compute()

        if z_error_value is not None:
            self.z_model_error.error_value = z_error_value
        if z_error_type is not None:
            self.z_model_error.error_type = z_error_type
        if z_floor is not None:
            self.z_model_error.floor = z_floor

        if t_error_value is not None:
            self.t_model_error.error_value = t_error_value
        if t_error_type is not None:
            self.t_model_error.error_type = t_error_type
        if t_floor is not None:
            self.t_model_error.floor = t_floor

        for station_path in self._iter_station_paths():
            station_ds = self.get_station(station_path)
            attrs = dict(station_ds.attrs)

            mt_obj = self._dataset_to_mt(station_ds)
            self._hydrate_metadata_from_cache(mt_obj, station_ds)

            mt_obj.compute_model_z_errors(**self.z_model_error.error_parameters)
            mt_obj.compute_model_t_errors(**self.t_model_error.error_parameters)

            out_ds = mt_obj._transfer_function
            out_ds.attrs = attrs
            self._set_station_dataset(station_path, out_ds)

    def estimate_starting_rho(self) -> None:
        """Estimate starting resistivity from all station data and plot summary curves."""
        import matplotlib.pyplot as plt

        self.compute()

        entries: list[dict[str, float]] = []
        for station_path in self._iter_station_paths():
            mt_obj = self.get_station(station_path, as_mt=True)
            for period, res_det in zip(mt_obj.period, mt_obj.Z.res_det):
                entries.append({"period": period, "res_det": res_det})

        res_df = pd.DataFrame(entries)

        mean_rho = res_df.groupby("period").mean()
        median_rho = res_df.groupby("period").median()

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1)
        (l1,) = ax.loglog(mean_rho.index, mean_rho.res_det, lw=2, color=(0.75, 0.25, 0))
        (l2,) = ax.loglog(
            median_rho.index, median_rho.res_det, lw=2, color=(0, 0.25, 0.75)
        )

        ax.loglog(
            mean_rho.index,
            np.repeat(mean_rho.res_det.mean(), mean_rho.shape[0]),
            ls="--",
            lw=2,
            color=(0.75, 0.25, 0),
        )
        ax.loglog(
            median_rho.index,
            np.repeat(median_rho.res_det.median(), median_rho.shape[0]),
            ls="--",
            lw=2,
            color=(0, 0.25, 0.75),
        )

        ax.set_xlabel("Period (s)", fontdict={"size": 12, "weight": "bold"})
        ax.set_ylabel("Resistivity (Ohm-m)", fontdict={"size": 12, "weight": "bold"})

        ax.legend(
            [l1, l2],
            [
                f"Mean = {mean_rho.res_det.mean():.1f}",
                f"Median = {median_rho.res_det.median():.1f}",
            ],
            loc="upper left",
        )
        ax.grid(which="both", ls="--", color=(0.75, 0.75, 0.75))
        ax.set_xlim((res_df.period.min(), res_df.period.max()))

        plt.show()

    def to_modem(self, data_filename: str | Path | None = None, **kwargs: Any) -> Any:
        """Create a ModEM Data object from the station collection.

        Parameters
        ----------
        data_filename : str or pathlib.Path, optional
            Path to write the ModEM data file.  When ``None`` (default) the
            file is not written.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`mtpy.modeling.modem.Data` (e.g. ``rotation_angle``,
            ``inv_mode``, ``formatting``).

        Returns
        -------
        mtpy.modeling.modem.Data
            Populated ModEM Data object with ``z_model_error`` and
            ``t_model_error`` set from the tree.

        Examples
        --------
        Create a data file and retrieve the Data object:

        >>> from mtpy.core import MTDataTree
        >>> tree = MTDataTree()
        >>> # tree.add_station(...)  # populate with MT objects first
        >>> tree.model_parameters = {"inv_mode": "1", "formatting": "1"}
        >>> modem_data = tree.to_modem(data_filename="ModEM_data.dat")
        >>> print(modem_data.center_point.latitude)
        """
        from mtpy.modeling.modem import Data

        modem_kwargs = dict(self.model_parameters)
        modem_kwargs.update(kwargs)

        modem_df = self._dataframe_with_relative_locations(
            impedance_units=self.impedance_units
        )
        if modem_df.empty:
            modem_df = self.to_dataframe(impedance_units=self.impedance_units)

        modem_data = Data(
            dataframe=modem_df,
            center_point=self.center_point,
            **modem_kwargs,
        )
        modem_data.z_model_error = self.z_model_error
        modem_data.t_model_error = self.t_model_error
        if data_filename is not None:
            modem_data.write_data_file(file_name=data_filename)

        return modem_data

    def from_modem(
        self, data_filename: str | Path, survey: str = "data", **kwargs: Any
    ) -> None:
        """Populate the tree by reading an existing ModEM data file.

        Station datasets, model-error parameters, the center point, and any
        top-level model parameters (those without a dot in the key) are all
        restored from the file.

        Parameters
        ----------
        data_filename : str or pathlib.Path
            Path to the ModEM ``.dat`` / ``.data`` file.
        survey : str, optional
            Survey label to assign to all imported stations,
            by default ``'data'``.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`mtpy.modeling.modem.Data`.

        Examples
        --------
        >>> from mtpy.core import MTDataTree
        >>> tree = MTDataTree()
        >>> tree.from_modem("ModEM_data.dat", survey="line1")
        >>> print(tree.survey_ids)
        """
        from mtpy.modeling.modem import Data

        modem_data = Data(**kwargs)
        mdf = modem_data.read_data_file(data_filename)
        mdf.dataframe.loc[:, "survey"] = survey

        self.from_mt_dataframe(mdf)
        self.z_model_error = ModelErrors(
            mode="impedance", **modem_data.z_model_error.error_parameters
        )
        self.t_model_error = ModelErrors(
            mode="tipper", **modem_data.t_model_error.error_parameters
        )
        self.data_rotation_angle = modem_data.rotation_angle
        self._center_lat = modem_data.center_point.latitude
        self._center_lon = modem_data.center_point.longitude
        self._center_elev = modem_data.center_point.elevation
        self.attrs["utm_epsg"] = modem_data.center_point.utm_epsg
        self.attrs["datum_crs"] = modem_data.center_point.datum_crs

        self.model_parameters = {
            key: value
            for key, value in modem_data.model_parameters.items()
            if "." not in key
        }

    def to_occam2d(self, data_filename: str | Path | None = None, **kwargs: Any) -> Any:
        """Create an Occam2D data object from the station collection.

        Parameters
        ----------
        data_filename : str or pathlib.Path, optional
            Path to write the Occam2D data file.  When ``None`` (default) the
            file is not written.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`mtpy.modeling.occam2d.Occam2DData` (e.g.
            ``model_mode``, ``profile_angle``, ``res_te_err``).

        Returns
        -------
        mtpy.modeling.occam2d.Occam2DData
            Populated Occam2D data object with ``profile_origin`` set from
            :attr:`center_point` when not supplied via *kwargs*.

        Notes
        -----
        All information is derived from the station dataframe.  The user
        should create the profile, interpolate, and estimate model errors
        from the tree before calling this method.

        Examples
        --------
        >>> from mtpy.core import MTDataTree
        >>> tree = MTDataTree()
        >>> # tree.add_station(...)  # populate first
        >>> occam_data = tree.to_occam2d(
        ...     data_filename="OccamDataFile.dat", model_mode="5"
        ... )
        """
        from mtpy.modeling.occam2d import Occam2DData

        occam2d_data = Occam2DData(**kwargs)
        occam2d_data.dataframe = self.to_dataframe()
        if occam2d_data.profile_origin is None:
            cp = self.center_point
            occam2d_data.profile_origin = (cp.east, cp.north)
        if data_filename is not None:
            occam2d_data.write_data_file(data_filename)
        return occam2d_data

    def from_occam2d(
        self,
        data_filename: str | Path,
        file_type: str = "data",
        **kwargs: Any,
    ) -> None:
        """Populate the tree by reading an existing Occam2D data file.

        After reading, ``profile_origin``, ``profile_angle``, and
        ``model_mode`` are stored in :attr:`model_parameters`.

        Parameters
        ----------
        data_filename : str or pathlib.Path
            Path to the Occam2D data file.
        file_type : str, optional
            ``'data'`` (default) or ``'response'``/``'model'``.  Controls the
            survey label (``'data'`` or ``'model'``) assigned to each row.
        **kwargs
            Additional keyword arguments forwarded to
            :class:`mtpy.modeling.occam2d.Occam2DData`.

        Examples
        --------
        Read a data file:

        >>> from mtpy.core import MTDataTree
        >>> tree = MTDataTree()
        >>> tree.from_occam2d("OccamDataFile.dat")
        >>> print(tree.station_ids)

        Read a response / model file:

        >>> tree.from_occam2d("OccamResponse.dat", file_type="response")
        """
        from mtpy.modeling.occam2d import Occam2DData

        occam2d_data = Occam2DData(**kwargs)
        occam2d_data.read_data_file(data_filename)
        if file_type in ["data"]:
            occam2d_data.dataframe["survey"] = "data"
        elif file_type in ["response", "model"]:
            occam2d_data.dataframe["survey"] = "model"
        self.from_dataframe(occam2d_data.dataframe)
        self.model_parameters["profile_origin"] = occam2d_data.profile_origin
        self.model_parameters["profile_angle"] = occam2d_data.profile_angle
        self.model_parameters["model_mode"] = occam2d_data.model_mode

    def to_simpeg_2d(self, **kwargs: Any) -> Any:
        """Create a SimPEG 2-D MT data object from the station collection.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            :class:`mtpy.modeling.simpeg.data_2d.Simpeg2DData`.  Common
            options include:

            include_elevation : bool
                Include station elevation in the receiver locations,
                by default ``True``.
            invert_te : bool
                Include TE-mode apparent resistivity and phase,
                by default ``True``.
            invert_tm : bool
                Include TM-mode apparent resistivity and phase,
                by default ``True``.

        Returns
        -------
        mtpy.modeling.simpeg.data_2d.Simpeg2DData
            Populated SimPEG 2-D data object.

        Notes
        -----
        The impedance units are converted to ``'ohm'`` automatically.
        The user should create the profile, interpolate, and estimate model
        errors from the tree before calling this method.

        Examples
        --------
        >>> from mtpy.core import MTDataTree
        >>> tree = MTDataTree()
        >>> # tree.add_station(...)  # populate first
        >>> simpeg_2d = tree.to_simpeg_2d(invert_te=True, invert_tm=False)
        """
        from mtpy.modeling.simpeg.data_2d import Simpeg2DData

        return Simpeg2DData(self.to_dataframe(impedance_units="ohm"), **kwargs)

    def to_simpeg_3d(self, **kwargs: Any) -> Any:
        """Create a SimPEG 3-D MT data object from the station collection.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments forwarded to
            :class:`mtpy.modeling.simpeg.data_3d.Simpeg3DData`.  Common
            options include:

            include_elevation : bool
                Include station elevation in the receiver locations,
                by default ``False``.
            geographic_coordinates : bool
                Use geographic (UTM) coordinates instead of model-relative
                coordinates, by default ``True``.
            invert_z_xx, invert_z_xy, invert_z_yx, invert_z_yy : bool
                Select which impedance tensor components to include,
                all default to ``True``.
            invert_t_zx, invert_t_zy : bool
                Select which tipper components to include,
                both default to ``True``.

        Returns
        -------
        mtpy.modeling.simpeg.data_3d.Simpeg3DData
            Populated SimPEG 3-D data object.

        Notes
        -----
        The impedance units are converted to ``'ohm'`` automatically.
        The user should interpolate and estimate model errors from the tree
        before calling this method.

        Examples
        --------
        >>> from mtpy.core import MTDataTree
        >>> tree = MTDataTree()
        >>> # tree.add_station(...)  # populate first
        >>> simpeg_3d = tree.to_simpeg_3d(invert_z_yy=False, include_elevation=True)
        """
        from mtpy.modeling.simpeg.data_3d import Simpeg3DData

        return Simpeg3DData(self.to_dataframe(impedance_units="ohm"), **kwargs)

    def add_white_noise(
        self, value: float, inplace: bool = True
    ) -> "MTDataTree | None":
        """Add white noise to the impedance and tipper of every station.

        Multiplies the real and imaginary parts of the transfer function by
        independent random factors drawn from
        ``1 ± U(0, value) `` and increments the transfer-function error by
        *value*.  Useful for generating synthetic test datasets.

        Parameters
        ----------
        value : float
            Noise level expressed as a decimal fraction (0–1) or as a
            percentage (>1).  Values greater than 1 are divided by 100
            automatically (e.g. ``10`` becomes ``0.10``, i.e. 10 %).
        inplace : bool, optional
            When ``True`` (default) each station dataset is modified in place
            and ``None`` is returned.  When ``False`` a new
            :class:`MTDataTree` containing noisy copies is returned and the
            original tree is left unchanged.

        Returns
        -------
        MTDataTree or None
            A new tree containing noisy station data when *inplace* is
            ``False``; ``None`` otherwise.

        Examples
        --------
        In-place noise addition:

        >>> from mtpy.core import MTDataTree
        >>> tree = MTDataTree()
        >>> # tree.add_station(...)  # populate first
        >>> tree.add_white_noise(5)  # adds 5 % noise in place

        Non-destructive copy with noise:

        >>> noisy_tree = tree.add_white_noise(0.05, inplace=False)
        """
        if value > 1:
            value = value / 100.0

        paths = self._iter_station_paths()

        if inplace:
            for path in paths:
                mt_obj = self.get_station(path, as_mt=True)
                mt_obj.add_white_noise(value)
                self.add_station(mt_obj)
            return None
        else:
            mt_list = []
            for path in paths:
                mt_obj = self.get_station(path, as_mt=True)
                mt_list.append(mt_obj.add_white_noise(value, inplace=False))
            new_tree = self.clone_empty()
            new_tree.add_stations(mt_list)
            return new_tree

    def estimate_spatial_static_shift(
        self,
        station_key: str,
        radius: float,
        period_min: float,
        period_max: float,
        radius_units: str = "m",
        shift_tolerance: float = 0.15,
    ) -> tuple[float, float]:
        """Estimate static shift from nearby stations."""
        nearby_keys = self.get_nearby_stations(station_key, radius, radius_units)
        if len(nearby_keys) == 0:
            return 1.0, 1.0

        nearby_paths = [self._resolve_station_path(key) for key in nearby_keys]
        md = self.get_subset(nearby_paths)

        local_site = self.get_station(
            self._resolve_station_path(station_key),
            as_mt=True,
        )

        interp_periods = local_site.period[
            np.where(
                (local_site.period >= period_min) & (local_site.period <= period_max)
            )
        ]

        local_site = local_site.interpolate(interp_periods)
        md.interpolate(interp_periods)

        df = md.to_dataframe()

        sx = np.nanmedian(df.res_xy) / np.nanmedian(local_site.Z.res_xy)
        sy = np.nanmedian(df.res_yx) / np.nanmedian(local_site.Z.res_yx)

        if 1 - shift_tolerance < sx < 1 + shift_tolerance:
            sx = 1.0
        if 1 - shift_tolerance < sy < 1 + shift_tolerance:
            sy = 1.0

        return sx, sy

    @property
    def n_stations(self) -> int:
        """Total number of stations in the collection."""
        self.compute()

        if self._index is not None:
            return self._index.n_stations()
        return len(self._iter_station_paths())

    @property
    def survey_ids(self) -> list[str]:
        """Unique survey IDs in the collection."""
        self.compute()

        if self._index is not None:
            return [row.name for row in self._index.all_surveys()]

        return list(
            {
                path.split("/", 3)[1]
                for path in self._iter_station_paths()
                if path.count("/") >= 3
            }
        )

    def get_survey(self, survey_id: str) -> "MTDataTree":
        """Get all stations belonging to a specific survey."""
        self.compute()

        station_list = [
            station_path
            for station_path in self._iter_station_paths()
            if station_path.startswith(
                f"{self.SURVEYS_NODE}/{survey_id}/{self.STATIONS_NODE}/"
            )
        ]
        return self.get_subset(station_list)

    def add_station(
        self,
        mt_obj: "MT | str | Path | list[MT | str | Path]",
        overwrite: bool = True,
        dataset_copy_mode: str | None = None,
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
        dataset_copy_mode : {'deep', 'shallow', 'none'}, optional
            Dataset copy behavior for station transfer-function storage.

        Returns
        -------
        str or list[str]
            Station node path for scalar inputs or list of paths for list
            inputs.
        """
        self.compute()

        if mt_obj is None:
            raise TypeError("mt_obj cannot be None")

        if isinstance(mt_obj, list):
            return self.add_stations(
                mt_obj,
                overwrite=overwrite,
                dataset_copy_mode=dataset_copy_mode,
            )

        (
            station_path,
            station,
            station_ds,
            metadata_objects,
        ) = self._coerce_and_prepare_station(
            mt_obj,
            dataset_copy_mode=dataset_copy_mode,
        )

        if self._path_exists(station_path) and not overwrite:
            raise KeyError(f"Station path already exists: {station_path}")

        self.tree[station_path] = xr.DataTree(name=station, dataset=station_ds)
        self._commit_cached_metadata(
            station_path,
            metadata_objects["survey"],
            metadata_objects["station"],
        )
        if self._index is not None:
            station_row, period_row = MTDataTreeIndexStore._extract_rows(
                station_path, station_ds
            )
            self._index.upsert_station(station_row)
            if period_row is not None:
                self._index.replace_station_period_rows(period_row)
            self._index.refresh_survey_aggregates(station_row.survey_name)
        return station_path

    def add_tf(
        self,
        tf: "MT | str | Path | list[MT | str | Path]",
        **kwargs: Any,
    ) -> str | list[str]:
        """Alias for add_station to mirror MTData API."""
        return self.add_station(tf, **kwargs)

    def add_stations(
        self,
        mt_objects: list["MT | str | Path"],
        overwrite: bool = True,
        dataset_copy_mode: str | None = None,
        precomputed_attrs: list[dict[str, Any] | None] | None = None,
    ) -> list[str]:
        """
        Bulk-add MT stations with optional precomputed attrs for fast ingest.

        Parameters
        ----------
        mt_objects : list
            List of MT objects, filename strings, or Paths.
        overwrite : bool, optional
            If False, raise if a station path already exists.
        dataset_copy_mode : {'deep', 'shallow', 'none'}, optional
            Dataset copy behavior for station transfer-function storage.
        precomputed_attrs : list[dict | None], optional
            Optional attrs payload aligned by index with mt_objects. When
            provided, these attrs are used directly and only canonical keys are
            enforced (survey/station and metadata refs).

        Returns
        -------
        list[str]
            Inserted station paths.
        """
        self.compute()

        if mt_objects is None:
            raise TypeError("mt_objects cannot be None")
        if not isinstance(mt_objects, list):
            raise TypeError("mt_objects must be a list")
        if not mt_objects:
            return []

        if precomputed_attrs is not None:
            if not isinstance(precomputed_attrs, list):
                raise TypeError("precomputed_attrs must be a list when provided")
            if len(precomputed_attrs) != len(mt_objects):
                raise ValueError("precomputed_attrs must match mt_objects length")

        prepared: list[tuple[str, str, xr.Dataset, dict[str, Any]]] = []
        seen_paths: set[str] = set()
        for index, mt_obj in enumerate(mt_objects):
            attrs = None
            if precomputed_attrs is not None:
                attrs = precomputed_attrs[index]
                if attrs is not None and not isinstance(attrs, dict):
                    raise TypeError("Each precomputed_attrs entry must be dict or None")

            (
                station_path,
                station,
                station_ds,
                metadata_objects,
            ) = self._coerce_and_prepare_station(
                mt_obj,
                dataset_copy_mode=dataset_copy_mode,
                precomputed_attrs=attrs,
            )
            if station_path in seen_paths and not overwrite:
                raise KeyError(f"Station path already exists: {station_path}")
            seen_paths.add(station_path)
            if self._path_exists(station_path) and not overwrite:
                raise KeyError(f"Station path already exists: {station_path}")
            prepared.append((station_path, station, station_ds, metadata_objects))

        parent_cache: dict[str, Any] = {}
        inserted_paths: list[str] = []
        for station_path, station, station_ds, metadata_objects in prepared:
            parent_path, child_name = station_path.rsplit("/", 1)
            parent_node = parent_cache.get(parent_path)
            if parent_node is None:
                try:
                    parent_node = self.tree[parent_path]
                except KeyError:
                    _, survey_name, _ = parent_path.split("/", 2)
                    survey_path = f"{self.SURVEYS_NODE}/{survey_name}"
                    if not self._path_exists(survey_path):
                        self.tree[survey_path] = xr.DataTree(
                            name=survey_name,
                            dataset=xr.Dataset(),
                        )
                    if not self._path_exists(parent_path):
                        self.tree[parent_path] = xr.DataTree(
                            name=self.STATIONS_NODE,
                            dataset=xr.Dataset(),
                        )
                    parent_node = self.tree[parent_path]
                parent_cache[parent_path] = parent_node

            parent_node[child_name] = xr.DataTree(
                name=station,
                dataset=station_ds,
            )
            self._commit_cached_metadata(
                station_path,
                metadata_objects["survey"],
                metadata_objects["station"],
            )
            inserted_paths.append(station_path)

        if self._index is not None:
            updated_surveys: set[str] = set()
            for station_path, _station, station_ds, _meta in prepared:
                station_row, period_row = MTDataTreeIndexStore._extract_rows(
                    station_path, station_ds
                )
                self._index.upsert_station(station_row)
                if period_row is not None:
                    self._index.replace_station_period_rows(period_row)
                updated_surveys.add(station_row.survey_name)
            for sv in updated_surveys:
                self._index.refresh_survey_aggregates(sv)

        return inserted_paths

    def get_station(self, station_key: str, as_mt: bool = False) -> xr.Dataset | "MT":
        """Return a station by tree path as dataset or reconstructed MT object."""
        self.compute(station_paths=[station_key])
        station_ds = self.tree[station_key].ds
        if as_mt:
            mt_obj = self._dataset_to_mt(station_ds)
            self._hydrate_metadata_from_cache(mt_obj, station_ds)
            return mt_obj
        return station_ds

    def remove_station(self, station_key: str) -> None:
        """Remove a station node from the tree."""
        self.compute()
        self._lazy_station_transforms.pop(station_key, None)
        self._clear_cached_metadata(station_key)
        if self._index is not None:
            self._index.delete_station_by_tree_path(station_key)
        if "/" not in station_key:
            del self.tree[station_key]
            return

        parent_path, child_name = station_key.rsplit("/", 1)
        del self.tree[parent_path][child_name]

    def get_subset(self, station_list: list[str]) -> "MTDataTree":
        """Return a new tree containing only the requested station paths."""
        subset = self.__class__(
            metadata_storage=self.metadata_storage,
            **dict(self.attrs),
        )
        for station_key in station_list:
            station_ds = self.get_station(station_key).copy()
            attrs = station_ds.attrs
            target_path = self._station_path(
                self._clean_name(attrs.get("survey"), "default"),
                self._clean_name(attrs.get("station"), "unknown_station"),
            )

            if self.metadata_storage == "cache":
                for metadata_kind in ["survey", "station"]:
                    cached_md = self._metadata_cache[metadata_kind].get(station_key)
                    if cached_md is None:
                        continue
                    subset._metadata_cache[metadata_kind][target_path] = cached_md
                    station_ds.attrs[f"{metadata_kind}_metadata_ref"] = target_path

            subset.tree[target_path] = xr.DataTree(
                name=target_path.rsplit("/", 1)[-1], dataset=station_ds
            )
        return subset

    def _set_station_dataset(self, station_path: str, station_ds: xr.Dataset) -> None:
        """Insert or replace a station dataset at its tree path."""
        parent_path, child_name = station_path.rsplit("/", 1)
        try:
            parent_node = self.tree[parent_path]
        except KeyError:
            _, survey_name, _ = parent_path.split("/", 2)
            survey_path = f"{self.SURVEYS_NODE}/{survey_name}"
            if not self._path_exists(survey_path):
                self.tree[survey_path] = xr.DataTree(
                    name=survey_name,
                    dataset=xr.Dataset(),
                )
            if not self._path_exists(parent_path):
                self.tree[parent_path] = xr.DataTree(
                    name=self.STATIONS_NODE,
                    dataset=xr.Dataset(),
                )
            parent_node = self.tree[parent_path]

        parent_node[child_name] = xr.DataTree(name=child_name, dataset=station_ds)

    @staticmethod
    def _interpolate_station_dataset(
        station_ds: xr.Dataset,
        new_periods: np.ndarray,
        **kwargs: Any,
    ) -> xr.Dataset:
        """Interpolate a stored station dataset with TFBase semantics."""
        from .transfer_function.base import TFBase

        target_periods = np.asarray(new_periods, dtype=float)
        attrs = dict(station_ds.attrs)

        if "period" not in station_ds.coords or not station_ds.data_vars:
            coords = {
                coord_name: coord.values
                for coord_name, coord in station_ds.coords.items()
                if coord_name != "period"
            }
            coords["period"] = target_periods
            interpolated_ds = xr.Dataset(coords=coords)
        elif target_periods.size == 0:
            interpolated_ds = station_ds.isel(period=slice(0, 0)).copy(deep=True)
        else:
            tf_obj = TFBase()
            # TFBase.interpolate reads source arrays and writes a new dataset,
            # so we can avoid deep-copying the input and the return-object copy.
            tf_obj.from_xarray(station_ds)
            tf_obj.interpolate(
                target_periods,
                inplace=True,
                **kwargs,
            )
            interpolated_ds = tf_obj.to_xarray()

        interpolated_ds.attrs.update(attrs)
        return interpolated_ds

    @staticmethod
    def _rotate_station_dataset(
        station_ds: xr.Dataset,
        rotation_angle: float | np.ndarray,
        coordinate_reference_frame: str = "ned",
    ) -> xr.Dataset:
        """Rotate impedance/tipper channel blocks with MT.rotate semantics."""
        from mtpy.utils.calculator import (
            rotate_matrix_with_errors,
            rotate_vector_with_errors,
        )

        attrs = dict(station_ds.attrs)
        rotated_ds = station_ds.copy(deep=True).load()

        if (
            "transfer_function" not in rotated_ds
            or "period" not in rotated_ds.coords
            or "output" not in rotated_ds.coords
            or "input" not in rotated_ds.coords
        ):
            rotated_ds.attrs.update(attrs)
            return rotated_ds

        n_periods = int(rotated_ds.sizes.get("period", 0))
        if n_periods == 0:
            rotated_ds.attrs.update(attrs)
            return rotated_ds

        if isinstance(rotation_angle, (float, int, str, np.floating, np.integer)):
            degree_angle = np.repeat(float(rotation_angle) % 360.0, n_periods)
        elif isinstance(rotation_angle, (list, tuple, np.ndarray)):
            angles = np.asarray(rotation_angle, dtype=float) % 360.0
            if angles.size == 1:
                degree_angle = np.repeat(float(angles[0]), n_periods)
            elif angles.size == n_periods:
                degree_angle = angles
            else:
                raise ValueError(
                    "angles must be the same size as periods "
                    f"{n_periods} not {angles.size}"
                )
        else:
            raise ValueError(
                f"Angle must be a valid number (in degrees) not {rotation_angle}"
            )

        crf = str(coordinate_reference_frame).lower()
        if crf in ["ned", "+"]:
            clockwise = True
        elif crf in ["enu", "-"]:
            clockwise = False
        else:
            raise ValueError(
                f"coordinate_reference_frame {coordinate_reference_frame} not understood."
            )

        output_labels = list(rotated_ds.coords["output"].values)
        input_labels = list(rotated_ds.coords["input"].values)

        z_outputs = MTDataTree._pick_channel_labels(
            output_labels, ["ex", "ey", "x", "y"], 2
        )
        z_inputs = MTDataTree._pick_channel_labels(
            input_labels, ["hx", "hy", "x", "y"], 2
        )
        if z_outputs is not None and z_inputs is not None:
            tf_block = (
                rotated_ds["transfer_function"]
                .sel(output=z_outputs, input=z_inputs)
                .values.copy()
            )
            rot_tf = np.zeros_like(tf_block, dtype=complex)
            for index, angle in enumerate(degree_angle):
                rot_tf[index], _ = rotate_matrix_with_errors(
                    tf_block[index],
                    angle,
                    clockwise=clockwise,
                )
            rotated_ds["transfer_function"].loc[
                dict(output=z_outputs, input=z_inputs)
            ] = rot_tf

            if "transfer_function_error" in rotated_ds:
                tf_error_block = (
                    rotated_ds["transfer_function_error"]
                    .sel(output=z_outputs, input=z_inputs)
                    .values.copy()
                )
                rot_tf_error = np.zeros_like(tf_error_block, dtype=float)
                for index, angle in enumerate(degree_angle):
                    _, rot_tf_error[index] = rotate_matrix_with_errors(
                        tf_block[index],
                        angle,
                        tf_error_block[index],
                        clockwise=clockwise,
                    )
                rotated_ds["transfer_function_error"].loc[
                    dict(output=z_outputs, input=z_inputs)
                ] = rot_tf_error

            if "transfer_function_model_error" in rotated_ds:
                tf_model_error_block = (
                    rotated_ds["transfer_function_model_error"]
                    .sel(output=z_outputs, input=z_inputs)
                    .values.copy()
                )
                rot_tf_model_error = np.zeros_like(tf_model_error_block, dtype=float)
                for index, angle in enumerate(degree_angle):
                    _, rot_tf_model_error[index] = rotate_matrix_with_errors(
                        tf_block[index],
                        angle,
                        tf_model_error_block[index],
                        clockwise=clockwise,
                    )
                rotated_ds["transfer_function_model_error"].loc[
                    dict(output=z_outputs, input=z_inputs)
                ] = rot_tf_model_error

        t_output = MTDataTree._pick_channel_labels(output_labels, ["hz", "z"], 1)
        t_inputs = MTDataTree._pick_channel_labels(
            input_labels, ["hx", "hy", "x", "y"], 2
        )
        if t_output is not None and t_inputs is not None:
            tf_tipper = (
                rotated_ds["transfer_function"]
                .sel(output=t_output, input=t_inputs)
                .values.copy()
            )
            rot_tipper = np.zeros_like(tf_tipper, dtype=complex)
            for index, angle in enumerate(degree_angle):
                rot_tipper[index], _ = rotate_vector_with_errors(
                    tf_tipper[index],
                    angle,
                    clockwise=clockwise,
                )
            rotated_ds["transfer_function"].loc[
                dict(output=t_output, input=t_inputs)
            ] = rot_tipper

            if "transfer_function_error" in rotated_ds:
                tipper_error = (
                    rotated_ds["transfer_function_error"]
                    .sel(output=t_output, input=t_inputs)
                    .values.copy()
                )
                rot_tipper_error = np.zeros_like(tipper_error, dtype=float)
                for index, angle in enumerate(degree_angle):
                    _, rot_tipper_error[index] = rotate_vector_with_errors(
                        tf_tipper[index],
                        angle,
                        tipper_error[index],
                        clockwise=clockwise,
                    )
                rotated_ds["transfer_function_error"].loc[
                    dict(output=t_output, input=t_inputs)
                ] = rot_tipper_error

            if "transfer_function_model_error" in rotated_ds:
                tipper_model_error = (
                    rotated_ds["transfer_function_model_error"]
                    .sel(output=t_output, input=t_inputs)
                    .values.copy()
                )
                rot_tipper_model_error = np.zeros_like(
                    tipper_model_error,
                    dtype=float,
                )
                for index, angle in enumerate(degree_angle):
                    _, rot_tipper_model_error[index] = rotate_vector_with_errors(
                        tf_tipper[index],
                        angle,
                        tipper_model_error[index],
                        clockwise=clockwise,
                    )
                rotated_ds["transfer_function_model_error"].loc[
                    dict(output=t_output, input=t_inputs)
                ] = rot_tipper_model_error

        rotated_ds.attrs.update(attrs)
        return rotated_ds

    def rotate(
        self,
        rotation_angle: float | np.ndarray,
        inplace: bool = True,
    ) -> "MTDataTree" | None:
        """Rotate all station datasets by a given angle."""
        tree_obj = self
        if not inplace:
            tree_obj = self.__class__(
                metadata_storage=self.metadata_storage,
                dataset_copy_mode=self.dataset_copy_mode,
                use_index=self._index is not None,
                index_db_path=self._index_db_path,
                **dict(self.attrs),
            )

        updated_surveys: set[str] = set()
        for station_path in self._iter_station_paths():
            station_ds = self.get_station(station_path)
            crf = station_ds.attrs.get(
                "coordinate_reference_frame",
                self.attrs.get("coordinate_reference_frame", "ned"),
            )
            rotated_ds = self._rotate_station_dataset(
                station_ds,
                rotation_angle,
                coordinate_reference_frame=crf,
            )
            tree_obj._set_station_dataset(station_path, rotated_ds)

            if tree_obj.metadata_storage == "cache":
                for metadata_kind in ["survey", "station"]:
                    cached_md = self._metadata_cache[metadata_kind].get(station_path)
                    if cached_md is not None:
                        tree_obj._metadata_cache[metadata_kind][
                            station_path
                        ] = cached_md

            if tree_obj._index is not None:
                station_row, period_row = MTDataTreeIndexStore._extract_rows(
                    station_path,
                    rotated_ds,
                )
                tree_obj._index.upsert_station(station_row)
                if period_row is not None:
                    tree_obj._index.replace_station_period_rows(period_row)
                updated_surveys.add(station_row.survey_name)

        if tree_obj._index is not None:
            for survey_name in updated_surveys:
                tree_obj._index.refresh_survey_aggregates(survey_name)

        if not inplace:
            return tree_obj
        return None

    def interpolate(
        self,
        new_periods: np.ndarray,
        f_type: str = "period",
        inplace: bool = True,
        bounds_error: bool = True,
        **kwargs: Any,
    ) -> "MTDataTree" | None:
        """Interpolate all station datasets onto a common period range."""
        if f_type not in ["frequency", "freq", "period", "per"]:
            raise ValueError(
                f"f_type must be either 'frequency' or 'period' not {f_type}"
            )

        target_periods = np.asarray(new_periods, dtype=float)
        if target_periods.ndim != 1:
            target_periods = target_periods.reshape(-1)
        if f_type in ["frequency", "freq"]:
            target_periods = 1.0 / target_periods

        tree_obj = self
        if not inplace:
            tree_obj = self.__class__(
                metadata_storage=self.metadata_storage,
                dataset_copy_mode=self.dataset_copy_mode,
                use_index=self._index is not None,
                index_db_path=self._index_db_path,
                **dict(self.attrs),
            )

        updated_surveys: set[str] = set()
        for station_path in self._iter_station_paths():
            station_ds = self.get_station(station_path)
            interp_periods = target_periods
            if bounds_error and "period" in station_ds.coords:
                station_periods = np.asarray(
                    station_ds.coords["period"].values, dtype=float
                )
                if station_periods.size > 0:
                    interp_periods = target_periods[
                        (target_periods <= station_periods.max())
                        & (target_periods >= station_periods.min())
                    ]

            interpolated_ds = self._interpolate_station_dataset(
                station_ds,
                interp_periods,
                **kwargs,
            )
            tree_obj._set_station_dataset(station_path, interpolated_ds)

            if tree_obj.metadata_storage == "cache":
                for metadata_kind in ["survey", "station"]:
                    cached_md = self._metadata_cache[metadata_kind].get(station_path)
                    if cached_md is not None:
                        tree_obj._metadata_cache[metadata_kind][
                            station_path
                        ] = cached_md

            if tree_obj._index is not None:
                station_row, period_row = MTDataTreeIndexStore._extract_rows(
                    station_path,
                    interpolated_ds,
                )
                if period_row is None:
                    tree_obj._index.delete_station_by_tree_path(station_path)
                tree_obj._index.upsert_station(station_row)
                if period_row is not None:
                    tree_obj._index.replace_station_period_rows(period_row)
                updated_surveys.add(station_row.survey_name)

        if tree_obj._index is not None:
            for survey_name in updated_surveys:
                tree_obj._index.refresh_survey_aggregates(survey_name)

        if not inplace:
            return tree_obj
        return None

    def interpolate_lazy(
        self,
        new_periods: np.ndarray,
        f_type: str = "period",
        inplace: bool = False,
        bounds_error: bool = True,
        **kwargs: Any,
    ) -> "MTDataTree":
        """Build deferred station interpolation transforms without materializing."""
        if f_type not in ["frequency", "freq", "period", "per"]:
            raise ValueError(
                f"f_type must be either 'frequency' or 'period' not {f_type}"
            )

        # Build lazy plans from realized source station datasets.
        self.compute()

        target_periods = np.asarray(new_periods, dtype=float)
        if target_periods.ndim != 1:
            target_periods = target_periods.reshape(-1)
        if f_type in ["frequency", "freq"]:
            target_periods = 1.0 / target_periods

        tree_obj = self
        if not inplace:
            tree_obj = self.__class__(
                metadata_storage=self.metadata_storage,
                dataset_copy_mode=self.dataset_copy_mode,
                use_index=False,
                index_db_path=self._index_db_path,
                **dict(self.attrs),
            )
            tree_obj._lazy_use_index = self._index is not None
        else:
            tree_obj._lazy_station_transforms.clear()

        for station_path in self._iter_station_paths():
            source_station_ds = self.get_station(station_path)

            interp_periods = target_periods
            if bounds_error and "period" in source_station_ds.coords:
                station_periods = np.asarray(
                    source_station_ds.coords["period"].values,
                    dtype=float,
                )
                if station_periods.size > 0:
                    interp_periods = target_periods[
                        (target_periods <= station_periods.max())
                        & (target_periods >= station_periods.min())
                    ]

            source_snapshot = source_station_ds.copy(deep=False)
            target_snapshot = np.asarray(interp_periods, dtype=float)
            interp_kwargs = dict(kwargs)

            def _transform(
                ds: xr.Dataset = source_snapshot,
                periods: np.ndarray = target_snapshot,
                op_kwargs: dict[str, Any] = interp_kwargs,
            ) -> xr.Dataset:
                return MTDataTree._interpolate_station_dataset(ds, periods, **op_kwargs)

            tree_obj._lazy_station_transforms[station_path] = _transform

            if not inplace:
                tree_obj._set_station_dataset(
                    station_path, source_snapshot.copy(deep=False)
                )

            if tree_obj.metadata_storage == "cache":
                for metadata_kind in ["survey", "station"]:
                    cached_md = self._metadata_cache[metadata_kind].get(station_path)
                    if cached_md is not None:
                        tree_obj._metadata_cache[metadata_kind][
                            station_path
                        ] = cached_md

        return tree_obj

    def apply_bounding_box(
        self, lon_min: float, lon_max: float, lat_min: float, lat_max: float
    ) -> "MTDataTree":
        """Return a new tree containing stations within a geographic bounding box."""
        if self._index is not None:
            station_keys = self._index.query_station_paths(
                lon_min=lon_min, lon_max=lon_max, lat_min=lat_min, lat_max=lat_max
            )
            return self.get_subset(station_keys)

        station_df = self.station_locations
        if station_df is None or station_df.empty:
            return self.__class__(**dict(self.attrs))

        bb_df = station_df.loc[
            (station_df.longitude >= lon_min)
            & (station_df.longitude <= lon_max)
            & (station_df.latitude >= lat_min)
            & (station_df.latitude <= lat_max)
        ]

        station_keys = [
            self._station_path(
                self._clean_name(survey, "default"),
                self._clean_name(station, "unknown_station"),
            )
            for survey, station in zip(bb_df.survey, bb_df.station)
        ]

        return self.get_subset(station_keys)

    def rebuild_index(self, index_db_path: str = ":memory:") -> None:
        """
        Build or replace the station index from the current tree contents.

        Enables the index if it was not already active.

        Parameters
        ----------
        index_db_path : str
            SQLite database path.  Defaults to ``":memory:"`` (in-process).
        """
        # Temporarily clear self._index so _iter_station_paths() uses the tree
        # walk, not the (new, empty) index that would otherwise be returned.
        saved = self._index
        self._index = None
        try:
            new_index = MTDataTreeIndexStore(index_db_path)
            new_index.rebuild_from_tree(self)
        except Exception:
            self._index = saved
            raise
        self._index = new_index

    def query_station_paths(
        self,
        survey: str | None = None,
        lat_min: float | None = None,
        lat_max: float | None = None,
        lon_min: float | None = None,
        lon_max: float | None = None,
        period_min: float | None = None,
        period_max: float | None = None,
    ) -> list[str]:
        """
        Return station tree paths matching filter criteria via the index.

        Requires the index to be enabled (``use_index=True`` or after calling
        :meth:`rebuild_index`).

        Parameters
        ----------
        survey, lat_min, lat_max, lon_min, lon_max, period_min, period_max
            See :meth:`MTDataTreeIndexStore.query_station_paths`.

        Returns
        -------
        list[str]
        """
        self.compute()
        if self._index is None:
            raise RuntimeError(
                "Index not enabled. Pass use_index=True to the constructor "
                "or call rebuild_index() first."
            )
        return self._index.query_station_paths(
            survey=survey,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            period_min=period_min,
            period_max=period_max,
        )

    def to_dataframe(
        self,
        utm_crs: Any | None = None,
        cols: list[str] | None = None,
        impedance_units: str = "mt",
    ) -> pd.DataFrame:
        """Convert all stations in the tree to a pandas DataFrame."""
        self.compute()
        station_paths = self._iter_station_paths()
        df_list = []
        for path in station_paths:
            station_ds = self.get_station(path)
            try:
                df_list.append(
                    self._station_dataset_to_dataframe(
                        station_ds,
                        utm_crs=utm_crs,
                        cols=cols,
                        impedance_units=impedance_units,
                    )
                )
            except Exception:
                # Fallback keeps behavior for unexpected/legacy dataset layouts.
                df_list.append(
                    self._dataset_to_mt(station_ds)
                    .to_dataframe(
                        utm_crs=utm_crs,
                        cols=cols,
                        impedance_units=impedance_units,
                    )
                    .dataframe
                )

        if not df_list:
            return pd.DataFrame()

        return pd.concat(df_list, ignore_index=True)

    def to_mt_dataframe(
        self, utm_crs: Any | None = None, impedance_units: str = "mt"
    ) -> MTDataFrame:
        """Create an MTDataFrame containing all stations in the tree."""
        return MTDataFrame(
            self.to_dataframe(utm_crs=utm_crs, impedance_units=impedance_units)
        )

    def from_dataframe(self, df: pd.DataFrame, impedance_units: str = "mt") -> None:
        """Populate the tree from a pandas DataFrame of MT station data."""
        from .mt import MT

        if df.empty:
            return

        group_cols = ["station"]
        if "survey" in df.columns:
            group_cols = ["survey", "station"]

        mt_objects = []
        for _, sdf in df.groupby(group_cols, sort=False):
            mt_object = MT(period=sdf.period.unique())
            mt_object.from_dataframe(sdf, impedance_units=impedance_units)
            mt_objects.append(mt_object)

        if mt_objects:
            self.add_stations(mt_objects)

    def from_mt_dataframe(
        self, mt_df: MTDataFrame, impedance_units: str = "mt"
    ) -> None:
        """Populate the tree from an MTDataFrame."""
        self.from_dataframe(mt_df.dataframe, impedance_units=impedance_units)

    def get_periods(self) -> np.ndarray:
        """Return sorted unique periods across all station datasets in the tree."""
        self.compute()
        periods: list[np.ndarray] = []

        def _walk(node: Any) -> None:
            ds = getattr(node, "ds", None)
            if isinstance(ds, xr.Dataset) and "period" in ds.coords:
                periods.append(np.asarray(ds.coords["period"].values, dtype=float))
            for child in getattr(node, "children", {}).values():
                _walk(child)

        _walk(self.tree)

        if not periods:
            return np.array([], dtype=float)

        unique_periods = np.unique(np.concatenate(periods))
        unique_periods.sort()
        return unique_periods

    def keys(self) -> list[str]:
        """Return immediate child node keys for quick inspection."""
        return list(self.tree.children.keys())
