# -*- coding: utf-8 -*-
"""
Scaffold for a tree-backed MT data container.

This class is an outline for migrating from OrderedDict-based MTData to an
Xarray tree representation for better scalability.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import xarray as xr

from .mt_data_tree_index import MTDataTreeIndexStore
from .mt_dataframe import MTDataFrame


if TYPE_CHECKING:
    from .mt import MT
    from .mt_data import MTData
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

    def __init__(
        self,
        tree: Any | None = None,
        metadata_storage: str = "cache",
        dataset_copy_mode: str = "shallow",
        use_index: bool = False,
        index_db_path: str = ":memory:",
        **attrs: Any,
    ) -> None:
        if hasattr(xr, "DataTree"):
            self._xarray_tree_cls = xr.DataTree
        elif hasattr(xr, "Tree"):
            self._xarray_tree_cls = xr.Tree
        else:
            raise ImportError(
                "xarray tree support is not available. Install a version of xarray "
                "that provides DataTree or Tree."
            )

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

        self.tree[station_path] = self._xarray_tree_cls(
            name=station, dataset=station_ds
        )
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
                        self.tree[survey_path] = self._xarray_tree_cls(
                            name=survey_name,
                            dataset=xr.Dataset(),
                        )
                    if not self._path_exists(parent_path):
                        self.tree[parent_path] = self._xarray_tree_cls(
                            name=self.STATIONS_NODE,
                            dataset=xr.Dataset(),
                        )
                    parent_node = self.tree[parent_path]
                parent_cache[parent_path] = parent_node

            parent_node[child_name] = self._xarray_tree_cls(
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
        station_ds = self.tree[station_key].ds
        if as_mt:
            mt_obj = self._dataset_to_mt(station_ds)
            self._hydrate_metadata_from_cache(mt_obj, station_ds)
            return mt_obj
        return station_ds

    def remove_station(self, station_key: str) -> None:
        """Remove a station node from the tree."""
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

            subset.tree[target_path] = subset._xarray_tree_cls(
                name=target_path.rsplit("/", 1)[-1], dataset=station_ds
            )
        return subset

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
        station_paths = self._iter_station_paths()
        df_list = []
        for path in station_paths:
            station_ds = self.tree[path].ds
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
