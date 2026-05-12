"""xarray DataTree accessor for MTData station trees."""

from __future__ import annotations

from typing import Callable

import xarray as xr

if not hasattr(xr, "register_datatree_accessor"):
    raise ImportError(
        "mtpy requires xarray >= 2024.10.0 for DataTree support "
        f"(installed: {xr.__version__}). "
        "Please upgrade: pip install 'xarray>=2024.10.0'"
    )


@xr.register_datatree_accessor("mt")
class MTDataTreeAccessor:
    """Convenience accessor for MT station trees.

    The accessor provides structural operations on the MTData DataTree,
    including station path resolution, station dataset read/write operations,
    and light-weight station-wise transforms.

    Notes
    -----
    This accessor intentionally stays structural and lightweight:
    path resolution, station dataset read/write, and small tree transforms.
    Higher-level workflows (metadata policy, indexing, lazy execution) remain
    in :class:`mtpy.core.mt_data.MTData`.

    Examples
    --------
    >>> import xarray as xr
    >>> tree = xr.DataTree()
    >>> _ = tree.mt.survey_names
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
        """Survey names under the canonical surveys node.

        Returns
        -------
        list of str
            Survey names found under ``surveys``.
        """
        try:
            return list(self._obj[self.SURVEYS_NODE].children.keys())
        except KeyError:
            return []

    @property
    def station_paths(self) -> list[str]:
        """Canonical station tree paths.

        Returns
        -------
        list of str
            Fully qualified station paths of the form
            ``surveys/<survey>/stations/<station>``.
        """
        return self._iter_station_paths()

    @property
    def short_station_paths(self) -> list[str]:
        """Compact station keys in ``survey/station`` form.

        Returns
        -------
        list of str
            Compact station identifiers derived from canonical station paths.
        """
        return [
            f"{path.split('/')[1]}/{path.split('/')[3]}" for path in self.station_paths
        ]

    def has_station(self, station_key: str) -> bool:
        """Check whether a station key resolves to an existing node.

        Parameters
        ----------
        station_key : str
            Station selector in canonical
            (``surveys/<survey>/stations/<station>``), short
            (``survey/station``), dotted (``survey.station``), or station-only
            form.

        Returns
        -------
        bool
            ``True`` when the station exists, otherwise ``False``.
        """
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
        """Return a station dataset by key.

        Parameters
        ----------
        station_key : str
            Station selector in canonical
            (``surveys/<survey>/stations/<station>``), short
            (``survey/station``), dotted (``survey.station``), or station-only
            form.
        copy : bool, default=False
            If ``True``, return a copied dataset.
        deep : bool, default=False
            Passed to :meth:`xarray.Dataset.copy` when ``copy=True``.

        Returns
        -------
        xarray.Dataset
            Dataset for the selected station.

        Examples
        --------
        >>> ds = tree.mt.get_station_dataset("survey_a/st01")
        >>> ds2 = tree.mt.get_station_dataset("survey_a.st01", copy=True)
        """
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
        """Set a station dataset.

        Parameters
        ----------
        station_key : str
            Station selector in canonical, short, dotted, or station-only form.
        station_ds : xarray.Dataset
            Dataset to assign to the station node.
        inplace : bool, default=True
            If ``True``, modify the current tree and return ``None``.
            If ``False``, return a modified deep copy.
        create_parents : bool, default=True
            If ``True``, create missing survey/stations parent nodes.

        Returns
        -------
        xarray.DataTree or None
            Modified DataTree when ``inplace=False``; otherwise ``None``.

        Examples
        --------
        >>> updated = ds.tf.rotate(10)
        >>> tree.mt.set_station_dataset("survey_a/st01", updated, inplace=True)
        """
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
        """Apply a station-wise dataset transform.

        Parameters
        ----------
        transform : callable
            Callable with signature ``transform(ds) -> xr.Dataset``.
        station_paths : list of str or None, default=None
            Optional station selectors. If ``None``, all stations are used.
        inplace : bool, default=False
            If ``True``, modify this tree and return ``None``.
            If ``False``, return a transformed deep copy.

        Returns
        -------
        xarray.DataTree or None
            Transformed DataTree when ``inplace=False``; otherwise ``None``.

        Examples
        --------
        >>> rotated = tree.mt.map_stations(lambda ds: ds.tf.rotate(15))
        >>> tree.mt.map_stations(lambda ds: ds.tf.rotate(15), inplace=True)
        """
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
        """Return a new tree with only selected stations.

        Parameters
        ----------
        station_paths : list of str
            Station selectors in canonical, short, dotted, or station-only
            form.
        deep : bool, default=False
            If ``True``, deep-copy station datasets into the output tree.

        Returns
        -------
        xarray.DataTree
            New tree containing only the selected station nodes while
            preserving top-level tree name and attributes.

        Examples
        --------
        >>> subset = tree.mt.select_stations(["survey_a/st01", "survey_a/st02"])
        """
        selected_paths = [self._resolve_station_path(path) for path in station_paths]

        out_tree = xr.DataTree(name=self._obj.name, dataset=self._obj.ds)
        out_tree.attrs.update(dict(self._obj.attrs))
        out_accessor = MTDataTreeAccessor(out_tree)

        for station_path in selected_paths:
            station_ds = self.get_station_dataset(station_path, copy=True, deep=deep)
            out_accessor._ensure_parent_nodes(out_tree, station_path)
            out_accessor.set_station_dataset(station_path, station_ds, inplace=True)

        return out_tree
