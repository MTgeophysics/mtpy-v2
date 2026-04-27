# -*- coding: utf-8 -*-
"""
SQLite-backed three-table index for MTDataTree.

The index provides O(1) station lookup by tree path and fast geographic /
period-range queries without scanning every xarray dataset node.

Schema
------
surveys
    id          INTEGER PRIMARY KEY
    name        TEXT NOT NULL UNIQUE
    lat_min     REAL
    lat_max     REAL
    lon_min     REAL
    lon_max     REAL
    n_stations  INTEGER DEFAULT 0

stations
    id          INTEGER PRIMARY KEY
    tree_path   TEXT NOT NULL UNIQUE   -- e.g. "surveys/s1/stations/stn1"
    survey_id   INTEGER REFERENCES surveys(id)
    name        TEXT NOT NULL
    latitude    REAL
    longitude   REAL
    elevation   REAL
    datum_epsg  TEXT
    east        REAL
    north       REAL
    utm_epsg    TEXT
    model_east  REAL
    model_north REAL
    model_elevation REAL
    profile_offset  REAL

station_period_index
    id          INTEGER PRIMARY KEY
    station_id  INTEGER NOT NULL REFERENCES stations(id)
    period_min  REAL NOT NULL
    period_max  REAL NOT NULL
    n_periods   INTEGER NOT NULL

All three tables are rebuilt by :meth:`MTDataTreeIndexStore.rebuild_from_tree`.
"""

from __future__ import annotations

import sqlite3
from dataclasses import asdict, dataclass, field
from typing import Any, TYPE_CHECKING


if TYPE_CHECKING:
    import xarray as xr


# ---------------------------------------------------------------------------
# Row dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SurveyRow:
    name: str
    lat_min: float | None = None
    lat_max: float | None = None
    lon_min: float | None = None
    lon_max: float | None = None
    n_stations: int = 0
    id: int | None = field(default=None, repr=False)


@dataclass
class StationRow:
    tree_path: str
    survey_name: str
    name: str
    latitude: float | None = None
    longitude: float | None = None
    elevation: float | None = None
    datum_epsg: str | None = None
    east: float | None = None
    north: float | None = None
    utm_epsg: str | None = None
    model_east: float | None = None
    model_north: float | None = None
    model_elevation: float | None = None
    profile_offset: float | None = None
    id: int | None = field(default=None, repr=False)


@dataclass
class StationPeriodRow:
    station_path: str
    period_min: float
    period_max: float
    n_periods: int
    id: int | None = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS surveys (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL UNIQUE,
    lat_min     REAL,
    lat_max     REAL,
    lon_min     REAL,
    lon_max     REAL,
    n_stations  INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS stations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    tree_path       TEXT    NOT NULL UNIQUE,
    survey_id       INTEGER REFERENCES surveys(id) ON DELETE CASCADE,
    name            TEXT    NOT NULL,
    latitude        REAL,
    longitude       REAL,
    elevation       REAL,
    datum_epsg      TEXT,
    east            REAL,
    north           REAL,
    utm_epsg        TEXT,
    model_east      REAL,
    model_north     REAL,
    model_elevation REAL,
    profile_offset  REAL
);

CREATE TABLE IF NOT EXISTS station_period_index (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    station_id  INTEGER NOT NULL REFERENCES stations(id) ON DELETE CASCADE,
    period_min  REAL    NOT NULL,
    period_max  REAL    NOT NULL,
    n_periods   INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_stations_survey
    ON stations(survey_id);
CREATE INDEX IF NOT EXISTS idx_stations_latlon
    ON stations(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_spi_station
    ON station_period_index(station_id);
"""


# ---------------------------------------------------------------------------
# Index store
# ---------------------------------------------------------------------------


class MTDataTreeIndexStore:
    """
    SQLite-backed three-table index for an :class:`MTDataTree`.

    Parameters
    ----------
    db_path : str, optional
        Filesystem path to the SQLite database.  Use the default ``":memory:"``
        for a transient in-process index (fastest, lost when the object is
        garbage-collected) or supply a file path for a persistent on-disk index.

    Examples
    --------
    >>> store = MTDataTreeIndexStore()           # in-memory
    >>> store = MTDataTreeIndexStore("idx.db")  # persistent
    >>> store.rebuild_from_tree(my_tree)
    >>> paths = store.query_station_paths(lat_min=30, lat_max=40)
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._db_path = db_path
        # check_same_thread=False is safe here because we serialize all access
        # through Python's GIL in typical single-threaded MT workflows.
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._apply_ddl()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "MTDataTreeIndexStore":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_ddl(self) -> None:
        self._conn.executescript(_DDL)
        self._conn.commit()

    def _survey_id(self, survey_name: str) -> int | None:
        """Return the id for *survey_name*, or None if not present."""
        row = self._conn.execute(
            "SELECT id FROM surveys WHERE name = ?", (survey_name,)
        ).fetchone()
        return row["id"] if row else None

    def _station_id(self, tree_path: str) -> int | None:
        """Return the id for *tree_path*, or None if not present."""
        row = self._conn.execute(
            "SELECT id FROM stations WHERE tree_path = ?", (tree_path,)
        ).fetchone()
        return row["id"] if row else None

    # ------------------------------------------------------------------
    # Upsert helpers
    # ------------------------------------------------------------------

    def upsert_survey(self, row: SurveyRow) -> int:
        """
        Insert or update a survey record.

        Parameters
        ----------
        row : SurveyRow
            Survey metadata.  ``n_stations`` is ignored here; use
            :meth:`refresh_survey_aggregates` to recompute it from live data.

        Returns
        -------
        int
            Row id of the inserted or updated survey.
        """
        cur = self._conn.execute(
            """
            INSERT INTO surveys (name, lat_min, lat_max, lon_min, lon_max)
            VALUES (:name, :lat_min, :lat_max, :lon_min, :lon_max)
            ON CONFLICT(name) DO UPDATE SET
                lat_min = excluded.lat_min,
                lat_max = excluded.lat_max,
                lon_min = excluded.lon_min,
                lon_max = excluded.lon_max
            RETURNING id
            """,
            asdict(row),
        )
        survey_id = cur.fetchone()["id"]
        self._conn.commit()
        return survey_id

    def upsert_station(self, row: StationRow) -> int:
        """
        Insert or update a station record.

        The survey is created (via :meth:`upsert_survey`) if it does not exist.

        Parameters
        ----------
        row : StationRow
            Station metadata.

        Returns
        -------
        int
            Row id of the inserted or updated station.
        """
        survey_id = self._survey_id(row.survey_name)
        if survey_id is None:
            survey_id = self.upsert_survey(SurveyRow(name=row.survey_name))

        cur = self._conn.execute(
            """
            INSERT INTO stations (
                tree_path, survey_id, name,
                latitude, longitude, elevation, datum_epsg,
                east, north, utm_epsg,
                model_east, model_north, model_elevation, profile_offset
            ) VALUES (
                :tree_path, :survey_id, :name,
                :latitude, :longitude, :elevation, :datum_epsg,
                :east, :north, :utm_epsg,
                :model_east, :model_north, :model_elevation, :profile_offset
            )
            ON CONFLICT(tree_path) DO UPDATE SET
                survey_id       = excluded.survey_id,
                name            = excluded.name,
                latitude        = excluded.latitude,
                longitude       = excluded.longitude,
                elevation       = excluded.elevation,
                datum_epsg      = excluded.datum_epsg,
                east            = excluded.east,
                north           = excluded.north,
                utm_epsg        = excluded.utm_epsg,
                model_east      = excluded.model_east,
                model_north     = excluded.model_north,
                model_elevation = excluded.model_elevation,
                profile_offset  = excluded.profile_offset
            RETURNING id
            """,
            {
                "tree_path": row.tree_path,
                "survey_id": survey_id,
                "name": row.name,
                "latitude": row.latitude,
                "longitude": row.longitude,
                "elevation": row.elevation,
                "datum_epsg": row.datum_epsg,
                "east": row.east,
                "north": row.north,
                "utm_epsg": row.utm_epsg,
                "model_east": row.model_east,
                "model_north": row.model_north,
                "model_elevation": row.model_elevation,
                "profile_offset": row.profile_offset,
            },
        )
        station_id = cur.fetchone()["id"]
        self._conn.commit()
        return station_id

    def replace_station_period_rows(self, row: StationPeriodRow) -> None:
        """
        Delete existing period-index rows for the station and insert a fresh one.

        Parameters
        ----------
        row : StationPeriodRow
            Period coverage summary for the station identified by
            ``row.station_path``.
        """
        station_id = self._station_id(row.station_path)
        if station_id is None:
            raise KeyError(
                f"Station not found in index: {row.station_path!r}. "
                "Call upsert_station() first."
            )
        with self._conn:
            self._conn.execute(
                "DELETE FROM station_period_index WHERE station_id = ?",
                (station_id,),
            )
            self._conn.execute(
                """
                INSERT INTO station_period_index
                    (station_id, period_min, period_max, n_periods)
                VALUES (?, ?, ?, ?)
                """,
                (station_id, row.period_min, row.period_max, row.n_periods),
            )

    def delete_station_by_tree_path(self, tree_path: str) -> bool:
        """
        Remove a station (and its period rows) from the index.

        Parameters
        ----------
        tree_path : str
            Station node path, e.g. ``"surveys/s1/stations/stn1"``.

        Returns
        -------
        bool
            True if a record was deleted, False if tree_path was not found.
        """
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM stations WHERE tree_path = ?", (tree_path,)
            )
        return cur.rowcount > 0

    def refresh_survey_aggregates(self, survey_name: str | None = None) -> None:
        """
        Recompute ``n_stations``, ``lat_min/max``, and ``lon_min/max`` for
        one or all surveys from live station data.

        Parameters
        ----------
        survey_name : str or None
            Survey to refresh.  Pass ``None`` to refresh all surveys.
        """
        where = ""
        params: tuple = ()
        if survey_name is not None:
            survey_id = self._survey_id(survey_name)
            if survey_id is None:
                return
            where = "WHERE s.id = ?"
            params = (survey_id,)

        with self._conn:
            self._conn.execute(
                f"""
                UPDATE surveys
                SET
                    n_stations = agg.cnt,
                    lat_min    = agg.lat_min,
                    lat_max    = agg.lat_max,
                    lon_min    = agg.lon_min,
                    lon_max    = agg.lon_max
                FROM (
                    SELECT
                        s.id          AS survey_id,
                        COUNT(st.id)  AS cnt,
                        MIN(st.latitude)  AS lat_min,
                        MAX(st.latitude)  AS lat_max,
                        MIN(st.longitude) AS lon_min,
                        MAX(st.longitude) AS lon_max
                    FROM surveys s
                    LEFT JOIN stations st ON st.survey_id = s.id
                    {where}
                    GROUP BY s.id
                ) agg
                WHERE surveys.id = agg.survey_id
                """,
                params,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

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
        Return station tree paths matching the supplied filter criteria.

        All filter arguments are optional and combine with AND logic.

        Parameters
        ----------
        survey : str, optional
            Filter to a specific survey name.
        lat_min, lat_max, lon_min, lon_max : float, optional
            Geographic bounding box.
        period_min, period_max : float, optional
            Require a station to have period coverage that *overlaps* the
            supplied period range.

        Returns
        -------
        list[str]
            Matching station tree paths.
        """
        clauses: list[str] = []
        params: list[Any] = []

        if survey is not None:
            clauses.append("sv.name = ?")
            params.append(survey)
        if lat_min is not None:
            clauses.append("st.latitude >= ?")
            params.append(lat_min)
        if lat_max is not None:
            clauses.append("st.latitude <= ?")
            params.append(lat_max)
        if lon_min is not None:
            clauses.append("st.longitude >= ?")
            params.append(lon_min)
        if lon_max is not None:
            clauses.append("st.longitude <= ?")
            params.append(lon_max)

        period_join = ""
        if period_min is not None or period_max is not None:
            period_join = "JOIN station_period_index spi ON spi.station_id = st.id"
            # Overlap condition: station's [period_min, period_max]
            # intersects requested [period_min, period_max].
            if period_min is not None:
                clauses.append("spi.period_max >= ?")
                params.append(period_min)
            if period_max is not None:
                clauses.append("spi.period_min <= ?")
                params.append(period_max)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        sql = f"""
            SELECT st.tree_path
            FROM stations st
            JOIN surveys sv ON sv.id = st.survey_id
            {period_join}
            {where}
            ORDER BY st.tree_path
        """
        rows = self._conn.execute(sql, params).fetchall()
        return [r["tree_path"] for r in rows]

    def all_station_paths(self) -> list[str]:
        """Return all station tree paths in insertion order."""
        rows = self._conn.execute(
            "SELECT tree_path FROM stations ORDER BY id"
        ).fetchall()
        return [r["tree_path"] for r in rows]

    def all_surveys(self) -> list[SurveyRow]:
        """Return all survey records."""
        rows = self._conn.execute("SELECT * FROM surveys ORDER BY name").fetchall()
        return [
            SurveyRow(
                id=r["id"],
                name=r["name"],
                lat_min=r["lat_min"],
                lat_max=r["lat_max"],
                lon_min=r["lon_min"],
                lon_max=r["lon_max"],
                n_stations=r["n_stations"],
            )
            for r in rows
        ]

    def station_record(self, tree_path: str) -> StationRow | None:
        """Return the StationRow for a given tree_path, or None if absent."""
        row = self._conn.execute(
            """
            SELECT st.*, sv.name AS survey_name
            FROM stations st
            JOIN surveys sv ON sv.id = st.survey_id
            WHERE st.tree_path = ?
            """,
            (tree_path,),
        ).fetchone()
        if row is None:
            return None
        return StationRow(
            id=row["id"],
            tree_path=row["tree_path"],
            survey_name=row["survey_name"],
            name=row["name"],
            latitude=row["latitude"],
            longitude=row["longitude"],
            elevation=row["elevation"],
            datum_epsg=row["datum_epsg"],
            east=row["east"],
            north=row["north"],
            utm_epsg=row["utm_epsg"],
            model_east=row["model_east"],
            model_north=row["model_north"],
            model_elevation=row["model_elevation"],
            profile_offset=row["profile_offset"],
        )

    def n_stations(self) -> int:
        """Total number of indexed stations."""
        return self._conn.execute("SELECT COUNT(*) FROM stations").fetchone()[0]

    def n_surveys(self) -> int:
        """Total number of indexed surveys."""
        return self._conn.execute("SELECT COUNT(*) FROM surveys").fetchone()[0]

    # ------------------------------------------------------------------
    # Bulk rebuild
    # ------------------------------------------------------------------

    def rebuild_from_tree(self, tree: Any) -> None:
        """
        Populate the index from an :class:`~mtpy.core.mt_data_tree.MTDataTree`.

        Existing data is replaced atomically.  The tree is scanned for station
        nodes and each dataset's ``.attrs`` dict is used to populate the three
        tables.

        Parameters
        ----------
        tree : MTDataTree
            The data tree to index.
        """
        station_paths = tree._iter_station_paths()
        survey_rows: dict[str, SurveyRow] = {}
        station_data: list[tuple[StationRow, StationPeriodRow | None]] = []

        for path in station_paths:
            station_ds = tree.get_station(path)
            station_row, period_row = self._extract_rows(path, station_ds)
            survey_name = station_row.survey_name
            if survey_name not in survey_rows:
                survey_rows[survey_name] = SurveyRow(name=survey_name)
            station_data.append((station_row, period_row))

        # Atomic replacement using a transaction.
        with self._conn:
            self._conn.execute("DELETE FROM station_period_index")
            self._conn.execute("DELETE FROM stations")
            self._conn.execute("DELETE FROM surveys")

            # Insert surveys and collect their ids.
            survey_ids: dict[str, int] = {}
            for sv_row in survey_rows.values():
                cur = self._conn.execute(
                    """
                    INSERT INTO surveys (name) VALUES (?)
                    RETURNING id
                    """,
                    (sv_row.name,),
                )
                survey_ids[sv_row.name] = cur.fetchone()["id"]

            # Insert stations.
            for station_row, period_row in station_data:
                sv_id = survey_ids[station_row.survey_name]
                cur = self._conn.execute(
                    """
                    INSERT INTO stations (
                        tree_path, survey_id, name,
                        latitude, longitude, elevation, datum_epsg,
                        east, north, utm_epsg,
                        model_east, model_north, model_elevation, profile_offset
                    ) VALUES (
                        :tree_path, :survey_id, :name,
                        :latitude, :longitude, :elevation, :datum_epsg,
                        :east, :north, :utm_epsg,
                        :model_east, :model_north, :model_elevation, :profile_offset
                    )
                    RETURNING id
                    """,
                    {
                        "tree_path": station_row.tree_path,
                        "survey_id": sv_id,
                        "name": station_row.name,
                        "latitude": station_row.latitude,
                        "longitude": station_row.longitude,
                        "elevation": station_row.elevation,
                        "datum_epsg": station_row.datum_epsg,
                        "east": station_row.east,
                        "north": station_row.north,
                        "utm_epsg": station_row.utm_epsg,
                        "model_east": station_row.model_east,
                        "model_north": station_row.model_north,
                        "model_elevation": station_row.model_elevation,
                        "profile_offset": station_row.profile_offset,
                    },
                )
                st_id = cur.fetchone()["id"]
                if period_row is not None:
                    self._conn.execute(
                        """
                        INSERT INTO station_period_index
                            (station_id, period_min, period_max, n_periods)
                        VALUES (?, ?, ?, ?)
                        """,
                        (
                            st_id,
                            period_row.period_min,
                            period_row.period_max,
                            period_row.n_periods,
                        ),
                    )

        # Recompute aggregates outside the bulk transaction.
        self.refresh_survey_aggregates()

    # ------------------------------------------------------------------
    # Dataset → row helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _crs_to_epsg(crs_value: Any) -> str | None:
        """Best-effort extraction of an EPSG string from a CRS-like value."""
        if crs_value is None:
            return None
        if isinstance(crs_value, str):
            return crs_value if crs_value.strip() else None
        # pyproj CRS object
        try:
            epsg = crs_value.to_epsg()
            return str(epsg) if epsg is not None else None
        except Exception:
            return str(crs_value)

    @classmethod
    def _extract_rows(
        cls, tree_path: str, station_ds: "xr.Dataset"
    ) -> tuple[StationRow, StationPeriodRow | None]:
        """
        Build a :class:`StationRow` and optional :class:`StationPeriodRow`
        from a station dataset and its attrs.
        """
        attrs = station_ds.attrs

        survey_name = attrs.get("survey") or "default"
        station_name = attrs.get("station") or tree_path.rsplit("/", 1)[-1]

        station_row = StationRow(
            tree_path=tree_path,
            survey_name=str(survey_name),
            name=str(station_name),
            latitude=_to_float(attrs.get("latitude")),
            longitude=_to_float(attrs.get("longitude")),
            elevation=_to_float(attrs.get("elevation")),
            datum_epsg=cls._crs_to_epsg(attrs.get("datum_crs")),
            east=_to_float(attrs.get("easting")),
            north=_to_float(attrs.get("northing")),
            utm_epsg=cls._crs_to_epsg(attrs.get("utm_crs")),
            model_east=_to_float(attrs.get("model_east")),
            model_north=_to_float(attrs.get("model_north")),
            model_elevation=_to_float(attrs.get("model_elevation")),
            profile_offset=_to_float(attrs.get("profile_offset")),
        )

        period_row: StationPeriodRow | None = None
        if "period" in station_ds.coords and station_ds.coords["period"].size > 0:
            import numpy as np

            periods = station_ds.coords["period"].values.astype(float)
            finite = periods[np.isfinite(periods)]
            if finite.size > 0:
                period_row = StationPeriodRow(
                    station_path=tree_path,
                    period_min=float(finite.min()),
                    period_max=float(finite.max()),
                    n_periods=int(finite.size),
                )

        return station_row, period_row


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _to_float(value: Any) -> float | None:
    """Convert *value* to float, returning None on failure."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
