# -*- coding: utf-8 -*-
"""End-to-end benchmark for interpolation at multiple station counts.

This script compares legacy MTData interpolation against MTData eager and
Dask-backed interpolation using MTData.interpolate_dask.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

import mtpy_data
import numpy as np

from mtpy.core import MTData
from mtpy.core.mt import MT
from mtpy.core.mt_data import MTData


def load_seed_mt_objects(n_seed: int = 8) -> list[MT]:
    """Load a small set of real MT objects to clone for larger benchmarks."""
    files = sorted(mtpy_data.PROFILE_LIST)[:n_seed]
    seeds: list[MT] = []
    for fn in files:
        mt_obj = MT(fn)
        mt_obj.read()
        seeds.append(mt_obj)
    return seeds


def build_mt_objects(n_stations: int, seeds: list[MT]) -> list[MT]:
    """Build a list of MT objects by copying seed objects and assigning unique ids."""
    mt_objects: list[MT] = []
    for idx in range(n_stations):
        seed = seeds[idx % len(seeds)]
        mt_obj = seed.copy()
        mt_obj.survey = "benchmark"
        mt_obj.station = f"st_{idx:04d}"
        mt_obj.survey_metadata.id = "benchmark"
        mt_obj.station_metadata.id = mt_obj.station
        mt_objects.append(mt_obj)
    return mt_objects


def time_eager(
    tree: MTData,
    target_periods: np.ndarray,
    method: str,
) -> float:
    """Time eager interpolation."""
    t0 = time.perf_counter()
    _ = tree.interpolate(
        target_periods,
        inplace=False,
        bounds_error=True,
        method=method,
    )
    return time.perf_counter() - t0


def time_mtdata(
    mt_data: MTData,
    target_periods: np.ndarray,
    method: str,
) -> float:
    """Time legacy MTData interpolation."""
    t0 = time.perf_counter()
    _ = mt_data.interpolate(
        target_periods,
        inplace=False,
        bounds_error=True,
        method=method,
    )
    return time.perf_counter() - t0


def time_dask(
    tree: MTData,
    target_periods: np.ndarray,
    method: str,
    chunks: dict[str, int],
    scheduler: str,
) -> float:
    """Time Dask interpolation including delayed graph execution."""
    t0 = time.perf_counter()
    _ = tree.interpolate_dask(
        target_periods,
        bounds_error=True,
        method=method,
        chunks=chunks,
        scheduler=scheduler,
        compute=True,
        inplace=False,
    )
    return time.perf_counter() - t0


def summarize(values: list[float]) -> dict[str, float]:
    """Return simple summary stats for a list of timings."""
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def benchmark_case(
    n_stations: int,
    repeats: int,
    n_periods: int,
    method: str,
    chunks: dict[str, int],
    scheduler: str,
    seeds: list[MT],
) -> dict[str, Any]:
    """Run one benchmark case for a station count and return timing metrics."""
    mt_objects = build_mt_objects(n_stations, seeds)

    build_times: list[float] = []
    eager_times: list[float] = []
    legacy_times: list[float] = []
    dask_times: list[float] = []

    for _ in range(repeats):
        t0 = time.perf_counter()
        tree = MTData()
        tree.add_stations(mt_objects)
        build_elapsed = time.perf_counter() - t0
        build_times.append(build_elapsed)

        periods = np.asarray(mt_objects[0].period, dtype=float)
        target_periods = np.geomspace(periods.min(), periods.max(), n_periods)

        mt_data = MTData()
        mt_data.add_station(mt_objects, compute_relative_location=False)

        legacy_times.append(time_mtdata(mt_data, target_periods, method=method))
        eager_times.append(time_eager(tree, target_periods, method=method))
        dask_times.append(
            time_dask(
                tree,
                target_periods,
                method=method,
                chunks=chunks,
                scheduler=scheduler,
            )
        )

    return {
        "stations": n_stations,
        "build": summarize(build_times),
        "legacy": summarize(legacy_times),
        "eager": summarize(eager_times),
        "dask": summarize(dask_times),
    }


def print_report(results: list[dict[str, Any]], repeats: int, scheduler: str) -> None:
    """Print benchmark results in a compact table."""
    print("\nMTData vs MTData interpolate benchmark")
    print(f"repeats={repeats} scheduler={scheduler}")
    print(
        "stations | build_med_s | mtdata_med_s | tree_med_s | dask_med_s | tree_vs_mtdata"
    )
    print("-" * 95)
    for row in results:
        legacy_med = row["legacy"]["median"]
        eager_med = row["eager"]["median"]
        dask_med = row["dask"]["median"]
        speedup = legacy_med / eager_med if eager_med > 0 else float("inf")
        print(
            f"{row['stations']:8d} | "
            f"{row['build']['median']:11.3f} | "
            f"{legacy_med:12.3f} | "
            f"{eager_med:10.3f} | "
            f"{dask_med:10.3f} | "
            f"{speedup:16.2f}x"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[50, 200, 500],
        help="Station counts to benchmark.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Repeats per station count.",
    )
    parser.add_argument(
        "--n-periods",
        type=int,
        default=48,
        help="Number of target periods for interpolation.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="slinear",
        help="Interpolation method passed to interpolate/interpolate_dask.",
    )
    parser.add_argument(
        "--chunk-period",
        type=int,
        default=16,
        help="Dask chunk size along period dimension.",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="threads",
        choices=["threads", "processes", "single-threaded"],
        help="Dask scheduler to use for interpolate_dask compute.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = load_seed_mt_objects()
    chunks = {"period": args.chunk_period}

    results: list[dict[str, Any]] = []
    for n_stations in args.sizes:
        print(f"Running benchmark for {n_stations} stations...")
        row = benchmark_case(
            n_stations=n_stations,
            repeats=args.repeats,
            n_periods=args.n_periods,
            method=args.method,
            chunks=chunks,
            scheduler=args.scheduler,
            seeds=seeds,
        )
        results.append(row)

    print_report(results, repeats=args.repeats, scheduler=args.scheduler)


if __name__ == "__main__":
    main()
