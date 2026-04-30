# -*- coding: utf-8 -*-
"""Profile dataframe conversion methods for MTData and MTData."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from typing import Callable

import mt_metadata

from mtpy import MT, MTData
from mtpy.core import MTData


def get_tf_file_list() -> list[str]:
    """Get transfer-function file paths exposed by mt_metadata."""
    return [
        value for key, value in mt_metadata.__dict__.items() if key.startswith("TF")
    ]


def load_mt_objects(n_stations: int) -> list[MT]:
    """Load and read MT objects from mt_metadata test files."""
    mt_objects: list[MT] = []
    for fn in sorted(get_tf_file_list())[:n_stations]:
        mt_obj = MT(fn)
        mt_obj.read()
        mt_objects.append(mt_obj)
    return mt_objects


def run_profile(
    label: str,
    func: Callable[[], object],
    repeat: int = 1,
    stats_filter: str = "mtpy/core",
) -> None:
    """Run cProfile and wall-time measurement for a callable."""
    total_wall = 0.0
    profiler = cProfile.Profile()

    for _ in range(repeat):
        start = time.perf_counter()
        profiler.enable()
        func()
        profiler.disable()
        total_wall += time.perf_counter() - start

    avg_wall = total_wall / repeat
    print(f"\n=== {label} ===")
    print(f"avg_wall_s={avg_wall:.6f} over repeat={repeat}")

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs().sort_stats("cumtime").print_stats(30)
    stats.print_stats(stats_filter, 30)
    print(stream.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile to_dataframe/from_dataframe in MTData and MTData"
    )
    parser.add_argument("--n-stations", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    print(f"Loading MT objects (n_stations={args.n_stations})...")
    mt_objects = load_mt_objects(args.n_stations)
    print(f"Loaded {len(mt_objects)} stations")

    md = MTData(mt_list=mt_objects)
    tree = MTData()
    tree.add_stations(mt_objects)

    base_df = md.to_dataframe()
    print(f"Base dataframe rows={len(base_df)} cols={len(base_df.columns)}")

    run_profile(
        "MTData.to_dataframe",
        lambda: md.to_dataframe(),
        repeat=args.repeat,
    )

    run_profile(
        "MTData.to_dataframe",
        lambda: tree.to_dataframe(),
        repeat=args.repeat,
    )

    run_profile(
        "MTData.from_dataframe",
        lambda: MTData().from_dataframe(base_df),
        repeat=args.repeat,
    )

    run_profile(
        "MTData.from_dataframe",
        lambda: MTData().from_dataframe(base_df),
        repeat=args.repeat,
    )


if __name__ == "__main__":
    main()
