#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone profiling runner for MTData.add_station / add_stations.

Usage
-----
Run from the repository root in the py313 conda environment::

    python tests/perf/run_profile.py                    # default: 300 objects
    python tests/perf/run_profile.py --n 1000           # override count
    python tests/perf/run_profile.py --save prof.prof   # save cProfile to file
    python tests/perf/run_profile.py --top 40           # show 40 cProfile rows

The script always prints a comparison table of all modes/APIs, followed by
a scaling-chunk breakdown and a cProfile hotspot table.
"""

from __future__ import annotations

import argparse
import cProfile
import pstats
import time
from io import StringIO
from typing import Any

import mt_metadata

from mtpy.core.mt import MT
from mtpy.core.mt_data import MTData


# ---------------------------------------------------------------------------
# Object construction helpers
# ---------------------------------------------------------------------------


def _make_basic_mt(i: int) -> MT:
    mt = MT()
    mt.survey = "profile_basic"
    mt.station = f"basic_{i:05d}"
    mt.latitude = 10.0
    mt.longitude = 20.0
    return mt


def _make_real_mt(template: MT, i: int) -> MT:
    mt = template.copy()
    mt.survey = "profile_real"
    mt.station = f"real_{i:05d}"
    return mt


# ---------------------------------------------------------------------------
# Core timing helper
# ---------------------------------------------------------------------------


def _run(
    label: str,
    tree: MTData,
    objs: list[MT],
    *,
    bulk: bool = False,
    precomputed_attrs: list[dict[str, Any] | None] | None = None,
) -> float:
    t0 = time.perf_counter()
    if bulk:
        tree.add_stations(objs, precomputed_attrs=precomputed_attrs)
    else:
        for mt in objs:
            tree.add_station(mt)
    dt = time.perf_counter() - t0
    n = len(objs)
    per_ms = dt / n * 1_000
    print(f"  {label:<50}  {dt:7.4f}s   {per_ms:8.3f} ms/call")
    return dt


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile MTData.add_station")
    parser.add_argument("--n", type=int, default=300, help="Number of stations")
    parser.add_argument("--save", type=str, default=None, help="Save .prof to file")
    parser.add_argument("--top", type=int, default=30, help="cProfile rows to show")
    args = parser.parse_args()

    n = args.n

    print(f"\n{'='*72}")
    print(f"MTData.add_station profiler   n={n}")
    print(f"{'='*72}")

    # ------------------------------------------------------------------
    # Build objects (not counted in timings below)
    # ------------------------------------------------------------------
    print("\nBuilding test objects…")
    basic_objs = [_make_basic_mt(i) for i in range(n)]

    mt_template = MT(mt_metadata.TF_EDI_CGG)
    mt_template.read()
    real_objs = [_make_real_mt(mt_template, i) for i in range(n)]

    pre = [
        {"latitude": mt.latitude, "longitude": mt.longitude, "elevation": mt.elevation}
        for mt in real_objs
    ]

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    print(f"\n{'─'*72}")
    print(f"  {'Scenario':<50}  {'Total':>8}   {'ms/call':>9}")
    print(f"{'─'*72}")

    _run(
        "basic / loop / cache / shallow",
        MTData(metadata_storage="cache", dataset_copy_mode="shallow"),
        basic_objs,
    )
    _run(
        "basic / loop / cache / none",
        MTData(metadata_storage="cache", dataset_copy_mode="none"),
        basic_objs,
    )

    print()
    _run(
        "realistic / loop / dict / deep  [legacy]",
        MTData(metadata_storage="dict", dataset_copy_mode="deep"),
        real_objs,
    )
    _run(
        "realistic / loop / dict / shallow",
        MTData(metadata_storage="dict", dataset_copy_mode="shallow"),
        real_objs,
    )
    _run(
        "realistic / loop / summary / shallow",
        MTData(metadata_storage="summary", dataset_copy_mode="shallow"),
        real_objs,
    )
    _run(
        "realistic / loop / cache / shallow",
        MTData(metadata_storage="cache", dataset_copy_mode="shallow"),
        real_objs,
    )
    _run(
        "realistic / loop / cache / none",
        MTData(metadata_storage="cache", dataset_copy_mode="none"),
        real_objs,
    )

    print()
    _run(
        "realistic / BULK / cache / shallow",
        MTData(metadata_storage="cache", dataset_copy_mode="shallow"),
        real_objs,
        bulk=True,
    )
    _run(
        "realistic / BULK / cache / none",
        MTData(metadata_storage="cache", dataset_copy_mode="none"),
        real_objs,
        bulk=True,
    )
    _run(
        "realistic / BULK+precomputed / cache / none",
        MTData(metadata_storage="cache", dataset_copy_mode="none"),
        real_objs,
        bulk=True,
        precomputed_attrs=pre,
    )

    print(f"{'─'*72}")

    # ------------------------------------------------------------------
    # Scaling breakdown
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print("Scaling breakdown: per-call ms by insertion chunk")
    print(f"{'─'*72}")
    chunk_size = max(1, n // 8)
    scale_objs = real_objs[:n]
    scale_tree = MTData(metadata_storage="cache", dataset_copy_mode="shallow")

    for start in range(0, len(scale_objs), chunk_size):
        chunk = scale_objs[start : start + chunk_size]
        t0 = time.perf_counter()
        for mt in chunk:
            scale_tree.add_station(mt)
        dt = time.perf_counter() - t0
        end = min(start + chunk_size - 1, n - 1)
        print(f"  stations {start:4d}–{end:4d}: {dt/len(chunk)*1_000:.3f} ms/call")

    # ------------------------------------------------------------------
    # cProfile hotspot table
    # ------------------------------------------------------------------
    print(f"\n{'='*72}")
    print(
        "cProfile (top {top} by cumtime) — realistic / BULK / cache / none".format(
            top=args.top
        )
    )
    print(f"{'─'*72}")

    prof_objs = [_make_real_mt(mt_template, i) for i in range(n)]
    prof_tree = MTData(metadata_storage="cache", dataset_copy_mode="none")

    pr = cProfile.Profile()
    pr.enable()
    prof_tree.add_stations(prof_objs)
    pr.disable()

    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
    ps.print_stats(args.top)
    print(s.getvalue())

    if args.save:
        pr.dump_stats(args.save)
        print(f"cProfile data saved to: {args.save}")


if __name__ == "__main__":
    main()
