# -*- coding: utf-8 -*-
"""
Performance tests for MTData.add_station and MTData.add_stations.

These tests are marked ``slow`` and are excluded from the default test run::

    pytest -m "not slow"    # normal CI pass
    pytest tests/perf/      # run only performance tests
    pytest tests/perf/ -v --tb=short    # verbose pass/fail with timing

Each test records wall-time and asserts against a loose regression ceiling.
The ceilings are intentionally generous so they do not fail on slower CI
runners while still catching accidental O(n²) regressions.

To capture a cProfile snapshot of any individual scenario pass
``--profile`` on the command line (handled by the custom option below).
"""

from __future__ import annotations

import cProfile
import pstats
import time
from io import StringIO
from typing import Any

import mt_metadata
import pytest

from mtpy.core.mt import MT
from mtpy.core.mt_data import MTData


# ---------------------------------------------------------------------------
# pytest option hook
# ---------------------------------------------------------------------------


def pytest_addoption(parser):
    """Add --profile flag to enable cProfile output for perf tests."""
    try:
        parser.addoption(
            "--profile",
            action="store_true",
            default=False,
            help="Print cProfile hotspot table for each performance test.",
        )
    except ValueError:
        # Option already registered (e.g., during collection in larger run)
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_mt_template():
    """Load one real MT object from mt_metadata's bundled EDI file."""
    tf_edi = mt_metadata.TF_EDI_CGG
    mt_obj = MT(tf_edi)
    mt_obj.read()
    return mt_obj


def _make_basic_mt(i: int) -> MT:
    mt = MT()
    mt.survey = "perf_basic"
    mt.station = f"basic_{i:05d}"
    mt.latitude = 10.0
    mt.longitude = 20.0
    return mt


def _make_realistic_mt(template: MT, survey: str, i: int) -> MT:
    mt = template.copy()
    mt.survey = survey
    mt.station = f"real_{i:05d}"
    return mt


def _timed_insert(
    mt_objects: list[MT],
    *,
    metadata_storage: str = "cache",
    dataset_copy_mode: str = "shallow",
    bulk: bool = False,
    precomputed_attrs: list[dict[str, Any] | None] | None = None,
    request: pytest.FixtureRequest | None = None,
) -> float:
    """Insert stations and return total wall-time in seconds."""
    tree = MTData(
        metadata_storage=metadata_storage,
        dataset_copy_mode=dataset_copy_mode,
    )

    profile_requested = (
        request is not None
        and hasattr(request.config, "getoption")
        and request.config.getoption("--profile", default=False)
    )

    pr = cProfile.Profile() if profile_requested else None

    if pr:
        pr.enable()

    t0 = time.perf_counter()
    if bulk:
        tree.add_stations(mt_objects, precomputed_attrs=precomputed_attrs)
    else:
        for mt in mt_objects:
            tree.add_station(mt)
    elapsed = time.perf_counter() - t0

    if pr:
        pr.disable()
        s = StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats("cumtime")
        ps.print_stats(30)
        print(f"\n{s.getvalue()}")

    return elapsed


# ---------------------------------------------------------------------------
# Helper: report per-call timing
# ---------------------------------------------------------------------------


def _report(label: str, elapsed: float, n: int) -> None:
    per_call_ms = elapsed / n * 1_000
    print(f"\n  {label}: {elapsed:.4f}s total, {per_call_ms:.3f} ms/call (n={n})")


# ---------------------------------------------------------------------------
# Regression ceilings (ms per call) – generous for slow CI runners
# ---------------------------------------------------------------------------

# Basic (no transfer function data) – only path/Dataset allocation overhead
CEILING_BASIC_MS = 20.0

# Realistic (with real transfer function data) per-call limit
CEILING_REALISTIC_LOOP_MS = 50.0
CEILING_REALISTIC_BULK_MS = 50.0
CEILING_REALISTIC_BULK_PRECOMPUTED_MS = 50.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestAddStationPerf:
    """Wall-time regression tests for MTData.add_station."""

    N_BASIC = 500
    N_REALISTIC = 200

    def test_basic_loop_cache_shallow(self, request):
        """Baseline: minimal MT objects, cache mode, shallow copy."""
        objs = [_make_basic_mt(i) for i in range(self.N_BASIC)]

        elapsed = _timed_insert(
            objs,
            metadata_storage="cache",
            dataset_copy_mode="shallow",
            request=request,
        )
        _report("basic_loop cache/shallow", elapsed, self.N_BASIC)

        per_call_ms = elapsed / self.N_BASIC * 1_000
        assert per_call_ms < CEILING_BASIC_MS, (
            f"add_station (basic) too slow: {per_call_ms:.2f} ms/call "
            f"(ceiling {CEILING_BASIC_MS} ms)"
        )

    def test_basic_loop_cache_none(self, request):
        """Minimal MT objects, cache mode, no-copy – fastest safe path."""
        objs = [_make_basic_mt(i) for i in range(self.N_BASIC)]

        elapsed = _timed_insert(
            objs,
            metadata_storage="cache",
            dataset_copy_mode="none",
            request=request,
        )
        _report("basic_loop cache/none", elapsed, self.N_BASIC)

        per_call_ms = elapsed / self.N_BASIC * 1_000
        assert per_call_ms < CEILING_BASIC_MS

    def test_realistic_loop_cache_shallow(self, real_mt_template, request):
        """Real MT objects, cache mode, shallow copy."""
        objs = [
            _make_realistic_mt(real_mt_template, "perf_real", i)
            for i in range(self.N_REALISTIC)
        ]

        elapsed = _timed_insert(
            objs,
            metadata_storage="cache",
            dataset_copy_mode="shallow",
            request=request,
        )
        _report("realistic_loop cache/shallow", elapsed, self.N_REALISTIC)

        per_call_ms = elapsed / self.N_REALISTIC * 1_000
        assert per_call_ms < CEILING_REALISTIC_LOOP_MS, (
            f"add_station (realistic) too slow: {per_call_ms:.2f} ms/call "
            f"(ceiling {CEILING_REALISTIC_LOOP_MS} ms)"
        )

    def test_realistic_loop_dict_shallow(self, real_mt_template, request):
        """Legacy dict mode baseline – expected to be slowest."""
        objs = [
            _make_realistic_mt(real_mt_template, "perf_dict", i)
            for i in range(self.N_REALISTIC)
        ]

        elapsed = _timed_insert(
            objs,
            metadata_storage="dict",
            dataset_copy_mode="shallow",
            request=request,
        )
        _report("realistic_loop dict/shallow", elapsed, self.N_REALISTIC)
        # No hard ceiling – this is a reference measurement only

    def test_realistic_loop_summary_shallow(self, real_mt_template, request):
        """Summary metadata mode."""
        objs = [
            _make_realistic_mt(real_mt_template, "perf_summary", i)
            for i in range(self.N_REALISTIC)
        ]

        elapsed = _timed_insert(
            objs,
            metadata_storage="summary",
            dataset_copy_mode="shallow",
            request=request,
        )
        _report("realistic_loop summary/shallow", elapsed, self.N_REALISTIC)

        per_call_ms = elapsed / self.N_REALISTIC * 1_000
        assert per_call_ms < CEILING_REALISTIC_LOOP_MS


@pytest.mark.slow
class TestAddStationsBulkPerf:
    """Wall-time regression tests for MTData.add_stations (bulk API)."""

    N = 300

    def _objs(self, template: MT, survey: str) -> list[MT]:
        return [_make_realistic_mt(template, survey, i) for i in range(self.N)]

    def test_bulk_cache_shallow(self, real_mt_template, request):
        """Bulk insert, cache mode, shallow copy."""
        objs = self._objs(real_mt_template, "perf_bulk_shallow")

        elapsed = _timed_insert(
            objs,
            metadata_storage="cache",
            dataset_copy_mode="shallow",
            bulk=True,
            request=request,
        )
        _report("bulk cache/shallow", elapsed, self.N)

        per_call_ms = elapsed / self.N * 1_000
        assert per_call_ms < CEILING_REALISTIC_BULK_MS

    def test_bulk_cache_none(self, real_mt_template, request):
        """Bulk insert, cache mode, no-copy – fastest safe path."""
        objs = self._objs(real_mt_template, "perf_bulk_none")

        elapsed = _timed_insert(
            objs,
            metadata_storage="cache",
            dataset_copy_mode="none",
            bulk=True,
            request=request,
        )
        _report("bulk cache/none", elapsed, self.N)

        per_call_ms = elapsed / self.N * 1_000
        assert per_call_ms < CEILING_REALISTIC_BULK_MS

    def test_bulk_cache_none_precomputed(self, real_mt_template, request):
        """Bulk insert with precomputed attrs, cache mode, no-copy."""
        objs = self._objs(real_mt_template, "perf_bulk_precomp")
        pre = [
            {
                "latitude": mt.latitude,
                "longitude": mt.longitude,
                "elevation": mt.elevation,
            }
            for mt in objs
        ]

        elapsed = _timed_insert(
            objs,
            metadata_storage="cache",
            dataset_copy_mode="none",
            bulk=True,
            precomputed_attrs=pre,
            request=request,
        )
        _report("bulk cache/none + precomputed", elapsed, self.N)

        per_call_ms = elapsed / self.N * 1_000
        assert per_call_ms < CEILING_REALISTIC_BULK_PRECOMPUTED_MS


@pytest.mark.slow
class TestAddStationScaling:
    """
    Confirm per-call cost stays flat as tree grows.

    Inserts in chunks and asserts no chunk is more than 2× slower
    than the first chunk (i.e. no super-linear growth up to N_TOTAL).
    """

    N_TOTAL = 1_000
    CHUNK = 200

    def test_per_call_does_not_degrade(self, real_mt_template, request):
        """Per-call time must not grow super-linearly with tree size."""
        objs = [
            _make_realistic_mt(real_mt_template, "perf_scale", i)
            for i in range(self.N_TOTAL)
        ]

        tree = MTData(metadata_storage="cache", dataset_copy_mode="shallow")
        chunk_ms: list[float] = []

        for start in range(0, self.N_TOTAL, self.CHUNK):
            chunk = objs[start : start + self.CHUNK]
            t0 = time.perf_counter()
            for mt in chunk:
                tree.add_station(mt)
            elapsed = time.perf_counter() - t0
            chunk_ms.append(elapsed / len(chunk) * 1_000)

        print("\n  scaling chunks (ms/call):", [f"{v:.2f}" for v in chunk_ms])

        # Allow up to 2× the first chunk's per-call cost across all chunks
        ceiling = chunk_ms[0] * 2
        worst = max(chunk_ms)
        assert worst < ceiling, (
            f"add_station scaling regression: worst chunk {worst:.2f} ms/call "
            f"exceeded 2x first-chunk ceiling {ceiling:.2f} ms/call. "
            f"Full profile: {chunk_ms}"
        )
