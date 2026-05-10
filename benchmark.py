"""
TRITON M47 — End-to-End Benchmark

Pushes a substantial AIS stream through the live engine, measures per-message
detection latency with nanosecond resolution, and reports the percentile
distribution. This is the single source of truth for any latency claim
made about M47.

Run:  python3 benchmark.py [n_messages] [n_vessels]
"""
from __future__ import annotations

import sys
import time
from collections import Counter

from perimeter_engine import PerimeterEngine, Alert
from geofences import all_fences
from ais_simulator import GulfTrafficSimulator


def run(n_messages: int = 100_000, n_vessels: int = 400) -> None:
    print("=" * 72)
    print(f"TRITON M47 — Streaming Perimeter Engine Benchmark")
    print("=" * 72)
    print(f"  messages:   {n_messages:,}")
    print(f"  vessels:    {n_vessels}")
    print(f"  fences:     {len(all_fences())}")
    print()

    engine = PerimeterEngine(fences=all_fences())
    sim = GulfTrafficSimulator(n_vessels=n_vessels, seed=42)

    alert_types = Counter()
    severities = Counter()
    fence_hits = Counter()

    def collect(a: Alert) -> None:
        alert_types[a.alert_type] += 1
        severities[a.severity] += 1
        if a.fence_id:
            fence_hits[a.fence_id] += 1

    engine.on_alert(collect)

    # Warm-up — JIT spatial index branches, fill caches
    print("  warming up (5,000 messages, not measured)...")
    for msg in sim.stream(n_messages=5_000):
        engine.ingest(msg)
    engine.reset_latency_telemetry()
    engine.alerts_emitted = 0
    alert_types.clear()
    severities.clear()
    fence_hits.clear()

    # Measured run
    print(f"  measuring {n_messages:,} messages...")
    sim2 = GulfTrafficSimulator(n_vessels=n_vessels, seed=42)
    # Skip past the warm-up window in the simulator
    for _ in sim2.stream(n_messages=5_000):
        pass

    wall_t0 = time.perf_counter()
    msg_count = 0
    for msg in sim2.stream(n_messages=n_messages):
        engine.ingest(msg)
        msg_count += 1
    wall_elapsed = time.perf_counter() - wall_t0

    # ============================================================
    print()
    print("─" * 72)
    print("RESULTS")
    print("─" * 72)
    print(f"  Wall time:           {wall_elapsed:.3f} s")
    print(f"  Throughput:          {msg_count / wall_elapsed:,.0f} msg/s")
    print(f"  Tracked vessels:     {len(engine.vessels)}")
    print(f"  Alerts emitted:      {engine.alerts_emitted:,}")
    print()

    lat = engine.latency_percentiles()
    print("  Per-message detection latency (end-to-end, single core):")
    print(f"    samples           {lat['samples']:>12,}")
    print(f"    mean              {lat['mean_us']:>10.2f} µs")
    print(f"    P50  (median)     {lat['p50_us']:>10.2f} µs")
    print(f"    P95               {lat['p95_us']:>10.2f} µs")
    print(f"    P99               {lat['p99_us']:>10.2f} µs")
    print(f"    P99.9             {lat['p999_us']:>10.2f} µs")
    print(f"    max               {lat['max_us']:>10.2f} µs")
    print()

    sub_ms_count = sum(1 for ns in engine._latencies_ns if ns < 1_000_000)
    sub_ms_pct = sub_ms_count / len(engine._latencies_ns) * 100
    print(f"  Messages detected in <1 ms:  {sub_ms_count:,} / {len(engine._latencies_ns):,} "
          f"({sub_ms_pct:.2f}%)")

    print()
    print("  Alerts by type:")
    for at, c in alert_types.most_common():
        print(f"    {at:24s} {c:>8,}")

    print()
    print("  Alerts by severity:")
    for sev in sorted(severities.keys(), reverse=True):
        bar = "█" * min(40, severities[sev] // max(1, max(severities.values()) // 40))
        print(f"    sev {sev}  {severities[sev]:>8,}  {bar}")

    print()
    print("  Top fences hit:")
    for fid, c in fence_hits.most_common(8):
        print(f"    {fid:24s} {c:>8,}")

    print()
    print("─" * 72)
    if lat['p50_us'] < 1000 and lat['p95_us'] < 1000:
        print(f"  ✅  Sub-millisecond detection at P50 AND P95.")
    elif lat['p50_us'] < 1000:
        print(f"  ⚠️   Sub-ms at P50 only. P95 = {lat['p95_us']:.1f}µs.")
    else:
        print(f"  ❌  P50 = {lat['p50_us']:.1f}µs — sub-ms claim NOT supported.")
    print("─" * 72)


if __name__ == "__main__":
    n_msgs = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    n_vessels = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    run(n_messages=n_msgs, n_vessels=n_vessels)
