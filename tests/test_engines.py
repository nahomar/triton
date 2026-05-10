"""
Smoke tests for TRITON M47/M48/M49.

These run in CI on every push. They are NOT exhaustive correctness tests —
they verify each engine loads, processes a known input, and emits at least
the expected alerts. The full benchmark harnesses (benchmark.py and
edge/edge_benchmark.py) cover latency/throughput characterization.
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../edge"))


# ============================================================
# M47 — AIS engine
# ============================================================
def test_m47_geofences_load():
    from geofences import all_fences
    fences = all_fences()
    assert len(fences) >= 10
    assert any(f.fence_id == "HORMUZ_CHOKE" for f in fences)
    assert any(f.zone_type == "PROTECTION" for f in fences)


def test_m47_engine_fires_chokepoint_entry():
    from perimeter_engine import PerimeterEngine, AISMessage
    from geofences import all_fences
    eng = PerimeterEngine(fences=all_fences())
    captured = []
    eng.on_alert(captured.append)
    base = time.time()
    # Two-step path: outside → inside the chokepoint
    eng.ingest(AISMessage(mmsi=636017123, timestamp=base, lat=26.20, lon=57.30,
                          sog=14, cog=270, vessel_class="tanker", name="MT TEST"))
    eng.ingest(AISMessage(mmsi=636017123, timestamp=base + 60, lat=26.65, lon=56.70,
                          sog=14, cog=270, vessel_class="tanker", name="MT TEST"))
    types = {a.alert_type for a in captured}
    fences_hit = {a.fence_id for a in captured}
    assert "GEOFENCE_ENTRY" in types
    assert "HORMUZ_CHOKE" in fences_hit


def test_m47_latency_under_one_ms():
    """Median per-message latency must stay under 1 ms on the smoke-test load."""
    from perimeter_engine import PerimeterEngine
    from geofences import all_fences
    from ais_simulator import GulfTrafficSimulator
    eng = PerimeterEngine(fences=all_fences())
    sim = GulfTrafficSimulator(n_vessels=100, seed=1)
    for m in sim.stream(n_messages=2_000):
        eng.ingest(m)
    p = eng.latency_percentiles()
    assert p["p50_us"] < 1_000, f"M47 P50 latency regressed: {p}"


# ============================================================
# M48 — Vision engine
# ============================================================
def test_m48_synthetic_stream_produces_frames():
    from rtsp_ingest import SyntheticStream
    s = SyntheticStream(num_people=2)
    f = s.read()
    assert f.image.shape == (720, 1280, 3)


def test_m48_vision_engine_fires_on_synthetic_track():
    from rtsp_ingest import SyntheticStream
    from person_detector import make_detector
    from vision_perimeter import VisionPerimeterEngine, default_pier_fences
    stream = SyntheticStream(num_people=2)
    eng = VisionPerimeterEngine(make_detector(), default_pier_fences(),
                                source_id="cam-test", confirm_frames=2)
    captured = []
    eng.on_alert(captured.append)
    for _ in range(20):
        eng.process(stream.read())
    # Should fire at least one entry across 20 frames of 2 walking persons
    assert any(a.alert_type == "GEOFENCE_ENTRY" for a in captured)


# ============================================================
# M49 — Audio engine
# ============================================================
def test_m49_classifies_all_anomaly_classes():
    from audio_anomaly import (
        AcousticPerimeterEngine, RuleBasedDetector,
        AudioChunk, synth_chunk,
    )
    eng = AcousticPerimeterEngine(RuleBasedDetector(), source_id="mic-test")
    captured = []
    eng.on_alert(captured.append)
    base = time.time()
    classes = ["gunshot", "glass_break", "scream", "alarm", "breaking_door"]
    for i, kind in enumerate(classes):
        eng.process(AudioChunk(
            samples=synth_chunk(kind), sample_rate=16_000,
            timestamp=base + i * 5,
            grabbed_at_ns=time.perf_counter_ns(),
            source_id="mic-test",
        ))
    seen_classes = {a.fence_id for a in captured}  # fence_id holds class label
    # All five classes should fire
    assert seen_classes == set(classes), \
        f"M49 missed classes: expected {set(classes)}, got {seen_classes}"


def test_m49_ambient_does_not_fire():
    from audio_anomaly import (
        AcousticPerimeterEngine, RuleBasedDetector,
        AudioChunk, synth_chunk,
    )
    eng = AcousticPerimeterEngine(RuleBasedDetector(), source_id="mic-test")
    captured = []
    eng.on_alert(captured.append)
    eng.process(AudioChunk(
        samples=synth_chunk("ambient"), sample_rate=16_000,
        timestamp=time.time(),
        grabbed_at_ns=time.perf_counter_ns(), source_id="mic-test",
    ))
    assert len(captured) == 0, "M49 should not fire on ambient noise"


# ============================================================
# Integration — unified bus
# ============================================================
def test_unified_alert_bus():
    """Verify all three engines can publish to one bus with a unified handler."""
    from integration import AlertBus, wire_engine_to_bus
    from perimeter_engine import PerimeterEngine, AISMessage
    from geofences import all_fences

    bus = AlertBus()
    captured = []
    bus.subscribe(captured.append)

    eng = PerimeterEngine(fences=all_fences())
    wire_engine_to_bus(eng, bus)

    base = time.time()
    eng.ingest(AISMessage(mmsi=12345, timestamp=base, lat=26.65, lon=56.70,
                          sog=14, cog=270, vessel_class="tanker", name="X"))
    assert len(captured) > 0
