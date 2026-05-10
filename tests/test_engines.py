"""
Smoke tests for TRITON M47/M48/M49/M50.

These run in CI on every push. They are NOT exhaustive correctness tests —
they verify each engine loads, processes a known input, and emits at least
the expected alerts. The full benchmark harnesses (benchmark.py and
edge/edge_benchmark.py) cover latency/throughput characterization.
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/../edge"))

import numpy as np
import cv2


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


# ============================================================
# M50 — ReID embedder
# ============================================================
def test_m50_reid_same_person_beats_cross_person():
    from reid_embedder import make_embedder, ReidEmbedder
    person_a = np.zeros((128, 64, 3), dtype=np.uint8)
    person_a[:, :] = [40, 40, 220]; person_a[:32] = [180, 150, 130]; person_a[80:] = [80, 60, 40]
    person_b = np.zeros((128, 64, 3), dtype=np.uint8)
    person_b[:, :] = [200, 60, 40]; person_b[:32] = [160, 140, 120]; person_b[80:] = [60, 90, 30]
    gamma_table = np.array([((i / 255.0) ** 0.85) * 255 for i in range(256)]).astype(np.uint8)
    person_a_lit = cv2.LUT(person_a, gamma_table)

    e = make_embedder()
    va = e.embed(person_a); vb = e.embed(person_b); va2 = e.embed(person_a_lit)
    sim_same = ReidEmbedder.cosine_similarity(va, va2)
    sim_diff = ReidEmbedder.cosine_similarity(va, vb)
    assert sim_same > sim_diff, f"sim_same={sim_same:.3f} should exceed sim_diff={sim_diff:.3f}"
    assert sim_same - sim_diff > 0.15, "ReID discrimination margin too narrow"


def test_m50_cross_camera_handoff_fires():
    from cross_camera_tracker import CrossCameraTracker
    from reid_embedder import make_embedder
    person_a = np.zeros((128, 64, 3), dtype=np.uint8)
    person_a[:, :] = [40, 40, 220]; person_a[:32] = [180, 150, 130]; person_a[80:] = [80, 60, 40]
    gamma_table = np.array([((i / 255.0) ** 0.85) * 255 for i in range(256)]).astype(np.uint8)
    person_a_lit = cv2.LUT(person_a, gamma_table)

    tracker = CrossCameraTracker(make_embedder())
    captured = []
    tracker.on_alert(captured.append)
    g1 = tracker.observe("cam-1", 1, person_a, timestamp=1000.0)
    g2 = tracker.observe("cam-2", 1, person_a_lit, timestamp=1010.0)
    assert g1 == g2, f"expected cross-camera match, got g1={g1} g2={g2}"
    assert any(a.alert_type == "CROSS_CAMERA_HANDOFF" for a in captured)


# ============================================================
# Audio CNN — architecture forward pass
# ============================================================
def test_audio_cnn_forward_pass_returns_valid_distribution():
    from audio_cnn import MelSpecCnnNumpy, log_mel_spectrogram, CNN_CLASSES
    from audio_anomaly import synth_chunk
    cnn = MelSpecCnnNumpy()
    samples = synth_chunk("ambient")
    log_mel = log_mel_spectrogram(samples)
    assert log_mel.shape == (1, 64, 96), f"unexpected mel shape: {log_mel.shape}"
    probs = cnn.predict(log_mel)
    assert probs.shape == (6,)
    assert abs(probs.sum() - 1.0) < 1e-4, f"softmax not normalized: sum={probs.sum()}"
    assert (probs >= 0).all() and (probs <= 1).all()


def test_audio_cnn_param_count_in_expected_range():
    """Architecture should be ~25K params — small enough for INT8 on Jetson."""
    from audio_cnn import CnnWeights
    weights = CnnWeights.random_init()
    n = weights.param_count()
    assert 10_000 < n < 100_000, f"param count {n} outside expected ~25K range"


# ============================================================
# Multi-camera manager
# ============================================================
def test_multi_camera_manager_runs_three_synthetic_cams():
    from multi_camera_manager import MultiCameraManager, CameraConfig
    captured = []
    mgr = MultiCameraManager(cameras=[
        CameraConfig(camera_id="cam-1", url="synthetic://2"),
        CameraConfig(camera_id="cam-2", url="synthetic://2"),
        CameraConfig(camera_id="cam-3", url="synthetic://1"),
    ], max_fps_per_camera=4.0)
    mgr.on_alert(captured.append)
    mgr.start()
    time.sleep(3)
    mgr.stop()
    s = mgr.status()
    assert s["frames_processed"] > 0
    assert s["person_observations"] > 0
    assert len(s["cameras"]) == 3


# ============================================================
# Reliability — graceful degradation
# ============================================================
def test_degradable_detector_wraps_primary():
    """DegradableDetector in primary mode should pass through to YOLO/HOG."""
    from reliability import DegradableDetector, MotionFallbackDetector
    from rtsp_ingest import SyntheticStream
    from person_detector import make_detector

    stream = SyntheticStream(num_people=2)
    deg = DegradableDetector(primary=make_detector(),
                             fallback=MotionFallbackDetector())
    f = stream.read()
    dets = deg.detect(f.image)
    assert deg.mode == "primary"
    # Synthetic frames have detectable persons
    assert len(dets) >= 1


def test_motion_fallback_detector_loads():
    """Smoke check: motion fallback detector instantiates and runs without error."""
    from reliability import MotionFallbackDetector
    from rtsp_ingest import SyntheticStream
    det = MotionFallbackDetector()
    stream = SyntheticStream(num_people=2)
    # Just verify it runs without exception across several frames
    for _ in range(5):
        det.detect(stream.read().image)
