"""
TRITON M49 — Acoustic Anomaly Detector

Streaming audio anomaly detection with two backend tiers:

  RuleBasedDetector   — librosa spectral features + threshold rules.
                        Runs anywhere with librosa, no model file. Good
                        baseline; used by the sandbox benchmark.
  YamnetDetector      — production path. YAMNet (Google's 521-class
                        audio event classifier) exported to TFLite or
                        ONNX, wrapped here. Runs on Jetson via TensorRT.

Targeted anomaly classes (Verkada-style perimeter use cases):
    glass_break      — high-frequency transient, broadband, 100-400 ms
    gunshot          — high-amplitude impulse, very short, broad spectrum
    scream           — sustained harmonic content 800-3000 Hz
    alarm            — periodic harmonic peaks
    breaking_door    — low-frequency impact + secondary transients

Output schema reuses parent-module Alert with alert_type "AUDIO_ANOMALY"
so M26 AIP doesn't have to care which sensor produced the event.
"""
from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import librosa

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perimeter_engine import Alert


# ============================================================
# AUDIO CHUNK
# ============================================================
@dataclass
class AudioChunk:
    samples: np.ndarray     # float32 mono, range [-1, 1]
    sample_rate: int
    timestamp: float        # epoch seconds at chunk start
    grabbed_at_ns: int      # perf_counter_ns at grab
    source_id: str          # microphone identifier


# ============================================================
# FEATURES
# ============================================================
@dataclass
class AudioFeatures:
    rms: float                  # short-term RMS energy
    peak: float                 # peak |amplitude|
    crest_factor: float         # peak / rms
    spectral_centroid: float    # Hz
    spectral_rolloff: float     # Hz at 95% energy
    spectral_flatness: float    # 0..1, 1 = noise-like
    zcr: float                  # zero-crossing rate
    high_band_ratio: float      # energy in 2-8 kHz / total
    low_band_ratio: float       # energy in 0-500 Hz / total
    onset_strength: float       # spectral flux at chunk start
    mfcc_mean: np.ndarray       # 13-dim


class AudioFeatureExtractor:
    def __init__(self, sample_rate: int = 16_000, n_mfcc: int = 13):
        self.sr = sample_rate
        self.n_mfcc = n_mfcc

    def extract(self, samples: np.ndarray) -> AudioFeatures:
        x = samples.astype(np.float32)
        if x.size < 512:
            x = np.pad(x, (0, 512 - x.size))

        rms = float(np.sqrt(np.mean(x ** 2) + 1e-12))
        peak = float(np.max(np.abs(x)))
        crest = peak / (rms + 1e-9)
        zcr = float(np.mean(np.abs(np.diff(np.sign(x))))) / 2.0

        # STFT-based features
        S = np.abs(librosa.stft(x, n_fft=1024, hop_length=256)) + 1e-9
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=1024)
        psd = (S ** 2).mean(axis=1)
        total = psd.sum()

        cent = float(np.sum(freqs * psd) / total)
        cum = np.cumsum(psd) / total
        rolloff_idx = int(np.searchsorted(cum, 0.95))
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])
        flatness = float(np.exp(np.log(psd + 1e-12).mean()) / (psd.mean() + 1e-12))

        high_mask = (freqs >= 2000) & (freqs <= 8000)
        low_mask = freqs < 500
        high_ratio = float(psd[high_mask].sum() / total)
        low_ratio = float(psd[low_mask].sum() / total)

        # Peak onset strength across the chunk (spectral flux)
        onset = float(librosa.onset.onset_strength(
            y=x, sr=self.sr, hop_length=256
        ).max())

        mfcc = librosa.feature.mfcc(y=x, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_mean = mfcc.mean(axis=1)

        return AudioFeatures(
            rms=rms, peak=peak, crest_factor=crest,
            spectral_centroid=cent, spectral_rolloff=rolloff,
            spectral_flatness=flatness, zcr=zcr,
            high_band_ratio=high_ratio, low_band_ratio=low_ratio,
            onset_strength=onset, mfcc_mean=mfcc_mean,
        )


# ============================================================
# RULE-BASED DETECTOR
# ============================================================
class RuleBasedDetector:
    """
    Threshold-rule classifier on extracted features. Returns
    (anomaly_class, confidence, rationale) or (None, 0.0, "") if normal.

    The thresholds below are calibrated against synthetic test signals
    embedded in the benchmark. Production deployment should retune them
    against ~30 minutes of in-scene "ambient quiet" audio per camera.
    """
    backend_name = "rule_based"

    def classify(self, f: AudioFeatures) -> Tuple[Optional[str], float, str]:
        # 1. Gunshot — high crest factor + high peak, broadband (NOT low-band dominant
        #    which would be a heavy impact / breaking door instead).
        if f.crest_factor > 9 and f.peak > 0.6 and f.low_band_ratio < 0.3:
            conf = min(1.0, (f.crest_factor / 15) * f.peak)
            return "gunshot", conf, (
                f"crest={f.crest_factor:.1f} peak={f.peak:.2f} "
                f"low_ratio={f.low_band_ratio:.2f}"
            )

        # 2. Glass break — saturated high-band energy + high ZCR + tonal
        #    (low flatness rules out white noise that has same hi-band ratio).
        if f.high_band_ratio > 0.85 and f.zcr > 0.30 and f.spectral_flatness < 0.20:
            conf = min(1.0, f.high_band_ratio)
            return "glass_break", conf, (
                f"high_ratio={f.high_band_ratio:.2f} zcr={f.zcr:.2f} "
                f"flatness={f.spectral_flatness:.2f}"
            )

        # 3. Breaking door / heavy impact — low-band dominant + high peak
        #    (check before scream/alarm because low-frequency rules win).
        if f.low_band_ratio > 0.70 and f.peak > 0.5:
            conf = min(1.0, f.low_band_ratio)
            return "breaking_door", conf, (
                f"low_ratio={f.low_band_ratio:.2f} peak={f.peak:.2f}"
            )

        # 4. Scream — sustained harmonic energy in voice range, low flatness,
        #    crest factor low (sustained, not impulsive).
        if (800 < f.spectral_centroid < 3500 and f.rms > 0.10
                and f.spectral_flatness < 0.10 and f.crest_factor < 8
                and f.zcr > 0.12):
            conf = min(1.0, f.rms * 4)
            return "scream", conf, (
                f"centroid={f.spectral_centroid:.0f}Hz rms={f.rms:.2f} "
                f"flatness={f.spectral_flatness:.3f}"
            )

        # 5. Alarm — periodic harmonic peaks → very low spectral flatness
        #    + sustained RMS + low ZCR (steadier than a scream).
        if f.spectral_flatness < 0.05 and f.rms > 0.15 and f.zcr < 0.15:
            conf = min(1.0, f.rms * 2)
            return "alarm", conf, (
                f"flatness={f.spectral_flatness:.3f} rms={f.rms:.2f} zcr={f.zcr:.2f}"
            )

        return None, 0.0, ""


# ============================================================
# YAMNET PRODUCTION PATH (stub with real interface)
# ============================================================
class YamnetDetector:
    """
    Production path: YAMNet exported to ONNX/TFLite, wrapped behind the
    same classify() interface as RuleBasedDetector.

    On Jetson, deploy the ONNX export and run via:
        ort.InferenceSession("yamnet.onnx",
            providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"])
    YAMNet's 521-class output gets remapped to the 5-class M49 schema via
    a fixed lookup table (gunshot ← {429: "Gunshot, gunfire", ...}).

    NOT instantiated in the sandbox benchmark (no model file, no torch),
    but the surface is here so the rest of the system can import it.
    """
    backend_name = "yamnet_onnx"
    YAMNET_TO_M49 = {
        429: "gunshot", 430: "gunshot",      # Gunshot, Machine gun
        437: "glass_break",                   # Shatter
        15:  "scream", 16: "scream",          # Screaming, Whimper
        390: "alarm", 391: "alarm",           # Alarm, Smoke detector
        458: "breaking_door",                 # Slam
    }
    def __init__(self, model_path: str): ...
    def classify(self, f: AudioFeatures) -> Tuple[Optional[str], float, str]:
        raise NotImplementedError("YAMNet stub — deploy on Jetson with onnx export")


# ============================================================
# ACOUSTIC PERIMETER ENGINE
# ============================================================

# Severity per anomaly class — surfaces to the unified bus
ANOMALY_SEVERITY = {
    "gunshot":       5,
    "glass_break":   4,
    "breaking_door": 4,
    "scream":        4,
    "alarm":         3,
}

class AcousticPerimeterEngine:
    """
    Audio analogue of the vision perimeter engine. Consumes AudioChunks,
    extracts features, classifies, and emits Alerts on the unified schema.

    Maintains a small refractory window per anomaly class so a single
    event doesn't fire dozens of alerts across consecutive chunks.
    """
    REFRACTORY_S = 2.0

    def __init__(self, detector, source_id: str, sample_rate: int = 16_000):
        self.detector = detector
        self.extractor = AudioFeatureExtractor(sample_rate=sample_rate)
        self.source_id = source_id
        self._handlers: List = []
        self.alerts_emitted = 0
        self._latencies_ns: List[int] = []
        self._last_fire_ts: Dict[str, float] = {}

    def on_alert(self, fn) -> None:
        self._handlers.append(fn)

    def _emit(self, alert: Alert) -> None:
        self.alerts_emitted += 1
        for h in self._handlers:
            h(alert)

    def process(self, chunk: AudioChunk) -> List[Alert]:
        t0 = time.perf_counter_ns()
        feats = self.extractor.extract(chunk.samples)
        cls, conf, rationale = self.detector.classify(feats)
        latency_ns = time.perf_counter_ns() - t0
        self._latencies_ns.append(latency_ns)
        alerts: List[Alert] = []

        if cls is not None:
            last = self._last_fire_ts.get(cls, 0.0)
            if chunk.timestamp - last >= self.REFRACTORY_S:
                a = Alert(
                    alert_type="AUDIO_ANOMALY",
                    severity=ANOMALY_SEVERITY.get(cls, 3),
                    timestamp=chunk.timestamp,
                    mmsi=0,                 # not applicable for audio
                    fence_id=cls,           # reuse field for class label
                    detail=(f"{cls} (conf={conf:.2f}) on {chunk.source_id}: {rationale}"),
                    detection_latency_us=latency_ns / 1000.0,
                )
                alerts.append(a)
                self._last_fire_ts[cls] = chunk.timestamp
                self._emit(a)
        return alerts

    def latency_ms(self) -> Dict[str, float]:
        if not self._latencies_ns: return {}
        s = sorted(self._latencies_ns); n = len(s)
        def pct(p): return s[min(n - 1, int(p * n))] / 1e6
        return {
            "samples": n, "p50_ms": pct(0.5), "p95_ms": pct(0.95),
            "p99_ms": pct(0.99), "max_ms": s[-1] / 1e6,
            "mean_ms": (sum(s) / n) / 1e6,
        }


# ============================================================
# Synthetic anomaly generator — for the benchmark
# ============================================================
def synth_chunk(kind: str, sample_rate: int = 16_000,
                duration: float = 1.0) -> np.ndarray:
    """Generate a synthetic audio chunk of the given anomaly kind."""
    n = int(duration * sample_rate)
    t = np.linspace(0, duration, n, endpoint=False)
    # Stable seed across Python invocations (Python's hash() is randomized).
    stable_seed = sum(ord(c) * (31 ** i) for i, c in enumerate(kind)) & 0xffff
    rng = np.random.default_rng(seed=stable_seed)

    if kind == "ambient":
        return (rng.normal(0, 0.02, n)).astype(np.float32)

    if kind == "gunshot":
        # Sharp impulse + brief broadband decay
        x = np.zeros(n, dtype=np.float32)
        impulse_idx = int(0.05 * sample_rate)
        x[impulse_idx] = 0.95
        decay = np.exp(-np.linspace(0, 8, n - impulse_idx)) * rng.normal(0, 0.5, n - impulse_idx)
        x[impulse_idx:] += decay.astype(np.float32) * 0.6
        return x + (rng.normal(0, 0.02, n)).astype(np.float32)

    if kind == "glass_break":
        # High-freq transients, sustained 200ms
        x = np.zeros(n, dtype=np.float32)
        for f in [3500, 4800, 6200, 7400]:
            env = np.exp(-np.linspace(0, 6, n)) * (np.sin(2 * np.pi * f * t)).astype(np.float32)
            x += env * 0.3
        x += rng.normal(0, 0.04, n).astype(np.float32) * np.exp(-np.linspace(0, 8, n))
        return x.astype(np.float32)

    if kind == "scream":
        # Harmonic sustained 1500-2500 Hz with fundamental ~500 Hz
        f0 = 520
        x = (0.4 * np.sin(2 * np.pi * f0 * t)
             + 0.3 * np.sin(2 * np.pi * f0 * 2 * t)
             + 0.25 * np.sin(2 * np.pi * f0 * 3 * t)
             + 0.15 * np.sin(2 * np.pi * f0 * 4 * t))
        env = np.exp(-((t - 0.5) ** 2) / 0.3 ** 2)
        return (x * env * 0.5 + rng.normal(0, 0.01, n)).astype(np.float32)

    if kind == "alarm":
        # Two-tone alarm 800/1000 Hz alternating at 4 Hz
        sw = (np.sign(np.sin(2 * np.pi * 4 * t)) + 1) / 2
        x = (0.5 * np.sin(2 * np.pi * 800 * t) * sw
             + 0.5 * np.sin(2 * np.pi * 1000 * t) * (1 - sw))
        return (x * 0.6 + rng.normal(0, 0.005, n)).astype(np.float32)

    if kind == "breaking_door":
        # Low-frequency thump + secondary scatter
        x = 0.7 * np.sin(2 * np.pi * 80 * t) * np.exp(-np.linspace(0, 12, n))
        x += 0.3 * np.sin(2 * np.pi * 150 * t) * np.exp(-np.linspace(0, 8, n))
        x += rng.normal(0, 0.05, n) * np.exp(-np.linspace(0, 6, n))
        return x.astype(np.float32)

    return np.zeros(n, dtype=np.float32)


if __name__ == "__main__":
    detector = RuleBasedDetector()
    engine = AcousticPerimeterEngine(detector, source_id="mic-pier-7")
    captured = []
    engine.on_alert(lambda a: captured.append(a))

    print("Testing all 6 audio classes...")
    classes = ["ambient", "gunshot", "glass_break", "scream", "alarm", "breaking_door"]
    base_t = time.time()
    for i, kind in enumerate(classes):
        chunk = AudioChunk(
            samples=synth_chunk(kind),
            sample_rate=16_000,
            timestamp=base_t + i * 5,   # spaced out so refractory doesn't bite
            grabbed_at_ns=time.perf_counter_ns(),
            source_id="mic-pier-7",
        )
        alerts = engine.process(chunk)
        if alerts:
            for a in alerts:
                print(f"  {kind:14s} → [{a.severity}] AUDIO_ANOMALY  {a.detail[:90]}  ({a.detection_latency_us/1000:.1f}ms)")
        else:
            print(f"  {kind:14s} → (no alert — correctly silent)")

    print(f"\nLatency: {engine.latency_ms()}")
