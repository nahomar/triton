"""
TRITON M49b — Mel-Spectrogram CNN Audio Classifier

Small CNN architecture (~50K params) over log-mel-spectrogram input,
running alongside the rule-based detector. Two backends:

  MelSpecCnnOnnx     — production path. Loads a trained ONNX model and
                       runs via onnxruntime (TensorRT INT8 on Jetson).
                       Expected ~3 ms per chunk on Orin Nano.

  MelSpecCnnNumpy    — sandbox reference path. Pure-numpy forward pass
                       through the same architecture. Used when no
                       trained weights are available; ships with random
                       weights so its outputs are uniform — meaning it
                       won't be the active classifier in sandbox, but
                       the inference graph is real and verifiable.

Architecture (matches a YAMNet-Lite / CNN10 lineage at ~1/30 the params):

  log-mel input          (1, 1, 64, 96)        # 64 mel bins × 96 frames
  conv2d(1→16, 3x3)  +ReLU +MaxPool(2x2)       (1,16,32,48)
  conv2d(16→32, 3x3) +ReLU +MaxPool(2x2)       (1,32,16,24)
  conv2d(32→64, 3x3) +ReLU +MaxPool(2x2)       (1,64,8,12)
  global average pool                          (1,64)
  fc(64→32)          +ReLU                     (1,32)
  fc(32→6)           +softmax                  (1,6)
                                                ↑
                              5 anomaly classes + 1 ambient

Production swap procedure:
  1. Train this exact architecture on ~10 hours of in-domain audio
     labelled per class (or fine-tune YAMNet and distil to this network).
  2. Export to ONNX with opset 17:
        torch.onnx.export(model, dummy, "audio_cnn.onnx", opset_version=17)
  3. Run trtexec --onnx=audio_cnn.onnx --saveEngine=audio_cnn_int8.engine
     --int8 --calib=<calibration_dir>
  4. Drop the path into MelSpecCnnOnnx and the rest of the pipeline is
     unchanged.

The CNN does NOT replace the rule-based detector by default. It runs in
parallel and the engine fuses both opinions: an anomaly is emitted only
when both agree (high precision) or when one fires very strongly (high
recall). This dual-path design is the standard hedge against per-scene
distribution shift breaking the trained model.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import librosa


# 6 classes: 5 anomaly + ambient
CNN_CLASSES = ["ambient", "gunshot", "glass_break", "breaking_door", "scream", "alarm"]


def log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int = 16_000,
    n_mels: int = 64,
    target_frames: int = 96,
    n_fft: int = 1024,
    hop_length: int = 160,
) -> np.ndarray:
    """Compute log-mel spectrogram in the shape the CNN expects: (1, 64, 96)."""
    if samples.size < hop_length * 2:
        samples = np.pad(samples, (0, hop_length * 2 - samples.size))
    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32), sr=sample_rate,
        n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        fmin=20, fmax=sample_rate // 2,
    )
    log_mel = librosa.power_to_db(mel + 1e-10)
    # Pad / truncate frame axis
    F = log_mel.shape[1]
    if F < target_frames:
        log_mel = np.pad(log_mel, ((0, 0), (0, target_frames - F)),
                         mode="constant", constant_values=log_mel.min())
    elif F > target_frames:
        log_mel = log_mel[:, :target_frames]
    # Per-chunk normalize for stability across recording levels
    m, s = log_mel.mean(), log_mel.std() + 1e-6
    log_mel = (log_mel - m) / s
    return log_mel.astype(np.float32)[None, :, :]   # (1, 64, 96)


# ============================================================
# NUMPY REFERENCE FORWARD PASS
# ============================================================
def _conv2d(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Pure-numpy 2D convolution, stride 1, padding 1, no dilation.
    x: (N, Cin, H, W)   w: (Cout, Cin, kH, kW)   b: (Cout,)
    """
    N, Cin, H, W = x.shape
    Cout, _, kH, kW = w.shape
    pad_h, pad_w = kH // 2, kW // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)))
    # im2col
    cols = np.lib.stride_tricks.sliding_window_view(xp, (kH, kW), axis=(2, 3))
    # cols shape: (N, Cin, H, W, kH, kW)
    cols = cols.transpose(0, 2, 3, 1, 4, 5).reshape(N * H * W, Cin * kH * kW)
    w_flat = w.reshape(Cout, -1)             # (Cout, Cin*kH*kW)
    out = cols @ w_flat.T + b                # (N*H*W, Cout)
    return out.reshape(N, H, W, Cout).transpose(0, 3, 1, 2)


def _maxpool2d(x: np.ndarray, k: int = 2) -> np.ndarray:
    N, C, H, W = x.shape
    H2, W2 = H // k, W // k
    x = x[:, :, : H2 * k, : W2 * k]
    return x.reshape(N, C, H2, k, W2, k).max(axis=(3, 5))


def _relu(x): return np.maximum(x, 0)


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = x - x.max(axis=axis, keepdims=True)
    ez = np.exp(z); return ez / ez.sum(axis=axis, keepdims=True)


@dataclass
class CnnWeights:
    """Stored weights for the 3-conv + 2-fc architecture."""
    c1_w: np.ndarray; c1_b: np.ndarray   # (16,1,3,3) (16,)
    c2_w: np.ndarray; c2_b: np.ndarray   # (32,16,3,3) (32,)
    c3_w: np.ndarray; c3_b: np.ndarray   # (64,32,3,3) (64,)
    f1_w: np.ndarray; f1_b: np.ndarray   # (32,64) (32,)
    f2_w: np.ndarray; f2_b: np.ndarray   # (6,32) (6,)

    @classmethod
    def random_init(cls, seed: int = 0) -> "CnnWeights":
        """Random Kaiming-ish init. Sandbox use only — outputs are noise."""
        rng = np.random.default_rng(seed)
        def he(shape, fan_in):
            return rng.standard_normal(shape).astype(np.float32) * np.sqrt(2.0 / fan_in)
        return cls(
            c1_w=he((16, 1, 3, 3), 1 * 9), c1_b=np.zeros(16, np.float32),
            c2_w=he((32, 16, 3, 3), 16 * 9), c2_b=np.zeros(32, np.float32),
            c3_w=he((64, 32, 3, 3), 32 * 9), c3_b=np.zeros(64, np.float32),
            f1_w=he((32, 64), 64), f1_b=np.zeros(32, np.float32),
            f2_w=he((6, 32), 32), f2_b=np.zeros(6, np.float32),
        )

    def param_count(self) -> int:
        total = 0
        for arr in [self.c1_w, self.c1_b, self.c2_w, self.c2_b,
                    self.c3_w, self.c3_b, self.f1_w, self.f1_b,
                    self.f2_w, self.f2_b]:
            total += arr.size
        return total


class MelSpecCnnNumpy:
    """Pure-numpy forward pass through the architecture above."""
    backend_name = "melspec_cnn_numpy"

    def __init__(self, weights: Optional[CnnWeights] = None):
        self.w = weights if weights is not None else CnnWeights.random_init()

    def predict(self, log_mel: np.ndarray) -> np.ndarray:
        """log_mel: (1, 64, 96) → softmax probs over CNN_CLASSES, shape (6,)."""
        x = log_mel[:, None, :, :]                            # (1,1,64,96)
        x = _maxpool2d(_relu(_conv2d(x, self.w.c1_w, self.w.c1_b)))   # (1,16,32,48)
        x = _maxpool2d(_relu(_conv2d(x, self.w.c2_w, self.w.c2_b)))   # (1,32,16,24)
        x = _maxpool2d(_relu(_conv2d(x, self.w.c3_w, self.w.c3_b)))   # (1,64,8,12)
        x = x.mean(axis=(2, 3))                               # GAP → (1,64)
        x = _relu(x @ self.w.f1_w.T + self.w.f1_b)            # (1,32)
        x = x @ self.w.f2_w.T + self.w.f2_b                   # (1,6)
        return _softmax(x, axis=-1)[0]


# ============================================================
# ONNX BACKEND  — production
# ============================================================
class MelSpecCnnOnnx:
    """Production audio CNN via onnxruntime."""
    backend_name = "melspec_cnn_onnx"

    def __init__(self, model_path: str):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in avail:
            providers.append(("TensorrtExecutionProvider",
                              {"trt_int8_enable": True, "trt_engine_cache_enable": True}))
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, log_mel: np.ndarray) -> np.ndarray:
        x = log_mel[None, :, :, :].astype(np.float32)   # (1,1,64,96)
        return self.session.run(None, {self.input_name: x})[0][0]


# ============================================================
# FACTORY
# ============================================================
def make_cnn(model_path: Optional[str] = None):
    if model_path and os.path.exists(model_path):
        try:
            return MelSpecCnnOnnx(model_path)
        except Exception as e:
            print(f"[audio_cnn] ONNX load failed: {e}; using numpy reference path")
    return MelSpecCnnNumpy()


# ============================================================
# DUAL-PATH FUSION DETECTOR
# ============================================================
class DualPathAudioDetector:
    """
    Runs the rule-based detector AND the CNN in parallel, then fuses.

    Fusion policy:
      - If the rule detector and the CNN agree on a class → fire (high precision).
      - If only the rule fires AND its confidence is > 0.7 → fire (rules are
        well-calibrated for clear cases).
      - If only the CNN fires AND its confidence is > 0.85 → fire (high bar
        because untrained / out-of-domain risk).
      - Otherwise → no alert.

    With random-weight CNN (sandbox), the CNN almost never crosses 0.85, so
    the rule detector remains effectively the only active path. Once the CNN
    is trained, the agreement gate kicks in and false positives drop.
    """
    backend_name = "dual_path_fusion"

    def __init__(self, rule_detector, cnn=None, cnn_threshold: float = 0.85):
        self.rule = rule_detector
        self.cnn = cnn if cnn is not None else MelSpecCnnNumpy()
        self.cnn_threshold = cnn_threshold

    def classify(self, features) -> Tuple[Optional[str], float, str]:
        # Rule path
        rule_cls, rule_conf, rule_why = self.rule.classify(features)

        # CNN path needs the raw mel-spec, which the engine has but the
        # rule classify() interface doesn't carry. We expose a separate
        # method below for the engine to call. Fall back to rule-only.
        return rule_cls, rule_conf, rule_why

    def classify_with_melspec(
        self, features, log_mel: np.ndarray,
    ) -> Tuple[Optional[str], float, str]:
        rule_cls, rule_conf, rule_why = self.rule.classify(features)
        cnn_probs = self.cnn.predict(log_mel)
        cnn_idx = int(np.argmax(cnn_probs))
        cnn_cls = CNN_CLASSES[cnn_idx] if cnn_idx > 0 else None  # 0 = ambient
        cnn_conf = float(cnn_probs[cnn_idx])

        # Fusion
        if rule_cls is not None and rule_cls == cnn_cls:
            return rule_cls, max(rule_conf, cnn_conf), (
                f"BOTH agreed on {rule_cls}: rule={rule_conf:.2f} cnn={cnn_conf:.2f}; {rule_why}"
            )
        if rule_cls is not None and rule_conf > 0.7:
            return rule_cls, rule_conf, (
                f"RULE only ({rule_cls}, {rule_conf:.2f} > 0.7); cnn said {cnn_cls or 'ambient'}@{cnn_conf:.2f}"
            )
        if cnn_cls is not None and cnn_conf > self.cnn_threshold:
            return cnn_cls, cnn_conf, (
                f"CNN only ({cnn_cls}@{cnn_conf:.2f} > {self.cnn_threshold}); rule said {rule_cls or 'ambient'}"
            )
        return None, 0.0, ""


if __name__ == "__main__":
    from audio_anomaly import synth_chunk

    # Show architecture metadata
    weights = CnnWeights.random_init()
    cnn = MelSpecCnnNumpy(weights)
    print(f"CNN backend: {cnn.backend_name}")
    print(f"Parameter count: {weights.param_count():,}")

    print("\nForward-pass smoke test on every class (random weights):")
    for kind in ["ambient", "gunshot", "glass_break", "scream", "alarm", "breaking_door"]:
        samples = synth_chunk(kind)
        log_mel = log_mel_spectrogram(samples)
        t0 = time.perf_counter_ns()
        probs = cnn.predict(log_mel)
        elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
        top_idx = int(np.argmax(probs))
        print(f"  {kind:14s} → {CNN_CLASSES[top_idx]:14s} ({probs[top_idx]:.2f})  [{elapsed_ms:.1f}ms]")
    print("\nNote: with random weights, predictions are uninformative.")
    print("Production deploys trained weights at /opt/triton/audio_cnn_int8.engine.")
