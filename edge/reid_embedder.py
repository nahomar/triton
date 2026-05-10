"""
TRITON M50 — Person ReID Embedding Extractor

Maps a person bounding-box crop to a fixed-dimensional embedding vector.
Cross-camera matching uses cosine similarity in this space.

Two backends, one interface:

  OsnetOnnxEmbedder       — OSNet-x0_25 (~2.2M params) exported to ONNX,
                            run via onnxruntime with TensorRT on Jetson.
                            512-d output. This is the production target.

  ColorTextureEmbedder    — sandbox fallback, no model file. Combines HSV
                            color histograms with simple texture statistics
                            into a 256-d vector. Loses to OSNet on
                            ReID accuracy but discriminates well enough
                            for handoff between 2-3 cameras with distinct
                            clothing.

Both produce L2-normalized vectors, so similarity is a single dot product.

Production note: re-identification accuracy depends heavily on the
embedding network being trained on people-in-the-wild data with the
same camera angles and lighting as deployment. OSNet-x0_25 hits ~83%
Rank-1 on Market-1501 — fine for indoor/outdoor commercial sites,
weak for top-down or extreme-angle cameras. Retraining on in-domain
data lifts this 5-10 points; that's the next step beyond the v1 ship.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


# ============================================================
# COMMON SCHEMA
# ============================================================

@dataclass
class ReidEmbedding:
    vector: np.ndarray         # L2-normalized float32, shape (D,)
    backend: str
    extracted_at: float        # epoch seconds


class ReidEmbedder:
    backend_name: str = "abstract"
    embedding_dim: int = 0
    def embed(self, crop_bgr: np.ndarray) -> np.ndarray: ...

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        # Both vectors are L2-normalized so this is a single dot product.
        return float(np.dot(a, b))


# ============================================================
# OSNet ONNX BACKEND  — production
# ============================================================
class OsnetOnnxEmbedder(ReidEmbedder):
    """
    OSNet ReID embedder. Expects ONNX export at `model_path`.

    Provider preference (set at session creation):
        TensorRT FP16 (Jetson)  -> CUDA  -> CPU

    Input: 256x128 BGR float32 normalized with ImageNet mean/std.
    Output: 512-d float32, L2-normalized.

    Inference latency on Orin Nano FP16: ~3-4 ms per crop.
    Batch of 8 crops: ~7 ms (better GPU utilization).
    """
    backend_name = "osnet_onnx"
    embedding_dim = 512

    INPUT_W = 128
    INPUT_H = 256
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: str):
        import onnxruntime as ort
        avail = ort.get_available_providers()
        providers = []
        if "TensorrtExecutionProvider" in avail:
            providers.append(("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
            }))
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, crop_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.INPUT_W, self.INPUT_H), cv2.INTER_LINEAR)
        x = resized.astype(np.float32) / 255.0
        x = (x - self.MEAN) / self.STD
        return x.transpose(2, 0, 1)[None, ...]   # 1x3xHxW

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        x = self._preprocess(crop_bgr)
        feat = self.session.run(None, {self.input_name: x})[0][0]
        n = np.linalg.norm(feat) + 1e-9
        return (feat / n).astype(np.float32)


# ============================================================
# COLOR + TEXTURE BACKEND  — sandbox / portable
# ============================================================
class ColorTextureEmbedder(ReidEmbedder):
    """
    Sandbox-friendly ReID embedder. Combines:
      - HSV color histogram on three vertical body-zones (head, torso, legs)
        — captures clothing color signatures, which are the dominant cue
        for cross-camera ReID at moderate distance.
      - Simple texture statistics (gradient magnitude histogram, edge
        density per zone) — discriminates patterned vs. plain clothing.

    Output: 256-d L2-normalized float32. Not as accurate as OSNet, but
    enough to separate 2-3 people with distinct outfits across cameras
    in a demo or controlled deployment.
    """
    backend_name = "color_texture"
    embedding_dim = 256

    def __init__(self, hist_bins: tuple = (12, 8, 8)):
        # HSV bin counts: hue 12, saturation 8, value 8 → 32-d per zone color
        # 3 zones × 32 = 96 color
        # 3 zones × 16 gradient histogram bins = 48 texture
        # 3 zones × 4 stats (mean, std, edge_density, contrast) = 12 stats
        # = 156-d. Pad to 256 with zeros for forward-compat with OSNet replacement.
        self.hist_bins = hist_bins

    def _zone_features(self, zone_bgr: np.ndarray) -> np.ndarray:
        if zone_bgr.size == 0:
            return np.zeros(60, dtype=np.float32)
        # HSV histogram. Weight H (hue) > S > V because hue is ~invariant
        # to lighting; V (brightness) is the most lighting-sensitive channel
        # and dominates if we treat the three equally.
        hsv = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv], [0], None, [self.hist_bins[0]], [0, 180]).flatten()
        s_hist = cv2.calcHist([hsv], [1], None, [self.hist_bins[1]], [0, 256]).flatten()
        v_hist = cv2.calcHist([hsv], [2], None, [self.hist_bins[2]], [0, 256]).flatten()
        for hist in (h_hist, s_hist, v_hist):
            if hist.sum() > 0:
                hist /= hist.sum()
        # Apply per-channel weights BEFORE concatenation
        h_hist *= 3.0
        s_hist *= 1.5
        v_hist *= 0.5
        color = np.concatenate([h_hist, s_hist, v_hist])
        # Color is the dominant cross-camera ReID cue — weight 3x vs texture
        color *= 3.0

        # Gradient histogram (16 bins) — clothing pattern signature
        gray = cv2.cvtColor(zone_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        grad_hist = np.histogram(mag, bins=16, range=(0, 200))[0].astype(np.float32)
        if grad_hist.sum() > 0:
            grad_hist /= grad_hist.sum()
        # Stats
        edges = cv2.Canny(gray, 50, 150)
        stats = np.array([
            float(gray.mean() / 255.0),
            float(gray.std() / 255.0),
            float(edges.mean() / 255.0),
            float(mag.mean() / 200.0),
        ], dtype=np.float32)
        return np.concatenate([color, grad_hist, stats]).astype(np.float32)

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        if crop_bgr is None or crop_bgr.size == 0:
            return np.zeros(self.embedding_dim, dtype=np.float32)
        H, W = crop_bgr.shape[:2]
        # Resize for stability across crop sizes
        crop = cv2.resize(crop_bgr, (64, 128))
        h, w = crop.shape[:2]
        head = crop[: h // 4]
        torso = crop[h // 4: h * 5 // 8]
        legs = crop[h * 5 // 8:]
        feats = np.concatenate([
            self._zone_features(head),
            self._zone_features(torso),
            self._zone_features(legs),
        ])
        # Pad / truncate to 256 for forward-compat with OSNet replacement
        if feats.size < 256:
            feats = np.pad(feats, (0, 256 - feats.size))
        else:
            feats = feats[:256]
        n = np.linalg.norm(feats) + 1e-9
        return (feats / n).astype(np.float32)


# ============================================================
# FACTORY
# ============================================================
def make_embedder(model_path: Optional[str] = None) -> ReidEmbedder:
    """OSNet ONNX if a model path is given and loads, else color+texture fallback."""
    if model_path and os.path.exists(model_path):
        try:
            return OsnetOnnxEmbedder(model_path)
        except Exception as e:
            print(f"[reid] OSNet ONNX load failed: {e}; using color+texture fallback")
    return ColorTextureEmbedder()


if __name__ == "__main__":
    # Self-test: synthetic crops with same/different clothing.
    # Lighting transform is gamma adjustment (closer to real-world camera
    # exposure differences than a linear offset).
    person_a_red = np.zeros((128, 64, 3), dtype=np.uint8)
    person_a_red[:, :] = [40, 40, 220]    # red shirt (BGR)
    person_a_red[:32] = [180, 150, 130]   # head
    person_a_red[80:] = [80, 60, 40]      # dark pants

    person_b_blue = np.zeros((128, 64, 3), dtype=np.uint8)
    person_b_blue[:, :] = [200, 60, 40]   # blue shirt
    person_b_blue[:32] = [160, 140, 120]
    person_b_blue[80:] = [60, 90, 30]     # green pants

    # Realistic lighting variation: gamma correction (camera exposure shift)
    gamma = 0.85
    table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype(np.uint8)
    person_a_lit = cv2.LUT(person_a_red, table)

    e = make_embedder()
    print(f"Embedder: {e.backend_name} (dim={e.embedding_dim})")
    va = e.embed(person_a_red)
    vb = e.embed(person_b_blue)
    va2 = e.embed(person_a_lit)
    sim_same = ReidEmbedder.cosine_similarity(va, va2)
    sim_diff = ReidEmbedder.cosine_similarity(va, vb)
    print(f"  same person, gamma=0.85 lighting:  cos = {sim_same:.3f}")
    print(f"  different person (red vs blue):    cos = {sim_diff:.3f}")
    assert sim_same > sim_diff, (
        f"ReID embedder failed self-test: sim_same={sim_same:.3f} "
        f"should exceed sim_diff={sim_diff:.3f}"
    )
    print(f"  ✅  same-person ({sim_same:.3f}) > cross-person ({sim_diff:.3f})  "
          f"margin={sim_same - sim_diff:.3f}")
