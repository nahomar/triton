"""
TRITON M50 — Cross-Camera Tracker

Maintains a global identity for each person across multiple cameras by
matching ReID embeddings. Per-camera engines emit local track IDs; this
module assigns a stable `global_id` that follows the person from camera
to camera.

Lifecycle:

  1. A camera's local engine reports a person crop + local_track_id.
  2. We embed the crop via the ReID embedder.
  3. Each known global identity has a small set of stored embeddings
     (one per camera, plus a recency-weighted "anchor" embedding).
  4. Match against all global identities by cosine similarity.
     Above MATCH_THRESHOLD: link this local track to that global ID.
     Below NEW_THRESHOLD: spawn a new global ID.
     In between: ambiguous; defer for additional frames.
  5. CROSS_CAMERA_HANDOFF alert fires when a global ID seen on camera A
     is matched on camera B within the handoff window.

Match thresholds depend on the embedding backend. OSNet ONNX gives well-
separated cosine similarity (same-person > 0.7, different < 0.4); the
color-texture fallback has narrower separation (~0.85 / ~0.45). Defaults
below are calibrated for the fallback so the same code works in both
sandbox and production; OSNet deployments should re-tune.

False-positive intuition (worth noting up front):
  - Two people in similar uniforms (e.g. event staff in matching shirts)
    will collide at the embedding level. Workaround: trust the local
    tracker first; only invoke ReID on appearance gaps > N seconds.
  - Lighting changes between cameras (indoor → outdoor) suppress similarity
    even for the same person. Workaround: per-camera lighting calibration
    or train OSNet on dual-domain data.
  - When in doubt, ambiguous matches DEFER rather than guess. False
    handoffs are worse than missed handoffs in this domain.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from reid_embedder import ReidEmbedder
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perimeter_engine import Alert


@dataclass
class GlobalIdentity:
    global_id: int
    first_seen: float
    last_seen: float
    last_camera: str
    # Per-camera most-recent embedding (so we can match against the right
    # camera's lighting/angle when the person re-appears).
    embeddings_by_camera: Dict[str, np.ndarray] = field(default_factory=dict)
    # Recency-weighted "anchor" embedding (used for cold cameras).
    anchor_embedding: Optional[np.ndarray] = None
    # Camera transit history (most recent N).
    camera_history: deque = field(default_factory=lambda: deque(maxlen=10))
    # Handoff bookkeeping
    n_handoffs: int = 0


class CrossCameraTracker:
    """
    Cross-camera tracker over an arbitrary set of cameras. Thread-safe is
    not implemented — this is intended to run on the main alert thread
    after a single per-frame call from each camera engine.

    Tunables:
        match_threshold     cosine sim above which we link to a known global ID
        new_threshold       cosine sim below which we spawn a new ID
                            (between the two: defer)
        gone_ttl_s          drop a global ID if no camera has seen it for this long
        handoff_window_s    only count cross-camera matches within this window
        anchor_alpha        EMA weight on anchor embedding update
    """
    def __init__(
        self,
        embedder: ReidEmbedder,
        match_threshold: float = 0.78,
        new_threshold: float = 0.55,
        gone_ttl_s: float = 90.0,
        handoff_window_s: float = 30.0,
        anchor_alpha: float = 0.3,
    ):
        self.embedder = embedder
        self.match_threshold = match_threshold
        self.new_threshold = new_threshold
        self.gone_ttl_s = gone_ttl_s
        self.handoff_window_s = handoff_window_s
        self.anchor_alpha = anchor_alpha

        self._next_id = 1
        self.identities: Dict[int, GlobalIdentity] = {}
        # Map (camera_id, local_track_id) → global_id while track is alive
        self._local_to_global: Dict[Tuple[str, int], int] = {}

        self._handlers: List = []
        self.alerts_emitted = 0
        # Telemetry
        self.matches = 0
        self.spawns = 0
        self.deferrals = 0
        self.handoffs = 0

    def on_alert(self, fn): self._handlers.append(fn)

    def _emit(self, a: Alert) -> None:
        self.alerts_emitted += 1
        for h in self._handlers: h(a)

    # --- main entry point --------------------------------------------------
    def observe(
        self,
        camera_id: str,
        local_track_id: int,
        crop_bgr: np.ndarray,
        timestamp: float,
    ) -> Optional[int]:
        """
        Inform the tracker about a person observation. Returns the global
        identity ID (existing or new), or None if the observation was
        deferred due to ambiguity.
        """
        # Fast path: this local track already matched to a global ID
        key = (camera_id, local_track_id)
        if key in self._local_to_global:
            gid = self._local_to_global[key]
            ident = self.identities.get(gid)
            if ident is not None:
                ident.last_seen = timestamp
                ident.last_camera = camera_id
                # Re-embed periodically to keep embeddings fresh.
                if (timestamp - ident.first_seen) % 3.0 < 0.5:
                    self._update_embedding(ident, camera_id, crop_bgr)
                return gid
            # Stale mapping; fall through to re-match.
            del self._local_to_global[key]

        emb = self.embedder.embed(crop_bgr)

        # Retire identities not seen in a while
        self._retire(now=timestamp)

        # Search for best match across known identities
        best_id, best_sim = -1, -1.0
        for gid, ident in self.identities.items():
            sim = self._best_similarity(emb, ident, camera_id)
            if sim > best_sim:
                best_sim = sim; best_id = gid

        # Decision
        if best_sim >= self.match_threshold:
            # Match — link this local track to that global identity
            self.matches += 1
            ident = self.identities[best_id]
            prior_camera = ident.last_camera
            ident.last_seen = timestamp
            ident.last_camera = camera_id
            self._update_embedding(ident, camera_id, crop_bgr, emb)
            ident.camera_history.append((camera_id, timestamp))
            self._local_to_global[key] = best_id

            # Cross-camera handoff alert
            if (prior_camera != camera_id and
                    timestamp - ident.last_seen <= self.handoff_window_s):
                self.handoffs += 1
                ident.n_handoffs += 1
                self._emit(Alert(
                    alert_type="CROSS_CAMERA_HANDOFF",
                    severity=3,
                    timestamp=timestamp,
                    mmsi=best_id,                  # reuse field for global_id
                    fence_id=None,
                    detail=(f"global #{best_id}: handoff "
                            f"{prior_camera} → {camera_id} (sim={best_sim:.2f})"),
                ))
            return best_id

        if best_sim < self.new_threshold:
            # New person — spawn a global identity
            self.spawns += 1
            gid = self._next_id; self._next_id += 1
            ident = GlobalIdentity(
                global_id=gid,
                first_seen=timestamp, last_seen=timestamp,
                last_camera=camera_id,
                anchor_embedding=emb.copy(),
            )
            ident.embeddings_by_camera[camera_id] = emb
            ident.camera_history.append((camera_id, timestamp))
            self.identities[gid] = ident
            self._local_to_global[key] = gid
            return gid

        # Ambiguous — defer (no commit). Caller should retry next frame.
        self.deferrals += 1
        return None

    def drop_local_track(self, camera_id: str, local_track_id: int) -> None:
        """Called by per-camera engines when a local track ends."""
        self._local_to_global.pop((camera_id, local_track_id), None)

    # --- internals ---------------------------------------------------------
    def _best_similarity(
        self, emb: np.ndarray, ident: GlobalIdentity, camera_id: str,
    ) -> float:
        # Prefer same-camera embedding if we've seen this person there before
        # (lighting/angle match). Otherwise use anchor.
        candidates = []
        if camera_id in ident.embeddings_by_camera:
            candidates.append(ident.embeddings_by_camera[camera_id])
        if ident.anchor_embedding is not None:
            candidates.append(ident.anchor_embedding)
        for cam_emb in ident.embeddings_by_camera.values():
            candidates.append(cam_emb)
        if not candidates: return -1.0
        sims = [float(np.dot(emb, c)) for c in candidates]
        return max(sims)

    def _update_embedding(
        self, ident: GlobalIdentity, camera_id: str,
        crop_bgr: np.ndarray, emb: Optional[np.ndarray] = None,
    ) -> None:
        if emb is None:
            emb = self.embedder.embed(crop_bgr)
        ident.embeddings_by_camera[camera_id] = emb
        if ident.anchor_embedding is None:
            ident.anchor_embedding = emb.copy()
        else:
            a = self.anchor_alpha
            new_anchor = (1 - a) * ident.anchor_embedding + a * emb
            n = np.linalg.norm(new_anchor) + 1e-9
            ident.anchor_embedding = (new_anchor / n).astype(np.float32)

    def _retire(self, now: float) -> None:
        stale = [gid for gid, ident in self.identities.items()
                 if now - ident.last_seen > self.gone_ttl_s]
        for gid in stale:
            del self.identities[gid]
            stale_keys = [k for k, v in self._local_to_global.items() if v == gid]
            for k in stale_keys: del self._local_to_global[k]

    def telemetry(self) -> dict:
        return {
            "n_identities_active": len(self.identities),
            "matches": self.matches, "spawns": self.spawns,
            "deferrals": self.deferrals, "handoffs": self.handoffs,
            "alerts_emitted": self.alerts_emitted,
        }


if __name__ == "__main__":
    from reid_embedder import make_embedder
    import cv2

    # Build two synthetic people
    person_a = np.zeros((128, 64, 3), dtype=np.uint8)
    person_a[:, :] = [40, 40, 220]; person_a[:32] = [180, 150, 130]; person_a[80:] = [80, 60, 40]
    person_b = np.zeros((128, 64, 3), dtype=np.uint8)
    person_b[:, :] = [200, 60, 40]; person_b[:32] = [160, 140, 120]; person_b[80:] = [60, 90, 30]
    gamma_table = np.array([((i / 255.0) ** 0.85) * 255 for i in range(256)]).astype(np.uint8)
    person_a_cam2 = cv2.LUT(person_a, gamma_table)

    tracker = CrossCameraTracker(make_embedder())
    captured = []
    tracker.on_alert(captured.append)

    # Camera 1 sees person A
    g1 = tracker.observe("cam-1", local_track_id=1, crop_bgr=person_a, timestamp=1000.0)
    # Camera 1 sees person B
    g2 = tracker.observe("cam-1", local_track_id=2, crop_bgr=person_b, timestamp=1000.5)
    # Camera 2 sees the same person A (different lighting)
    g3 = tracker.observe("cam-2", local_track_id=1, crop_bgr=person_a_cam2, timestamp=1010.0)
    # Camera 2 sees a fresh person B
    g4 = tracker.observe("cam-2", local_track_id=2, crop_bgr=person_b, timestamp=1011.0)

    print(f"cam-1 person A → global #{g1}")
    print(f"cam-1 person B → global #{g2}")
    print(f"cam-2 person A → global #{g3}  (same as cam-1?  {g3 == g1})")
    print(f"cam-2 person B → global #{g4}  (same as cam-1?  {g4 == g2})")
    print(f"Telemetry: {tracker.telemetry()}")
    print(f"Alerts:")
    for a in captured:
        print(f"  [{a.severity}] {a.alert_type} {a.detail}")

    assert g3 == g1, f"expected cross-camera match for A, got g3={g3} g1={g1}"
    assert g4 == g2, f"expected cross-camera match for B, got g4={g4} g2={g2}"
    print("✅  cross-camera handoff working")
