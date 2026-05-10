"""
TRITON M48 — Vision Perimeter

Image-space perimeter detection. Sits between the person detector and the
unified TRITON AlertBus. Maintains per-person tracks across frames using a
simple centroid tracker with predictive matching, then fires GEOFENCE_ENTRY
alerts when a tracked foot-point crosses into a camera fence polygon.

This is the visual analogue of the AIS perimeter engine in the parent
module — same alert schema, same severity scale, same bus. M26 AIP
consumes both indistinguishably; the only thing different is the fence
coordinate system (image pixels here vs WGS84 lon/lat there) and the
sensor.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from person_detector import Detection
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from perimeter_engine import Alert  # reuse the parent-module alert schema


# ============================================================
# CAMERA FENCE
# ============================================================

@dataclass
class CameraFence:
    """
    Polygonal fence in image pixel coordinates (origin top-left).
    `polygon`: ordered list of (x, y) vertices.
    Use OpenCV's pointPolygonTest for inside-test (works fine at this scale).
    """
    fence_id: str
    name: str
    zone_type: str          # RESTRICTED, AFTER_HOURS, PERIMETER_LINE, etc.
    severity: int           # 1..5 to align with parent-module fences
    polygon: np.ndarray     # shape (N, 1, 2) int32 — OpenCV format
    camera_id: str
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_points(cls, fence_id: str, name: str, zone_type: str,
                    severity: int, points: List[Tuple[int, int]],
                    camera_id: str, **meta) -> "CameraFence":
        poly = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
        return cls(
            fence_id=fence_id, name=name, zone_type=zone_type,
            severity=severity, polygon=poly, camera_id=camera_id,
            metadata=meta,
        )

    def contains(self, x: float, y: float) -> bool:
        return cv2.pointPolygonTest(self.polygon, (float(x), float(y)), False) >= 0


# ============================================================
# CENTROID TRACKER  — maintains person identity across frames
# ============================================================

@dataclass
class _Track:
    track_id: int
    last_seen_frame: int
    cx: float
    cy: float
    foot_x: float
    foot_y: float
    velocity: Tuple[float, float] = (0.0, 0.0)
    age_frames: int = 0
    inside_fences: set = field(default_factory=set)


class CentroidTracker:
    """
    Greedy centroid matcher with constant-velocity prediction.
    Hungarian assignment would be ideal but adds scipy dependency; greedy is
    fine when crowds are sparse, which is the typical perimeter-cam case.
    """
    def __init__(self, max_disappeared: int = 15, max_distance: float = 100.0):
        self.next_id = 1
        self.tracks: Dict[int, _Track] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._frame_idx = 0

    def update(self, detections: List[Detection]) -> Dict[int, _Track]:
        self._frame_idx += 1

        # Predict each track forward by velocity (very cheap)
        for t in self.tracks.values():
            t.cx += t.velocity[0]
            t.cy += t.velocity[1]
            t.foot_x += t.velocity[0]
            t.foot_y += t.velocity[1]

        # Greedy match detections to tracks by predicted-position distance
        unmatched = list(range(len(detections)))
        used_tracks = set()
        matches: List[Tuple[int, int]] = []  # (track_id, det_idx)

        for tid, t in list(self.tracks.items()):
            best_idx = -1; best_dist = self.max_distance + 1
            for i in unmatched:
                d = detections[i]
                dist = math.hypot(d.cx - t.cx, d.cy - t.cy)
                if dist < best_dist:
                    best_dist = dist; best_idx = i
            if best_idx >= 0 and best_dist <= self.max_distance:
                matches.append((tid, best_idx))
                used_tracks.add(tid)
                unmatched.remove(best_idx)

        # Update matched tracks
        for tid, di in matches:
            d = detections[di]
            t = self.tracks[tid]
            new_cx, new_cy = d.cx, d.cy
            t.velocity = (new_cx - t.cx + t.velocity[0]) / 2, (new_cy - t.cy + t.velocity[1]) / 2
            t.cx, t.cy = new_cx, new_cy
            t.foot_x, t.foot_y = d.foot_point
            t.last_seen_frame = self._frame_idx
            t.age_frames += 1

        # Spawn new tracks for unmatched detections
        for i in unmatched:
            d = detections[i]
            self.tracks[self.next_id] = _Track(
                track_id=self.next_id,
                last_seen_frame=self._frame_idx,
                cx=d.cx, cy=d.cy,
                foot_x=d.foot_point[0], foot_y=d.foot_point[1],
                age_frames=1,
            )
            self.next_id += 1

        # Retire stale tracks
        stale = [tid for tid, t in self.tracks.items()
                 if self._frame_idx - t.last_seen_frame > self.max_disappeared]
        for tid in stale:
            del self.tracks[tid]

        return self.tracks


# ============================================================
# VISION PERIMETER ENGINE
# ============================================================

class VisionPerimeterEngine:
    """
    Per-camera perimeter engine. Wraps detector + tracker + fence library
    and emits Alert events on the parent-module schema.

    Wire-up:
        engine = VisionPerimeterEngine(detector, fences, source_id="cam-pier-7")
        engine.on_alert(my_handler)
        for frame in stream: engine.process(frame)
    """
    def __init__(
        self,
        detector,
        fences: List[CameraFence],
        source_id: str,
        confirm_frames: int = 2,
    ):
        self.detector = detector
        self.fences = fences
        self.source_id = source_id
        self.tracker = CentroidTracker()
        self.confirm_frames = confirm_frames     # require N frames inside before alerting
        self._handlers: List = []
        self._inside_streak: Dict[Tuple[int, str], int] = {}
        self.alerts_emitted = 0
        self._latencies_ns: List[int] = []

    def on_alert(self, fn) -> None:
        self._handlers.append(fn)

    def _emit(self, alert: Alert) -> None:
        self.alerts_emitted += 1
        for h in self._handlers:
            h(alert)

    def process(self, frame) -> List[Alert]:
        """
        Process one Frame. Returns alerts emitted by this frame.
        Latency is measured from start of detector.detect() to end of dispatch.
        """
        t0 = time.perf_counter_ns()
        detections = self.detector.detect(frame.image)
        tracks = self.tracker.update(detections)

        alerts: List[Alert] = []
        active_tids = set(tracks.keys())
        # Dropped track entries should be cleared from streak counters
        stale_keys = [k for k in self._inside_streak if k[0] not in active_tids]
        for k in stale_keys:
            del self._inside_streak[k]

        for tid, t in tracks.items():
            for fence in self.fences:
                inside = fence.contains(t.foot_x, t.foot_y)
                key = (tid, fence.fence_id)
                streak = self._inside_streak.get(key, 0)

                if inside:
                    self._inside_streak[key] = streak + 1
                    # Fire on the Nth confirming frame, once per fence-entry
                    if streak + 1 == self.confirm_frames:
                        alerts.append(Alert(
                            alert_type="GEOFENCE_ENTRY",
                            severity=fence.severity,
                            timestamp=frame.wall_timestamp,
                            mmsi=tid,        # reuse the field for track_id
                            fence_id=fence.fence_id,
                            detail=(f"person track #{tid} entered "
                                    f"{fence.name} on {fence.camera_id}"),
                        ))
                else:
                    if streak > 0:
                        # Was inside, now outside — emit exit only if the entry had fired
                        if streak >= self.confirm_frames:
                            alerts.append(Alert(
                                alert_type="GEOFENCE_EXIT",
                                severity=max(1, fence.severity - 2),
                                timestamp=frame.wall_timestamp,
                                mmsi=tid,
                                fence_id=fence.fence_id,
                                detail=f"person track #{tid} exited {fence.name}",
                            ))
                        self._inside_streak[key] = 0

        latency_ns = time.perf_counter_ns() - t0
        self._latencies_ns.append(latency_ns)

        for a in alerts:
            a.detection_latency_us = latency_ns / 1000.0
            self._emit(a)

        return alerts

    def latency_ms(self) -> Dict[str, float]:
        if not self._latencies_ns: return {}
        s = sorted(self._latencies_ns); n = len(s)
        def pct(p): return s[min(n - 1, int(p * n))] / 1e6
        return {
            "samples": n,
            "p50_ms": pct(0.5),
            "p95_ms": pct(0.95),
            "p99_ms": pct(0.99),
            "max_ms": s[-1] / 1e6,
            "mean_ms": (sum(s) / n) / 1e6,
            "fps_equivalent": 1000.0 / ((sum(s) / n) / 1e6),
        }

    @staticmethod
    def render_overlay(frame_img: np.ndarray, fences: List[CameraFence],
                       tracks: Dict[int, _Track]) -> np.ndarray:
        """Return a debug-overlay image with fences and tracked persons drawn."""
        out = frame_img.copy()
        for f in fences:
            color = (0, 0, 255) if f.severity >= 4 else (0, 200, 200)
            cv2.polylines(out, [f.polygon], isClosed=True, color=color, thickness=2)
            x, y = f.polygon[0][0]
            cv2.putText(out, f.fence_id, (int(x), int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for tid, t in tracks.items():
            cv2.circle(out, (int(t.foot_x), int(t.foot_y)), 6, (0, 255, 0), -1)
            cv2.putText(out, f"#{tid}", (int(t.cx), int(t.cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return out


# ============================================================
# Default fence set — example camera at a port pier
# ============================================================
def default_pier_fences(camera_id: str = "cam-bandar-abbas-pier-7") -> List[CameraFence]:
    return [
        CameraFence.from_points(
            fence_id="PIER_RESTRICTED",
            name="Restricted Pier Apron",
            zone_type="RESTRICTED",
            severity=4,
            points=[(200, 350), (1100, 350), (1100, 720), (200, 720)],
            camera_id=camera_id,
        ),
        CameraFence.from_points(
            fence_id="LOADING_DOCK",
            name="Loading Dock After-Hours Zone",
            zone_type="AFTER_HOURS",
            severity=3,
            points=[(400, 400), (900, 400), (900, 600), (400, 600)],
            camera_id=camera_id,
        ),
        CameraFence.from_points(
            fence_id="WATER_LINE",
            name="Water-Line Approach",
            zone_type="PERIMETER_LINE",
            severity=5,
            points=[(0, 620), (1280, 620), (1280, 720), (0, 720)],
            camera_id=camera_id,
        ),
    ]


if __name__ == "__main__":
    from rtsp_ingest import SyntheticStream
    from person_detector import make_detector

    stream = SyntheticStream(num_people=2)
    detector = make_detector()
    fences = default_pier_fences()
    engine = VisionPerimeterEngine(detector, fences, source_id="cam-pier-7",
                                   confirm_frames=2)
    captured = []
    engine.on_alert(lambda a: captured.append(a))

    print("Processing 30 synthetic frames...")
    for i in range(30):
        frame = stream.read()
        engine.process(frame)

    print(f"\n{engine.alerts_emitted} alerts emitted across 30 frames")
    for a in captured[:8]:
        print(f"  [{a.severity}] {a.alert_type:14s} {a.detail}  ({a.detection_latency_us/1000:.1f}ms)")
    print(f"\nLatency: {engine.latency_ms()}")
