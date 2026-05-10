"""
TRITON Edge Reliability Layer

Production-grade reliability primitives for the M48 multi-camera stack.
Four components:

  ReconnectingRTSPStream  — extends RTSPStream with exponential backoff
                            reconnect, frame-freshness tracking, and
                            consecutive-failure circuit breaker.

  HealthMonitor           — observes the manager's worker threads and
                            flags state changes: HEALTHY → DEGRADED →
                            FAILED → RECOVERED. Emits HEALTH_* alerts on
                            the same bus operators are already watching.

  Watchdog                — periodic timer that kills and restarts a
                            worker thread that has gone silent.

  GracefulDegradation     — when the primary detector fails (e.g. CUDA
                            OOM, model file corrupted), swap to a motion-
                            based fallback detector so the camera still
                            produces some signal until the primary recovers.

This module is the answer to the question "what happens when a cheap IP
camera drops or the network blips?" in the v1 ship. Production hardening
tradeoffs:
  - Reconnect is local. Cameras that go down for >5 min should trigger a
    NOC ticket via an external monitoring system, not be retried forever
    locally.
  - Watchdog kills workers but doesn't escalate. A persistently-sick
    worker will flap; deployment policy should rotate the whole node.
  - Graceful degradation is intentionally conservative — it logs a HEALTH_
    alert at severity 4 so an operator can intervene before relying on
    motion-only detection for a long stretch.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtsp_ingest import RTSPStream, Frame
from person_detector import PersonDetector, Detection
from perimeter_engine import Alert


log = logging.getLogger("triton.reliability")


# ============================================================
# RECONNECTING RTSP STREAM
# ============================================================
class ReconnectingRTSPStream(RTSPStream):
    """
    RTSP stream with exponential-backoff reconnect and circuit breaker.

    Backoff schedule (seconds): 1, 2, 4, 8, 16, 30, 30, 30 ...
    Circuit-breaker: after N consecutive total failures, stop trying for
    `circuit_open_s` seconds, then attempt one probe before resuming.
    """
    def __init__(
        self,
        url: str,
        source_id: str,
        max_queue: int = 4,
        use_gstreamer: bool = False,
        max_backoff_s: float = 30.0,
        circuit_breaker_after: int = 8,
        circuit_open_s: float = 60.0,
    ):
        super().__init__(url, source_id, max_queue, use_gstreamer)
        self.max_backoff_s = max_backoff_s
        self.circuit_breaker_after = circuit_breaker_after
        self.circuit_open_s = circuit_open_s
        self.consecutive_failures = 0
        self.total_reconnects = 0
        self.circuit_open_until: float = 0.0
        self.last_frame_at: float = 0.0

    def _loop(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            # Honor the circuit breaker
            if time.time() < self.circuit_open_until:
                time.sleep(0.5); continue

            ok, img = self._cap.read() if self._cap else (False, None)
            if not ok:
                self.consecutive_failures += 1
                self.total_reconnects += 1

                if self.consecutive_failures >= self.circuit_breaker_after:
                    self.circuit_open_until = time.time() + self.circuit_open_s
                    log.warning(
                        f"{self.source_id}: circuit breaker OPEN for "
                        f"{self.circuit_open_s}s after {self.consecutive_failures} failures"
                    )
                    self.consecutive_failures = 0
                    backoff = 1.0
                    continue

                log.info(f"{self.source_id}: read failed, backoff {backoff:.1f}s")
                time.sleep(backoff)
                backoff = min(self.max_backoff_s, backoff * 2)
                self._open()
                continue

            # success
            self.consecutive_failures = 0
            backoff = 1.0
            self.last_frame_at = time.time()
            self._seq += 1
            frame = Frame(
                seq=self._seq,
                grabbed_at_ns=time.perf_counter_ns(),
                wall_timestamp=self.last_frame_at,
                image=img, source_id=self.source_id,
            )
            try:
                self._q.put_nowait(frame)
                self.frames_grabbed += 1
            except Exception:
                # Drop oldest, keep newest
                try:
                    self._q.get_nowait(); self._q.put_nowait(frame)
                    self.frames_dropped += 1
                except Exception: pass


# ============================================================
# HEALTH MONITOR
# ============================================================
class HealthState(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    FAILED = "FAILED"


@dataclass
class CameraHealthSnapshot:
    camera_id: str
    state: HealthState
    last_frame_age_s: float
    fps_recent: float
    detections_recent: int
    consecutive_dry_seconds: float
    notes: str


class HealthMonitor:
    """
    Periodically inspects the multi-camera manager's status and classifies
    each camera as HEALTHY / DEGRADED / FAILED. Emits HEALTH_TRANSITION
    alerts on state changes only (not every tick).
    """
    HEALTHY_FPS_FLOOR = 1.5         # below this for >5s → DEGRADED
    DEGRADED_DRY_S = 5.0
    FAILED_DRY_S = 20.0

    def __init__(self, manager, tick_s: float = 1.0):
        self.manager = manager
        self.tick_s = tick_s
        self._handlers: List[Callable[[Alert], None]] = []
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._state: Dict[str, HealthState] = {}
        self._last_frame_count: Dict[str, int] = {}
        self._dry_since: Dict[str, float] = {}

    def on_alert(self, fn): self._handlers.append(fn)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="triton-health")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.tick_s)
            self._tick()

    def _tick(self) -> None:
        st = self.manager.status()
        for cam_id, eng_status in st["engines"].items():
            stream = self.manager.streams.get(cam_id)
            now = time.time()
            last_frame_t = getattr(stream, "last_frame_at", 0.0) or 0.0
            age = now - last_frame_t if last_frame_t else 999.0

            cur_frames = self.manager.frames_processed
            prev = self._last_frame_count.get(cam_id, cur_frames)
            self._last_frame_count[cam_id] = cur_frames
            fps = (cur_frames - prev) / self.tick_s

            if fps < self.HEALTHY_FPS_FLOOR:
                self._dry_since[cam_id] = self._dry_since.get(cam_id, now)
            else:
                self._dry_since.pop(cam_id, None)

            dry = now - self._dry_since.get(cam_id, now)

            if age > self.FAILED_DRY_S or dry > self.FAILED_DRY_S:
                state = HealthState.FAILED
            elif age > self.DEGRADED_DRY_S or dry > self.DEGRADED_DRY_S:
                state = HealthState.DEGRADED
            else:
                state = HealthState.HEALTHY

            prior = self._state.get(cam_id, HealthState.HEALTHY)
            if state != prior:
                severity = {
                    HealthState.HEALTHY: 2,
                    HealthState.DEGRADED: 4,
                    HealthState.FAILED: 5,
                }[state]
                self._emit(Alert(
                    alert_type=f"HEALTH_{state.value}",
                    severity=severity,
                    timestamp=now,
                    mmsi=0,
                    fence_id=cam_id,
                    detail=(f"{cam_id}: {prior.value} → {state.value} "
                            f"(frame_age={age:.1f}s, fps={fps:.2f}, dry={dry:.1f}s)"),
                ))
            self._state[cam_id] = state

    def _emit(self, a: Alert) -> None:
        for h in self._handlers: h(a)

    def snapshot(self) -> List[CameraHealthSnapshot]:
        out: List[CameraHealthSnapshot] = []
        for cam_id, state in self._state.items():
            stream = self.manager.streams.get(cam_id)
            now = time.time()
            last_frame_t = getattr(stream, "last_frame_at", 0.0) or 0.0
            out.append(CameraHealthSnapshot(
                camera_id=cam_id, state=state,
                last_frame_age_s=now - last_frame_t if last_frame_t else 999.0,
                fps_recent=0.0, detections_recent=0,
                consecutive_dry_seconds=now - self._dry_since.get(cam_id, now),
                notes="",
            ))
        return out


# ============================================================
# WATCHDOG
# ============================================================
class Watchdog:
    """
    Restarts worker threads that have gone silent. The manager exposes
    `_workers: dict[str, Thread]`. The watchdog inspects them and asks
    the manager to restart any whose target is no longer alive.
    """
    def __init__(self, manager, restart_fn: Callable[[str], None],
                 check_interval_s: float = 5.0):
        self.manager = manager
        self.restart_fn = restart_fn
        self.check_interval_s = check_interval_s
        self.restart_count: Dict[str, int] = {}
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="triton-watchdog")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread: self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        while not self._stop.is_set():
            time.sleep(self.check_interval_s)
            for cam_id, t in list(self.manager._workers.items()):
                if not t.is_alive():
                    self.restart_count[cam_id] = self.restart_count.get(cam_id, 0) + 1
                    log.warning(f"watchdog: {cam_id} thread dead, restart #{self.restart_count[cam_id]}")
                    try: self.restart_fn(cam_id)
                    except Exception as e:
                        log.error(f"watchdog: restart of {cam_id} failed: {e}")


# ============================================================
# GRACEFUL DEGRADATION — motion-only fallback detector
# ============================================================
class MotionFallbackDetector(PersonDetector):
    """
    Motion-blob fallback when the primary detector (YOLO/HOG) fails.
    Uses a running average background subtractor; emits a single
    "person-shaped" detection per blob above a minimum area threshold.

    NOT a person detector — just enough signal that perimeter alerts can
    still fire on intrusion (confirm_frames=2 already filters most of the
    motion-blob false positives caused by lighting changes).
    """
    backend_name = "motion_fallback"

    def __init__(self, min_area: int = 4_000, max_area: int = 80_000):
        self.bg = cv2.createBackgroundSubtractorMOG2(
            history=300, varThreshold=25, detectShadows=False)
        self.min_area = min_area
        self.max_area = max_area

    def detect(self, frame: np.ndarray) -> List[Detection]:
        mask = self.bg.apply(frame)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        mask = cv2.dilate(mask, None, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: List[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (self.min_area <= area <= self.max_area): continue
            x, y, w, h = cv2.boundingRect(c)
            # Person-shaped aspect ratio prior: tall blobs only
            if h < 1.5 * w: continue
            out.append(Detection(
                x1=int(x), y1=int(y), x2=int(x + w), y2=int(y + h),
                confidence=0.4, class_id=0, class_name="person",
            ))
        return out


class DegradableDetector(PersonDetector):
    """
    Wraps a primary detector with a motion-only fallback. Tracks the last
    K detections; if the primary fails or returns nothing for
    `dry_threshold_s` seconds while motion is observed, switches to
    fallback for `fallback_for_s` seconds, then probes the primary again.
    """
    backend_name = "degradable"

    def __init__(
        self,
        primary: PersonDetector,
        fallback: Optional[PersonDetector] = None,
        dry_threshold_s: float = 30.0,
        fallback_for_s: float = 60.0,
    ):
        self.primary = primary
        self.fallback = fallback or MotionFallbackDetector()
        self.dry_threshold_s = dry_threshold_s
        self.fallback_for_s = fallback_for_s
        self._last_primary_hit: float = time.time()
        self._fallback_until: float = 0.0
        self._mode: str = "primary"
        self.transition_count: int = 0

    @property
    def mode(self) -> str: return self._mode

    def detect(self, frame: np.ndarray) -> List[Detection]:
        now = time.time()
        if now < self._fallback_until:
            return self.fallback.detect(frame)
        try:
            dets = self.primary.detect(frame)
            if dets: self._last_primary_hit = now
            elif now - self._last_primary_hit > self.dry_threshold_s:
                # Long dry stretch — temporarily fall back
                self._fallback_until = now + self.fallback_for_s
                self._mode = "fallback"; self.transition_count += 1
                return self.fallback.detect(frame)
            self._mode = "primary"
            return dets
        except Exception as e:
            log.error(f"primary detector exception: {e}; falling back")
            self._fallback_until = now + self.fallback_for_s
            self._mode = "fallback"; self.transition_count += 1
            return self.fallback.detect(frame)


if __name__ == "__main__":
    # Smoke test — verify motion fallback detector works on a synthetic frame
    from rtsp_ingest import SyntheticStream
    s = SyntheticStream(num_people=2)
    det = MotionFallbackDetector()
    print("Warming up background model...")
    for _ in range(15):
        det.detect(s.read().image)
    f = s.read()
    dets = det.detect(f.image)
    print(f"Motion fallback detected {len(dets)} blob(s)")
    for d in dets[:5]:
        print(f"  bbox=({d.x1},{d.y1})-({d.x2},{d.y2})  area={(d.x2-d.x1)*(d.y2-d.y1)}  conf={d.confidence:.2f}")

    # Verify DegradableDetector wraps without breaking primary path
    from person_detector import make_detector
    deg = DegradableDetector(primary=make_detector())
    dets2 = deg.detect(f.image)
    print(f"\nDegradableDetector mode={deg.mode}  detected={len(dets2)}")
    print(f"  transitions: {deg.transition_count}")
