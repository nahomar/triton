"""
TRITON M48b — Multi-Camera Manager

Orchestrates N camera feeds:
  - one RTSPStream per camera (or SyntheticStream in sandbox)
  - one VisionPerimeterEngine per camera (image-space fences + local tracker)
  - one shared CrossCameraTracker (ReID across cameras)
  - all alerts flow to the unified TRITON AlertBus

Each camera runs in its own worker thread; the cross-camera tracker is
single-threaded and serialized via a mutex (the embedding step is the only
contention point and it's brief).

Production deployment:
  - Edge nodes typically host 2-4 cameras per Jetson Orin Nano.
  - One Python process per node hosts the manager; restart policy is
    handled by systemd or kubernetes (the reliability module sets up
    the inner watchdog and graceful degradation).

The "cheap IP cameras + webcam + phone" target is supported by giving the
RTSPStream constructor different URLs per CameraConfig — the manager
doesn't care whether the source is a Hikvision RTSP, a v4l2 webcam
(rtsp://0/dev/video0 with a small wrapper), or a phone via DroidCam
(rtsp://<phone-ip>:1935/h264.sdp).
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rtsp_ingest import RTSPStream, SyntheticStream, Frame
from person_detector import make_detector, PersonDetector
from vision_perimeter import (
    VisionPerimeterEngine, CameraFence, default_pier_fences,
)
from cross_camera_tracker import CrossCameraTracker
from reid_embedder import make_embedder
from perimeter_engine import Alert


@dataclass
class CameraConfig:
    camera_id: str
    url: str                      # rtsp:// URL, or "synthetic://N" for sandbox
    fences: List[CameraFence] = field(default_factory=list)
    use_gstreamer: bool = False   # set True on Jetson for NVDEC pipeline


class MultiCameraManager:
    """
    Spawns and runs N camera pipelines, sharing one ReID tracker.
    """
    def __init__(
        self,
        cameras: List[CameraConfig],
        detector: Optional[PersonDetector] = None,
        cross_tracker: Optional[CrossCameraTracker] = None,
        max_fps_per_camera: float = 10.0,
        crop_padding: int = 8,
    ):
        self.cameras = cameras
        self.detector = detector or make_detector()
        self.cross_tracker = cross_tracker or CrossCameraTracker(make_embedder())
        self.max_fps = max_fps_per_camera
        self.crop_padding = crop_padding

        self.engines: Dict[str, VisionPerimeterEngine] = {}
        self.streams: Dict[str, object] = {}
        self._workers: Dict[str, threading.Thread] = {}
        self._stop = threading.Event()
        self._cross_lock = threading.Lock()

        self._handlers: List[Callable[[Alert], None]] = []
        self.frames_processed = 0
        self.detections_total = 0
        self.handoffs_total = 0

    def on_alert(self, fn: Callable[[Alert], None]) -> None:
        self._handlers.append(fn)
        # Forward cross-tracker alerts (CROSS_CAMERA_HANDOFF) too
        self.cross_tracker.on_alert(fn)

    def _emit(self, a: Alert) -> None:
        for h in self._handlers: h(a)

    # --- per-camera worker ------------------------------------------------
    def _camera_worker(self, cfg: CameraConfig) -> None:
        engine = self.engines[cfg.camera_id]
        stream = self.streams[cfg.camera_id]
        engine.on_alert(self._emit)
        min_dt = 1.0 / self.max_fps if self.max_fps > 0 else 0.0
        last_t = 0.0

        while not self._stop.is_set():
            frame = stream.read(timeout=1.0) if hasattr(stream, "read") else stream.read()
            if frame is None:
                time.sleep(0.05)
                continue
            now = time.perf_counter()
            if now - last_t < min_dt:
                continue
            last_t = now

            # Run vision engine; capture detections that produced crops.
            engine.process(frame)
            self.frames_processed += 1

            # ReID: feed each currently tracked person to the cross-camera tracker
            for tid, t in engine.tracker.tracks.items():
                # Crop a person box from the current frame using the track centroid
                # and the most recent detection size if available.
                crop = self._extract_crop(frame.image, t)
                if crop is None: continue
                with self._cross_lock:
                    gid = self.cross_tracker.observe(
                        camera_id=cfg.camera_id,
                        local_track_id=tid,
                        crop_bgr=crop,
                        timestamp=frame.wall_timestamp,
                    )
                if gid is not None: self.detections_total += 1

    def _extract_crop(self, img: np.ndarray, track) -> Optional[np.ndarray]:
        """Center a 64x128 crop on the track's centroid as a stand-in when we
        don't have the original detection bbox handy."""
        H, W = img.shape[:2]
        cx, cy = int(track.cx), int(track.cy)
        w, h = 64, 128
        x1 = max(0, cx - w // 2 - self.crop_padding)
        y1 = max(0, cy - h // 2 - self.crop_padding)
        x2 = min(W, cx + w // 2 + self.crop_padding)
        y2 = min(H, cy + h // 2 + self.crop_padding)
        if x2 - x1 < 16 or y2 - y1 < 32: return None
        return img[y1:y2, x1:x2]

    # --- lifecycle --------------------------------------------------------
    def start(self) -> None:
        for cfg in self.cameras:
            if cfg.url.startswith("synthetic://"):
                n_people = int(cfg.url.split("//")[1] or "2")
                stream = SyntheticStream(source_id=cfg.camera_id, num_people=n_people)
            else:
                stream = RTSPStream(cfg.url, source_id=cfg.camera_id,
                                    use_gstreamer=cfg.use_gstreamer)
                if not stream.start():
                    print(f"[manager] failed to open {cfg.url}; skipping {cfg.camera_id}")
                    continue
            self.streams[cfg.camera_id] = stream

            engine = VisionPerimeterEngine(
                detector=self.detector,
                fences=cfg.fences or default_pier_fences(camera_id=cfg.camera_id),
                source_id=cfg.camera_id,
                confirm_frames=2,
            )
            self.engines[cfg.camera_id] = engine

            t = threading.Thread(target=self._camera_worker, args=(cfg,),
                                 daemon=True, name=f"cam-{cfg.camera_id}")
            self._workers[cfg.camera_id] = t
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for t in self._workers.values():
            t.join(timeout=2.0)
        for s in self.streams.values():
            if hasattr(s, "stop"): s.stop()

    def status(self) -> dict:
        return {
            "cameras": list(self.engines.keys()),
            "frames_processed": self.frames_processed,
            "person_observations": self.detections_total,
            "cross_tracker": self.cross_tracker.telemetry(),
            "engines": {
                cid: {
                    "alerts_emitted": eng.alerts_emitted,
                    "tracks_active": len(eng.tracker.tracks),
                    "latency_ms": eng.latency_ms(),
                }
                for cid, eng in self.engines.items()
            },
        }


if __name__ == "__main__":
    # Sandbox demo: 3 synthetic "cameras" with the same scene type.
    captured: List[Alert] = []
    manager = MultiCameraManager(cameras=[
        CameraConfig(camera_id="cam-pier-7",  url="synthetic://2"),
        CameraConfig(camera_id="cam-pier-8",  url="synthetic://2"),
        CameraConfig(camera_id="cam-gate-1",  url="synthetic://1"),
    ], max_fps_per_camera=4.0)

    manager.on_alert(captured.append)
    manager.start()
    print("Running 3-camera demo for 4 seconds...")
    time.sleep(4)
    manager.stop()

    print(f"\nCaptured {len(captured)} alerts across all cameras")
    by_type: dict = {}
    for a in captured:
        by_type[a.alert_type] = by_type.get(a.alert_type, 0) + 1
    for at, c in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"  {at:24s} {c:>4d}")

    handoffs = [a for a in captured if a.alert_type == "CROSS_CAMERA_HANDOFF"]
    print(f"\nCross-camera handoffs: {len(handoffs)}")
    for a in handoffs[:5]:
        print(f"  {a.detail}")

    print(f"\nManager status:")
    s = manager.status()
    print(f"  frames processed:        {s['frames_processed']}")
    print(f"  person observations:     {s['person_observations']}")
    print(f"  cross-tracker telemetry: {s['cross_tracker']}")
