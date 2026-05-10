"""
TRITON M48 — RTSP Frame Ingest

Threaded RTSP frame grabber. On Jetson production this connects to an actual
camera stream (e.g. rtsp://admin:pw@10.0.1.42:554/Streaming/Channels/101).
For testing without a camera, a `SyntheticStream` produces frames with an
embedded synthetic person silhouette so the downstream detector / fence
pipeline has something real to work with.

Design notes:
  - Grabber runs in its own thread so OpenCV's blocking `read()` never stalls
    the main inference loop.
  - Bounded queue with drop-on-full policy: prefer freshness over completeness.
    Stale frames are useless for intrusion alerting.
  - Each frame is timestamped with `time.perf_counter_ns()` at grab time so
    end-to-end latency (camera → alert) is measurable downstream.
"""
from __future__ import annotations

import threading
import time
import queue
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class Frame:
    seq: int
    grabbed_at_ns: int      # perf_counter_ns at the moment of capture
    wall_timestamp: float   # epoch seconds (for log/alert correlation)
    image: np.ndarray       # BGR uint8 HxWx3
    source_id: str          # camera identifier (e.g. "cam-bandar-abbas-pier-7")


class RTSPStream:
    """
    Threaded RTSP/MJPEG/file-source video grabber.

    Production usage:
        s = RTSPStream("rtsp://...:554/...", source_id="cam-pier-7")
        s.start()
        frame = s.read(timeout=0.2)

    On Jetson, set `use_gstreamer=True` to invoke the hardware-accelerated
    NVDEC pipeline through GStreamer (rtspsrc → rtph264depay → nvv4l2decoder
    → nvvidconv → BGRx → appsink). That bypasses ffmpeg-on-CPU entirely.
    """
    GSTREAMER_TEMPLATE = (
        "rtspsrc location={url} latency=0 buffer-mode=auto ! "
        "rtph264depay ! h264parse ! nvv4l2decoder ! "
        "nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! "
        "video/x-raw, format=BGR ! appsink drop=1 sync=false"
    )

    def __init__(
        self,
        url: str,
        source_id: str,
        max_queue: int = 4,
        use_gstreamer: bool = False,
    ):
        self.url = url
        self.source_id = source_id
        self.use_gstreamer = use_gstreamer
        self._q: queue.Queue[Frame] = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._seq = 0
        self.frames_grabbed = 0
        self.frames_dropped = 0

    def _open(self) -> bool:
        if self.use_gstreamer:
            pipe = self.GSTREAMER_TEMPLATE.format(url=self.url)
            self._cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
        else:
            self._cap = cv2.VideoCapture(self.url)
        return self._cap is not None and self._cap.isOpened()

    def _loop(self) -> None:
        while not self._stop.is_set():
            ok, img = self._cap.read() if self._cap else (False, None)
            if not ok:
                # Reconnect on failure (cameras drop, networks blip)
                time.sleep(0.5)
                self._open()
                continue
            self._seq += 1
            frame = Frame(
                seq=self._seq,
                grabbed_at_ns=time.perf_counter_ns(),
                wall_timestamp=time.time(),
                image=img,
                source_id=self.source_id,
            )
            try:
                self._q.put_nowait(frame)
                self.frames_grabbed += 1
            except queue.Full:
                # Drop oldest, keep newest — staleness is worse than gap.
                try:
                    _ = self._q.get_nowait()
                    self._q.put_nowait(frame)
                    self.frames_dropped += 1
                except queue.Empty:
                    pass

    def start(self) -> bool:
        if not self._open():
            return False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return True

    def read(self, timeout: float = 1.0) -> Optional[Frame]:
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()


# ============================================================
# Synthetic stream — for benchmarks without a real camera.
# Produces a moving person silhouette over a noisy background.
# ============================================================
class SyntheticStream:
    """
    Generates a 1280x720 BGR stream at a target FPS with one or more synthetic
    person silhouettes walking diagonally across the frame. Used by the
    benchmark harness so the full pipeline has real pixel data to process.
    """
    def __init__(
        self,
        source_id: str = "synthetic-cam-01",
        width: int = 1280,
        height: int = 720,
        fps: float = 30.0,
        num_people: int = 2,
    ):
        self.source_id = source_id
        self.W = width
        self.H = height
        self.fps = fps
        self.num_people = num_people
        self._seq = 0
        self._t0 = time.time()

    def _draw_person(self, img: np.ndarray, cx: int, cy: int, scale: float) -> None:
        # Crude humanoid silhouette: head circle + body rectangle + legs
        h = int(160 * scale)
        w = int(60 * scale)
        cv2.circle(img, (cx, cy - h // 2), int(15 * scale), (60, 60, 60), -1)
        cv2.rectangle(img, (cx - w // 2, cy - h // 2 + int(15 * scale)),
                      (cx + w // 2, cy + h // 4), (50, 50, 50), -1)
        cv2.rectangle(img, (cx - w // 4, cy + h // 4),
                      (cx + w // 4, cy + h // 2), (40, 40, 40), -1)

    def read(self, timeout: float = 1.0) -> Frame:
        self._seq += 1
        # Background: low-amplitude noise + dark gradient
        img = np.random.randint(20, 60, (self.H, self.W, 3), dtype=np.uint8)
        cv2.rectangle(img, (0, self.H - 100), (self.W, self.H), (15, 25, 35), -1)

        # Walk people diagonally
        t = self._seq / self.fps
        for i in range(self.num_people):
            phase = i * 0.3 + t * 0.05
            cx = int(self.W * (0.2 + 0.6 * (phase % 1.0)))
            cy = int(self.H * (0.6 + 0.1 * np.sin(t * 0.5 + i)))
            scale = 0.9 + 0.2 * np.sin(t * 0.3 + i)
            self._draw_person(img, cx, cy, scale)

        return Frame(
            seq=self._seq,
            grabbed_at_ns=time.perf_counter_ns(),
            wall_timestamp=time.time(),
            image=img,
            source_id=self.source_id,
        )

    def start(self) -> bool: return True
    def stop(self) -> None: pass


if __name__ == "__main__":
    s = SyntheticStream(num_people=2)
    frames = [s.read() for _ in range(5)]
    print(f"Synthetic stream: produced {len(frames)} frames")
    for f in frames:
        print(f"  seq={f.seq:3d}  shape={f.image.shape}  source={f.source_id}")
