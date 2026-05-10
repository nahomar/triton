"""
TRITON M48 — Person Detector (dual backend)

One interface, two backends:

  YoloOnnxDetector — YOLOv8n ONNX with TensorrtExecutionProvider on Jetson
                     (or CUDAExecutionProvider on dGPU, CPU fallback on x86)
                     Production deployment target.

  HOGDetector      — OpenCV HOG + linear SVM with default people SVM weights.
                     Portable, no model file required, runs anywhere with
                     OpenCV. Used for the sandbox benchmark.

Both return the same `Detection` schema so the rest of the pipeline doesn't
care which backend is loaded. The `make_detector` factory picks the best
available backend at startup.

Jetson production tuning notes (out-of-band, do not affect this code):
  - Export YOLOv8n.pt → yolov8n.onnx → yolov8n.engine via trtexec
      trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16
  - For INT8: run trtexec with --int8 and a calibration cache built from
    ~200 representative frames from the deployment scene.
  - Place 1 of the 2 DLAs (Deep Learning Accelerator cores) on the engine
    for ~30% GPU offload: --useDLACore=0 --allowGPUFallback
  - Expect 5-8 ms inference per 640x640 frame on Orin Nano FP16, 2-3 ms on
    AGX Orin INT8.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


# ============================================================
# COMMON SCHEMA
# ============================================================

@dataclass
class Detection:
    x1: int               # bbox in original image pixels
    y1: int
    x2: int
    y2: int
    confidence: float     # 0..1
    class_id: int         # 0 = person
    class_name: str = "person"

    @property
    def cx(self) -> float: return (self.x1 + self.x2) / 2.0
    @property
    def cy(self) -> float: return (self.y1 + self.y2) / 2.0
    @property
    def foot_point(self) -> tuple:
        # The "ground contact" point — what the perimeter fence cares about
        return (int((self.x1 + self.x2) / 2), int(self.y2))


class PersonDetector:
    """Abstract interface."""
    backend_name: str = "abstract"
    def detect(self, frame: np.ndarray) -> List[Detection]: ...
    def warmup(self, frame: np.ndarray) -> None: self.detect(frame)


# ============================================================
# YOLOv8 ONNX BACKEND  — Jetson production path
# ============================================================
class YoloOnnxDetector(PersonDetector):
    """
    YOLOv8n ONNX inference via onnxruntime.
    Provider preference: TensorRT (Jetson) → CUDA → CPU.
    Input: 640x640 RGB float32 normalized 0..1.
    Output (YOLOv8 ONNX): [1, 84, 8400] — 4 bbox + 80 class scores per anchor.
    """
    backend_name = "yolov8_onnx"

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.35,
        nms_iou: float = 0.5,
        target_classes: tuple = (0,),  # person only
    ):
        import onnxruntime as ort
        providers = []
        avail = ort.get_available_providers()
        if "TensorrtExecutionProvider" in avail:
            providers.append(("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": "/var/cache/triton/trt",
            }))
        if "CUDAExecutionProvider" in avail:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape  # [1,3,640,640]
        self.conf = conf_threshold
        self.iou = nms_iou
        self.target_classes = set(target_classes)

    @staticmethod
    def _letterbox(img: np.ndarray, new_shape: int = 640) -> tuple:
        H, W = img.shape[:2]
        r = min(new_shape / H, new_shape / W)
        nh, nw = int(round(H * r)), int(round(W * r))
        top = (new_shape - nh) // 2
        left = (new_shape - nw) // 2
        canvas = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
        resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        canvas[top:top + nh, left:left + nw] = resized
        return canvas, r, left, top

    def _preprocess(self, img: np.ndarray):
        canvas, r, dx, dy = self._letterbox(img, 640)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None, ...].astype(np.float32) / 255.0
        return tensor, r, dx, dy

    def _postprocess(self, output: np.ndarray, r: float, dx: int, dy: int,
                     orig_h: int, orig_w: int) -> List[Detection]:
        # YOLOv8 ONNX raw: [1, 84, 8400]; transpose to [8400, 84]
        preds = output[0].T
        cls_scores = preds[:, 4:]
        cls_ids = cls_scores.argmax(axis=1)
        confs = cls_scores.max(axis=1)
        keep = (confs >= self.conf) & np.isin(cls_ids, list(self.target_classes))
        if not keep.any():
            return []
        boxes_xywh = preds[keep, :4]
        confs = confs[keep]
        cls_ids = cls_ids[keep]
        # cx,cy,w,h → x1,y1,x2,y2
        x1 = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2
        y1 = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2
        x2 = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2
        y2 = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2
        # Undo letterbox
        x1 = (x1 - dx) / r; x2 = (x2 - dx) / r
        y1 = (y1 - dy) / r; y2 = (y2 - dy) / r
        x1 = np.clip(x1, 0, orig_w); x2 = np.clip(x2, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h); y2 = np.clip(y2, 0, orig_h)
        # NMS via OpenCV
        boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        idxs = cv2.dnn.NMSBoxes(boxes, confs.tolist(), self.conf, self.iou)
        out: List[Detection] = []
        for i in (idxs.flatten() if len(idxs) else []):
            out.append(Detection(
                x1=int(x1[i]), y1=int(y1[i]), x2=int(x2[i]), y2=int(y2[i]),
                confidence=float(confs[i]),
                class_id=int(cls_ids[i]), class_name="person",
            ))
        return out

    def detect(self, frame: np.ndarray) -> List[Detection]:
        H, W = frame.shape[:2]
        tensor, r, dx, dy = self._preprocess(frame)
        out = self.session.run(None, {self.input_name: tensor})[0]
        return self._postprocess(out, r, dx, dy, H, W)


# ============================================================
# OpenCV HOG BACKEND  — sandbox / portable fallback
# ============================================================
class HOGDetector(PersonDetector):
    backend_name = "opencv_hog"

    def __init__(
        self,
        win_stride: tuple = (8, 8),
        padding: tuple = (8, 8),
        scale: float = 1.05,
        hit_threshold: float = 0.0,
    ):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.win_stride = win_stride
        self.padding = padding
        self.scale = scale
        self.hit_threshold = hit_threshold

    def detect(self, frame: np.ndarray) -> List[Detection]:
        # Downscale for HOG speed if large
        H, W = frame.shape[:2]
        max_dim = 640
        if max(H, W) > max_dim:
            r = max_dim / max(H, W)
            small = cv2.resize(frame, (int(W * r), int(H * r)))
        else:
            small = frame; r = 1.0
        rects, weights = self.hog.detectMultiScale(
            small, winStride=self.win_stride, padding=self.padding,
            scale=self.scale, hitThreshold=self.hit_threshold,
        )
        out: List[Detection] = []
        for (x, y, w, h), wt in zip(rects, weights):
            x1, y1, x2, y2 = int(x / r), int(y / r), int((x + w) / r), int((y + h) / r)
            out.append(Detection(
                x1=x1, y1=y1, x2=x2, y2=y2,
                confidence=float(wt[0] if hasattr(wt, '__len__') else wt),
                class_id=0, class_name="person",
            ))
        return out


# ============================================================
# FACTORY
# ============================================================
def make_detector(model_path: Optional[str] = None) -> PersonDetector:
    """
    Pick the best available detector. If a YOLO ONNX path is given and
    onnxruntime can load it, use that; otherwise fall back to HOG.
    """
    if model_path and os.path.exists(model_path):
        try:
            return YoloOnnxDetector(model_path)
        except Exception as e:
            print(f"[detector] YOLO ONNX load failed: {e}; falling back to HOG")
    return HOGDetector()


if __name__ == "__main__":
    # Verify HOG runs on a synthetic frame
    from rtsp_ingest import SyntheticStream
    s = SyntheticStream(num_people=3)
    f = s.read()
    det = make_detector()
    print(f"Backend: {det.backend_name}")
    t0 = time.perf_counter_ns()
    detections = det.detect(f.image)
    elapsed_ms = (time.perf_counter_ns() - t0) / 1e6
    print(f"Inference: {elapsed_ms:.1f} ms — {len(detections)} person(s) detected")
    for d in detections[:5]:
        print(f"  bbox=({d.x1},{d.y1})-({d.x2},{d.y2})  conf={d.confidence:.2f}  foot={d.foot_point}")
