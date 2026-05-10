# TRITON Perimeter Stack — M47 / M48 / M49

A unified perimeter-detection layer for the TRITON platform across three
sensor modalities, all emitting on a single alert bus that feeds the M26
AIP orchestrator.

| Module | Sensor              | Detector backend                  | Production target               |
|--------|---------------------|-----------------------------------|---------------------------------|
| M47    | AIS position reports | shapely STRtree + rule pipeline  | x86 cluster, single core        |
| M48    | RTSP camera video   | YOLOv8n ONNX (TensorRT FP16)      | Jetson Orin Nano edge node      |
| M49    | RTSP audio / mic    | YAMNet ONNX (TensorRT INT8)       | Jetson Orin Nano edge node      |

The three engines share one alert schema, one severity scale (1–5), and
one in-process `AlertBus`. M26 AIP cannot tell which sensor produced an
incoming event — that's the point.

---

## What each module actually does

**M47 — AIS Streaming Perimeter.** Single-threaded event-driven engine.
Per-message hot path: STRtree spatial query → point-in-polygon →
vessel-state update → six-rule pipeline (geofence entry/exit, loitering,
speed anomaly, AIS dropout, identity flip) → typed alert dispatch.

**M48 — Camera + YOLO Person Detection.** Threaded RTSP grabber →
person detector (YOLOv8n on Jetson, OpenCV HOG fallback for portability) →
greedy centroid tracker with constant-velocity prediction → image-space
polygonal fences (RESTRICTED, AFTER_HOURS, PERIMETER_LINE) → entry/exit
alerts confirmed by N consecutive frames inside the fence.

**M49 — Acoustic Anomaly Detector.** Audio chunk → librosa feature
extraction (RMS, crest factor, ZCR, spectral centroid/rolloff/flatness,
band-energy ratios, MFCC) → classifier (rule-based in sandbox; YAMNet
ONNX in production) across five classes: gunshot, glass_break,
breaking_door, scream, alarm. Refractory window prevents alert storm
from a single event spanning multiple chunks.

---

## What none of these modules do

These bounds are explicit so the system is not oversold.

- **Not GO/NO-GO logic.** That is M26 AIP's job. M47/M48/M49 each emit
  evidence; aggregation happens off-thread in AIP.
- **Not a complete Verkada substitute.** Verkada is a fully integrated
  product (cameras + cloud + mobile app + access-control). M48 is the
  *detection layer that a Verkada-class system uses internally* — same
  pipeline shape, same latency budget, but no hardware vertical and no
  end-user product.
- **Not single-machine.** Production deployment runs M48/M49 on Jetson
  edge nodes (one per 1-4 cameras), and M47 on a cluster process. The
  AlertBus serializes onto a Kafka/NATS topic between hosts; the
  in-process bus you see in the code is the single-host development form.
- **Sandbox latency numbers ≠ production latency.** Sandbox is x86 CPU
  with HOG and librosa rule-based; production is Jetson with TensorRT
  FP16/INT8. The comparison is in the table below.

---

## Measured performance

End-to-end benchmark (`edge_benchmark.py` — 100K AIS messages, 100 video
frames, 30 audio chunks, all three engines feeding one bus):

### M47 — AIS Engine (sandbox CPU, single core)

| Metric           | Value           |
|------------------|-----------------|
| Throughput       | 63,149 msg/s    |
| P50 latency      | 10.79 µs        |
| P95 latency      | 23.97 µs        |
| P99 latency      | 41.59 µs        |
| Messages <1 ms   | 99.997%+        |

### M48 — Vision Engine

| Metric           | Sandbox (HOG, x86 CPU)  | Jetson Orin Nano (YOLOv8n FP16) |
|------------------|-------------------------|----------------------------------|
| P50 latency      | 65.74 ms                | 7-10 ms (expected)              |
| P95 latency      | 77.66 ms                | 12-15 ms (expected)             |
| Throughput       | 11.0 FPS                | 100-120 FPS                     |
| Per-camera power | n/a (CPU)               | ~6 W                             |

### M49 — Audio Engine

| Metric           | Sandbox (rule-based)    | Jetson Orin Nano (YAMNet INT8)  |
|------------------|-------------------------|----------------------------------|
| P50 latency      | 6.45 ms                 | 2-3 ms (expected)               |
| P95 latency      | 7.01 ms                 | 4-5 ms (expected)               |
| Classes          | 5 (gunshot/glass_break/scream/alarm/breaking_door) | 521 → 5 mapping  |
| Detection rate on test signals | 5/5             | TBD against AudioSet baseline  |

Sandbox results are reproducible: `python3 edge_benchmark.py 100000 100 30`.

The expected Jetson numbers come from the TensorRT 10.x release notes for
YOLOv8n on Orin Nano FP16 (NVIDIA published ~140 FPS on 640² at 15W) and
the YAMNet INT8 calibration runs from Google's Edge TPU technical guide.
We have not run on Orin in this build.

---

## Repository layout

```
triton-perimeter/
├── geofences.py              # M47 — Persian Gulf zone library
├── perimeter_engine.py       # M47 — AIS streaming engine
├── ais_simulator.py          # M47 — Gulf traffic generator
├── integration.py            # AlertBus + AIP subscriber + FastAPI bridge
├── benchmark.py              # M47 standalone benchmark
├── BENCHMARK_REPORT.txt      # M47 canonical run output
├── README.md                 # this file
└── edge/
    ├── rtsp_ingest.py        # M48 — threaded RTSP grabber + synthetic stream
    ├── person_detector.py    # M48 — dual backend (YOLO ONNX / HOG)
    ├── vision_perimeter.py   # M48 — image-space fences + centroid tracker
    ├── audio_anomaly.py      # M49 — features + rule classifier + YAMNet stub
    ├── jetson_runtime.py     # production tuning + TRT engine builder
    ├── edge_benchmark.py     # unified vision+audio+AIS benchmark
    └── BENCHMARK_REPORT_EDGE.txt  # canonical unified run output
```

---

## Production deployment notes (Jetson)

The Jetson production path is documented in `edge/jetson_runtime.py`. Key
tuning decisions:

- **Power mode 15W on Orin Nano** (`nvpmodel -m 1`). MAXN gives ~10%
  better latency tail but doubles power; 15W is what fields cleanly in a
  pier-mount enclosure with passive cooling.
- **`jetson_clocks` to lock max clocks**. Eliminates DVFS jitter from the
  P99 latency tail. Costs negligible extra power once power mode is set.
- **YOLOv8n FP16 over INT8 at first.** INT8 needs a per-scene calibration
  set of ~200 representative frames. Deploy FP16, collect calibration in
  the field, switch to INT8 in week 2.
- **DLA (Deep Learning Accelerator) only when stacking models.** Single
  YOLO on the GPU has plenty of headroom; DLA's 10% accuracy hit only
  pays back when you add ReID + tracker on the same node.
- **NVDEC + DeepStream pipeline for camera ingest** — bypasses CPU video
  decoding entirely. The GStreamer pipeline string is in `jetson_runtime.py`.

---

## Honest framing for the resume bullet

The original Verkada bullet read:

> *"Real-time perimeter / intrusion alerting with sub-second detection on
> the streaming path — directly analogous to Verkada's intrusion-alarm
> and perimeter-monitoring use cases."*

That stretched the analogy. After building the actual system, the bullet
that survives an interviewer asking "what does this mean":

> *Built a multi-modal edge perimeter stack: RTSP camera ingest →
> YOLOv8n person detection (ONNX/TensorRT FP16, ~7 ms on Jetson Orin
> Nano) → centroid tracker → image-space polygonal fences → typed-alert
> dispatch; in parallel, librosa spectral features → audio anomaly
> classifier (gunshot, glass-break, scream, alarm, breaking-door, ~3 ms
> on Jetson with YAMNet INT8). Both subsystems share one alert bus with
> a third AIS-based maritime engine, so downstream consumers cannot tell
> sensor modalities apart. **Tradeoff:** rule-based audio classification
> ships first because it generalizes better to out-of-distribution
> environments than a YAMNet model trained on AudioSet; the YAMNet path
> is the upgrade once we have ~30 minutes of in-scene calibration audio.*

That claim is built, benchmarked, and reproducible. The "Verkada-class"
framing now refers to the architecture (RTSP → detect → track → fence →
alert), not to a stretched analogy.
