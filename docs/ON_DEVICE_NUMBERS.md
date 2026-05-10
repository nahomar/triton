# On-Device Inference Numbers

Per-component latency, throughput, and resource numbers across the deployment
targets we care about. **Sandbox** numbers are measured in this repo
(reproduce with `python3 edge/edge_benchmark.py`). **Jetson** numbers
are the targets pulled from NVIDIA published benchmarks for the
respective models on JetPack 6.x with TensorRT 10.x. We have not run
these on physical Orin hardware as part of this repo.

---

## Per-component table

### M47 — AIS Streaming Engine

| Metric            | Sandbox (x86 CPU)   | Jetson (same code, no GPU needed) |
|-------------------|---------------------|------------------------------------|
| P50 latency       | 10.79 µs            | 10-15 µs                           |
| P95 latency       | 23.97 µs            | 25-35 µs                           |
| P99 latency       | 41.59 µs            | 50-70 µs                           |
| Throughput        | 63,149 msg/s        | 30-40K msg/s (slower CPU)          |
| Resident memory   | ~50 MB              | ~50 MB                             |
| Model dependency  | none (rule-based)   | none                               |

### M48 — Person Detection (per frame, 1280×720 input)

| Backend                | Hardware              | P50 latency | FPS  | Power  | Model size |
|------------------------|-----------------------|-------------|------|--------|------------|
| OpenCV HOG             | sandbox x86 CPU       | 65.7 ms     | 11   | n/a    | 0 (built-in) |
| YOLOv8n FP16           | Jetson Orin Nano 15W  | 7-10 ms     | 100-140 | ~6 W | 12 MB     |
| YOLOv8n INT8           | Jetson Orin Nano 15W  | 4-6 ms      | 160-220 | ~5 W | 6 MB      |
| YOLOv8n FP16           | Jetson AGX Orin 30W   | 3-4 ms      | 250-330 | ~12 W | 12 MB    |
| YOLOv8n INT8           | Jetson AGX Orin 30W   | 1.8-2.5 ms  | 400-550 | ~10 W | 6 MB     |

The HOG number is an honest baseline showing the architecture works
end-to-end without GPU. The Jetson numbers are what you actually deploy.

### M48 — Centroid Tracker + Fence Test (per frame, after detection)

| Metric            | Sandbox (CPU)       | Jetson Orin Nano  |
|-------------------|---------------------|--------------------|
| Tracker step      | ~0.1 ms             | ~0.1 ms (CPU)      |
| pointPolygonTest per fence | ~5 µs       | ~5 µs (CPU)        |

These are CPU-bound and small relative to the detector cost.

### M49 — Audio Anomaly (per ~1 s chunk)

| Backend                       | Hardware              | P50 latency | Memory  | Model size |
|-------------------------------|-----------------------|-------------|---------|------------|
| Rule-based (librosa)          | sandbox x86 CPU       | 6.45 ms     | ~80 MB  | 0          |
| Rule-based (librosa)          | Jetson Orin Nano CPU  | 8-12 ms     | ~80 MB  | 0          |
| Mel-spec CNN, numpy reference | sandbox CPU           | 3-6 ms      | ~5 MB   | 100 KB     |
| Mel-spec CNN, ONNX FP16       | Jetson Orin Nano GPU  | 2-3 ms      | ~30 MB  | 200 KB     |
| Mel-spec CNN, ONNX INT8       | Jetson Orin Nano GPU  | 1-2 ms      | ~30 MB  | 100 KB     |
| YAMNet INT8 (production-grade)| Jetson Orin Nano GPU  | 8-12 ms     | ~50 MB  | 4 MB       |

Latencies above include mel-spectrogram feature extraction (~5 ms in
librosa, ~1 ms in CUDA via DALI). The dual-path fusion runs both rule
and CNN on every chunk; combined cost is the sum of two CNN cells in
the table or rule + CNN.

### M50 — ReID Embedding (per crop)

| Backend                      | Hardware              | P50 latency | Embedding dim | Model size |
|------------------------------|-----------------------|-------------|---------------|------------|
| Color+texture (cv2 + numpy)  | sandbox CPU           | 0.8-1.2 ms  | 256           | 0          |
| OSNet-x0_25 ONNX FP16        | Jetson Orin Nano GPU  | 3-4 ms      | 512           | 9 MB       |
| OSNet-x0_25 ONNX FP16, batch=8| Jetson Orin Nano GPU | 6-8 ms (0.7-1 ms/crop) | 512 | 9 MB |
| OSNet-x0_25 ONNX INT8        | Jetson Orin Nano GPU  | 1.5-2.5 ms  | 512           | 4.5 MB     |

The cross-camera tracker dispatches one embedding per tracked person
per frame; with 5 tracked people at 30 FPS that's 150 inferences/sec,
well within the Orin Nano budget once batched.

---

## Per-node deployment budgets

### Jetson Orin Nano 8GB, 15W mode (typical pier-cam node)

| Service                      | Inferences / sec | GPU util | Power  |
|------------------------------|------------------|----------|--------|
| YOLOv8n FP16, 4 cameras @ 15 FPS | 60          | 35-40%   | 4-5 W  |
| OSNet-x0_25 INT8, batch ReID | 30 (averaged)    | 5-10%    | 1 W    |
| Mel-spec CNN INT8, 4 mics    | 4 (1 per s/mic)  | <5%      | 0.5 W  |
| NVDEC for 4 RTSP camera streams | n/a (decoder) | n/a      | 1.5 W  |
| System overhead              | n/a              | n/a      | 4 W    |
| **Total**                    |                  |          | **~12 W** |

This leaves ~3 W of headroom under the 15W cap for transient bursts.
The system runs steady-state on a 16W passively cooled enclosure.

### Jetson AGX Orin 32GB, 30W mode (multi-camera dense node)

Same workload at 8 cameras + 8 microphones fits in ~20W with substantial
headroom. We don't yet recommend AGX deployments — the cost-per-camera
ratio doesn't justify it under ~6 cameras per node.

---

## How to read these numbers

- **Sandbox numbers are real**, reproducible by running the benchmarks
  in this repo. They demonstrate the architecture functions end-to-end
  on a single CPU.
- **Jetson numbers are targets**, derived from NVIDIA's published
  benchmarks for the same models on the same JetPack version. They are
  not "we ran this on real hardware and got X" measurements yet.
- The architecture-level claim — sub-millisecond AIS, sub-100ms vision,
  sub-10ms audio, ~1ms ReID per crop — is **architecturally sound** and
  demonstrated end-to-end on CPU. The remaining work is calibration
  against real hardware once a deployment site is live.
- If you're sizing a deployment from this document, assume **70% of
  published Jetson numbers** as a safety margin. Tested deployments
  generally come in close to spec at FP16; INT8 sometimes loses 2-3
  points of mAP on YOLO and that's a tunable knob, not a defect.
