# TRITON

**Transit Risk Intelligence & Tactical Operations Network**

A multi-layer intelligence platform with two pillars:

1. **Convoy Escort** — strategic maritime decision-support for the Strait
   of Hormuz crisis. 44 modules (M1–M44) covering vessel detection,
   atmospheric ducting, mine probability, route optimization, insurance
   viability, AIP orchestrator, MARL swarm, XAI consensus, and full
   GO/NO-GO decision pipeline. FastAPI backend with 60 endpoints +
   Gotham-style operator dashboard.

2. **Edge Perimeter** — streaming sensor-fusion perimeter detection.
   M47 AIS, M48 multi-camera with YOLO + cross-camera ReID, M49 audio
   anomaly via mel-spec CNN, M50 cross-camera identity tracking, plus
   a production reliability layer.

Both pillars emit on the same alert schema and severity scale (1–5).
The convoy_escort M26 AIP orchestrator consumes evidence from any
sensor; edge_perimeter feeds M26 the same way internal modules do.

---

## Repo layout

```
triton/
├── README.md                         # this file (unified overview)
├── LICENSE, .gitignore, pyproject.toml, requirements.txt
│
├── convoy_escort/                    # Pillar 1 — strategic platform (v10)
│   ├── backend/
│   │   ├── main.py                   # FastAPI, 60 endpoints
│   │   ├── data_integration.py       # 13 connectors (NOAA, EIA,
│   │   │                             #   MarineTraffic, CENTCOM, ...)
│   │   ├── requirements.txt
│   │   └── models/                   # M1-M44 (28 .py files, ~9,500 LOC)
│   ├── triton-dashboard-v10.html     # Gotham-style operator dashboard
│   ├── update_march20.py             # Day-20 intelligence patch
│   ├── BUILD_PLAN.md, HCTIP_BUILD_PLAN.md
│   └── README.md                     # convoy escort detailed docs
│
├── (root)                            # Pillar 2 — Edge Perimeter (M47)
├── geofences.py                      # M47 — Persian Gulf zone library
├── perimeter_engine.py               # M47 — AIS streaming engine
├── ais_simulator.py                  # M47 — Gulf traffic generator
├── integration.py                    # AlertBus + AIP subscriber + FastAPI
├── benchmark.py                      # M47 standalone benchmark
├── BENCHMARK_REPORT.txt              # M47 canonical run output
│
├── edge/                             # Pillar 2 — Edge Perimeter (M48-M50)
│   ├── rtsp_ingest.py                # M48 RTSP grabber + synthetic stream
│   ├── person_detector.py            # M48 dual backend (YOLO ONNX / HOG)
│   ├── vision_perimeter.py           # M48 image-space fences + tracker
│   ├── multi_camera_manager.py       # M48 N-camera orchestrator
│   ├── audio_anomaly.py              # M49 features + rule classifier
│   ├── audio_cnn.py                  # M49 mel-spec CNN + dual-path fusion
│   ├── reid_embedder.py              # M50 OSNet + color-texture
│   ├── cross_camera_tracker.py       # M50 global ID across cameras
│   ├── reliability.py                # production reconnect / health / watchdog
│   ├── jetson_runtime.py             # production tuning + TRT engine builder
│   ├── edge_benchmark.py             # unified vision+audio+AIS benchmark
│   └── BENCHMARK_REPORT_EDGE.txt
│
├── docs/
│   ├── FALSE_POSITIVES.md            # honest catalogue of FP causes
│   └── ON_DEVICE_NUMBERS.md          # latency/throughput/power tables
│
└── tests/
    └── test_engines.py               # 15 smoke tests, run in CI
```

---

## Module map across both pillars

| # | Module | Pillar | Function |
|---|--------|--------|----------|
| M1 | VesselDetector | convoy | XGBoost AIS anomaly detection |
| M2 | Atmosphere | convoy | Smith-Weintraub tropospheric ducting |
| M3 | MineProbability | convoy | Bayesian 400-cell mine posterior |
| M4 | RouteOptimizer | convoy | Multi-objective A* convoy routing |
| M5 | Enforcement | convoy | Selective enforcement tracking |
| M6 | DisguisedVessel | convoy | Behavioural shadow-fleet analytics |
| M7 | RiskScore | convoy | Transit risk composite |
| M8 | ThreatEngagement | convoy | Per-waypoint engagement timeline |
| M9 | ConvoyScheduler | convoy | Throughput scheduling |
| M10–12 | ExecutionModules | convoy | Comms, MCM, adversary game theory |
| M13 | BabAlMandeb | convoy | Dual-chokepoint coordination |
| M14 | SubmarineThreat | convoy | Ghadir-class threat model |
| M15 | GpsWarfare | convoy | GPS spoof/denial zones |
| M16–17 | InsuranceTransit | convoy | P&I viability + IRGC permission |
| M18–20 | StrategicModules | convoy | Bypass, stranded vessels, coalition |
| M21–22 | StrikeDegradation | convoy | BDA + LUCAS arsenal |
| M23–25 | AdvancedModules | convoy | Cascade, reinforcement, targeting |
| M26 | AIP Orchestrator | convoy | 4-agent GO/NO-GO synthesis |
| M27–29 | Nexus / Quantum / NKE | convoy | Coalition + PNT + non-kinetic |
| M30–32 | WarpSpeed Foundry | convoy | Economic twin + industrial base |
| M33–34 | SwarmXAI | convoy | MARL swarm + XAI consensus |
| M35–40 | LastMile | convoy | Bio-acoustic, crew, ROE, MLS guard |
| M41–44 | BleedingEdge | convoy | Alignment, DOGE, regime, DAGIR |
| **M47** | **AIS Perimeter** | **edge** | **Streaming geofence engine** |
| **M48** | **Vision Perimeter** | **edge** | **RTSP + YOLO + tracker + fences** |
| **M49** | **Acoustic Anomaly** | **edge** | **Mel-spec CNN + rule fusion** |
| **M50** | **Cross-Camera ReID** | **edge** | **Embedding-based identity matching** |

The numbering gap (M45–46 not present here) is from the v11 work in
ceasefire_bypass.py which lives in a separate branch; the v10 trunk
imported here is the canonical platform.

---

## Quick start

### Edge perimeter (root + edge/)

```bash
pip install -r requirements.txt
python3 edge/edge_benchmark.py 100000 100 30
python3 -m pytest tests/ -v
```

Verified sandbox numbers in `BENCHMARK_REPORT_EDGE.txt`:

| Engine | P50 | Throughput |
|---|---|---|
| M47 AIS | 10.79 µs | 63K msg/s |
| M48 Vision | 65.74 ms (HOG/CPU) | 11 FPS — Jetson FP16 target ~140 FPS |
| M49 Audio | 6.45 ms (rule-based) | — |
| M50 ReID | 0.8-1.2 ms (color-texture) | — |

### Convoy escort (convoy_escort/)

```bash
cd convoy_escort
pip install -r backend/requirements.txt
cd backend && uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` for all 60 endpoints with Swagger UI,
and `convoy_escort/triton-dashboard-v10.html` for the operator console.

By default everything runs in **synthetic mode** — no API keys needed.
Set `TRITON_LIVE=true` and provide keys in `convoy_escort/backend/.env`
to switch any of the 13 connectors to live data. See
`convoy_escort/README.md` for the connector key list.

---

## How the pillars connect

```
                    ┌─────────────────────────────────┐
                    │   M26 AIP Orchestrator (convoy) │
                    │   GO / NO-GO scenario branching  │
                    └───────────────▲─────────────────┘
                                    │ unified Alert schema
                ┌───────────────────┼───────────────────┐
                │                   │                   │
        ┌───────┴────────┐  ┌───────┴────────┐  ┌──────┴──────┐
        │ Convoy modules │  │ M47 AIS engine │  │ M48 / M49 / │
        │   M1–M25,      │  │   geofences    │  │ M50 edge    │
        │   M27–M44      │  │   on streaming │  │ nodes per   │
        │                │  │   AIS          │  │ camera/mic  │
        └────────────────┘  └────────────────┘  └─────────────┘
                  ▲                  ▲                  ▲
                  │                  │                  │
            13 data           AIS feed (real      RTSP cameras
           connectors           or simulated)     + microphones
        (NOAA, EIA, etc.)
```

The convoy_escort M26 AIP orchestrator is the synthesis point. Edge
perimeter modules emit on the same `AlertBus` schema that M26 already
consumes, so they slot in as additional evidence sources without
rewiring the orchestrator.

---

## What's measured vs what's synthetic

**Honest framing — important to keep this clear in interviews:**

| Component | Status |
|---|---|
| Edge perimeter latency (M47 / M48 / M49 / M50) | Measured on sandbox CPU, reproducible with `edge/edge_benchmark.py` |
| Edge perimeter Jetson numbers | Targets from NVIDIA published benchmarks, not yet run on physical Orin |
| Audio CNN weights | Architecture only; sandbox ships with random weights, production deploys trained ONNX |
| OSNet ReID weights | Production path interface; sandbox uses color+texture fallback |
| Convoy escort (M1-M44) | Synthetic data by default; 13 connectors wired for live data, none verified end-to-end against live APIs in sandbox |
| Hormuz "March 20" intelligence in dashboard | Hand-curated from CNN/Reuters reporting at the time of v10 build |

The platform demonstrates the **architecture** and **decision logic**
end-to-end. The remaining work is calibration against real hardware
(Jetson) and live API integration testing.

---

## License

MIT. See `LICENSE`.
