# HORMUZ CORRIDOR TRANSIT INTELLIGENCE PLATFORM (HCTIP)
# Palantir FDSE Year Application — Build Plan
# Author: Nahom Woldegebriel

## ARCHITECTURE OVERVIEW

```
┌─────────────────────────────────────────────────────────┐
│                    FRONTEND (React + Mapbox)             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐  │
│  │ Vessel   │ │ Atmos    │ │ Mine     │ │ Escort    │  │
│  │ Map      │ │ Overlay  │ │ Heatmap  │ │ Route     │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └─────┬─────┘  │
│       └────────────┼────────────┼──────────────┘        │
│                    ▼                                     │
│            ┌──────────────┐                              │
│            │ Dashboard    │  Timeline / Risk Score / API │
│            └──────┬───────┘                              │
└───────────────────┼─────────────────────────────────────┘
                    │ REST API
┌───────────────────┼─────────────────────────────────────┐
│                   ▼  BACKEND (FastAPI)                   │
│  ┌─────────────────────────────────────────────────┐    │
│  │              API Gateway (FastAPI)               │    │
│  │  /vessels  /atmosphere  /mines  /route  /risk    │    │
│  └──┬──────────┬───────────┬────────┬────────┬─────┘    │
│     ▼          ▼           ▼        ▼        ▼          │
│  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐     │
│  │ M1   │  │ M2   │  │ M3   │  │ M4   │  │ M5-7 │     │
│  │Vessel│  │Atmos │  │Mine  │  │Route │  │Extra │     │
│  │Detect│  │Threat│  │Prob  │  │Optim │  │Mods  │     │
│  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘  └──┬───┘     │
│     ▼         ▼         ▼         ▼         ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │           DATA LAYER (SQLite + GeoJSON)          │    │
│  │  AIS data │ HRRR weather │ GEBCO depth │ etc    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## TECH STACK

- **Frontend:** React + TypeScript + Mapbox GL JS (free tier)
- **Backend:** FastAPI (Python)
- **ML Models:** XGBoost, scikit-learn (anomaly detection, classification)
- **Optimization:** Google OR-Tools (route optimization)
- **Data:** AIS (MarineTraffic CSV exports), HRRR/NAM (NOAA API), GEBCO (bathymetry), HYCOM (ocean currents), Global Fishing Watch
- **Maps:** Mapbox GL JS with custom layers
- **Deployment:** Local for demo (Docker optional)
- **Video:** Screen recording + voiceover

## DATA SOURCES (ALL PUBLIC / FREE)

1. **AIS Vessel Data**
   - MarineTraffic historical CSV exports (free tier gives Hormuz area)
   - UN Global Platform AIS data
   - Alternative: Generate realistic synthetic AIS data based on known traffic patterns

2. **Atmospheric / Weather**
   - NOAA HRRR (High-Resolution Rapid Refresh) — same API you used at T-Mobile
   - NOAA GFS for broader coverage
   - Compute: refractivity, K-factor, duct height, radio horizon extension

3. **Bathymetry (depth charts)**
   - GEBCO (General Bathymetric Chart of the Oceans) — free NetCDF download
   - Used for mine probability (mines deployed at specific depths)

4. **Ocean Currents**
   - HYCOM (Hybrid Coordinate Ocean Model) — free
   - Used for mine drift prediction

5. **Fishing Vessel Baselines**
   - Global Fishing Watch — free API
   - Establishes "normal" fishing behavior for disguised vessel detection

6. **Historical Mine Data**
   - ONI (Office of Naval Intelligence) reports on 1988 Tanker War mine placements
   - Publicly documented chokepoints and mining patterns

## BUILD PLAN

### PHASE 1: Data Pipeline + Core Models (Day 1-2)

**Day 1 — Data Ingestion & Preprocessing**
- [ ] Set up project structure (monorepo: /backend, /frontend, /data, /models)
- [ ] Download/generate AIS data for Strait of Hormuz area
- [ ] Download HRRR atmospheric data for Persian Gulf region
- [ ] Download GEBCO bathymetry for strait
- [ ] Build data preprocessing pipeline (pandas + geopandas)
- [ ] Create SQLite database with vessel_tracks, weather_forecasts, bathymetry tables

**Day 2 — ML Models**
- [ ] Module 1: Train XGBoost vessel anomaly detector
  - Features: speed, heading, heading_variance, distance_to_shipping_lane,
    ais_gap_duration, time_of_day, vessel_type, proximity_to_known_threats
  - Labels: normal_transit, suspicious, hostile_pattern
  - Port your T-Mobile anomaly detection code — same architecture
- [ ] Module 2: Atmospheric propagation model
  - Port your T-Mobile tropoducting code
  - Input: HRRR refractivity profiles over strait
  - Output: radar_range_extension_factor, duct_probability, sensor_effectiveness_score
  - 48-hour forecast grid over strait geography
- [ ] Module 3: Bayesian mine probability model
  - Priors: depth (GEBCO), distance to shipping lane, historical mine locations
  - Update: ocean current drift (HYCOM), known mine-layer tracks
  - Output: probability grid over strait (lat/lon cells)

### PHASE 2: Backend API (Day 3)

- [ ] FastAPI app with endpoints:
  ```
  GET  /api/vessels          — current vessel positions + classifications
  GET  /api/vessels/anomalies — flagged suspicious contacts
  GET  /api/atmosphere       — 48hr atmospheric threat overlay
  GET  /api/mines            — mine probability heatmap
  POST /api/route/optimize   — compute optimal escort route
  GET  /api/risk/score       — insurance risk score for given transit params
  GET  /api/enforcement      — selective enforcement tracker (flag state analysis)
  ```
- [ ] Module 4: Route optimizer endpoint
  - Inputs: origin, destination, escort_assets, time_window
  - Constraints: avoid high mine-prob cells, prefer low-atmospheric-threat windows,
    route around hostile contacts, stay within escort defensive envelope
  - Use OR-Tools constrained optimization (port from T-Mobile drive test tool)
  - Output: route (list of waypoints), departure_time, threat_exposure_score, confidence
- [ ] Module 7: Risk score API
  - Inputs: flag_state, vessel_type, transit_time, escort_level
  - Output: risk_score (0-100), confidence_interval, contributing_factors

### PHASE 3: Frontend Dashboard (Day 4-5)

- [ ] React app with Mapbox GL JS
- [ ] Map layers:
  1. **Base layer:** Strait of Hormuz satellite imagery
  2. **Vessel layer:** Colored dots (green=friendly, yellow=unknown, red=hostile)
     with popup showing vessel details + anomaly score
  3. **Atmospheric overlay:** Color-coded grid showing Iranian sensor effectiveness
     (red=extended range/high threat, blue=degraded/safe window)
  4. **Mine heatmap:** Semi-transparent probability overlay
     (red=high probability, transparent=low)
  5. **Escort route:** Animated line showing optimal transit path
  6. **Selective enforcement:** Side panel showing flag-state transit history
- [ ] Dashboard panels:
  - 48-hour atmospheric threat timeline (chart)
  - Vessel classification summary (counts by type)
  - Risk score calculator (interactive form)
  - "Atmospheric Advantage Windows" — highlighted time blocks when conditions favor transit
- [ ] Responsive design for demo recording

### PHASE 4: Integration + Polish (Day 6)

- [ ] Connect frontend to backend API
- [ ] Add real-time simulation mode (replay AIS data at accelerated speed)
- [ ] Add "Escort Planning" mode — click to set origin/destination, see route computed
- [ ] Polish UI: Palantir-inspired dark theme, clean typography
- [ ] Test full workflow end-to-end
- [ ] Generate demo scenarios:
  - Scenario A: High-threat window (ducting event + multiple hostile contacts)
  - Scenario B: Atmospheric advantage window (sensors degraded, mines avoided)

### PHASE 5: Video Recording (Day 7)

- [ ] Script the 3-minute demo narration (see below)
- [ ] Screen record the dashboard walkthrough
- [ ] Record voiceover (raw, authentic — they said "drop the script")
- [ ] Upload as unlisted YouTube video
- [ ] Also record Prompt 1 and Prompt 2 videos if doing Option 2

## 3-MINUTE DEMO SCRIPT

```
[0:00 - 0:20] HOOK
"It's March 2026. The Strait of Hormuz is a kill box. Iran is using mines,
drone boats disguised as fishing vessels, and encrypted drone swarms to block
20% of global oil. The US Navy says they're 'not ready' to escort tankers.
The reason: there's no integrated tool that fuses maritime, atmospheric, and
subsurface threat data into one decision. I built one."

[0:20 - 0:50] MODULE 1: VESSEL DETECTION
[Show map with AIS data flowing in, vessel dots color-coded]
"Live AIS data classifies every contact in the strait. My anomaly detector
— the same XGBoost architecture I deployed at T-Mobile across 100,000 sites —
flags vessels with suspicious behavior: AIS dark periods, speed inconsistent
with fishing, formation patterns matching Iranian fast-boat swarms.
This one [highlight red dot] is broadcasting as a fishing vessel but its
heading variance and proximity to the shipping lane say otherwise."

[0:50 - 1:30] MODULE 2: ATMOSPHERIC THREAT OVERLAY
[Toggle atmospheric layer on map — color-coded grid appears]
"This is my unfair advantage. I hold 4 patents in RF propagation prediction.
This layer shows when atmospheric ducting conditions extend Iranian coastal
radar and missile seeker range — and when they're degraded.
[Point to red zone] Right now, ducting over the northern channel extends
Iranian Noor missile seeker range by 40%.
[Point to blue zone] But in 14 hours, conditions flip. This window
[highlight on timeline] is an atmospheric advantage — Iranian sensors degraded,
friendly radar enhanced. This layer doesn't exist in any current military tool."

[1:30 - 2:00] MODULE 3: MINE MAP + MODULE 4: ROUTE
[Toggle mine heatmap — show probability overlay]
"Bayesian mine probability map. Priors from depth charts, historical mining
patterns from the 1988 Tanker War, and tracked mine-layer positions.
Updates continuously as mines are found or mine-layers destroyed.
[Click 'Plan Escort' button — animated route appears]
The optimizer computes the safest route: southern channel, depart at 0400
during the atmospheric advantage window, avoid these high-probability mine
cells, route around the flagged contacts. Threat exposure score: 23 out of
100. Confidence: 78%."

[2:00 - 2:30] MODULE 5 + 7: ENFORCEMENT + INSURANCE API
[Show side panel with flag-state data]
"Iran is selectively allowing Chinese and Indian ships through while blocking
Western allies. This tracker reveals the pattern — which flags, what times,
what conditions. And here [show API response] — the same threat data exposed
as an API. Lloyd's of London queries this to price transit risk in real-time.
That's not just a military tool — that's a commercial product."

[2:30 - 3:00] CLOSE
"I built 2 commercial products at T-Mobile that are now sold to SpaceX,
Ericsson, and Nokia. I built this prototype in [X] days using public data.
With Palantir's Gotham, real DoD sensor feeds, and classified threat data,
this becomes the operational tool that reopens Hormuz.
I have 4 patents in this exact domain. I want to build it for real."
```

## FILE STRUCTURE

```
hormuz-platform/
├── README.md
├── docker-compose.yml
├── backend/
│   ├── main.py                 # FastAPI app
│   ├── requirements.txt
│   ├── models/
│   │   ├── vessel_detector.py  # Module 1: XGBoost anomaly detection
│   │   ├── atmosphere.py       # Module 2: Atmospheric propagation
│   │   ├── mine_probability.py # Module 3: Bayesian mine map
│   │   ├── route_optimizer.py  # Module 4: Escort route optimization
│   │   ├── enforcement.py      # Module 5: Selective enforcement tracker
│   │   ├── disguised_vessel.py # Module 6: Fishing boat anomaly detector
│   │   └── risk_score.py       # Module 7: Insurance risk API
│   ├── data/
│   │   ├── ingest_ais.py       # AIS data pipeline
│   │   ├── ingest_weather.py   # HRRR/NAM data pipeline
│   │   ├── ingest_bathymetry.py # GEBCO depth data
│   │   └── generate_synthetic.py # Synthetic data for demo
│   └── db/
│       └── schema.sql          # SQLite schema
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/
│   │   │   ├── Map.tsx         # Mapbox GL map with all layers
│   │   │   ├── VesselLayer.tsx # Vessel dots + popups
│   │   │   ├── AtmosOverlay.tsx # Atmospheric threat grid
│   │   │   ├── MineHeatmap.tsx # Mine probability overlay
│   │   │   ├── EscortRoute.tsx # Animated route line
│   │   │   ├── Timeline.tsx    # 48hr atmospheric forecast chart
│   │   │   ├── RiskPanel.tsx   # Risk score calculator
│   │   │   └── EnforcementPanel.tsx # Flag state tracker
│   │   └── api/
│   │       └── client.ts       # API client
│   └── public/
│       └── index.html
├── data/
│   ├── ais/                    # AIS CSV exports
│   ├── weather/                # HRRR GRIB files
│   ├── bathymetry/             # GEBCO NetCDF
│   └── reference/              # ONI reports, historical mine locations
└── models/
    ├── vessel_anomaly.joblib   # Trained XGBoost model
    ├── fishing_classifier.joblib
    └── atmosphere_model.joblib
```

## KEY DECISIONS

1. **Synthetic vs Real Data:** Use real GEBCO bathymetry and HRRR weather.
   Generate synthetic AIS data that mimics real Hormuz traffic patterns
   (easier than dealing with MarineTraffic rate limits for demo).

2. **Atmospheric Model:** Port directly from T-Mobile tropoducting code.
   Same physics (Modified Paulus-Jeske model or equivalent), different geography.

3. **Route Optimization:** Port from T-Mobile drive test tool. Replace road
   network with maritime grid. Replace drive-time constraints with
   threat-exposure constraints.

4. **Demo Mode:** Pre-compute scenarios so the demo runs smoothly.
   Real-time computation is impressive but risky for a 3-min video.

5. **Palantir Framing:** Even though you're building standalone, frame
   everything as "ready to deploy on Foundry/Gotham." Use their terminology:
   "ontology" for your data model, "actions" for your API endpoints,
   "operational view" for your dashboard.
