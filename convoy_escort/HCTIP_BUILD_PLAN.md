# Hormuz Corridor Transit Intelligence Platform (HCTIP)
## Complete Build Plan — 13 Modules, 5 Phases

**Author:** Nahom Woldegebriel  
**Date:** March 18, 2026  
**Status:** Backend complete, all 13 modules tested ✓  
**Target:** Palantir FDSE Year Program — Option 1 (Build)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    HCTIP Platform                        │
├─────────────────────────────────────────────────────────┤
│  PHASE 1: INTELLIGENCE        PHASE 2: PLANNING        │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ M1: Dark Vessel      │    │ M4: Route Optimizer  │   │
│  │ M2: Atmospheric      │    │ M7: Insurance Risk   │   │
│  │ M3: Mine Probability │    │     Score API        │   │
│  │ M5: Selective Enforce │    └──────────────────────┘   │
│  │ M6: Disguised Vessel │                               │
│  └──────────────────────┘                               │
├─────────────────────────────────────────────────────────┤
│  PHASE 3: EXECUTION          PHASE 4: SUSTAINMENT       │
│  ┌──────────────────────┐    ┌──────────────────────┐   │
│  │ M8: Threat Engage    │    │ M9: Convoy Scheduler │   │
│  │ M10: Comms Resilience│    │                      │   │
│  │ M11: MCM Formation   │    └──────────────────────┘   │
│  │ M12: Adversary Model │                               │
│  └──────────────────────┘                               │
├─────────────────────────────────────────────────────────┤
│  PHASE 5: STRATEGIC                                     │
│  ┌──────────────────────────────────────────────────┐   │
│  │ M13: Bab al-Mandeb / Houthi Threat Extension     │   │
│  │      Combined chokepoint risk (Hormuz + Red Sea) │   │
│  │      Insurance routing: Suez vs Cape of Good Hope│   │
│  └──────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────┤
│           /api/mission/plan (Master Endpoint)           │
│              Integrates all 12 modules                  │
└─────────────────────────────────────────────────────────┘
```

---

## Module Summary

### Phase 1 — Intelligence Layer

| Module | Name | Method | Output |
|--------|------|--------|--------|
| M1 | Dark Vessel Detection | XGBoost anomaly scoring (7 factors: AIS gap, speed delta, flag risk, proximity, heading variance, draught mismatch, name changes) | 108 vessels tracked, 9 anomalies flagged |
| M2 | Atmospheric Threat Overlay | Smith-Weintraub refractivity → M-profile → K-factor ducting computation (from T-Mobile patents on tropospheric propagation) | 3 advantage windows in 48hrs, duct strength 0.95 |
| M3 | Bayesian Mine Probability | Prior (depth × history × mine-layer tracks) × likelihood updates → posterior grid | 400 cells, max P=43.6% |
| M5 | Selective Enforcement Tracker | Real-world data: which flags Iran allows vs attacks | CN/IN/TR allowed, US/MH/LR blocked |
| M6 | Disguised Vessel Detector | Fishing boat anomaly detection (speed, AIS pattern, proximity to shipping lanes) | Flags IRGC suicide boats disguised as fishing vessels |

### Phase 2 — Planning Layer

| Module | Name | Method | Output |
|--------|------|--------|--------|
| M4 | Escort Route Optimizer | 3 candidate routes (northern/central/southern), constrained optimization over threat + atmospheric + mine layers | Northern route, risk 74.1/100, confidence 73% |
| M7 | Insurance Risk Score API | Composite: flag_risk × atmospheric × mine × contacts × escort_modifier | US/VLCC/DDG escort: 32.0 — "TRANSIT FEASIBLE" |

### Phase 3 — Execution Layer

| Module | Name | Method | Output |
|--------|------|--------|--------|
| M8 | Threat Engagement Timeline | Simulates inbound weapons at each waypoint: launch origin, weapon type, time-to-impact, defensive system selection, intercept probability | Survivability 0.898, 3 critical waypoints |
| M9 | Convoy Throughput Scheduler | Asset rotation (8 DDGs, 3 LCS), atmospheric window constraints, mine clearance progress → max daily throughput | 2 convoys/day, 6 tankers, 2.4M bbl/day (12% pre-war) |
| M10 | Communications Resilience | C2 link availability at each waypoint: MUOS, AEHF, Starshield, shore relays (degraded), CVN relay, E-2D airborne | All waypoints HIGH resilience (satellite-dominant) |
| M11 | MCM Formation Planner | Mine sweep scheduling: LCS with AN/AQS-20A + MH-60S helos sweep ahead of convoy, 3-pass cumulative detection 97.3% | 7 high-risk segments, 6.4h sweep time |
| M12 | Adversary Reaction Model | Game-theoretic prediction: 6 Iranian response options, probability adjusted by escort strength, atmospheric conditions, remaining Iranian capacity | Swarm attack 52%, selective targeting 46% |

### Phase 4 — Sustainment

Convoy scheduler also answers the strategic question: **when does oil flow resume?**
- Escort operations feasible: ~March 28 (10 days)
- Independent transit safe: ~March 31 (13 days)
- Assumes attrition +3%/day, mine clearance +5%/day

### Phase 5 — Strategic

| Module | Name | Method | Output |
|--------|------|--------|--------|
| M13 | Bab al-Mandeb Threat Extension | Houthi attack capability modeling (ASBM, ASCM, drones, USVs, mines) × 6 launch zones × flag-based targeting. Combined chokepoint risk: P(incident) = 1-(1-P_hormuz)(1-P_redsea). Insurance premium comparison: Suez vs Cape routing. | Red Sea max risk 100/100 for US-flagged. Combined Suez risk 100%. Cape alternative: 34.5% risk, $3.78M insurance savings. |

---

## 17 Core Calculations

1. **Smith-Weintraub Refractivity** (M2) — N = 77.6(P/T) + 3.73×10⁵(e/T²)
2. **Modified Refractivity M-Profile** (M2) — M(z) = N(z) + 157z, detect ducting
3. **K-Factor** (M2) — K = 1/(1 - R·dN/dh), radar propagation factor
4. **Bayesian Posterior** (M3) — P(mine|obs) ∝ P(obs|mine) × P(mine), grid update
5. **XGBoost Anomaly Score** (M1) — 7-feature ensemble, threshold 0.7
6. **Route Risk Optimization** (M4) — weighted sum over threat/atmospheric/mine layers per waypoint
7. **Composite Insurance Risk** (M7) — multiplicative model: flag × atmos × mine × contacts × escort
8. **Time-to-Impact** (M8) — distance / (Mach × 343 m/s), per weapon per waypoint
9. **Defensive System Selection** (M8) — optimal weapon-defense pairing by reaction window
10. **Survivability Estimation** (M8) — defend_ratio × 0.95, capped at 0.99
11. **Convoy Throughput** (M9) — min(DDG_limited, window_limited, MCM_limited, air_limited)
12. **Cumulative Mine Detection** (M11) — 1 - (1 - P_detect)^passes
13. **Adversary Probability Adjustment** (M12) — base × capability_remaining × strategic_factors
14. **Comms Bandwidth Assessment** (M10) — sum of available links, partitioned by jam-resistance
15. **Combined Chokepoint Risk** (M13) — P(incident) = 1 - (1-P_hormuz)(1-P_redsea)
16. **Flag-Based Targeting Model** (M13) — Houthi prioritization: US/UK/IL ×2.0, CN/RU ×0.3
17. **War Risk Insurance Premium** (M13) — risk_score → premium % of hull value → Suez vs Cape cost comparison

---

## Data Sources (6)

1. **AIS Vessel Tracking** — Marine Traffic / Global Fishing Watch
2. **HRRR/NAM Weather** — NOAA (temperature, humidity, pressure profiles)
3. **GEBCO Bathymetry** — Ocean depth grid
4. **HYCOM Ocean Currents** — Surface current vectors
5. **Global Fishing Watch** — Fishing vessel patterns (anomaly baseline)
6. **ONI/CSIS Historical Reports** — Mine deployment history, attack patterns

---

## API Endpoints

| Endpoint | Method | Module(s) |
|----------|--------|-----------|
| `/api/vessels` | GET | M1 |
| `/api/vessels/anomalies` | GET | M1 |
| `/api/atmosphere/overlay` | GET | M2 |
| `/api/atmosphere/windows` | GET | M2 |
| `/api/mines/grid` | GET | M3 |
| `/api/route/optimize` | POST | M4 |
| `/api/enforcement/patterns` | GET | M5 |
| `/api/disguised/flagged` | GET | M6 |
| `/api/risk/score` | POST | M7 |
| `/api/engagement/simulate` | POST | M8 |
| `/api/convoy/schedule` | GET | M9 |
| `/api/comms/assess` | POST | M10 |
| `/api/mcm/plan` | POST | M11 |
| `/api/adversary/predict` | POST | M12 |
| `/api/redsea/assess` | GET | M13 |
| `/api/chokepoints/combined` | POST | M13 + M7 |
| `/api/mission/plan` | POST | ALL 13 |
| `/api/situational-picture` | GET | M1-7 |

---

## Test Output (March 18, 2026)

```
═══════════════════════════════════════════════
  HCTIP FULL MISSION PLAN — ALL 13 MODULES
═══════════════════════════════════════════════

PHASE 1 — INTELLIGENCE (Modules 1-3, 5-6)
  Vessel picture: 108 total, 9 anomalies
  Atmosphere: duct=0.95, 3 advantage windows
  Mine map: 400 cells, max P=0.4365
  Enforcement: 5 flags allowed

PHASE 2 — PLANNING (Modules 4, 7)
  Route: northern, risk 74.1/100, confidence 73%
  Departure: 2026-03-18T00:00:00

PHASE 3 — EXECUTION (Modules 8-12)
  [M8]  Survivability: 0.898, 3 critical waypoints
        TRANSIT HIGH RISK — Recommend additional escorts
  [M10] COMMS ADEQUATE — jam-resistant links available
  [M11] MCM: 7 high-risk segments, sweep 6.4h
  [M12] Adversary: Swarm attack (52%)
        Counter: DDG defensive fire + helicopter standoff

PHASE 4 — SUSTAINMENT (Module 9)
  Convoys/day: 2
  Tankers/day: 6
  Oil: 2.4M bbl/day
  Pre-war restored: 12.0%
  Binding constraint: Atmospheric advantage windows
  Escort ops feasible: March 28
  Independent transit: March 31

PHASE 5 — STRATEGIC (Module 13)
  Red Sea max risk: 100/100 (US-flagged, unescorted)
  Critical zones: 8 of 10 waypoints
  Hormuz+Suez combined risk: 100%
  Hormuz+Cape combined risk: 34.5%
  Insurance: Suez $3,900,000 vs Cape $120,000
  RECOMMEND CAPE ROUTING for Europe-bound tankers

═══════════════════════════════════════════════
  ALL 13 MODULES OPERATIONAL ✓
  6/6 GAPS FILLED ✓
═══════════════════════════════════════════════
```

---

## 3-Minute Demo Script

**[0:00-0:20] Hook**
"It's March 18, 2026. The US has been at war with Iran for 18 days. The Strait of Hormuz is closed. 20 million barrels a day aren't flowing. Oil is above $95. The President promised tanker escorts. Zero have happened. I built the tool that makes them possible."

**[0:20-0:50] Intelligence Picture**
Show map with all 6 data layers lighting up. "Module 1 tracks 108 vessels — 9 flagged as potential threats. Module 2 uses atmospheric physics from my T-Mobile patents — the same tropoducting math I built for satellite interference — to find windows where Iranian radar is degraded. Module 3 builds a Bayesian mine map with 43% probability hotspots."

**[0:50-1:20] Planning**
Route appears on map. "Module 4 optimizes three routes. Northern is best — risk 74 out of 100 with a DDG escort. Module 5 shows Iran's selective enforcement — China and India pass through freely. US and Marshall Islands get attacked. Module 7 scores insurance risk for Lloyd's underwriters."

**[1:20-2:00] Execution — The New Modules**
Threat timeline animates along route. "This is what happens during the transit. Module 8 simulates every Iranian weapon that can reach each waypoint — Noor missiles from Bandar Abbas, Shahed drones from Qeshm, fast boats from the IRGCN flotilla — and maps each to a defensive response. SM-2 for missiles, SeaRAM for drones, Hellfire for boats. Survivability: 89.8%. Module 10 checks comms — satellite links hold, shore relays are degraded from Iranian strikes. Module 11 positions mine sweepers 5 nautical miles ahead with a 6.4-hour pre-sweep. Module 12 predicts Iran's response using game theory — 52% chance of a swarm attack, and here's the counter."

**[2:00-2:30] Sustainment**
Dashboard shows throughput numbers. "The strategic question: when does oil flow again? Module 9 says with 8 destroyers rotating 2-at-a-time escorts during atmospheric windows, CENTCOM pushes 6 tankers per day — 2.4 million barrels, 12% of pre-war traffic. That matches CNN's analysis. Full escort operations feasible by March 28. Independent transit by March 31."

**[2:30-2:50] Strategic — The Second Chokepoint**
"But here's what nobody else is modeling. Ships that clear Hormuz bound for Europe still have to transit the Red Sea through Bab al-Mandeb — where Houthis have attacked 130 ships since 2023. Module 13 extends the platform to cover both chokepoints. For a US-flagged VLCC: combined Hormuz+Suez risk is 100%. Cape of Good Hope routing drops it to 34.5% — and saves $3.78 million in war risk insurance. This is the analysis Lloyd's underwriters and CENTCOM planners both need."

**[2:50-3:00] Close**
"13 modules. 17 core calculations. 6 data sources. Two chokepoints. One API call — `/api/mission/plan` — returns the complete escort plan. I built this in [X] days. With Palantir's Gotham, real AIS feeds, and classified sensor data, this becomes the operational system that reopens the world's most dangerous waterways."

---

## File Structure

```
hormuz-platform/
├── backend/
│   ├── main.py                      # FastAPI app, all endpoints
│   ├── models/
│   │   ├── vessel_detector.py       # M1: Dark vessel detection
│   │   ├── atmosphere.py            # M2: Atmospheric threat overlay
│   │   ├── mine_probability.py      # M3: Bayesian mine map
│   │   ├── route_optimizer.py       # M4: Escort route optimizer
│   │   ├── enforcement.py           # M5, M6, M7: Enforcement + disguised + risk
│   │   ├── disguised_vessel.py      # Re-export from enforcement
│   │   ├── risk_score.py            # Re-export from enforcement
│   │   ├── threat_engagement.py     # M8: Threat engagement timeline
│   │   ├── convoy_scheduler.py      # M9: Convoy throughput scheduler
│   │   ├── execution_modules.py     # M10, M11, M12: Comms, MCM, adversary
│   │   └── bab_al_mandeb.py         # M13: Houthi / Red Sea threat + combined chokepoint
│   └── requirements.txt
├── frontend/                        # TODO: React + Mapbox GL JS
└── HCTIP_BUILD_PLAN.md              # This file
```

---

## Next Steps

- [ ] Build frontend (React + Mapbox GL JS) — map with vessel/atmospheric/mine/route/engagement layers
- [ ] Pre-compute demo scenarios (high-threat vs advantage window)
- [ ] Record 3-minute YouTube demo (unlisted)
- [ ] Submit to Palantir
