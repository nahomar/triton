# TRITON v10 — Transit Risk Intelligence & Tactical Operations Network

**Maritime convoy escort planning platform for the Strait of Hormuz crisis**
**March 20, 2026 — Day 20 of Operation Epic Fury**

Built by Nahom Woldegebriel | ASU CS+Stats | 4 Patents | NeurIPS 2025
Palantir FDSE Application Demo

---

## What This Is

TRITON is a 44-module intelligence platform that fuses military survivability,
insurance viability, economic cascade risk, and coalition politics into a single
GO/NO-GO recommendation for maritime convoy escort through the Strait of Hormuz.

**Key finding:** The AIP (AI Planning) engine returns NO-GO — not because military
risk is too high (89.8% survivability), but because all P&I insurance clubs have
cancelled Gulf coverage. No ship sails without coverage regardless of military escort.
The DIA confirms Iran can hold the strait for 1-6 months. The binding constraint
is commercial, not kinetic.

This insight — that the insurance gap, not Iranian missiles, is the actual blocker —
was identified by TRITON on Day 18. CNN confirmed the Pentagon reached the same
conclusion on Day 20 (March 20, 2026).

---

## Architecture

```
44 Modules | 60 API Endpoints | 8,154 Lines | 13 Data Connectors
7 Foundry Ontology Types | 5 AIP Logic Functions
```

### 12 Operational Phases

| Phase | Modules | Description |
|-------|---------|-------------|
| P1 Intelligence | M1-M6 | Vessel detection, atmospheric ducting, mine mapping, enforcement, disguised vessels |
| P2 Planning | M4, M7 | Route optimization, insurance risk scoring |
| P3 Execution | M8-M12 | Threat engagement, convoy scheduling, comms, MCM, adversary game theory |
| P4 Sustainment | M9 | Convoy throughput scheduling |
| P5 Strategic | M13 | Bab al-Mandeb dual chokepoint, Cape vs Suez |
| P6 Domain | M14-M15 | Submarine threat (Ghadir-class), GPS/GNSS warfare |
| P7 Commercial | M16-M17 | Insurance viability, IRGC permission transit |
| P8 Context | M18-M20 | Bypass pipelines, stranded vessels, coalition planning |
| P9 Strike | M21-M22 | Strike degradation, LUCAS arsenal model |
| P10 Economic | M23-M25 | Bypass targeting, production cascade, reinforcement |
| P11 Predictive | M26-M34 | AIP orchestrator, Monte Carlo, MARL swarm, XAI consensus |
| P12 Platform | M35-M44 | Bio-acoustic, crew readiness, ROE, adversarial filter, DOGE, regime model |

### Data Integration Layer — 13 Connectors

Every module pulls data through a connector with the same interface:
- `connector.fetch()` → routes to live or synthetic based on `TRITON_LIVE` env var
- `connector.fetch_live()` → real API call
- `connector.fetch_synthetic()` → current hardcoded data (default)

Set `TRITON_LIVE=true` and provide API keys to switch any connector to live data.
Connectors that fail gracefully fall back to synthetic.

---

## Quick Start

### 1. Clone and install

```bash
tar -xzf triton-v10-mar20.tar.gz
cd triton-platform
pip install -r backend/requirements.txt
```

### 2. Run the API server (synthetic mode — no keys needed)

```bash
cd backend
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000/docs` — all 60 endpoints with Swagger UI.

### 3. Open the dashboard

Open `triton-dashboard-v10.html` in any browser. The Leaflet map with
CartoDB dark_matter tiles loads immediately. All 4 toolbar modes work:
Layers → Objects → Analysis → Timeline.

### 4. Run data integration test

```bash
python backend/data_integration.py
```

Shows all 13 connectors in SYNTHETIC mode with status for each.

---

## Going Live — API Setup Instructions

### Tier 1: Free, No Key Required

These work immediately on any internet-connected machine.

#### NOAA HRRR (Atmospheric Ducting — M2)

```bash
# No key needed. HRRR is free public data from NOAA NOMADS.
# The connector fetches GRIB2 files and computes Smith-Weintraub refractivity.
# Install GRIB parser:
pip install pygrib

# Test manually:
curl "https://nomads.ncep.noaa.gov/cgi-bin/filter_hrrr_2d.pl?\
file=hrrr.t12z.wrfsfcf00.grib2&\
lev_2_m_above_ground=on&var_TMP=on&var_RH=on&\
subregion=&leftlon=54&rightlon=58&toplat=28&bottomlat=25&\
dir=%2Fhrrr.20260320%2Fconus" -o hormuz_atmo.grib2
```

#### World Bank API (Economic Data — M30)

```bash
# No key needed. Free public API.
# Test:
curl "https://api.worldbank.org/v2/country/IRN;SAU;ARE;QAT/indicator/NY.GDP.MKTP.CD?format=json&date=2025"
```

#### USASpending.gov (DOGE Fiscal — M42)

```bash
# No key needed. Federal spending data.
# Test:
curl "https://api.usaspending.gov/api/v2/agency/097/awards/"
```

#### GPSJam.org (GPS Interference — M15)

```bash
# No key needed. Crowdsourced GPS interference from ADS-B data.
# Open in browser: https://gpsjam.org/?lat=26.5&lon=56.3&z=8
```

#### CENTCOM Press Releases (Strike BDA — M21)

```bash
# No key needed. Public press releases.
# In production: RSS scrape + NLP extraction.
curl "https://www.centcom.mil/MEDIA/PRESS-RELEASES/"
```

### Tier 2: Free With API Key (5 Minutes to Set Up)

#### Yahoo Finance — Oil Prices (M7, M16, M24, M30, M42)

```bash
# No key needed for basic quotes. The connector uses the free chart endpoint.
# Test:
curl "https://query1.finance.yahoo.com/v8/finance/chart/BZ=F?interval=1d&range=1mo"

# For more reliable access, use Alpha Vantage:
# Sign up: https://www.alphavantage.co/support/#api-key (free, 25 req/day)
export ALPHA_VANTAGE_KEY=your_key_here
```

#### EIA API — Energy Production Data (M18, M23, M24)

```bash
# Sign up: https://www.eia.gov/opendata/register.php (free, instant)
export EIA_API_KEY=your_key_here

# Test:
curl "https://api.eia.gov/v2/petroleum/pri/spt/data/?api_key=$EIA_API_KEY&frequency=daily&data[0]=value&facets[series][]=RBRTE&sort[0][column]=period&sort[0][direction]=desc&length=30"
```

#### OpenWeather (Supplementary Atmospheric — M2)

```bash
# Sign up: https://openweathermap.org/api (free tier, 1000 calls/day)
export OPENWEATHER_API_KEY=your_key_here

# Test (Strait of Hormuz):
curl "https://api.openweathermap.org/data/2.5/weather?lat=26.5&lon=56.3&appid=$OPENWEATHER_API_KEY"
```

### Tier 3: Paid / Enterprise

#### MarineTraffic (AIS Vessel Tracking — M1, M5, M6, M19)

```
Sign up: https://www.marinetraffic.com/en/ais-api-services
Tier needed: PS07 (Extended Ship Info) — starts at ~$50/month
                                                          
export MARINETRAFFIC_API_KEY=your_key_here

# The connector uses:
# GET /exportvessel/v:8/{KEY}/MINLAT:25/MAXLAT:28/MINLON:54/MAXLON:58/
#     timespan:60/msgtype:extended/protocol:jsono

# Returns: MMSI, IMO, SHIP_NAME, LAT, LON, SPEED, HEADING, FLAG, SHIPTYPE
# Rate limit: 1 request/second
# Cost: ~$0.05 per vessel per query
```

#### Windward AI (Maritime Intelligence — M1, M5, M6)

```
Enterprise only. Contact: https://www.windward.ai/
Provides: Behavioral analytics, sanctions screening, dark fleet detection
The Windward data is what powers the XGBoost anomaly detector in M1.
Their API returns risk scores, AIS gap analysis, and vessel behavioral profiles.
No public pricing — enterprise contract required.
```

#### Planet Labs (SAR Imagery — M3, M11, M38)

```
Sign up: https://www.planet.com/products/
Tier needed: Planet Tasking (SkySat) or Archive (SuperDove)
Cost: $5K-50K/month depending on area and frequency

export PLANET_API_KEY=your_key_here

# The connector uses:
# POST /data/v1/quick-search
# Body: item_types=["SkySatCollect"], geometry=Hormuz polygon, cloud_cover<0.1
# Returns: image IDs for download, then run change detection vs pre-war baseline

# For mine detection: compare pre-Feb-28 imagery vs current.
# Objects >2m appearing in shipping lanes = probable mine.
# Requires image processing pipeline (OpenCV or similar).
```

#### Lloyd's List Intelligence (Insurance — M7, M16)

```
Enterprise subscription: https://www.lloydslist.com/
No public API. Data available via:
1. Lloyd's List Intelligence portal (web scraping with subscription)
2. Bloomberg Terminal: BWRI <GO> for war risk index
3. Reuters Eikon: <0#WARRISK> for war risk premiums

export BLOOMBERG_API_KEY=your_key_here  # If using Bloomberg B-PIPE

# Alternative free signal: monitor UKMTO advisories
# https://www.ukmto.org/ — public maritime security advisories
```

### Tier 4: Classified (Requires Security Clearance + Network)

These connectors only activate when deployed on classified networks.

#### CENTCOM BDA (M8, M12, M21, M29, M37)

```
Network: SIPR (SECRET)
Access: Requires TS/SCI clearance + CENTCOM J2 read access
Protocol: REST API on SIPR enclave or STANAG 4559 message bus

export SIPR_ACCESS=true  # On classified terminal only

# Provides: Real-time battle damage assessment, weapon expenditure,
# target status, confirmed kills vs unconfirmed
```

#### Five Eyes CENTRIX / Stone Ghost (M14, M20, M27, M35, M40)

```
Network: JWICS or CENTRIX (TS/SCI/FVEY)
Access: Requires TS/SCI + FVEY access + Project Overmatch enrollment

export JWICS_ACCESS=true
export CENTRIX_ACCESS=true

# Provides: Coalition sonar fusion (5,800km²), multistatic submarine tracks,
# allied air detection, shared targeting data
# Protocol: NATO FMN (Federated Mission Networking) standards
```

---

## Activating Live Mode

Once you have API keys set:

```bash
# Set master switch
export TRITON_LIVE=true

# Set available keys
export MARINETRAFFIC_API_KEY=abc123
export PLANET_API_KEY=def456
export EIA_API_KEY=ghi789
export OPENWEATHER_API_KEY=jkl012

# Run
cd backend
uvicorn main:app --reload --port 8000

# Check connector status
python data_integration.py
```

Output will show which connectors are LIVE vs SYNTHETIC:

```
● AIS Vessel Tracking            LIVE       UNCLASSIFIED
● NOAA HRRR Atmospheric          LIVE       UNCLASSIFIED
● Planet Labs SAR Imagery        LIVE       UNCLASSIFIED
● Oil Price & Insurance          LIVE       UNCLASSIFIED
○ Insurance & War Risk Pricing   SYNTHETIC  UNCLASSIFIED    ← No Bloomberg key
● GPS Interference Detection     LIVE       UNCLASSIFIED
● Stranded Vessel Tracker        LIVE       UNCLASSIFIED
● Economic Intelligence          LIVE       UNCLASSIFIED
○ Crew Wearable Data             SYNTHETIC  UNCLASSIFIED    ← Needs sailor OAuth
● Federal Spending Data          LIVE       UNCLASSIFIED
● Strike BDA Intelligence        LIVE       UNCLASSIFIED
○ CENTCOM Classified BDA         SYNTHETIC  SECRET          ← Needs SIPR
○ Five Eyes Coalition Nexus      SYNTHETIC  TS // SCI       ← Needs JWICS
```

Connectors that fail (network error, bad key, timeout) automatically fall back
to synthetic data. The system never crashes — it degrades gracefully.

---

## Foundry Deployment Architecture

In a Palantir Foundry deployment, the architecture maps to:

### Ontology Object Types

| Object Type | Properties | Primary Key | Sources |
|-------------|-----------|-------------|---------|
| Vessel | mmsi, imo, name, lat, lon, speed, heading, flag, type, status, last_update | mmsi | MarineTraffic, Windward |
| MineZone | lat, lon, probability, source, confirmed, detection_method | zone_id | Planet Labs SAR |
| SubmarineContact | lat, lon, radius_km, classification, confidence, last_detection | contact_id | FVEY sonar fusion |
| ConvoyMission | mission_id, route, risk_score, survivability, aip_decision, escort_assets, status | mission_id | AIP Orchestrator |
| AtmosphericWindow | start_time, end_time, duct_factor, advantage_level, waypoints_affected | window_id | NOAA HRRR |
| InsuranceStatus | club_name, coverage_active, war_risk_premium, cancellation_date | club_id | Lloyd's List |
| EconomicImpact | chain_name, status, gdp_at_risk, days_to_impact, current_indicator | chain_id | World Bank, EIA |

### Code Repositories (Foundry)

```
triton-transforms/       # Data ingestion transforms
  ais_ingest.py          # MarineTraffic → Vessel objects
  hrrr_ingest.py         # NOAA GRIB2 → AtmosphericWindow objects
  sar_ingest.py          # Planet Labs → MineZone objects
  oil_ingest.py          # Yahoo Finance → EconomicImpact objects

triton-models/           # ML model code
  vessel_anomaly.py      # XGBoost 7-feature anomaly detector
  mine_bayesian.py       # Bayesian posterior mine probability grid
  ducting_model.py       # Smith-Weintraub refractivity computation
  swarm_monte_carlo.py   # 10K sim/sec swarm engagement model

triton-logic/            # AIP Logic Functions
  assess_route_risk.py   # Route risk scoring
  compute_survivability.py # Kinetic + crew-adjusted survivability
  check_insurance.py     # Insurance viability gate
  cascade_risk.py        # Economic cascade chain triggers
  aip_decision.py        # 4-agent vote → GO/NO-GO

triton-actions/          # Foundry Actions (user-triggered)
  plan_convoy.py         # Create ConvoyMission from parameters
  update_bda.py          # Ingest new strike BDA
  recalculate_aip.py     # Re-run AIP with updated data
```

### AIP Logic Functions

```python
@aip_logic_function
def aip_decision(
    route_risk: float,         # From assess_route_risk
    survivability: float,       # From compute_survivability
    insurance_viable: bool,     # From check_insurance
    cascade_gdp: float,         # From cascade_risk
    crew_factor: float,         # From M36
) -> str:
    """
    4 agents vote independently:
    1. Kinetic Agent: GO if survivability > 0.7
    2. Insurance Agent: GO if insurance_viable
    3. Economic Agent: GO if cascade_gdp < $100B
    4. Crew Agent: GO if crew_factor > 0.85
    
    Decision: GO requires unanimous. One NO-GO = NO-GO.
    Current result: Insurance Agent vetoes. AIP: NO-GO.
    """
```

---

## March 20, 2026 Intelligence Summary

| Metric | Day 18 (Mar 18) | Day 20 (Mar 20) | Delta |
|--------|-----------------|-----------------|-------|
| Oil (Brent) | $102/bbl | $110/bbl | +$8 |
| Threat Index | 51.5/100 | 49.2/100 | -2.3 |
| Naval Destroyed | 100+ | 120+ | +20 |
| Combat Flights | 6,000+ | 7,500+ | +1,500 |
| Transits (daily) | 8 | 10 | +2 |
| Stranded Vessels | 150 | 155+ | +5 |
| Cumulative Cost | $9B | $10B | +$1B |
| Regime Stability | 0.538 | 0.525 | -0.013 |
| IRGC C2 | 0.40 | 0.35 | -0.05 |
| BROI | 360:1 | 363:1 | +3 |
| AIP Decision | NO-GO | NO-GO | No change |

### New March 20 Intelligence
- **DIA Assessment:** Iran can keep strait shut 1-6 months (CNN, 4 hours ago)
- **Marines:** 2,500 deploying from San Diego + 3 warships (AP, March 20)
- **Ras Laffan:** Israel struck South Pars → Iran retaliated on Ras Laffan LNG. 17% LNG capacity lost. 5-year repair (QatarEnergy).
- **Goldman Sachs:** Warns Brent could reach $147/bbl (all-time high) if disruptions lengthen
- **IRGC Vetting:** Formal registration system under development for selective transit (Lloyd's List)
- **A-10 Warthogs:** Confirmed in anti-ship role in strait (CENTCOM)
- **IEA SPR:** 32-nation emergency release. US committed 172M barrels.
- **Ground Troops:** Experts say may be required to fully reopen strait (CNN)
- **UK:** "Will not be drawn into wider war" — PM Starmer. Sent planners only.

---

## File Structure

```
triton-platform/
├── README.md                          # This file
├── update_march20.py                  # Intelligence update script
├── backend/
│   ├── main.py                        # FastAPI — 60 endpoints
│   ├── data_integration.py            # 13 data connectors + Foundry ontology
│   ├── requirements.txt
│   └── models/
│       ├── vessel_detector.py         # M1: XGBoost anomaly detection
│       ├── atmosphere.py              # M2: Smith-Weintraub ducting
│       ├── mine_probability.py        # M3: Bayesian posterior grid
│       ├── route_optimizer.py         # M4: Multi-objective A*
│       ├── enforcement.py             # M5: Selective enforcement tracker
│       ├── disguised_vessel.py        # M6: Behavioral analytics
│       ├── risk_score.py              # Transit risk scoring
│       ├── insurance_transit.py       # M7/M16/M17: Insurance + permission
│       ├── threat_engagement.py       # M8: Per-waypoint timeline
│       ├── convoy_scheduler.py        # M9: Throughput scheduling
│       ├── execution_modules.py       # M10-M12: Comms, MCM, adversary
│       ├── bab_al_mandeb.py           # M13: Dual chokepoint
│       ├── submarine_threat.py        # M14: Ghadir-class model
│       ├── gps_warfare.py             # M15: Spoof/denial zones
│       ├── strategic_modules.py       # M18-M20: Bypass, stranded, coalition
│       ├── strike_degradation.py      # M21-M22: BDA + LUCAS
│       ├── advanced_modules.py        # M23-M25: Arsenal, cascade, reinforcement
│       ├── aip_orchestrator.py        # M26: 4-agent GO/NO-GO
│       ├── nexus_quantum_nke.py       # M27-M29: Coalition, PNT, NKE
│       ├── warpspeed_foundry.py       # M30-M32: Econ twin, Ender's, industrial
│       ├── swarm_xai.py              # M33-M34: MARL swarm, XAI consensus
│       ├── module_upgrades.py         # Connective tissue upgrades
│       ├── last_mile.py               # M35-M40: Bio-acoustic to MLS guard
│       └── bleeding_edge.py           # M41-M44: Alignment, DOGE, regime, DAGIR
└── triton-dashboard-v10.html          # Gotham-style frontend
```

---

## License

Built for Palantir FDSE application demonstration purposes.
Uses publicly available data sources and open-source libraries.
No classified information is contained in this repository.

---

*"The military problem is solvable. The insurance problem is the binding constraint."*
*— TRITON AIP Engine, Day 18, confirmed by Pentagon Day 20*
