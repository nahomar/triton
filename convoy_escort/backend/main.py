"""
Transit Risk Intelligence & Tactical Operations Network (TRITON)
Main FastAPI Application
Author: Nahom Woldegebriel
"""

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
import uvicorn

from models.vessel_detector import VesselDetector
from models.atmosphere import AtmosphericThreatModel
from models.mine_probability import MineProbabilityModel
from models.route_optimizer import EscortRouteOptimizer
from models.enforcement import SelectiveEnforcementTracker
from models.disguised_vessel import DisguisedVesselDetector
from models.risk_score import TransitRiskScorer
from models.threat_engagement import ThreatEngagementTimeline
from models.convoy_scheduler import ConvoyScheduler
from models.execution_modules import CommsResilienceLayer, MCMFormationPlanner, AdversaryReactionModel
from models.bab_al_mandeb import BabAlMandebThreatModel
from models.submarine_threat import SubmarineThreatLayer
from models.gps_warfare import GPSWarfareLayer
from models.insurance_transit import InsuranceViabilityLayer, PermissionTransitModel
from models.strategic_modules import BypassPipelineModel, StrandedVesselLayer, CoalitionEscortPlanner
from models.strike_degradation import StrikeDegradationLayer
from models.advanced_modules import UndergroundArsenalModel, BypassInfraTargeting, ProductionCascadeModel, ReinforcementTracker
from models.aip_orchestrator import AIPOrchestrator
from models.nexus_quantum_nke import CoalitionNexusLayer, QuantumPNTModule, NonKineticEffectsCell
from models.warpspeed_foundry import MacroEconomicDigitalTwin, EndersFoundry, WarpSpeedIndustrial
from models.swarm_xai import SwarmForge, XAIConsensusGate
from models.module_upgrades import NKEWindowIntegration, WarpSpeedAutoDownrate, EconomicCascadeChain, CoalitionSecureSharing
from models.last_mile import BioAcousticFingerprinter, CrewReadinessModel, DynamicROEGuard, AdversarialDataFilter, MagazineFeedbackLoop, MLSCoalitionGuard
from models.bleeding_edge import ModelAlignmentVerifier, DOGEFiscalGuardrail, RegimeFractalityModel, OpenDAGIRSandbox

app = FastAPI(
    title="Transit Risk Intelligence & Tactical Operations Network",
    description="Multi-domain threat fusion for Strait of Hormuz escort planning",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Initialize Models ──
vessel_detector = VesselDetector()
atmos_model = AtmosphericThreatModel()
mine_model = MineProbabilityModel()
route_optimizer = EscortRouteOptimizer()
enforcement_tracker = SelectiveEnforcementTracker()
disguised_detector = DisguisedVesselDetector()
risk_scorer = TransitRiskScorer()
threat_engagement = ThreatEngagementTimeline()
convoy_scheduler = ConvoyScheduler()
comms_layer = CommsResilienceLayer()
mcm_planner = MCMFormationPlanner()
adversary_model = AdversaryReactionModel()
bab_al_mandeb = BabAlMandebThreatModel()
submarine_layer = SubmarineThreatLayer()
gps_warfare = GPSWarfareLayer()
insurance_layer = InsuranceViabilityLayer()
permission_model = PermissionTransitModel()
bypass_pipeline = BypassPipelineModel()
stranded_vessels = StrandedVesselLayer()
coalition_planner = CoalitionEscortPlanner()
strike_degradation = StrikeDegradationLayer()
underground_arsenal = UndergroundArsenalModel()
bypass_targeting = BypassInfraTargeting()
production_cascade = ProductionCascadeModel()
reinforcement_tracker = ReinforcementTracker()
aip_orchestrator = AIPOrchestrator()
coalition_nexus = CoalitionNexusLayer()
quantum_pnt = QuantumPNTModule()
nke_cell = NonKineticEffectsCell()
economic_twin = MacroEconomicDigitalTwin()
enders_foundry = EndersFoundry()
warpspeed = WarpSpeedIndustrial()
swarm_forge = SwarmForge()
xai_gate = XAIConsensusGate()
nke_windows = NKEWindowIntegration()
ws_downrate = WarpSpeedAutoDownrate()
econ_cascade = EconomicCascadeChain()
coalition_sharing = CoalitionSecureSharing()
bio_acoustic = BioAcousticFingerprinter()
crew_readiness = CrewReadinessModel()
roe_guard = DynamicROEGuard()
adversarial_filter = AdversarialDataFilter()
magazine_loop = MagazineFeedbackLoop()
mls_guard = MLSCoalitionGuard()
alignment_verifier = ModelAlignmentVerifier()
doge_fiscal = DOGEFiscalGuardrail()
regime_model = RegimeFractalityModel()
dagir_sandbox = OpenDAGIRSandbox()


# ── Request/Response Models ──
class EscortRequest(BaseModel):
    origin_lat: float = 25.3  # Gulf side
    origin_lon: float = 56.2
    dest_lat: float = 25.5    # Oman side
    dest_lon: float = 56.8
    escort_assets: int = 2     # Number of DDGs
    time_window_hours: int = 48
    vessel_type: str = "VLCC"

class RiskRequest(BaseModel):
    flag_state: str = "US"
    vessel_type: str = "tanker"
    transit_time: str = "2026-03-18T04:00:00Z"
    escort_level: str = "DDG"  # none, DDG, full_convoy


# ══════════════════════════════════════
# MODULE 1: Vessel Detection & Classification
# ══════════════════════════════════════

@app.get("/api/vessels")
async def get_vessels():
    """Get all tracked vessel positions with classifications."""
    return vessel_detector.get_all_vessels()

@app.get("/api/vessels/anomalies")
async def get_anomalies(threshold: float = Query(0.7, description="Anomaly score threshold")):
    """Get vessels flagged as suspicious or hostile."""
    return vessel_detector.get_anomalies(threshold)


# ══════════════════════════════════════
# MODULE 2: Atmospheric Threat Overlay
# ══════════════════════════════════════

@app.get("/api/atmosphere")
async def get_atmosphere():
    """Get 48-hour atmospheric threat forecast over the strait."""
    return atmos_model.get_threat_overlay()

@app.get("/api/atmosphere/windows")
async def get_advantage_windows():
    """Get atmospheric advantage windows for safe transit."""
    return atmos_model.get_advantage_windows()


# ══════════════════════════════════════
# MODULE 3: Mine Probability Map
# ══════════════════════════════════════

@app.get("/api/mines")
async def get_mine_probability():
    """Get Bayesian mine probability heatmap."""
    return mine_model.get_probability_grid()

@app.post("/api/mines/update")
async def update_mine_evidence(lat: float, lon: float, event_type: str):
    """Update mine probability with new evidence (found/cleared/safe_transit)."""
    return mine_model.update_posterior(lat, lon, event_type)


# ══════════════════════════════════════
# MODULE 4: Escort Route Optimizer
# ══════════════════════════════════════

@app.post("/api/route/optimize")
async def optimize_route(request: EscortRequest):
    """Compute optimal escort route given all threat layers."""
    vessels = vessel_detector.get_anomalies(0.5)
    atmosphere = atmos_model.get_threat_overlay()
    mines = mine_model.get_probability_grid()
    
    return route_optimizer.optimize(
        origin=(request.origin_lat, request.origin_lon),
        destination=(request.dest_lat, request.dest_lon),
        escort_assets=request.escort_assets,
        time_window_hours=request.time_window_hours,
        vessel_type=request.vessel_type,
        hostile_contacts=vessels,
        atmospheric_threat=atmosphere,
        mine_probability=mines,
    )


# ══════════════════════════════════════
# MODULE 5: Selective Enforcement Tracker
# ══════════════════════════════════════

@app.get("/api/enforcement")
async def get_enforcement_patterns():
    """Analyze which flag states Iran allows through and when."""
    return enforcement_tracker.get_patterns()


# ══════════════════════════════════════
# MODULE 6: Disguised Vessel Detector
# ══════════════════════════════════════

@app.get("/api/vessels/disguised")
async def get_disguised_vessels():
    """Detect fishing vessels with anomalous behavior patterns."""
    return disguised_detector.get_flagged_vessels()


# ══════════════════════════════════════
# MODULE 7: Insurance Risk Score API
# ══════════════════════════════════════

@app.post("/api/risk/score")
async def get_risk_score(request: RiskRequest):
    """Compute real-time transit risk score for insurance pricing."""
    atmosphere = atmos_model.get_current_threat_level()
    mines = mine_model.get_route_risk()
    hostile_count = len(vessel_detector.get_anomalies(0.7).get("vessels", []))
    
    return risk_scorer.compute(
        flag_state=request.flag_state,
        vessel_type=request.vessel_type,
        transit_time=request.transit_time,
        escort_level=request.escort_level,
        atmospheric_threat=atmosphere,
        mine_risk=mines,
        hostile_contacts=hostile_count,
    )


# ══════════════════════════════════════
# MODULE 8: Threat Engagement Timeline
# ══════════════════════════════════════

@app.post("/api/engagement/simulate")
async def simulate_engagement(request: EscortRequest):
    """Simulate inbound threat sequence for a convoy transit."""
    route_result = await optimize_route(request)
    best_route = route_result["recommended_route"]
    duct = atmos_model.get_current_threat_level() * 3  # Convert to duct factor
    
    return threat_engagement.simulate_transit(
        route_waypoints=best_route["waypoints"],
        escort_assets=["DDG"] * request.escort_assets,
        duct_factor=max(0.5, duct),
    )


# ══════════════════════════════════════
# MODULE 9: Convoy Throughput Scheduler
# ══════════════════════════════════════

@app.get("/api/convoy/schedule")
async def get_convoy_schedule(
    mine_clearance: float = Query(0.3, description="Mine clearance progress 0-1"),
    threat_attrition: float = Query(0.75, description="Iranian capability destroyed 0-1"),
):
    """Compute maximum sustainable convoy throughput for next 48 hours."""
    windows = atmos_model.get_advantage_windows()
    return convoy_scheduler.compute_daily_throughput(
        atmospheric_windows=windows,
        mine_clearance_progress=mine_clearance,
        threat_attrition=threat_attrition,
    )


# ══════════════════════════════════════
# MODULE 10: Communications Resilience
# ══════════════════════════════════════

@app.post("/api/comms/assess")
async def assess_comms(request: EscortRequest):
    """Assess C2 communications resilience along transit route."""
    route_result = await optimize_route(request)
    best_route = route_result["recommended_route"]
    duct = atmos_model.get_current_threat_level() * 3
    
    return comms_layer.assess_comms_coverage(
        route_waypoints=best_route["waypoints"],
        duct_factor=max(0.5, duct),
    )


# ══════════════════════════════════════
# MODULE 11: MCM Formation Planner
# ══════════════════════════════════════

@app.post("/api/mcm/plan")
async def plan_mcm(request: EscortRequest):
    """Plan mine countermeasure formation and sweep schedule."""
    route_result = await optimize_route(request)
    best_route = route_result["recommended_route"]
    mines = mine_model.get_probability_grid()
    
    return mcm_planner.plan_formation(
        route_waypoints=best_route["waypoints"],
        mine_probability=mines,
    )


# ══════════════════════════════════════
# MODULE 12: Adversary Reaction Model
# ══════════════════════════════════════

@app.post("/api/adversary/predict")
async def predict_adversary_response(request: EscortRequest):
    """Predict Iranian response to convoy transit using game theory."""
    windows = atmos_model.get_advantage_windows()
    has_window = len(windows.get("windows", [])) > 0
    
    return adversary_model.predict_response({
        "escort_assets": request.escort_assets,
        "vessels": 3,  # Default convoy size
        "vessel_type": request.vessel_type,
        "atmospheric_advantage": has_window,
    })


# ══════════════════════════════════════
# MODULE 13: Bab al-Mandeb / Houthi Threat
# ══════════════════════════════════════

@app.get("/api/redsea/assess")
async def assess_red_sea(
    vessel_type: str = Query("VLCC"),
    flag_state: str = Query("US"),
    escort: bool = Query(False),
):
    """Assess Houthi threat for Red Sea / Bab al-Mandeb transit."""
    return bab_al_mandeb.assess_red_sea_transit(vessel_type, flag_state, escort)


@app.post("/api/chokepoints/combined")
async def combined_chokepoint_risk(request: EscortRequest):
    """Combined risk: Hormuz + Red Sea for Europe-bound tankers."""
    risk = risk_scorer.compute(
        flag_state=request.flag_state or "US",
        vessel_type=request.vessel_type,
        transit_time=datetime.now().isoformat(),
        escort_level="DDG" if request.escort_assets >= 2 else "none",
        atmospheric_threat=atmos_model.get_current_threat_level(),
        mine_risk=mine_model.get_route_risk(),
        hostile_contacts=vessel_detector.get_anomalies(0.7)["count"],
    )
    hormuz_risk = risk["composite_risk_score"]
    return bab_al_mandeb.compute_combined_chokepoint_risk(
        hormuz_risk=hormuz_risk,
        vessel_type=request.vessel_type,
        flag_state=request.flag_state or "US",
        destination="Europe",
    )


# ══════════════════════════════════════
# MODULE 14: Submarine Threat
# ══════════════════════════════════════

@app.post("/api/submarine/assess")
async def assess_submarine_threat(request: EscortRequest):
    """Assess Ghadir-class midget submarine threat along route."""
    route_result = await optimize_route(request)
    best_route = route_result["recommended_route"]
    return submarine_layer.assess_submarine_threat(best_route["waypoints"], request.escort_assets)


# ══════════════════════════════════════
# MODULE 15: GPS/GNSS Warfare
# ══════════════════════════════════════

@app.post("/api/gps/assess")
async def assess_gps_warfare(request: EscortRequest):
    """Assess GPS jamming/spoofing interference along route."""
    route_result = await optimize_route(request)
    best_route = route_result["recommended_route"]
    return gps_warfare.assess_navigation_integrity(
        best_route["waypoints"], request.vessel_type, has_military_escort=request.escort_assets > 0
    )


# ══════════════════════════════════════
# MODULE 16: Insurance & Commercial Viability
# ══════════════════════════════════════

@app.post("/api/insurance/viability")
async def assess_insurance_viability(request: EscortRequest):
    """Assess commercial viability — can ships actually sail?"""
    risk = risk_scorer.compute(
        flag_state=request.flag_state or "US", vessel_type=request.vessel_type,
        transit_time=datetime.now().isoformat(),
        escort_level="DDG" if request.escort_assets >= 2 else "none",
        atmospheric_threat=atmos_model.get_current_threat_level(),
        mine_risk=mine_model.get_route_risk(),
        hostile_contacts=vessel_detector.get_anomalies(0.7)["count"],
    )
    return insurance_layer.assess_commercial_viability(
        military_risk_score=risk["composite_risk_score"],
        escort_level="DDG" if request.escort_assets >= 2 else "none",
        vessel_type=request.vessel_type,
        flag_state=request.flag_state or "US",
    )


# ══════════════════════════════════════
# MODULE 17: Permission-Based Transit
# ══════════════════════════════════════

@app.get("/api/permissions/diplomatic")
async def get_diplomatic_picture():
    """Get full diplomatic transit permission picture."""
    return permission_model.get_full_diplomatic_picture()

@app.get("/api/permissions/{flag_state}")
async def get_flag_permission(flag_state: str):
    """Check transit permission for a specific flag state."""
    return permission_model.assess_transit_permission(flag_state.upper())


# ══════════════════════════════════════
# MODULE 18: Bypass Pipelines
# ══════════════════════════════════════

@app.get("/api/bypass/status")
async def get_bypass_status():
    """Get bypass pipeline status and supply gap analysis."""
    return bypass_pipeline.compute_bypass_status()


# ══════════════════════════════════════
# MODULE 19: Stranded Vessels
# ══════════════════════════════════════

@app.get("/api/stranded/situation")
async def get_stranded_situation():
    """Get stranded vessel count, locations, and evacuation plan."""
    return stranded_vessels.assess_stranded_situation()


# ══════════════════════════════════════
# MODULE 20: Coalition Escort Planning
# ══════════════════════════════════════

@app.get("/api/coalition/scenarios")
async def get_coalition_scenarios():
    """Model coalition scenarios and throughput impact."""
    return coalition_planner.compute_coalition_scenarios()


# ══════════════════════════════════════
# MODULE 21: Strike Degradation
# ══════════════════════════════════════

@app.get("/api/strike/degradation")
async def get_strike_degradation(day: int = Query(18)):
    """Current Iranian capability after US strike campaign."""
    return strike_degradation.compute_current_threat_level(day)


# ══════════════════════════════════════
# MODULE 22: Underground Arsenal
# ══════════════════════════════════════

@app.get("/api/arsenal/underground")
async def get_underground_arsenal():
    """Assess Iran's hidden tunnel-based weapon stockpiles."""
    return underground_arsenal.assess_hidden_arsenal()


# ══════════════════════════════════════
# MODULE 23: Bypass Infrastructure Targeting
# ══════════════════════════════════════

@app.get("/api/bypass/attacks")
async def get_bypass_attacks():
    """Track Iranian attacks on bypass pipeline infrastructure."""
    return bypass_targeting.assess_bypass_vulnerability()


# ══════════════════════════════════════
# MODULE 24: Production Cascade
# ══════════════════════════════════════

@app.get("/api/cascade/status")
async def get_cascade_status():
    """Economic cascade: force majeures, oil price, offline production."""
    return production_cascade.get_cascade_status()


# ══════════════════════════════════════
# MODULE 25: Reinforcement Tracker
# ══════════════════════════════════════

@app.get("/api/reinforcements/posture")
async def get_force_posture():
    """Current and inbound military force posture."""
    return reinforcement_tracker.get_force_posture()


# ══════════════════════════════════════
# MODULE 26: AIP Agentic Orchestration
# ══════════════════════════════════════

@app.post("/api/aip/evaluate")
async def aip_evaluate_mission(request: EscortRequest):
    """Multi-agent AI evaluation of mission plan."""
    route_result = await optimize_route(request)
    best = route_result["recommended_route"]
    wp = best["waypoints"]
    duct = max(0.5, atmos_model.get_current_threat_level() * 3)
    eng = threat_engagement.simulate_transit(wp, ["DDG"] * request.escort_assets, duct)
    gps = gps_warfare.assess_navigation_integrity(wp, request.vessel_type, request.escort_assets > 0)
    perm = permission_model.assess_transit_permission(request.flag_state or "US")
    
    mission_params = {
        "survivability": eng["summary"]["estimated_survivability"],
        "critical_waypoints": eng["summary"]["critical_count"],
        "duct_factor": duct,
        "asw_detection_prob": 0.35,
        "insurance_gap": True,
        "flag_state": request.flag_state or "US",
        "transit_permission": perm["permission_status"],
        "escort_nations": ["US"],
        "gps_denied_waypoints": gps["summary"]["gps_denied_waypoints"],
    }
    evaluation = aip_orchestrator.evaluate_mission(mission_params)
    monte_carlo = aip_orchestrator.monte_carlo_simulation(mission_params, 5000)
    return {"evaluation": evaluation, "monte_carlo": monte_carlo}


@app.post("/api/aip/montecarlo")
async def run_monte_carlo(request: EscortRequest, simulations: int = Query(5000)):
    """Run Monte Carlo simulation for mission survivability."""
    mission_params = {
        "survivability": 0.898, "critical_waypoints": 3, "duct_factor": 1.2,
        "asw_detection_prob": 0.35, "insurance_gap": True, "flag_state": request.flag_state or "US",
        "transit_permission": "denied", "escort_nations": ["US"], "gps_denied_waypoints": 5,
        "mine_probability": 0.05,
    }
    return aip_orchestrator.monte_carlo_simulation(mission_params, simulations)


# ══════════════════════════════════════
# MODULE 27: Coalition Nexus
# ══════════════════════════════════════

@app.get("/api/nexus/sensors")
async def get_coalition_sensors(nations: str = Query("US")):
    """Coalition sensor fusion picture."""
    nation_list = [n.strip() for n in nations.split(",")]
    return coalition_nexus.assess_coalition_sensor_picture(nation_list)


# ══════════════════════════════════════
# MODULE 28: Quantum PNT
# ══════════════════════════════════════

@app.get("/api/pnt/resilience")
async def get_pnt_resilience(gps_denied: bool = Query(True)):
    """Assess PNT resilience in GPS-denied environment."""
    return quantum_pnt.assess_pnt_resilience("convoy", gps_denied)


# ══════════════════════════════════════
# MODULE 29: Non-Kinetic Effects
# ══════════════════════════════════════

@app.post("/api/nke/plan")
async def generate_nke_plan(request: EscortRequest):
    """Generate synchronized cyber-EW plan for convoy transit."""
    route_result = await optimize_route(request)
    wp = route_result["recommended_route"]["waypoints"]
    return nke_cell.generate_nke_plan(datetime.now().isoformat(), wp)


# ══════════════════════════════════════
# MODULE 30: Economic Digital Twin
# ══════════════════════════════════════

@app.get("/api/economic/cost")
async def get_economic_cost(days_closed: int = Query(18)):
    """Price of Inaction — economic cost of not escorting."""
    return economic_twin.compute_price_of_inaction(days_closed)


# ══════════════════════════════════════
# MODULE 31: Ender's Foundry
# ══════════════════════════════════════

@app.get("/api/foundry/status")
async def get_foundry_status():
    """Sim-dev/sim-ops feedback loop status."""
    return enders_foundry.get_foundry_status()


# ══════════════════════════════════════
# MODULE 32: Warp Speed Industrial
# ══════════════════════════════════════

@app.get("/api/warpspeed/magazine")
async def get_magazine_depth(convoys: int = Query(8), days: int = Query(14)):
    """Assess munitions magazine depth for planned operations."""
    return warpspeed.assess_magazine_depth(convoys, days)


# ══════════════════════════════════════
# MODULE 33: MARL Swarm Optimizer
# ══════════════════════════════════════

@app.post("/api/swarm/optimize")
async def optimize_swarm(request: EscortRequest):
    """Compute optimal LUCAS drone-to-threat pairing."""
    threats = [
        {"type": "shahed_136", "bearing": 45, "distance_nm": 8},
        {"type": "usv_suicide", "bearing": 120, "distance_nm": 4},
        {"type": "fast_boat", "bearing": 210, "distance_nm": 6},
        {"type": "noor_ascm", "bearing": 350, "distance_nm": 12},
    ]
    return swarm_forge.optimize_swarm_defense(threats, available_drones=24)

@app.get("/api/swarm/simulate")
async def simulate_swarm(n: int = Query(10000)):
    """Monte Carlo simulation of swarm defense."""
    return swarm_forge.simulate_swarm_engagement(n)


# ══════════════════════════════════════
# MODULE 34: XAI Consensus Gate
# ══════════════════════════════════════

@app.post("/api/xai/explain")
async def explain_risk(request: EscortRequest):
    """Explainable risk decomposition with human-readable reasoning."""
    risk = risk_scorer.compute(
        flag_state=request.flag_state or "US", vessel_type=request.vessel_type,
        transit_time=datetime.now().isoformat(),
        escort_level="DDG" if request.escort_assets >= 2 else "none",
        atmospheric_threat=atmos_model.get_current_threat_level(),
        mine_risk=mine_model.get_route_risk(),
        hostile_contacts=vessel_detector.get_anomalies(0.7)["count"],
    )
    return xai_gate.explain_risk_score(risk["composite_risk_score"], risk["components"])

@app.post("/api/xai/consensus")
async def check_consensus(request: EscortRequest):
    """Multi-model consensus check — detects hallucination and data poisoning."""
    return xai_gate.consensus_check({
        "flag_state": request.flag_state or "US",
        "escort_assets": request.escort_assets,
        "duct_factor": atmos_model.get_current_threat_level() * 3,
        "gps_denied_waypoints": 5,
        "survivability": 0.898,
        "threat_attrition": 0.75,
    })


# ══════════════════════════════════════
# UPGRADE: NKE Windows + Warp Speed + Cascade + Sharing
# ══════════════════════════════════════

@app.post("/api/nke/windows")
async def get_nke_windows(request: EscortRequest):
    """Electronic silence windows synchronized with route."""
    route_result = await optimize_route(request)
    return nke_windows.compute_electronic_silence_windows(route_result["recommended_route"]["waypoints"])

@app.get("/api/warpspeed/downrate")
async def get_auto_downrate():
    """Auto-adjust survivability based on magazine depth."""
    mag = warpspeed.assess_magazine_depth(8, 14)
    return ws_downrate.auto_adjust_survivability(0.898, mag["munitions_status"])

@app.get("/api/economic/cascades")
async def get_cascade_chains(days: int = Query(18)):
    """Industrial supply chain cascade analysis."""
    return econ_cascade.compute_cascade_impact(days)

@app.get("/api/nexus/sharing")
async def get_sharing_posture(nations: str = Query("US,UK,France")):
    """Coalition secure data sharing posture."""
    return coalition_sharing.assess_sharing_posture([n.strip() for n in nations.split(",")])


# ══════════════════════════════════════
# MODULE 35: Bio-Acoustic Fingerprinting
# ══════════════════════════════════════

@app.get("/api/acoustic/detection")
async def get_acoustic_detection(ddgs: int = Query(2), p8a: bool = Query(True)):
    """Realistic Ghadir detection probability with bio-acoustic denoising."""
    return bio_acoustic.assess_detection_probability(ddgs, p8a)


# ══════════════════════════════════════
# MODULE 36: Crew Readiness
# ══════════════════════════════════════

@app.get("/api/crew/readiness")
async def get_crew_readiness():
    """Assess crew fatigue and performance degradation on escort vessels."""
    return crew_readiness.assess_crew_readiness(["DDG_Spruance", "DDG_McFaul"])


# ══════════════════════════════════════
# MODULE 37: ROE Guard
# ══════════════════════════════════════

@app.post("/api/roe/check")
async def check_roe(request: EscortRequest):
    """Vet mission plan against Rules of Engagement and IHL."""
    return roe_guard.vet_mission_plan({
        "anomalies_to_engage": vessel_detector.get_anomalies(0.7)["count"],
        "pid_confirmed": False,
        "lucas_preemptive_strikes": 0,
        "disguised_vessels_detected": 3,
        "vessel_type": request.vessel_type,
        "escort_mode": "defensive",
    })


# ══════════════════════════════════════
# MODULE 38: Adversarial Data Filter
# ══════════════════════════════════════

@app.get("/api/integrity/mines")
async def verify_mine_report():
    """Cross-check mine report against independent data streams."""
    return adversarial_filter.filter_mine_report()


# ══════════════════════════════════════
# MODULE 39: Magazine Feedback Loop
# ══════════════════════════════════════

@app.post("/api/magazine/check")
async def check_magazine_reroute(request: EscortRequest):
    """Check inventory and force reroute if insufficient."""
    return magazine_loop.check_and_reroute(
        mission_plan={"route_name": "northern", "lucas_screen_size": 24, "critical_waypoints": 3,
                       "sm2_per_critical_wp": 8, "sonobuoy_per_mission": 200},
        available_inventory={"lucas": 12, "sm2": 547, "sonobuoys": 5000},
        route_options=[
            {"name": "northern", "critical_waypoints": 3},
            {"name": "central", "critical_waypoints": 5},
            {"name": "southern", "critical_waypoints": 2},
        ],
    )


# ══════════════════════════════════════
# MODULE 40: MLS Coalition Guard
# ══════════════════════════════════════

@app.get("/api/mls/architecture")
async def get_mls_architecture():
    """Multi-level security architecture for coalition data sharing."""
    return mls_guard.get_mls_architecture()

@app.post("/api/mls/access")
async def check_access(nation: str = Query("France")):
    """Check if a nation can access a classified data object."""
    sample_object = {"classification": "TS//SCI//FVEY", "type": "submarine_track",
                      "lat": 26.612, "lon": 56.453, "depth_m": 45, "bearing": 180,
                      "source": "AN/SQQ-89", "aegis_track_id": "T-4417",
                      "time": "2026-03-18T14:32:00Z"}
    return mls_guard.enforce_classification(nation, sample_object)


# ══════════════════════════════════════
# MODULE 41: Model Alignment Verification
# ══════════════════════════════════════

@app.post("/api/alignment/verify")
async def verify_alignment(request: EscortRequest):
    """Verify AIP agents aren't self-censoring lawful actions."""
    return alignment_verifier.verify_alignment({
        "aip_decision": "NO-GO", "survivability": 0.898,
    })


# ══════════════════════════════════════
# MODULE 42: DOGE Fiscal Guardrail
# ══════════════════════════════════════

@app.post("/api/fiscal/broi")
async def compute_broi(request: EscortRequest):
    """Budgetary Return on Investment for escort mission."""
    return doge_fiscal.compute_broi({
        "transit_hours": 6, "escort_ddgs": request.escort_assets,
        "escort_lcs": 1, "lucas_screen": 24, "sm2_budget": 8,
        "p8a_sorties": 2, "mcm_sweeps": 1, "tankers_in_convoy": 3,
    })


# ══════════════════════════════════════
# MODULE 43: Regime Fractality
# ══════════════════════════════════════

@app.get("/api/regime/stability")
async def get_regime_stability():
    """Iranian regime stability and fracture trigger analysis."""
    return regime_model.assess_regime_stability()


# ══════════════════════════════════════
# MODULE 44: Open DAGIR Sandbox
# ══════════════════════════════════════

@app.get("/api/dagir/plugins")
async def get_dagir_plugins():
    """Available third-party algorithms for rapid onboarding."""
    return dagir_sandbox.list_available_plugins()

@app.get("/api/dagir/onboard/{plugin_name}")
async def onboard_plugin(plugin_name: str):
    """Simulate onboarding a specific plugin."""
    return dagir_sandbox.simulate_plugin_onboarding(plugin_name)


# ══════════════════════════════════════
# MASTER: Full Escort Mission Plan
# ══════════════════════════════════════

@app.post("/api/mission/plan")
async def generate_full_mission_plan(request: EscortRequest):
    """Generate complete escort mission plan integrating all 12 modules."""
    # Phase 1: Intelligence (Modules 1-3, 5-6)
    vessels = vessel_detector.get_all_vessels()
    anomalies = vessel_detector.get_anomalies(0.7)
    disguised = disguised_detector.get_flagged_vessels()
    atmosphere = atmos_model.get_threat_overlay()
    windows = atmos_model.get_advantage_windows()
    mines = mine_model.get_probability_grid()
    enforcement = enforcement_tracker.get_patterns()
    
    # Phase 2: Planning (Modules 4, 7)
    route = route_optimizer.optimize(
        origin=(request.origin_lat, request.origin_lon),
        destination=(request.dest_lat, request.dest_lon),
        escort_assets=request.escort_assets,
        time_window_hours=request.time_window_hours,
        vessel_type=request.vessel_type,
        hostile_contacts=anomalies,
        atmospheric_threat=atmosphere,
        mine_probability=mines,
    )
    best_route = route["recommended_route"]
    
    risk = risk_scorer.compute(
        flag_state="US", vessel_type=request.vessel_type,
        transit_time=route["recommended_departure"],
        escort_level="DDG" if request.escort_assets >= 2 else "none",
        atmospheric_threat=atmos_model.get_current_threat_level(),
        mine_risk=mine_model.get_route_risk(),
        hostile_contacts=anomalies["count"],
    )
    
    # Phase 3: Execution (Modules 8-12)
    duct = atmos_model.get_current_threat_level() * 3
    
    engagement = threat_engagement.simulate_transit(
        route_waypoints=best_route["waypoints"],
        escort_assets=["DDG"] * request.escort_assets,
        duct_factor=max(0.5, duct),
    )
    
    comms = comms_layer.assess_comms_coverage(
        route_waypoints=best_route["waypoints"],
        duct_factor=max(0.5, duct),
    )
    
    mcm = mcm_planner.plan_formation(
        route_waypoints=best_route["waypoints"],
        mine_probability=mines,
    )
    
    adversary = adversary_model.predict_response({
        "escort_assets": request.escort_assets,
        "vessels": 3,
        "vessel_type": request.vessel_type,
        "atmospheric_advantage": len(windows.get("windows", [])) > 0,
    })
    
    convoy = convoy_scheduler.compute_daily_throughput(windows)
    
    # Phase 5: Strategic (Module 13)
    red_sea = bab_al_mandeb.assess_red_sea_transit(request.vessel_type, request.flag_state or "US", has_escort=False)
    combined_chokepoints = bab_al_mandeb.compute_combined_chokepoint_risk(
        hormuz_risk=risk["composite_risk_score"],
        vessel_type=request.vessel_type,
        flag_state=request.flag_state or "US",
        destination="Europe",
    )
    
    # Phase 6: Domain Awareness (Modules 14-15)
    sub_threat = submarine_layer.assess_submarine_threat(best_route["waypoints"], request.escort_assets)
    nav_integrity = gps_warfare.assess_navigation_integrity(
        best_route["waypoints"], request.vessel_type, has_military_escort=request.escort_assets > 0
    )
    
    # Phase 7: Commercial Viability (Modules 16-17)
    commercial = insurance_layer.assess_commercial_viability(
        military_risk_score=risk["composite_risk_score"],
        escort_level="DDG" if request.escort_assets >= 2 else "none",
        vessel_type=request.vessel_type,
        flag_state=request.flag_state or "US",
    )
    diplomatic = permission_model.get_full_diplomatic_picture()
    
    # Phase 8: Operational Context (Modules 18-20)
    bypass = bypass_pipeline.compute_bypass_status()
    stranded = stranded_vessels.assess_stranded_situation()
    coalition = coalition_planner.compute_coalition_scenarios()
    
    # Phase 9: Strike Campaign (Modules 21-22)
    degradation = strike_degradation.compute_current_threat_level(18)
    arsenal = underground_arsenal.assess_hidden_arsenal()
    
    # Phase 10: Economic Warfare (Modules 23-25)
    bypass_attacks = bypass_targeting.assess_bypass_vulnerability()
    cascade = production_cascade.get_cascade_status()
    reinforcements = reinforcement_tracker.get_force_posture()
    
    # Phase 11: Predictive Mission Planning (Module 26)
    mission_params = {
        "survivability": engagement["summary"]["estimated_survivability"],
        "critical_waypoints": engagement["summary"]["critical_count"],
        "duct_factor": duct,
        "asw_detection_prob": 0.35,
        "insurance_gap": True,
        "flag_state": request.flag_state or "US",
        "transit_permission": "denied" if (request.flag_state or "US") in ["US", "UK", "IL"] else "negotiating",
        "escort_nations": ["US"],
        "gps_denied_waypoints": nav_integrity["summary"]["gps_denied_waypoints"],
        "mine_probability": 0.05,
    }
    aip_eval = aip_orchestrator.evaluate_mission(mission_params)
    monte_carlo = aip_orchestrator.monte_carlo_simulation(mission_params, 1000)
    
    # Phase 12: Platform Capabilities (Modules 27-34)
    nexus = coalition_nexus.assess_coalition_sensor_picture(["US"])
    pnt = quantum_pnt.assess_pnt_resilience("convoy", gps_denied=True)
    nke = nke_cell.generate_nke_plan(datetime.now().isoformat(), best_route["waypoints"])
    econ = economic_twin.compute_price_of_inaction(18)
    foundry = enders_foundry.get_foundry_status()
    magazine = warpspeed.assess_magazine_depth(8, 14)
    
    # M33: Swarm Forge
    sample_threats = [
        {"type": "shahed_136", "bearing": 45, "distance_nm": 8},
        {"type": "usv_suicide", "bearing": 120, "distance_nm": 4},
        {"type": "fast_boat", "bearing": 210, "distance_nm": 6},
    ]
    swarm = swarm_forge.optimize_swarm_defense(sample_threats, 24)
    swarm_mc = swarm_forge.simulate_swarm_engagement(1000)
    
    # M34: XAI Consensus
    consensus = xai_gate.consensus_check(mission_params)
    
    # Upgrades
    nke_win = nke_windows.compute_electronic_silence_windows(best_route["waypoints"])
    mag_adjust = ws_downrate.auto_adjust_survivability(engagement["summary"]["estimated_survivability"], magazine["munitions_status"])
    cascades = econ_cascade.compute_cascade_impact(18)
    sharing = coalition_sharing.assess_sharing_posture(["US", "UK"])
    
    return {
        "mission_id": f"TRITON-{datetime.now().strftime('%Y%m%d-%H%M')}",
        "classification": "UNCLASSIFIED // PROTOTYPE",
        "modules_integrated": 44,
        "phase_1_intelligence": {
            "vessel_picture": {"total": vessels["count"], "anomalies": anomalies["count"], "disguised": disguised["count"]},
            "atmospheric_conditions": {"duct_strength": atmosphere["current_duct_strength"], "advantage_windows": windows["count"]},
            "mine_threat": {"cells_flagged": mines["count"], "max_probability": mines["max_probability"]},
            "enforcement_pattern": {"allowed_flags": enforcement["pattern_summary"]["consistently_allowed"]},
        },
        "phase_2_planning": {
            "recommended_route": {"name": best_route["route_name"], "risk_score": best_route["risk_score"]},
            "recommended_departure": route["recommended_departure"],
            "transit_risk": risk,
            "escort_assessment": route["escort_assessment"],
        },
        "phase_3_execution": {
            "threat_engagement": {
                "survivability": engagement["summary"]["estimated_survivability"],
                "critical_waypoints": engagement["summary"]["critical_count"],
                "recommendation": engagement["recommendation"],
            },
            "comms_resilience": comms["recommendation"],
            "mcm_plan": {
                "sweep_time_hours": mcm["sweep_plan"]["estimated_sweep_time_hours"],
                "high_risk_segments": mcm["segment_count"],
            },
            "adversary_prediction": {
                "most_likely": adversary["most_likely"]["scenario"],
                "probability": adversary["most_likely"]["probability"],
                "counter": adversary["most_likely"]["recommended_counter"],
            },
        },
        "phase_4_sustainment": {
            "daily_throughput": convoy["daily_throughput"],
            "binding_constraint": convoy["constraints"]["binding_constraint"],
            "time_to_restore": convoy["time_to_restore_full_traffic"],
        },
        "phase_5_strategic": {
            "red_sea_threat": {
                "max_risk": red_sea["summary"]["max_risk_score"],
                "critical_zones": red_sea["summary"]["critical_zones"],
                "recommendation": red_sea["recommendation"],
            },
            "combined_chokepoint_risk": {
                "hormuz_suez_risk": combined_chokepoints["route_comparison"]["hormuz_suez"]["combined_risk"],
                "hormuz_cape_risk": combined_chokepoints["route_comparison"]["hormuz_cape"]["combined_risk"],
                "routing_recommendation": combined_chokepoints["recommendation"],
            },
        },
        "phase_6_domain_awareness": {
            "submarine_threat": {
                "ghadir_operational": sub_threat["fleet_assessment"]["ghadir_operational"],
                "torpedo_tubes": sub_threat["fleet_assessment"]["total_torpedo_tubes"],
                "asw_quality": sub_threat["asw_environment"]["quality"],
                "additional_mine_risk": sub_threat["mine_laying_risk"]["potential_additional_mines"],
                "recommendation": sub_threat["asw_recommendation"],
            },
            "gps_warfare": {
                "denied_waypoints": nav_integrity["summary"]["gps_denied_waypoints"],
                "spoofed_waypoints": nav_integrity["summary"]["gps_spoofed_waypoints"],
                "ais_reliable": not nav_integrity["summary"]["ais_unreliable_throughout"],
                "overall_nav_risk": nav_integrity["summary"]["overall_nav_risk"],
                "recommendation": nav_integrity["recommendation"],
            },
        },
        "phase_7_commercial_viability": {
            "viability": commercial["viability"],
            "reason": commercial["reason"],
            "insurance_gap": commercial["insurance_status"]["insurance_gap"],
            "pi_clubs_cancelled": commercial["insurance_status"]["pi_clubs_cancelled"],
            "total_exposure": commercial["financial_exposure"]["total_exposure"],
            "diplomatic_picture": {
                "iran_policy": diplomatic["iran_stated_policy"],
                "non_iranian_transits_today": diplomatic["total_non_iranian_transits_march_18"],
                "denied_flags": diplomatic["denied_flags"],
                "granted_flags": diplomatic["granted_flags"],
            },
        },
        "phase_8_operational_context": {
            "bypass_pipelines": {
                "total_flow_mbpd": bypass["total_bypass_flow_mbpd"],
                "supply_gap_mbpd": bypass["supply_gap_mbpd"],
                "supply_gap_pct": bypass["supply_gap_pct"],
                "stranded_production_mbpd": bypass["stranded_production_mbpd"],
            },
            "stranded_fleet": {
                "vessels": stranded["stranded_vessels"],
                "crew_at_risk": stranded["crew_at_risk"],
                "cargo_value": stranded["financial_impact"]["cargo_value_stranded"],
                "evacuation_convoys": stranded["evacuation_plan"]["convoys_needed"],
            },
            "coalition_status": {
                "current": "US only",
                "us_only_throughput": coalition["scenarios"]["us_alone"]["oil_mbpd"],
                "full_coalition_throughput": coalition["scenarios"]["full_coalition"]["oil_mbpd"],
                "political_reality": coalition["political_reality"],
            },
        },
        "phase_9_strike_campaign": {
            "threat_degradation": {
                "overall_threat_index": degradation["overall_threat_index"],
                "combat_flights": degradation["operation_epic_fury"]["combat_flights"],
                "naval_destroyed": degradation["operation_epic_fury"]["naval_vessels_destroyed"],
                "escort_feasible": degradation["projection"]["escort_ops_feasible"],
                "lucas_deployed": True,
            },
            "underground_arsenal": {
                "hidden_usv": arsenal["estimated_hidden_stockpile"]["usv_explosive"],
                "hidden_mines": arsenal["estimated_hidden_stockpile"]["naval_mines"],
                "bda_confidence": arsenal["bda_confidence"],
                "risk": arsenal["risk_to_convoy"][:100],
            },
        },
        "phase_10_economic_warfare": {
            "bypass_attacks": {
                "total_attacks_on_bypass": bypass_attacks["total_attacks"],
                "facilities_shut_down": bypass_attacks["facilities_shut_down"],
                "adjusted_bypass_mbpd": bypass_attacks["adjusted_bypass_capacity"]["total_adjusted"],
                "yanbu_status": bypass_attacks["yanbu_vulnerability"]["status"],
            },
            "production_cascade": {
                "force_majeures": len(cascade["force_majeure_declarations"]),
                "oil_price_current": cascade["oil_price"]["current"],
                "oil_price_peak": cascade["oil_price"]["peak"],
                "offline_mbpd": cascade["current_offline_mbpd"],
            },
            "force_posture": {
                "current_ddg": reinforcements["current_forces"]["ddg"],
                "current_ssn": reinforcements["current_forces"]["ssn"],
                "inbound_units": len(reinforcements["inbound_reinforcements"]),
                "escort_readiness": reinforcements["escort_readiness"][:80],
            },
        },
        "phase_11_predictive_planning": {
            "aip_recommendation": aip_eval["unified_recommendation"],
            "aip_reason": aip_eval["reason"],
            "flag_summary": aip_eval["flag_summary"],
            "monte_carlo": {
                "simulations": monte_carlo["n_simulations"],
                "mean_survivability": monte_carlo["survivability_distribution"]["mean"],
                "p5_survivability": monte_carlo["survivability_distribution"]["p5"],
                "go_probability": monte_carlo["mission_feasible_confidence"],
            },
        },
        "phase_12_platform_capabilities": {
            "coalition_sensors": {
                "sonar_coverage_km2": nexus["fusion_gains"]["sonar_coverage_km2"],
                "air_detection_km": nexus["fusion_gains"]["air_detection_range_km"],
            },
            "pnt_resilience": pnt["current_best_option"][:80],
            "non_kinetic_effects": f"{nke['critical_targets']} critical emitters, {len(nke['synchronization_timeline'])} timeline steps",
            "nke_silence_windows": {
                "count": len(nke_win["electronic_silence_windows"]),
                "coverage_pct": nke_win["coverage_pct"],
            },
            "economic_cost_of_inaction": econ["cumulative_economic_cost"]["cumulative_18_days_usd"],
            "cascade_chains": {
                "triggered": cascades["triggered_count"],
                "gdp_at_risk": cascades["total_gdp_at_risk"],
            },
            "foundry_cycle": foundry["cycle_time"],
            "magazine_depth": magazine["overall_assessment"][:80],
            "magazine_auto_adjust": {
                "base_survivability": mag_adjust["base_survivability"],
                "adjusted_survivability": mag_adjust["adjusted_survivability"],
                "supply_alert": mag_adjust["supply_chain_alert"],
            },
            "swarm_defense": {
                "formation": swarm["formation_name"],
                "drones_committed": swarm["drones_committed"],
                "fleet_survivability": swarm["fleet_survivability_with_swarm"],
                "mc_improvement": swarm_mc["swarm_improvement"],
            },
            "xai_consensus": {
                "status": consensus["consensus"],
                "mean_risk": consensus["mean_risk"],
                "divergence": consensus["max_divergence"],
                "confidence": consensus["confidence"][:60],
                "data_poison_risk": consensus["data_poison_risk"],
            },
            "coalition_sharing": {
                "nations_connected": len(sharing["nation_access"]),
                "highest_level": "FVEY_TS_SCI",
            },
        },
        "planning_timestamp": datetime.now().isoformat(),
    }


# ══════════════════════════════════════
# INTEGRATED: Full Situational Picture
# ══════════════════════════════════════

@app.get("/api/situation")
async def get_full_situation():
    """Get the complete integrated operational picture."""
    return {
        "vessels": vessel_detector.get_all_vessels(),
        "anomalies": vessel_detector.get_anomalies(0.7),
        "disguised": disguised_detector.get_flagged_vessels(),
        "atmosphere": atmos_model.get_threat_overlay(),
        "advantage_windows": atmos_model.get_advantage_windows(),
        "mines": mine_model.get_probability_grid(),
        "enforcement": enforcement_tracker.get_patterns(),
        "timestamp": "2026-03-18T12:00:00Z",
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
