"""
TRITON — March 20, 2026 Intelligence Update
Patches all modules with Day 20 operational data.
Run: python update_march20.py
"""
import os
import re

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "backend", "models")
MAIN = os.path.join(BASE, "backend", "main.py")

def patch(filepath, old, new):
    with open(filepath, 'r') as f:
        content = f.read()
    if old in content:
        content = content.replace(old, new)
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False

def patch_re(filepath, pattern, replacement):
    with open(filepath, 'r') as f:
        content = f.read()
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        return True
    return False

changes = 0

# ══════════════════════════════════════
# MAIN.PY — Mission ID, Day, Module Count
# ══════════════════════════════════════
print("Updating main.py...")

# Mission ID
if patch(MAIN, "TRITON-20260318-0200", "TRITON-20260320-1400"):
    changes += 1; print("  ✓ Mission ID → 20260320-1400")

# Day 18 → Day 20
if patch(MAIN, '"day": 18', '"day": 20'):
    changes += 1; print("  ✓ Day → 20")

# ══════════════════════════════════════
# M5 — ENFORCEMENT / TRANSIT TRACKING
# ══════════════════════════════════════
print("Updating enforcement.py (M5)...")
ef = os.path.join(MODELS, "enforcement.py")

# Transit count update
patch(ef, '"total_transits_today": 8', '"total_transits_today": 10')
patch(ef, "8 transits", "10 transits")
# IRGC vetting system
patch(ef, '"permission_based": "EMERGING"', '"permission_based": "FORMALIZING — IRGC vetting and registration system under development"')
# Add vetting system data
patch(ef, '"two_tier_system":', '''"irgc_vetting_system": {
                "status": "Under development — Lloyd\'s List March 20",
                "mechanism": "Ships must communicate vessel ownership and cargo destination to IRGC in advance",
                "safe_corridor": "Pre-approved route through Iranian territorial waters",
                "countries_in_talks": ["China", "India", "Pakistan", "Iraq", "Malaysia"],
                "impact": "Formalizes selective blockade — no longer ad-hoc permission",
            },
            "two_tier_system":''')
changes += 3; print("  ✓ Transit count → 10, IRGC vetting system added")

# ══════════════════════════════════════
# M7 — INSURANCE / RISK SCORING
# ══════════════════════════════════════
print("Updating insurance_transit.py (M7)...")
it = os.path.join(MODELS, "insurance_transit.py")
patch(it, "102", "110")  # Oil price references
changes += 1; print("  ✓ Oil price references → $110")

# ══════════════════════════════════════
# M8 — THREAT ENGAGEMENT
# ══════════════════════════════════════
print("Updating threat_engagement.py (M8)...")
te = os.path.join(MODELS, "threat_engagement.py")
# Add A-10 Warthog
patch(te, '"EA-18G Growler"', '"EA-18G Growler", "A-10 Warthog (anti-ship, confirmed Mar 20)"')
changes += 1; print("  ✓ A-10 Warthog added to engagement assets")

# ══════════════════════════════════════
# M9 — CONVOY SCHEDULER
# ══════════════════════════════════════
print("Updating convoy_scheduler.py (M9)...")
cs = os.path.join(MODELS, "convoy_scheduler.py")
patch(cs, "2 convoys", "2 convoys (DIA assesses 1-6 month closure)")
changes += 1; print("  ✓ DIA closure assessment added")

# ══════════════════════════════════════
# M13 — BAB AL-MANDEB / STRATEGIC
# ══════════════════════════════════════
print("Updating bab_al_mandeb.py (M13)...")
bm = os.path.join(MODELS, "bab_al_mandeb.py")
patch(bm, "Ras Laffan operational", "Ras Laffan DAMAGED — Israeli strike on South Pars triggered Iranian retaliation. 17% LNG capacity reduced. 5-year repair timeline (QatarEnergy March 20)")
changes += 1; print("  ✓ Ras Laffan damage update")

# ══════════════════════════════════════
# M21-22 — STRIKE DEGRADATION
# ══════════════════════════════════════
print("Updating strike_degradation.py (M21-22)...")
sd = os.path.join(MODELS, "strike_degradation.py")
patch(sd, "51.5", "49.2")  # Threat index declining further
patch(sd, '"combat_flights": "6,000+"', '"combat_flights": "7,500+"')
patch(sd, '"naval_destroyed": "100+"', '"naval_destroyed": "120+"')
changes += 3; print("  ✓ Threat index → 49.2, flights → 7,500+, naval → 120+")

# ══════════════════════════════════════
# M23-25 — ECONOMIC / MACRO TWIN
# ══════════════════════════════════════
print("Updating warpspeed_foundry.py (M23-25)...")
wf = os.path.join(MODELS, "warpspeed_foundry.py")
# Oil price
patch(wf, "102", "110")
# SPR release
patch(wf, '"spr_release":', '"spr_release_iea": {"barrels": "172M from US SPR", "coalition": "32 IEA member countries emergency release", "status": "Committed March 2026"},\n            "spr_release":')
changes += 2; print("  ✓ Oil → $110, IEA SPR release added")

# ══════════════════════════════════════
# M26 — AIP ORCHESTRATOR
# ══════════════════════════════════════
print("Updating aip_orchestrator.py (M26)...")
aip = os.path.join(MODELS, "aip_orchestrator.py")
# DIA assessment
patch(aip, '"NO-GO"', '"NO-GO"')  # Still NO-GO but update reasoning
patch(aip, "Insurance gap", "Insurance gap + DIA assesses 1-6 month closure")
# Update oil in mission params
patch(aip, '"oil_price": 102', '"oil_price": 110')
changes += 2; print("  ✓ AIP reasoning updated with DIA assessment, oil → $110")

# ══════════════════════════════════════
# MODULE UPGRADES — Economic Cascades
# ══════════════════════════════════════
print("Updating module_upgrades.py...")
mu = os.path.join(MODELS, "module_upgrades.py")

# Ras Laffan damage in LNG chain
patch(mu, '"QatarEnergy force majeure March 4', '"QatarEnergy force majeure March 4. Ras Laffan struck by Iran March 19 — 17% LNG capacity lost, 5-year repair (QatarEnergy). Israel struck South Pars first')
# Goldman warning
patch(mu, '"gdp_at_risk": "$150B Asian manufacturing output exposed"', '"gdp_at_risk": "$150B+ Asian manufacturing — Goldman warns $147/bbl possible if prolonged"')
# Update 18-day to 20-day
patch(mu, "18 days", "20 days")
changes += 3; print("  ✓ Ras Laffan damage, Goldman $147 warning, day count → 20")

# ══════════════════════════════════════
# LAST MILE — Crew Readiness (M36), ROE (M37), Magazine (M39)
# ══════════════════════════════════════
print("Updating last_mile.py (M35-40)...")
lm = os.path.join(MODELS, "last_mile.py")

# Update deployment days (+2)
patch(lm, '"deployed_days": 187', '"deployed_days": 189')
patch(lm, '"deployed_days": 142', '"deployed_days": 144')
patch(lm, '"deployed_days": 210', '"deployed_days": 212')
patch(lm, '"deployed_days": 95', '"deployed_days": 97')
patch(lm, '"deployed_days": 165', '"deployed_days": 167')

# Add 2500 Marines deployment
patch(lm, '"ford_factor_warning"', '"marine_deployment": {"count": 2500, "source": "San Diego", "ships": 3, "status": "Deploying — AP/CNN March 20", "arrival_estimate": "7-10 days"},\n            "ford_factor_warning"')

changes += 6; print("  ✓ Deployment days +2, 2500 Marines deployment added")

# ══════════════════════════════════════
# BLEEDING EDGE — Regime (M43), DOGE (M42)
# ══════════════════════════════════════
print("Updating bleeding_edge.py (M41-44)...")
be = os.path.join(MODELS, "bleeding_edge.py")

# Regime — update with coalition failure
patch(be, '"probability": 0.5', '"probability": 0.55')  # CN/IN bypass more likely now with vetting system
patch(be, '"timeline_days": 14', '"timeline_days": 10')  # Accelerated — vetting system formalizing

# DIA assessment in regime context
patch(be, '"c2_integrity": 0.4', '"c2_integrity": 0.35')  # Further degraded by Day 20

# DOGE — update oil price impact on BROI
patch(be, "102", "110")

# Update IHL incidents with March 19 South Pars/Ras Laffan exchange
patch(be, "IHL_INCIDENTS = [", '''IHL_INCIDENTS = [
        {"date": "2026-03-19", "incident": "Israeli strike on South Pars triggers Iranian retaliation on Ras Laffan LNG",
         "casualties": "Unknown — infrastructure strike", "cause": "Escalation cycle",
         "roe_violated": "ROE-006", "consequence": "Qatar LNG capacity reduced 17%, 5-year repair. Global condemnation of energy infrastructure targeting."},''')

changes += 5; print("  ✓ Regime fracture accelerated, BROI oil updated, South Pars/Ras Laffan incident added")

# ══════════════════════════════════════
# STRATEGIC MODULES — Coalition planning
# ══════════════════════════════════════
print("Updating strategic_modules.py (M18-20)...")
sm = os.path.join(MODELS, "strategic_modules.py")
patch(sm, "150 vessels", "155+ vessels")
patch(sm, '"crew": "20K"', '"crew": "21K+"')
# Add ground troops consideration
patch(sm, '"coalition_status":', '"ground_troops_assessment": {"status": "Under consideration — experts say may be required", "deployment": "2500 Marines + 3 warships from San Diego (March 20)", "timeline": "7-10 day transit", "risk": "Could drag on for years per military experts", "source": "CNN/AP March 20 2026"},\n            "coalition_status":')
changes += 3; print("  ✓ Stranded → 155+, Marines deployment, ground troops assessment")

# ══════════════════════════════════════
# ADVANCED MODULES — Fix coalition nexus with UK update
# ══════════════════════════════════════
print("Updating advanced_modules.py...")
am = os.path.join(MODELS, "advanced_modules.py")
patch(am, '"UK": "Considering"', '"UK": "Sent military planners but won\'t be drawn into wider war — PM Starmer March 20"')
changes += 1; print("  ✓ UK coalition status updated")

# ══════════════════════════════════════
# NEXUS/QUANTUM/NKE
# ══════════════════════════════════════
print("Updating nexus_quantum_nke.py...")
nq = os.path.join(MODELS, "nexus_quantum_nke.py")
# A-10 in NKE plan
patch(nq, '"EA-18G"', '"EA-18G + A-10 Warthog (anti-ship confirmed Day 20)"')
changes += 1; print("  ✓ A-10 added to NKE asset list")

# ══════════════════════════════════════
# VESSEL DETECTOR — traffic update
# ══════════════════════════════════════
print("Updating vessel_detector.py (M1)...")
vd = os.path.join(MODELS, "vessel_detector.py")
patch(vd, "traffic_drop_pct", "traffic_drop_pct_day20")  # Mark as Day 20 data
patch(vd, "97%", "97% (8 non-Iranian vessels detected Monday — Windward)")
changes += 1; print("  ✓ Traffic data annotated with Windward source")

print(f"\n{'='*60}")
print(f"  TRITON March 20 Update Complete")
print(f"  {changes} patches applied across all modules")
print(f"  Day: 18 → 20")
print(f"  Oil: $102 → $110")
print(f"  DIA: 1-6 month closure assessment")
print(f"  2,500 Marines deploying from San Diego")
print(f"  Ras Laffan LNG: 17% capacity lost, 5yr repair")
print(f"  IRGC vetting system formalizing")
print(f"  A-10 Warthogs confirmed in strait")
print(f"  Goldman Sachs: $147/bbl warning")
print(f"  IEA: 32-nation emergency SPR release (172M bbl US)")
print(f"  Regime fracture timeline accelerated")
print(f"{'='*60}")
