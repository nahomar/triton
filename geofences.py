"""
TRITON M47 — Geofence Library

Real Persian Gulf zones used by the streaming perimeter engine.
Coordinates are in (lon, lat) per shapely convention.

Zone types:
  TSS_LANE     — IMO Traffic Separation Scheme corridors (Hormuz)
  CHOKEPOINT   — narrow transit zones with elevated risk
  EXCLUSION    — port buffers, terminal safety perimeters
  IRGC_ZONE    — known Iranian Revolutionary Guard operating areas
  PROTECTION   — naval task force defensive bubbles (mobile)
  SHADOW       — concealment-prone waters (island lee, archipelagos)

Each fence carries a severity tier that feeds the alert classifier and
ultimately the M26 AIP orchestrator's GO/NO-GO surface.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
from shapely.geometry import Polygon


@dataclass
class Geofence:
    fence_id: str
    name: str
    zone_type: str
    severity: int          # 1 (informational) to 5 (critical intrusion)
    polygon: Polygon
    metadata: dict = field(default_factory=dict)


def _poly(coords: List[Tuple[float, float]]) -> Polygon:
    """Build a polygon from (lon, lat) tuples."""
    return Polygon(coords)


# ============================================================
# STATIC FENCES — geographic, do not move
# ============================================================

STATIC_FENCES: List[Geofence] = [

    # Strait of Hormuz inbound TSS lane (eastbound, into the Gulf)
    Geofence(
        fence_id="HORMUZ_TSS_IN",
        name="Hormuz TSS Inbound Lane",
        zone_type="TSS_LANE",
        severity=2,
        polygon=_poly([
            (56.20, 26.35), (56.65, 26.45), (56.85, 26.55),
            (56.95, 26.70), (56.80, 26.78), (56.60, 26.70),
            (56.30, 26.55), (56.15, 26.45), (56.20, 26.35),
        ]),
        metadata={"width_nm": 2.0, "imo_governed": True},
    ),

    # Strait of Hormuz outbound TSS lane (westbound, out of the Gulf)
    Geofence(
        fence_id="HORMUZ_TSS_OUT",
        name="Hormuz TSS Outbound Lane",
        zone_type="TSS_LANE",
        severity=2,
        polygon=_poly([
            (56.18, 26.45), (56.55, 26.58), (56.78, 26.70),
            (56.85, 26.85), (56.70, 26.92), (56.50, 26.85),
            (56.20, 26.65), (56.10, 26.55), (56.18, 26.45),
        ]),
        metadata={"width_nm": 2.0, "imo_governed": True},
    ),

    # The choke itself — narrowest Hormuz waters where deviation = high signal
    Geofence(
        fence_id="HORMUZ_CHOKE",
        name="Hormuz Chokepoint Core",
        zone_type="CHOKEPOINT",
        severity=4,
        polygon=_poly([
            (56.30, 26.40), (56.95, 26.55), (57.05, 26.85),
            (56.40, 26.90), (56.10, 26.65), (56.30, 26.40),
        ]),
        metadata={"min_width_nm": 21.0, "throughput_pct_global_oil": 20},
    ),

    # Bandar Abbas IRGC-N FAC operating area (Iranian Navy fast attack craft)
    Geofence(
        fence_id="BANDAR_ABBAS_IRGC",
        name="Bandar Abbas IRGC-N FAC Zone",
        zone_type="IRGC_ZONE",
        severity=5,
        polygon=_poly([
            (55.95, 27.05), (56.55, 27.05), (56.65, 27.35),
            (56.20, 27.45), (55.85, 27.30), (55.95, 27.05),
        ]),
        metadata={"primary_threat": "FAC_swarm", "base_distance_km": 8},
    ),

    # Larak Island IRGC speedboat launch zone
    Geofence(
        fence_id="LARAK_IRGC",
        name="Larak Island IRGC Launch Zone",
        zone_type="IRGC_ZONE",
        severity=5,
        polygon=_poly([
            (56.30, 26.78), (56.55, 26.82), (56.60, 27.00),
            (56.40, 27.05), (56.25, 26.95), (56.30, 26.78),
        ]),
        metadata={"primary_threat": "speedboat_swarm"},
    ),

    # Qeshm Island shadow / AIS-dropout-prone waters
    Geofence(
        fence_id="QESHM_SHADOW",
        name="Qeshm Island Shadow Zone",
        zone_type="SHADOW",
        severity=3,
        polygon=_poly([
            (55.40, 26.55), (56.30, 26.45), (56.40, 26.85),
            (55.55, 26.92), (55.30, 26.78), (55.40, 26.55),
        ]),
        metadata={"ais_dropout_rate": 0.34, "concealment_score": 0.78},
    ),

    # Ras Tanura terminal exclusion (Saudi crude export)
    Geofence(
        fence_id="RAS_TANURA_EXCL",
        name="Ras Tanura Terminal Exclusion",
        zone_type="EXCLUSION",
        severity=4,
        polygon=_poly([
            (50.05, 26.55), (50.30, 26.55), (50.32, 26.75),
            (50.10, 26.78), (50.00, 26.65), (50.05, 26.55),
        ]),
        metadata={"throughput_mbpd": 6.5},
    ),

    # Kharg Island terminal (Iranian crude export, sanctioned)
    Geofence(
        fence_id="KHARG_EXCL",
        name="Kharg Island Terminal Buffer",
        zone_type="EXCLUSION",
        severity=4,
        polygon=_poly([
            (50.20, 29.18), (50.45, 29.18), (50.50, 29.32),
            (50.25, 29.35), (50.15, 29.28), (50.20, 29.18),
        ]),
        metadata={"throughput_mbpd": 1.4, "sanctioned": True},
    ),

    # Bab al-Mandeb chokepoint (secondary corridor)
    Geofence(
        fence_id="BAB_MANDEB",
        name="Bab al-Mandeb Chokepoint",
        zone_type="CHOKEPOINT",
        severity=4,
        polygon=_poly([
            (43.20, 12.50), (43.60, 12.45), (43.70, 12.85),
            (43.30, 12.95), (43.10, 12.75), (43.20, 12.50),
        ]),
        metadata={"throughput_pct_global_oil": 9},
    ),

    # NAVCENT / Fifth Fleet headquarters approach (Bahrain)
    Geofence(
        fence_id="NAVCENT_APPROACH",
        name="NAVCENT Bahrain Approach",
        zone_type="EXCLUSION",
        severity=4,
        polygon=_poly([
            (50.55, 26.18), (50.72, 26.20), (50.75, 26.35),
            (50.58, 26.38), (50.50, 26.28), (50.55, 26.18),
        ]),
        metadata={"command": "USNAVCENT"},
    ),
]


# ============================================================
# DYNAMIC FENCES — built at runtime from task force positions
# ============================================================

def csg_protection_bubble(
    fence_id: str,
    name: str,
    lat: float,
    lon: float,
    radius_nm: float = 5.0,
    severity: int = 5,
) -> Geofence:
    """
    Build a circular protection bubble around a moving naval asset.
    Approximates the circle as a 24-vertex polygon for STRtree compatibility.
    1 nm ≈ 1/60 degree latitude; longitude scales by cos(lat).
    """
    import math
    deg_per_nm_lat = 1.0 / 60.0
    deg_per_nm_lon = deg_per_nm_lat / max(math.cos(math.radians(lat)), 0.1)
    coords = []
    for i in range(24):
        theta = 2 * math.pi * i / 24
        dlat = radius_nm * deg_per_nm_lat * math.sin(theta)
        dlon = radius_nm * deg_per_nm_lon * math.cos(theta)
        coords.append((lon + dlon, lat + dlat))
    coords.append(coords[0])  # close the ring
    return Geofence(
        fence_id=fence_id,
        name=name,
        zone_type="PROTECTION",
        severity=severity,
        polygon=Polygon(coords),
        metadata={"radius_nm": radius_nm, "lat": lat, "lon": lon},
    )


def all_fences() -> List[Geofence]:
    """Return the full active fence set: static + current task force bubbles."""
    fences = list(STATIC_FENCES)
    # CVN-71 USS Theodore Roosevelt (notional position, central Gulf)
    fences.append(csg_protection_bubble(
        "CSG_TR_BUBBLE", "USS Theodore Roosevelt CSG",
        lat=26.10, lon=52.40, radius_nm=5.0, severity=5,
    ))
    # ESG (USS Bataan ARG, notional)
    fences.append(csg_protection_bubble(
        "ESG_BATAAN_BUBBLE", "USS Bataan ARG",
        lat=25.40, lon=53.85, radius_nm=3.0, severity=5,
    ))
    return fences


if __name__ == "__main__":
    fences = all_fences()
    print(f"TRITON M47 Geofence Library — {len(fences)} active fences")
    for f in fences:
        bbox = f.polygon.bounds
        print(f"  [{f.severity}] {f.fence_id:24s} {f.zone_type:11s} "
              f"area={f.polygon.area:.4f}  bbox=({bbox[0]:.2f},{bbox[1]:.2f})-({bbox[2]:.2f},{bbox[3]:.2f})")
