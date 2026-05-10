"""
TRITON M47 — AIS Stream Simulator

Generates realistic Persian Gulf AIS traffic for benchmarking and demos.

Vessel population (calibrated to AIS observability of the Gulf):
  - 60% commercial transit (cargo + tanker on TSS lanes)
  - 15% port approach / anchorage
  - 10% fishing dhow / coastal
  -  8% naval coalition
  -  4% IRGC fast attack (mostly silent, occasional bursts)
  -  3% adversarial actors (going dark, identity flips, loitering)

Output is an iterator over AISMessage with monotonically non-decreasing
timestamps, suitable for replaying through PerimeterEngine.ingest().
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Iterator, List, Tuple

from perimeter_engine import AISMessage


def _bearing_step(lat: float, lon: float, course_deg: float, sog_kn: float, dt_s: float):
    """Move (lat, lon) forward by course/speed for dt seconds. Flat-earth OK at this scale."""
    nm = sog_kn * (dt_s / 3600.0)
    deg_per_nm_lat = 1.0 / 60.0
    deg_per_nm_lon = deg_per_nm_lat / max(math.cos(math.radians(lat)), 0.1)
    rad = math.radians(course_deg)
    dlat = nm * deg_per_nm_lat * math.cos(rad)
    dlon = nm * deg_per_nm_lon * math.sin(rad)
    return lat + dlat, lon + dlon


@dataclass
class _Vessel:
    mmsi: int
    name: str
    vessel_class: str
    lat: float
    lon: float
    sog: float
    cog: float
    profile: str          # "transit", "anchor", "fishing", "naval", "irgc", "adversary"
    # adversarial flags
    will_go_dark: bool = False
    dark_until: float = 0.0
    will_flip_identity: bool = False
    flipped: bool = False
    # transit endpoint to give natural exit
    target_lat: float = 0.0
    target_lon: float = 0.0


class GulfTrafficSimulator:
    def __init__(self, n_vessels: int = 300, seed: int = 42, start_time: float = 1_710_000_000.0):
        self.rng = random.Random(seed)
        self.start_time = start_time
        self.now = start_time
        self.vessels: List[_Vessel] = []
        self._spawn(n_vessels)

    def _spawn(self, n: int) -> None:
        # Quotas
        n_transit = int(n * 0.60)
        n_anchor = int(n * 0.15)
        n_fishing = int(n * 0.10)
        n_naval = int(n * 0.08)
        n_irgc = int(n * 0.04)
        n_adversary = n - (n_transit + n_anchor + n_fishing + n_naval + n_irgc)

        for _ in range(n_transit):
            self.vessels.append(self._mk_transit())
        for _ in range(n_anchor):
            self.vessels.append(self._mk_anchor())
        for _ in range(n_fishing):
            self.vessels.append(self._mk_fishing())
        for _ in range(n_naval):
            self.vessels.append(self._mk_naval())
        for _ in range(n_irgc):
            self.vessels.append(self._mk_irgc())
        for _ in range(n_adversary):
            self.vessels.append(self._mk_adversary())

    def _new_mmsi(self) -> int:
        return self.rng.randint(200_000_000, 799_999_999)

    def _mk_transit(self) -> _Vessel:
        """Cargo or tanker on Hormuz TSS, either eastbound or westbound."""
        is_tanker = self.rng.random() < 0.55
        eastbound = self.rng.random() < 0.5
        if eastbound:
            lat = self.rng.uniform(26.45, 26.92); lon = 55.50 + self.rng.uniform(-0.5, 0.0)
            cog = 90 + self.rng.uniform(-15, 15)
            tlat, tlon = lat, 57.5
        else:
            lat = self.rng.uniform(26.35, 26.78); lon = 57.50 + self.rng.uniform(0.0, 0.5)
            cog = 270 + self.rng.uniform(-15, 15)
            tlat, tlon = lat, 55.0
        return _Vessel(
            mmsi=self._new_mmsi(),
            name=f"MV {self.rng.choice(['ATLAS','HORIZON','PEGASUS','ORION','POLARIS','VEGA'])}-{self.rng.randint(100,999)}",
            vessel_class="tanker" if is_tanker else "cargo",
            lat=lat, lon=lon, sog=self.rng.uniform(11, 16), cog=cog,
            profile="transit", target_lat=tlat, target_lon=tlon,
        )

    def _mk_anchor(self) -> _Vessel:
        """Anchored vessel at Fujairah or Bandar Abbas approach."""
        if self.rng.random() < 0.5:
            lat, lon = 25.15 + self.rng.uniform(-0.1, 0.1), 56.40 + self.rng.uniform(-0.1, 0.1)
        else:
            lat, lon = 27.20 + self.rng.uniform(-0.1, 0.1), 56.30 + self.rng.uniform(-0.1, 0.1)
        return _Vessel(
            mmsi=self._new_mmsi(), name=f"ANCHOR-{self.rng.randint(100,999)}",
            vessel_class="cargo", lat=lat, lon=lon, sog=0.0, cog=0.0, profile="anchor",
        )

    def _mk_fishing(self) -> _Vessel:
        lat = self.rng.uniform(25.0, 27.5); lon = self.rng.uniform(54.0, 56.5)
        return _Vessel(
            mmsi=self._new_mmsi(), name=f"FV-{self.rng.randint(100,999)}",
            vessel_class="fishing", lat=lat, lon=lon,
            sog=self.rng.uniform(2, 6), cog=self.rng.uniform(0, 360), profile="fishing",
        )

    def _mk_naval(self) -> _Vessel:
        # Cluster around the CSG bubble (~26.10, 52.40)
        lat = 26.10 + self.rng.uniform(-0.4, 0.4); lon = 52.40 + self.rng.uniform(-0.4, 0.4)
        return _Vessel(
            mmsi=self._new_mmsi(), name=f"USS-{self.rng.randint(50,99)}",
            vessel_class="naval", lat=lat, lon=lon,
            sog=self.rng.uniform(8, 18), cog=self.rng.uniform(0, 360), profile="naval",
        )

    def _mk_irgc(self) -> _Vessel:
        # Launches near Bandar Abbas or Larak
        if self.rng.random() < 0.5:
            lat, lon = 27.15 + self.rng.uniform(-0.1, 0.1), 56.20 + self.rng.uniform(-0.1, 0.1)
        else:
            lat, lon = 26.85 + self.rng.uniform(-0.05, 0.05), 56.40 + self.rng.uniform(-0.05, 0.05)
        return _Vessel(
            mmsi=self._new_mmsi(), name="",  # IRGC FAC don't broadcast names
            vessel_class="naval", lat=lat, lon=lon,
            sog=self.rng.uniform(20, 35), cog=self.rng.uniform(0, 360), profile="irgc",
        )

    def _mk_adversary(self) -> _Vessel:
        """Adversarial: shadow fleet tanker that may go dark or flip identity."""
        lat = self.rng.uniform(26.4, 26.9); lon = self.rng.uniform(55.5, 57.0)
        v = _Vessel(
            mmsi=self._new_mmsi(),
            name=self.rng.choice(["GHOST RIDER", "PHANTOM SEA", "DARK PASSAGE"]),
            vessel_class="tanker", lat=lat, lon=lon,
            sog=self.rng.uniform(10, 14), cog=270.0, profile="adversary",
            target_lat=lat, target_lon=55.0,
        )
        v.will_go_dark = self.rng.random() < 0.6
        v.will_flip_identity = self.rng.random() < 0.4
        v.dark_until = self.start_time + self.rng.uniform(120, 600)
        return v

    def _step_vessel(self, v: _Vessel, dt: float) -> None:
        if v.profile in ("anchor",):
            # Slight drift only
            v.lat += self.rng.uniform(-1e-5, 1e-5)
            v.lon += self.rng.uniform(-1e-5, 1e-5)
            return
        if v.profile == "fishing":
            # Slow random walk; reverse course occasionally
            if self.rng.random() < 0.05:
                v.cog = (v.cog + self.rng.uniform(-90, 90)) % 360
        if v.profile in ("transit", "adversary"):
            # Steer toward target
            dlat = v.target_lat - v.lat
            dlon = v.target_lon - v.lon
            v.cog = (math.degrees(math.atan2(dlon, dlat))) % 360
        if v.profile == "irgc":
            # Erratic course changes (FAC behavior)
            if self.rng.random() < 0.2:
                v.cog = (v.cog + self.rng.uniform(-60, 60)) % 360
        v.lat, v.lon = _bearing_step(v.lat, v.lon, v.cog, v.sog, dt)

    def stream(self, n_messages: int, message_interval_s: float = 10.0) -> Iterator[AISMessage]:
        """
        Emit messages one at a time. Each tick advances every vessel by
        message_interval_s seconds and emits one message per vessel.
        """
        emitted = 0
        while emitted < n_messages:
            self.now += message_interval_s
            for v in self.vessels:
                if emitted >= n_messages:
                    return
                self._step_vessel(v, message_interval_s)

                # Adversarial: going dark — skip messages during dark window
                if v.will_go_dark and self.now < v.dark_until:
                    continue

                # Adversarial: identity flip after some time
                effective_name = v.name
                if v.will_flip_identity and not v.flipped and self.now > self.start_time + 800:
                    v.name = self.rng.choice(["BLUE STAR", "OCEAN CALM", "PACIFIC LIGHT"])
                    v.flipped = True
                    effective_name = v.name

                # IRGC FAC broadcast probability is low (they often run AIS-off)
                if v.profile == "irgc" and self.rng.random() < 0.6:
                    continue

                yield AISMessage(
                    mmsi=v.mmsi,
                    timestamp=self.now,
                    lat=v.lat, lon=v.lon,
                    sog=v.sog, cog=v.cog,
                    vessel_class=v.vessel_class,
                    name=effective_name,
                )
                emitted += 1


if __name__ == "__main__":
    sim = GulfTrafficSimulator(n_vessels=20, seed=1)
    for i, msg in enumerate(sim.stream(n_messages=10)):
        print(f"  {i:2d} mmsi={msg.mmsi}  pos=({msg.lat:.3f},{msg.lon:.3f})  "
              f"sog={msg.sog:.1f}kn  class={msg.vessel_class}  name='{msg.name}'")
