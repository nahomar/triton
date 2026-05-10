"""
TRITON M47 — Integration Adapter

Exposes the streaming engine to the rest of the TRITON platform:

  1. `AlertBus`        — in-process pub/sub for downstream modules
  2. `aip_subscriber`  — pushes severity-≥4 alerts into M26 AIP's input queue
  3. FastAPI router    — REST snapshot endpoint + WebSocket live feed

The M47 engine is intentionally domain-restricted: it emits typed perimeter
events. It does NOT make GO/NO-GO decisions. M26 AIP consumes the M47 alert
stream as ONE of many evidence sources (alongside M7 insurance, M3 mine
posterior, M21 BDA, etc.). This boundary is what keeps the streaming hot
path at sub-millisecond latency — strategic reasoning happens off the
ingest path.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Callable, List, Optional

from perimeter_engine import Alert, AISMessage, PerimeterEngine
from geofences import all_fences


class AlertBus:
    """Minimal in-process pub/sub. One bus instance per TRITON deployment."""
    def __init__(self) -> None:
        self._subs: List[Callable[[Alert], None]] = []
        self._async_subs: List[asyncio.Queue] = []

    def subscribe(self, fn: Callable[[Alert], None]) -> None:
        self._subs.append(fn)

    def subscribe_async(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=10_000)
        self._async_subs.append(q)
        return q

    def publish(self, alert: Alert) -> None:
        for fn in self._subs:
            fn(alert)
        for q in self._async_subs:
            try:
                q.put_nowait(alert)
            except asyncio.QueueFull:
                # Drop on slow consumer rather than back-pressure the hot path.
                pass


def make_aip_subscriber(aip_input_queue) -> Callable[[Alert], None]:
    """
    Returns a handler that forwards severity-≥4 alerts into the M26 AIP
    orchestrator's input queue. AIP runs on a slower cadence (every few
    seconds) and aggregates streaming evidence into its scenario branches.
    """
    def handler(alert: Alert) -> None:
        if alert.severity >= 4:
            try:
                aip_input_queue.put_nowait({
                    "source": "M47",
                    "alert_type": alert.alert_type,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp,
                    "mmsi": alert.mmsi,
                    "fence_id": alert.fence_id,
                    "detail": alert.detail,
                })
            except Exception:
                pass
    return handler


# ============================================================
# FastAPI bridge
# ============================================================
def build_router(engine: PerimeterEngine, bus: AlertBus):
    """
    Returns an APIRouter to mount inside the existing TRITON FastAPI app.
    Endpoints:
        GET  /m47/status           — engine snapshot (vessels, fences, latency)
        GET  /m47/fences           — list active geofences
        GET  /m47/vessels          — list tracked vessels
        WS   /m47/alerts/stream    — push live alert feed to the dashboard
    """
    try:
        from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    except ImportError:
        return None

    router = APIRouter(prefix="/m47", tags=["m47-perimeter"])

    @router.get("/status")
    def status():
        return {
            "module": "M47",
            "name": "Streaming Perimeter Engine",
            "fences_active": len(engine.fences),
            "vessels_tracked": len(engine.vessels),
            "alerts_emitted": engine.alerts_emitted,
            "latency": engine.latency_percentiles(),
        }

    @router.get("/fences")
    def fences():
        return [
            {
                "fence_id": f.fence_id,
                "name": f.name,
                "zone_type": f.zone_type,
                "severity": f.severity,
                "bbox": list(f.polygon.bounds),
            }
            for f in engine.fences
        ]

    @router.get("/vessels")
    def vessels():
        return [
            {
                "mmsi": vs.mmsi,
                "msg_count": vs.msg_count,
                "last_seen": vs.last_seen,
                "last_position": vs.last_position,
                "last_sog": vs.last_sog,
                "inside_fences": list(vs.inside_fences),
            }
            for vs in list(engine.vessels.values())[:200]   # cap for snapshot
        ]

    @router.websocket("/alerts/stream")
    async def alerts_ws(ws: WebSocket):
        await ws.accept()
        q = bus.subscribe_async()
        try:
            while True:
                alert: Alert = await q.get()
                await ws.send_text(json.dumps({
                    "type": alert.alert_type,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp,
                    "mmsi": alert.mmsi,
                    "fence_id": alert.fence_id,
                    "detail": alert.detail,
                    "latency_us": alert.detection_latency_us,
                }))
        except WebSocketDisconnect:
            return

    return router


def wire_engine_to_bus(engine: PerimeterEngine, bus: AlertBus) -> None:
    """Convenience: route engine alerts straight onto the bus."""
    engine.on_alert(bus.publish)


if __name__ == "__main__":
    # Plumbing test — no FastAPI server, just verify the bus dispatch.
    bus = AlertBus()
    engine = PerimeterEngine(fences=all_fences())
    wire_engine_to_bus(engine, bus)

    aip_inbox = []
    class FakeQueue:
        def put_nowait(self, item): aip_inbox.append(item)
    bus.subscribe(make_aip_subscriber(FakeQueue()))

    # Push a tanker through the choke
    import time
    base = time.time()
    for i, (lon, lat) in enumerate([(57.2, 26.4), (56.7, 26.65), (56.4, 26.85)]):
        engine.ingest(AISMessage(
            mmsi=636017123, timestamp=base + i*60, lat=lat, lon=lon,
            sog=14.0, cog=270, vessel_class="tanker", name="MT TEST",
        ))

    print(f"AIP received {len(aip_inbox)} severity-≥4 alerts:")
    for a in aip_inbox:
        print(f"  [{a['severity']}] {a['alert_type']:18s} {a['detail']}")
