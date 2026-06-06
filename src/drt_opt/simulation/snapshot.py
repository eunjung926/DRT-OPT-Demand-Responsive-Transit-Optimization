from __future__ import annotations

from drt_opt.metrics.collector import MetricsCollector
from drt_opt.models.stop import Stop
from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.simulation.recorder import SimSnapshot
from drt_opt.simulation.state import SimulationState


def _min_to_label(minutes: float) -> str:
    h = int(minutes // 60) % 24
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"


def _stop_coords(stops_by_id: dict[str, Stop], stop_id: str) -> tuple[float, float]:
    s = stops_by_id[stop_id]
    return s.lat, s.lon


def _interpolate_vehicle(
    vehicle,
    stops_by_id: dict[str, Stop],
    current_time: float,
    travel: TravelMatrix,
) -> tuple[float, float]:
    """Interpolate vehicle position along active route for smooth animation."""
    base_lat, base_lon = _stop_coords(stops_by_id, vehicle.current_location)

    if not vehicle.route:
        return base_lat, base_lon

    prev_stop = vehicle.current_location
    prev_time = vehicle.busy_until - 0.5

    for rs in vehicle.route:
        if rs.arrival_time <= current_time:
            prev_stop = rs.stop_id
            prev_time = rs.depart_time
            continue

        if rs.arrival_time > current_time >= prev_time:
            total = rs.arrival_time - prev_time
            if total <= 0:
                return _stop_coords(stops_by_id, rs.stop_id)

            frac = (current_time - prev_time) / total
            lat1, lon1 = _stop_coords(stops_by_id, prev_stop)
            lat2, lon2 = _stop_coords(stops_by_id, rs.stop_id)
            return lat1 + (lat2 - lat1) * frac, lon1 + (lon2 - lon1) * frac

        break

    return base_lat, base_lon


def build_snapshot(
    state: SimulationState,
    metrics: MetricsCollector,
    stops_by_id: dict[str, Stop],
    travel: TravelMatrix,
    algorithm: str,
) -> SimSnapshot:
    vehicles_data = []
    for v in state.vehicles:
        lat, lon = _interpolate_vehicle(v, stops_by_id, state.current_time, travel)
        route_coords = []
        for rs in v.route[:12]:
            c = stops_by_id.get(rs.stop_id)
            if c:
                route_coords.append([c.lat, c.lon])

        vehicles_data.append(
            {
                "id": v.id,
                "lat": round(lat, 6),
                "lon": round(lon, 6),
                "onboard": len(v.onboard),
                "route": route_coords,
            }
        )

    recent = []
    for r in state.requests[-12:]:
        o = stops_by_id.get(r.origin_stop_id)
        d = stops_by_id.get(r.dest_stop_id)
        if o and d:
            recent.append(
                {
                    "id": r.id,
                    "status": r.status,
                    "origin": [o.lat, o.lon],
                    "dest": [d.lat, d.lon],
                    "wait": round(r.pickup_time - r.request_time, 1)
                    if r.pickup_time is not None
                    else None,
                }
            )

    pending_markers = []
    rejected_markers = []
    for r in state.requests:
        o = stops_by_id.get(r.origin_stop_id)
        if not o:
            continue
        pt = {"id": r.id, "lat": o.lat, "lon": o.lon}
        if r.status == "pending":
            pending_markers.append(pt)
        elif r.status == "rejected":
            rejected_markers.append(pt)

    pending_count = len(pending_markers)

    wait_avg = (
        sum(metrics.wait_times) / len(metrics.wait_times) if metrics.wait_times else 0.0
    )
    success = (
        metrics.assigned_requests / metrics.total_requests if metrics.total_requests else 0.0
    )

    return SimSnapshot(
        time_min=state.current_time,
        time_label=_min_to_label(state.current_time),
        vehicles=vehicles_data,
        recent_requests=recent,
        metrics={
            "total_requests": metrics.total_requests,
            "assigned": metrics.assigned_requests,
            "rejected": metrics.rejected_requests,
            "served": metrics.served_requests,
            "pending": pending_count,
            "avg_wait_min": round(wait_avg, 2),
            "success_rate": round(success, 4),
            "distance_km": round(metrics.total_distance_km, 1),
        },
        pending_markers=pending_markers[-25:],
        rejected_markers=rejected_markers[-40:],
    )
