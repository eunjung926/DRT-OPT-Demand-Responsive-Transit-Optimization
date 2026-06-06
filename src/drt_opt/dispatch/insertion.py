from __future__ import annotations

import copy

from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.models.request import Request
from drt_opt.models.route import RouteStop


def enumerate_insertions(
    route: list[RouteStop],
    request: Request,
) -> list[list[RouteStop]]:
    """Generate all valid pickup-before-dropoff insertion candidates."""
    n = len(route)
    candidates: list[list[RouteStop]] = []

    for i in range(n + 1):
        for j in range(i + 1, n + 2):
            new_route: list[RouteStop] = []
            for k in range(n + 2):
                if k == i:
                    new_route.append(
                        RouteStop(
                            stop_id=request.origin_stop_id,
                            action="pickup",
                            request_id=request.id,
                        )
                    )
                elif k == j:
                    new_route.append(
                        RouteStop(
                            stop_id=request.dest_stop_id,
                            action="dropoff",
                            request_id=request.id,
                        )
                    )
                else:
                    orig_k = k if k < i else (k - 1 if k < j else k - 2)
                    if orig_k < n:
                        new_route.append(route[orig_k].copy())

            candidates.append(new_route)

    return candidates


def simulate_route_times(
    route: list[RouteStop],
    travel: TravelMatrix,
    start_location: str,
    start_time: float,
    dwell_time_min: float = 0.5,
) -> list[RouteStop]:
    """Forward-simulate arrival/departure times along a route."""
    if not route:
        return []

    result: list[RouteStop] = []
    prev_stop = start_location
    t = start_time

    for stop in route:
        rs = stop.copy()
        if prev_stop != rs.stop_id:
            t += travel.time(prev_stop, rs.stop_id)
        rs.arrival_time = t
        rs.depart_time = t + dwell_time_min
        t = rs.depart_time
        prev_stop = rs.stop_id
        result.append(rs)

    return result


def route_travel_distance(
    route: list[RouteStop],
    start_location: str,
    travel: TravelMatrix,
) -> float:
    if not route:
        return 0.0
    total = travel.distance(start_location, route[0].stop_id)
    for i in range(1, len(route)):
        total += travel.distance(route[i - 1].stop_id, route[i].stop_id)
    return total


def get_pickup_time(route: list[RouteStop], request_id: str) -> float | None:
    for stop in route:
        if stop.request_id == request_id and stop.action == "pickup":
            return stop.arrival_time
    return None


def get_dropoff_time(route: list[RouteStop], request_id: str) -> float | None:
    for stop in route:
        if stop.request_id == request_id and stop.action == "dropoff":
            return stop.arrival_time
    return None


def passenger_travel_time(route: list[RouteStop], request_id: str) -> float | None:
    pickup = get_pickup_time(route, request_id)
    dropoff = get_dropoff_time(route, request_id)
    if pickup is None or dropoff is None:
        return None
    return dropoff - pickup
