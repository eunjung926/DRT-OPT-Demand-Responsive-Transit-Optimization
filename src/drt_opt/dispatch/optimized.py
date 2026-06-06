from __future__ import annotations

from drt_opt.dispatch.base import AssignmentResult, Dispatcher
from drt_opt.dispatch.constraints import total_detour_time, validate_candidate
from drt_opt.dispatch.insertion import (
    enumerate_insertions,
    get_pickup_time,
    route_travel_distance,
    simulate_route_times,
)
from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.models.request import Request
from drt_opt.models.vehicle import Vehicle


def demand_bonus(route: list, demand_map: dict[str, float]) -> float:
    return sum(demand_map.get(s.stop_id, 0.0) for s in route)


def compute_cost(
    request: Request,
    old_route: list,
    new_route: list,
    onboard_requests: dict[str, Request],
    old_distance: float,
    new_distance: float,
    demand_map: dict[str, float],
    weights: dict[str, float],
) -> float:
    pickup = get_pickup_time(new_route, request.id)
    wait = (pickup - request.request_time) if pickup is not None else 999.0
    detour = total_detour_time(old_route, new_route, onboard_requests)
    added_dist = max(0.0, new_distance - old_distance)
    bonus = demand_bonus(new_route, demand_map)

    return (
        weights["wait"] * wait
        + weights["detour"] * detour
        + weights["distance"] * added_dist
        - weights["demand"] * bonus
    )


class OptimizedDispatcher(Dispatcher):
    """Proposed: insertion enumeration + constraints + demand-weighted scoring."""

    def __init__(self, config: dict):
        self.capacity = config["vehicle"]["capacity"]
        self.dwell_time = config["simulation"]["dwell_time_min"]
        self.weights = config["weights"]
        self.max_new_wait = config["constraints"]["max_new_wait_min"]
        self.max_existing_delay = config["constraints"]["max_existing_delay_min"]
        self.max_vehicle_distance_min = 30.0
        self.max_route_stops = 20

    def assign(
        self,
        request: Request,
        vehicles: list[Vehicle],
        travel: TravelMatrix,
        current_time: float,
        demand_map: dict[str, float],
    ) -> AssignmentResult:
        best: AssignmentResult | None = None

        for vehicle in vehicles:
            dist_to_origin = travel.time(vehicle.current_location, request.origin_stop_id)
            if dist_to_origin > self.max_vehicle_distance_min:
                continue

            old_route = [s.copy() for s in vehicle.route]
            if len(old_route) > self.max_route_stops:
                continue

            old_distance = route_travel_distance(old_route, vehicle.current_location, travel)
            start_time = max(current_time, vehicle.busy_until)
            candidates = enumerate_insertions(old_route, request)

            for candidate in candidates:
                timed = simulate_route_times(
                    candidate,
                    travel,
                    vehicle.current_location,
                    start_time,
                    self.dwell_time,
                )

                if not validate_candidate(
                    old_route,
                    timed,
                    request,
                    vehicle.onboard,
                    self.capacity,
                    self.max_new_wait,
                    self.max_existing_delay,
                ):
                    continue

                new_distance = route_travel_distance(timed, vehicle.current_location, travel)
                cost = compute_cost(
                    request,
                    old_route,
                    timed,
                    vehicle.onboard,
                    old_distance,
                    new_distance,
                    demand_map,
                    self.weights,
                )

                if best is None or cost < best.cost:
                    best = AssignmentResult(
                        success=True,
                        vehicle=vehicle,
                        route=timed,
                        cost=cost,
                    )

        if best is None:
            return AssignmentResult(success=False)
        return best
