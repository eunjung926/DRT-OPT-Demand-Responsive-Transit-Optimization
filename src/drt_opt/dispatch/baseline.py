from __future__ import annotations

from drt_opt.dispatch.base import AssignmentResult, Dispatcher
from drt_opt.dispatch.insertion import simulate_route_times
from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.models.request import Request
from drt_opt.models.route import RouteStop
from drt_opt.models.vehicle import Vehicle


class NearestVehicleDispatcher(Dispatcher):
    """Baseline: assign to nearest vehicle with end-of-route append."""

    def __init__(self, config: dict):
        self.capacity = config["vehicle"]["capacity"]
        self.dwell_time = config["simulation"]["dwell_time_min"]

    def assign(
        self,
        request: Request,
        vehicles: list[Vehicle],
        travel: TravelMatrix,
        current_time: float,
        demand_map: dict[str, float],
    ) -> AssignmentResult:
        sorted_vehicles = sorted(
            vehicles,
            key=lambda v: travel.time(v.current_location, request.origin_stop_id),
        )

        for vehicle in sorted_vehicles:
            pickup = RouteStop(
                stop_id=request.origin_stop_id,
                action="pickup",
                request_id=request.id,
            )
            dropoff = RouteStop(
                stop_id=request.dest_stop_id,
                action="dropoff",
                request_id=request.id,
            )
            new_route = [s.copy() for s in vehicle.route] + [pickup, dropoff]

            start_time = max(current_time, vehicle.busy_until)
            timed_route = simulate_route_times(
                new_route,
                travel,
                vehicle.current_location,
                start_time,
                self.dwell_time,
            )

            if len(vehicle.onboard) + 1 > self.capacity:
                continue

            pickup_time = timed_route[-2].arrival_time
            if pickup_time - request.request_time > 60:
                continue

            return AssignmentResult(success=True, vehicle=vehicle, route=timed_route)

        return AssignmentResult(success=False)
