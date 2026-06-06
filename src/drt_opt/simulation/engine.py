from __future__ import annotations

import heapq
from copy import deepcopy
from typing import Callable

from drt_opt.dispatch.base import Dispatcher
from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.metrics.collector import MetricsCollector
from drt_opt.models.request import Request
from drt_opt.models.stop import Stop
from drt_opt.models.vehicle import Vehicle
from drt_opt.simulation.events import EventType, SimulationEvent
from drt_opt.simulation.state import SimulationState


def create_vehicles(
    count: int,
    capacity: int,
    stops: list[Stop],
    start_time: float,
) -> list[Vehicle]:
    vehicles: list[Vehicle] = []
    for i in range(count):
        stop = stops[i % len(stops)]
        vehicles.append(
            Vehicle(
                id=f"V{i:02d}",
                capacity=capacity,
                current_location=stop.id,
                busy_until=start_time,
            )
        )
    return vehicles


class Simulator:
    def __init__(
        self,
        config: dict,
        stops: list[Stop],
        travel: TravelMatrix,
        demand_map: dict[str, float],
        dispatcher: Dispatcher,
    ):
        self.config = config
        self.stops = stops
        self.travel = travel
        self.demand_map = demand_map
        self.dispatcher = dispatcher
        self.dwell_time = config["simulation"]["dwell_time_min"]
        self.start_time = config["simulation"]["start_time_min"]
        self.end_time = config["simulation"]["end_time_min"]
        self._seq = 0

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def run(
        self,
        requests: list[Request],
        on_step: Callable | None = None,
    ) -> MetricsCollector:
        state = SimulationState(
            current_time=self.start_time,
            vehicles=create_vehicles(
                self.config["vehicle"]["count"],
                self.config["vehicle"]["capacity"],
                self.stops,
                self.start_time,
            ),
            requests=deepcopy(requests),
            stops=self.stops,
            travel=self.travel,
            demand_map=self.demand_map,
        )

        metrics = MetricsCollector(
            vehicle_count=len(state.vehicles),
            sim_duration=self.end_time - self.start_time,
            capacity=self.config["vehicle"]["capacity"],
        )

        event_queue: list[SimulationEvent] = []

        for req in state.requests:
            heapq.heappush(
                event_queue,
                SimulationEvent.create(
                    req.request_time,
                    self._next_seq(),
                    EventType.REQUEST,
                    {"request_id": req.id},
                ),
            )

        heapq.heappush(
            event_queue,
            SimulationEvent.create(self.end_time, self._next_seq(), EventType.SIMULATION_END, {}),
        )

        while event_queue:
            event = heapq.heappop(event_queue)
            state.current_time = event.time

            if event.event_type == EventType.SIMULATION_END:
                break

            if event.event_type == EventType.REQUEST:
                self._handle_request(state, event.payload["request_id"], metrics, event_queue)

            elif event.event_type == EventType.VEHICLE_ARRIVAL:
                self._handle_arrival(state, event.payload, metrics)

            if on_step:
                on_step(state, metrics)

        for vehicle in state.vehicles:
            metrics.total_distance_km += vehicle.total_distance_km

        metrics.final_vehicles = state.vehicles
        metrics.final_requests = state.requests
        metrics.stops = state.stops
        metrics.finalize(state.requests)

        if on_step:
            on_step(state, metrics)

        return metrics

    def _handle_request(
        self,
        state: SimulationState,
        request_id: str,
        metrics: MetricsCollector,
        event_queue: list[SimulationEvent],
    ) -> None:
        request = state.get_request(request_id)
        if request is None or request.status != "pending":
            return

        metrics.total_requests += 1
        result = self.dispatcher.assign(
            request,
            state.vehicles,
            self.travel,
            state.current_time,
            state.demand_map,
        )

        if not result.success or result.vehicle is None or result.route is None:
            request.status = "rejected"
            metrics.rejected_requests += 1
            return

        vehicle = state.get_vehicle(result.vehicle.id)
        if vehicle is None:
            request.status = "rejected"
            metrics.rejected_requests += 1
            return

        added_distance = self._compute_route_distance(vehicle.current_location, result.route)
        vehicle.route = result.route
        vehicle.route_version += 1
        vehicle.total_distance_km += added_distance

        request.status = "assigned"
        request.assigned_vehicle_id = vehicle.id
        metrics.assigned_requests += 1
        metrics.route_history.append((vehicle.id, [s.copy() for s in result.route]))

        version = vehicle.route_version
        for i, stop in enumerate(vehicle.route):
            heapq.heappush(
                event_queue,
                SimulationEvent.create(
                    stop.arrival_time,
                    self._next_seq(),
                    EventType.VEHICLE_ARRIVAL,
                    {
                        "vehicle_id": vehicle.id,
                        "stop_index": i,
                        "route_version": version,
                    },
                ),
            )

    def _handle_arrival(
        self,
        state: SimulationState,
        payload: dict,
        metrics: MetricsCollector,
    ) -> None:
        vehicle = state.get_vehicle(payload["vehicle_id"])
        if vehicle is None:
            return

        if payload.get("route_version") != vehicle.route_version:
            return

        stop_index = payload["stop_index"]
        if stop_index >= len(vehicle.route):
            return

        stop = vehicle.route[stop_index]
        if abs(stop.arrival_time - state.current_time) > 0.01:
            return

        vehicle.current_location = stop.stop_id
        vehicle.busy_until = stop.depart_time

        if stop.action == "pickup" and stop.request_id:
            request = state.get_request(stop.request_id)
            if request and request.pickup_time is None:
                request.pickup_time = stop.arrival_time
                vehicle.onboard[stop.request_id] = request
                metrics.record_wait(stop.arrival_time - request.request_time)

        elif stop.action == "dropoff" and stop.request_id:
            request = state.get_request(stop.request_id)
            if request and request.status != "served":
                request.dropoff_time = stop.arrival_time
                request.status = "served"
                vehicle.onboard.pop(stop.request_id, None)
                metrics.served_requests += 1

        metrics.record_utilization(len(vehicle.onboard))

        if stop_index == len(vehicle.route) - 1:
            vehicle.route = []

    def _compute_route_distance(self, start_location: str, route: list) -> float:
        if not route:
            return 0.0
        total = self.travel.distance(start_location, route[0].stop_id)
        for i in range(1, len(route)):
            total += self.travel.distance(route[i - 1].stop_id, route[i].stop_id)
        return total
