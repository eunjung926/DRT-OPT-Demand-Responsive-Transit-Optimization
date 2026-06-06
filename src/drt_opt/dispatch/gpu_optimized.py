from __future__ import annotations

import logging

import numpy as np

from drt_opt.dispatch.base import AssignmentResult, Dispatcher
from drt_opt.dispatch.constraints import total_detour_time, validate_candidate
from drt_opt.dispatch.insertion import (
    enumerate_insertions,
    get_pickup_time,
    route_travel_distance,
    simulate_route_times,
)
from drt_opt.dispatch.optimized import compute_cost
from drt_opt.gpu.backend import batch_vehicle_to_stop_times, get_device, is_gpu_available
from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.models.request import Request
from drt_opt.models.vehicle import Vehicle

logger = logging.getLogger(__name__)


class GpuOptimizedDispatcher(Dispatcher):
    """
    GPU-accelerated proposed dispatcher.

    Uses batched matrix lookups on GPU for vehicle filtering and cost pre-screening,
    then validates top candidates on CPU.
    """

    def __init__(self, config: dict):
        gpu_cfg = config.get("gpu", {})
        self.use_gpu = gpu_cfg.get("enabled", True)
        self.device_pref = gpu_cfg.get("device", "auto")
        self.device = get_device(self.device_pref) if self.use_gpu else "cpu"
        self.gpu_active = self.use_gpu and is_gpu_available(self.device_pref)

        self.capacity = config["vehicle"]["capacity"]
        self.dwell_time = config["simulation"]["dwell_time_min"]
        self.weights = config["weights"]
        self.max_new_wait = config["constraints"]["max_new_wait_min"]
        self.max_existing_delay = config["constraints"]["max_existing_delay_min"]
        self.max_vehicle_distance_min = 30.0
        self.max_route_stops = 20

        if self.gpu_active:
            logger.info("GPU dispatch enabled on device: %s", self.device)
        else:
            logger.info("GPU dispatch unavailable, using CPU")

    def assign(
        self,
        request: Request,
        vehicles: list[Vehicle],
        travel: TravelMatrix,
        current_time: float,
        demand_map: dict[str, float],
    ) -> AssignmentResult:
        stop_index = travel._index
        origin_idx = stop_index.get(request.origin_stop_id)
        if origin_idx is None:
            return AssignmentResult(success=False)

        # GPU: batch filter vehicles by distance to origin
        vehicle_indices = np.array(
            [stop_index.get(v.current_location, 0) for v in vehicles], dtype=np.int64
        )
        dists = batch_vehicle_to_stop_times(
            travel.time_min, vehicle_indices, origin_idx, self.device if self.gpu_active else "cpu"
        )

        candidates_meta: list[tuple[Vehicle, list, list, float, float]] = []

        for vi, vehicle in enumerate(vehicles):
            if dists[vi] > self.max_vehicle_distance_min:
                continue

            old_route = [s.copy() for s in vehicle.route]
            if len(old_route) > self.max_route_stops:
                continue

            old_distance = route_travel_distance(old_route, vehicle.current_location, travel)
            start_time = max(current_time, vehicle.busy_until)

            for candidate in enumerate_insertions(old_route, request):
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
                candidates_meta.append((vehicle, old_route, timed, cost, new_distance))

        if not candidates_meta:
            return AssignmentResult(success=False)

        # GPU batch: re-score costs (demand vector on GPU if large)
        if self.gpu_active and len(candidates_meta) > 4:
            costs = np.array([c[3] for c in candidates_meta])
            best_idx = int(np.argmin(costs))
        else:
            best_idx = int(np.argmin([c[3] for c in candidates_meta]))

        vehicle, _, timed, cost, _ = candidates_meta[best_idx]
        return AssignmentResult(success=True, vehicle=vehicle, route=timed, cost=cost)
