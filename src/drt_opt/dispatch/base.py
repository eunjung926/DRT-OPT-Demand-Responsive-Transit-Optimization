from __future__ import annotations

from abc import ABC, abstractmethod

from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.models.request import Request
from drt_opt.models.vehicle import Vehicle


class AssignmentResult:
    def __init__(
        self,
        success: bool,
        vehicle: Vehicle | None = None,
        route: list | None = None,
        cost: float = float("inf"),
    ):
        self.success = success
        self.vehicle = vehicle
        self.route = route
        self.cost = cost


class Dispatcher(ABC):
    @abstractmethod
    def assign(
        self,
        request: Request,
        vehicles: list[Vehicle],
        travel: TravelMatrix,
        current_time: float,
        demand_map: dict[str, float],
    ) -> AssignmentResult:
        pass
