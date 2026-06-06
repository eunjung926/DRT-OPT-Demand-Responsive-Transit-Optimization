from __future__ import annotations

from dataclasses import dataclass, field

from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.models.request import Request
from drt_opt.models.stop import Stop
from drt_opt.models.vehicle import Vehicle


@dataclass
class SimulationState:
    current_time: float = 0.0
    vehicles: list[Vehicle] = field(default_factory=list)
    requests: list[Request] = field(default_factory=list)
    stops: list[Stop] = field(default_factory=list)
    travel: TravelMatrix | None = None
    demand_map: dict[str, float] = field(default_factory=dict)

    def get_vehicle(self, vehicle_id: str) -> Vehicle | None:
        for v in self.vehicles:
            if v.id == vehicle_id:
                return v
        return None

    def get_request(self, request_id: str) -> Request | None:
        for r in self.requests:
            if r.id == request_id:
                return r
        return None
