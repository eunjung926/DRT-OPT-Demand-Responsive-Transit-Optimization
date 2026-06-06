from dataclasses import dataclass, field

from drt_opt.models.request import Request
from drt_opt.models.route import RouteStop


@dataclass
class Vehicle:
    id: str
    capacity: int
    current_location: str
    route: list[RouteStop] = field(default_factory=list)
    onboard: dict[str, Request] = field(default_factory=dict)
    total_distance_km: float = 0.0
    busy_until: float = 0.0
    route_version: int = 0

    def copy(self) -> "Vehicle":
        return Vehicle(
            id=self.id,
            capacity=self.capacity,
            current_location=self.current_location,
            route=[s.copy() for s in self.route],
            onboard=dict(self.onboard),
            total_distance_km=self.total_distance_km,
            busy_until=self.busy_until,
            route_version=self.route_version,
        )

    @property
    def current_onboard(self) -> int:
        return len(self.onboard)
