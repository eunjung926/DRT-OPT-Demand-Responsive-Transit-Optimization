from dataclasses import dataclass
from typing import Literal


RouteAction = Literal["pickup", "dropoff", "pass_through"]


@dataclass
class RouteStop:
    stop_id: str
    arrival_time: float = 0.0
    depart_time: float = 0.0
    action: RouteAction = "pass_through"
    request_id: str | None = None

    def copy(self) -> "RouteStop":
        return RouteStop(
            stop_id=self.stop_id,
            arrival_time=self.arrival_time,
            depart_time=self.depart_time,
            action=self.action,
            request_id=self.request_id,
        )
