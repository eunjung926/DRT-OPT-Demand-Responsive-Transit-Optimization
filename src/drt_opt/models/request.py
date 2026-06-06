from dataclasses import dataclass, field
from typing import Literal


RequestStatus = Literal["pending", "assigned", "served", "rejected"]


@dataclass
class Request:
    id: str
    origin_stop_id: str
    dest_stop_id: str
    request_time: float
    status: RequestStatus = "pending"
    assigned_vehicle_id: str | None = None
    pickup_time: float | None = None
    dropoff_time: float | None = None
