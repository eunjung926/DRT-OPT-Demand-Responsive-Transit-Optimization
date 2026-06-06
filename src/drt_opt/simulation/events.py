from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class EventType(Enum):
    REQUEST = auto()
    VEHICLE_ARRIVAL = auto()
    SIMULATION_END = auto()


EVENT_PRIORITY = {
    EventType.VEHICLE_ARRIVAL: 0,
    EventType.REQUEST: 1,
    EventType.SIMULATION_END: 2,
}


@dataclass(order=True)
class SimulationEvent:
    time: float
    priority: int
    seq: int
    event_type: EventType = field(compare=False)
    payload: dict = field(default_factory=dict, compare=False)

    @classmethod
    def create(
        cls,
        time: float,
        seq: int,
        event_type: EventType,
        payload: dict | None = None,
    ) -> "SimulationEvent":
        return cls(
            time=time,
            priority=EVENT_PRIORITY[event_type],
            seq=seq,
            event_type=event_type,
            payload=payload or {},
        )
