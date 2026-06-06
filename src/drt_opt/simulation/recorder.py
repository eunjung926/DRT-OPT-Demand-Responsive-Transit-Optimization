from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SimSnapshot:
    time_min: float
    time_label: str
    vehicles: list[dict]
    recent_requests: list[dict]
    metrics: dict
    pending_markers: list[dict] = field(default_factory=list)
    rejected_markers: list[dict] = field(default_factory=list)


@dataclass
class RecordedSimulation:
    algorithm: str
    frames: list[SimSnapshot] = field(default_factory=list)
    final_metrics: dict = field(default_factory=dict)


StepCallback = Callable[[SimSnapshot], None]
