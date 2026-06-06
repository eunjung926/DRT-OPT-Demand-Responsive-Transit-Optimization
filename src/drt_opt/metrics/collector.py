from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimulationMetrics:
    total_requests: int = 0
    assigned_requests: int = 0
    rejected_requests: int = 0
    served_requests: int = 0
    avg_wait_time_min: float = 0.0
    dispatch_success_rate: float = 0.0
    vehicle_utilization: float = 0.0
    total_distance_km: float = 0.0

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "assigned_requests": self.assigned_requests,
            "rejected_requests": self.rejected_requests,
            "served_requests": self.served_requests,
            "avg_wait_time_min": round(self.avg_wait_time_min, 3),
            "dispatch_success_rate": round(self.dispatch_success_rate, 4),
            "vehicle_utilization": round(self.vehicle_utilization, 4),
            "total_distance_km": round(self.total_distance_km, 3),
        }


class MetricsCollector:
    def __init__(self, vehicle_count: int, sim_duration: float, capacity: int = 16):
        self.vehicle_count = vehicle_count
        self.sim_duration = sim_duration
        self.capacity = capacity
        self.total_requests = 0
        self.assigned_requests = 0
        self.rejected_requests = 0
        self.served_requests = 0
        self.wait_times: list[float] = []
        self.utilization_samples: list[int] = []
        self.total_distance_km = 0.0
        self._metrics: SimulationMetrics | None = None
        self.final_vehicles: list = []
        self.final_requests: list = []
        self.stops: list = []
        self.route_history: list[tuple[str, list]] = []

    def record_wait(self, wait_min: float) -> None:
        self.wait_times.append(wait_min)

    def record_utilization(self, onboard: int) -> None:
        self.utilization_samples.append(onboard)

    def finalize(self, requests: list) -> SimulationMetrics:
        avg_wait = sum(self.wait_times) / len(self.wait_times) if self.wait_times else 0.0
        success_rate = (
            self.assigned_requests / self.total_requests if self.total_requests > 0 else 0.0
        )

        if self.utilization_samples:
            avg_onboard = sum(self.utilization_samples) / len(self.utilization_samples)
            utilization = avg_onboard / self.capacity
        else:
            utilization = 0.0

        self._metrics = SimulationMetrics(
            total_requests=self.total_requests,
            assigned_requests=self.assigned_requests,
            rejected_requests=self.rejected_requests,
            served_requests=self.served_requests,
            avg_wait_time_min=avg_wait,
            dispatch_success_rate=success_rate,
            vehicle_utilization=min(1.0, utilization),
            total_distance_km=self.total_distance_km,
        )
        return self._metrics

    @property
    def metrics(self) -> SimulationMetrics:
        if self._metrics is None:
            return SimulationMetrics()
        return self._metrics
