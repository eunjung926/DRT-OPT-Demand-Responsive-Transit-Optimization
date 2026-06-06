from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


@dataclass
class TravelMatrix:
    stop_ids: list[str]
    time_min: np.ndarray
    distance_km: np.ndarray
    _index: dict[str, int] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        self._index = {sid: i for i, sid in enumerate(self.stop_ids)}

    @classmethod
    def from_stops(cls, stops: list, speed_kmh: float = 25.0) -> TravelMatrix:
        ids = [s.id for s in stops]
        n = len(ids)
        dist = np.zeros((n, n))
        for i, s1 in enumerate(stops):
            for j, s2 in enumerate(stops):
                if i == j:
                    dist[i, j] = 0.0
                else:
                    dist[i, j] = haversine_km(s1.lat, s1.lon, s2.lat, s2.lon)
        time = (dist / speed_kmh) * 60.0
        return cls(stop_ids=ids, time_min=time, distance_km=dist)

    def _idx(self, a: str, b: str) -> tuple[int, int]:
        return self._index[a], self._index[b]

    def time(self, from_stop: str, to_stop: str) -> float:
        i, j = self._idx(from_stop, to_stop)
        return float(self.time_min[i, j])

    def distance(self, from_stop: str, to_stop: str) -> float:
        i, j = self._idx(from_stop, to_stop)
        return float(self.distance_km[i, j])

    def subset(self, stop_ids: list[str]) -> TravelMatrix:
        indices = [self._index[s] for s in stop_ids]
        sub_time = self.time_min[np.ix_(indices, indices)]
        sub_dist = self.distance_km[np.ix_(indices, indices)]
        return TravelMatrix(stop_ids=list(stop_ids), time_min=sub_time, distance_km=sub_dist)
