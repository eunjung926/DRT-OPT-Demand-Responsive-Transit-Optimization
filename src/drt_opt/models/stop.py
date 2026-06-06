from dataclasses import dataclass


@dataclass
class Stop:
    id: str
    name: str
    lat: float
    lon: float
    demand_weight: float = 0.0
    district: str = ""
