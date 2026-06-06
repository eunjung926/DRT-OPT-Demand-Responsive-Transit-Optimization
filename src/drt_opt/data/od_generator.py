from __future__ import annotations

import numpy as np

from drt_opt.models.request import Request
from drt_opt.models.stop import Stop


def _hourly_rate(base_rate: float, hour: int, peak_hours: list[int], peak_multiplier: float) -> float:
    if hour in peak_hours:
        return base_rate * peak_multiplier
    return base_rate


def generate_od_requests(
    stops: list[Stop],
    start_time_min: float,
    end_time_min: float,
    requests_per_hour: float = 30.0,
    peak_hours: list[int] | None = None,
    peak_multiplier: float = 2.0,
    seed: int = 0,
) -> list[Request]:
    """Generate a stream of ride requests weighted by stop demand."""
    rng = np.random.default_rng(seed)
    peak_hours = peak_hours or [7, 8, 18, 19]

    stop_ids = [s.id for s in stops]
    weights = np.array([max(s.demand_weight, 1e-9) for s in stops])
    weights /= weights.sum()

    requests: list[Request] = []
    t = start_time_min
    req_id = 0

    while t < end_time_min:
        hour = int(t // 60) % 24
        rate = _hourly_rate(requests_per_hour, hour, peak_hours, peak_multiplier)
        interval = rng.exponential(60.0 / rate) if rate > 0 else 60.0
        t += interval
        if t >= end_time_min:
            break

        origin_idx = rng.choice(len(stop_ids), p=weights)
        dest_candidates = [i for i in range(len(stop_ids)) if i != origin_idx]
        dest_weights = weights[dest_candidates]
        dest_weights /= dest_weights.sum()
        dest_idx = rng.choice(dest_candidates, p=dest_weights)

        requests.append(
            Request(
                id=f"R{req_id:05d}",
                origin_stop_id=stop_ids[origin_idx],
                dest_stop_id=stop_ids[dest_idx],
                request_time=float(t),
            )
        )
        req_id += 1

    return requests
