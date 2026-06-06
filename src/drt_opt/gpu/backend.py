"""GPU backend for batched dispatch evaluation."""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


def get_device(preference: str = "auto") -> str:
    """Return torch device string: cuda, mps, or cpu."""
    if not TORCH_AVAILABLE:
        return "cpu"

    if preference == "cpu":
        return "cpu"
    if preference == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preference == "mps" and torch.backends.mps.is_available():
        return "mps"

    if preference == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    return "cpu"


def is_gpu_available(preference: str = "auto") -> bool:
    return get_device(preference) != "cpu"


def to_tensor(array: np.ndarray, device: str):
    if not TORCH_AVAILABLE:
        return array
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def batch_route_times(
    time_matrix: np.ndarray,
    route_indices: np.ndarray,
    start_indices: np.ndarray,
    start_times: np.ndarray,
    dwell_time: float,
    device: str = "cpu",
) -> np.ndarray:
    """
    Batch-compute pickup arrival times for the first stop in each padded route.

    route_indices: (batch, max_len) int, -1 padded
    start_indices: (batch,) int — vehicle current stop index
    start_times: (batch,) float
    Returns: (batch,) arrival time at first route stop (index 0 of each route)
    """
    batch_size, max_len = route_indices.shape
    if not TORCH_AVAILABLE or device == "cpu":
        return _batch_route_times_numpy(
            time_matrix, route_indices, start_indices, start_times, dwell_time
        )

    tm = to_tensor(time_matrix, device)
    routes = torch.as_tensor(route_indices, dtype=torch.long, device=device)
    starts = torch.as_tensor(start_indices, dtype=torch.long, device=device)
    times = torch.as_tensor(start_times, dtype=torch.float32, device=device)

    valid = routes >= 0
    t = times.clone()

    prev = starts.clone()
    for k in range(max_len):
        curr = routes[:, k]
        has = valid[:, k]
        leg = tm[prev, curr.clamp(min=0)]
        leg = torch.where(has, leg, torch.zeros_like(leg))
        t = t + leg
        t = torch.where(has, t + dwell_time, t)
        prev = torch.where(has, curr, prev)

    return t.cpu().numpy()


def _batch_route_times_numpy(
    time_matrix: np.ndarray,
    route_indices: np.ndarray,
    start_indices: np.ndarray,
    start_times: np.ndarray,
    dwell_time: float,
) -> np.ndarray:
    batch_size, max_len = route_indices.shape
    result = np.zeros(batch_size, dtype=np.float64)
    for b in range(batch_size):
        t = start_times[b]
        prev = start_indices[b]
        for k in range(max_len):
            curr = route_indices[b, k]
            if curr < 0:
                break
            t += time_matrix[prev, curr]
            t += dwell_time
            prev = curr
        result[b] = t
    return result


def batch_vehicle_to_stop_times(
    time_matrix: np.ndarray,
    vehicle_indices: np.ndarray,
    target_index: int,
    device: str = "cpu",
) -> np.ndarray:
    """Vectorized lookup: travel time from each vehicle location to target stop."""
    if not TORCH_AVAILABLE or device == "cpu":
        return time_matrix[vehicle_indices, target_index]

    tm = to_tensor(time_matrix, device)
    vi = torch.as_tensor(vehicle_indices, dtype=torch.long, device=device)
    return tm[vi, target_index].cpu().numpy()
