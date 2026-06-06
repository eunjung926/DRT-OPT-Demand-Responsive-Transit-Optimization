from __future__ import annotations

from copy import deepcopy

from drt_opt.config import load_config
from drt_opt.data.loader import load_processed_network
from drt_opt.data.od_generator import generate_od_requests
from drt_opt.dispatch.baseline import NearestVehicleDispatcher
from drt_opt.dispatch.gpu_optimized import GpuOptimizedDispatcher
from drt_opt.dispatch.optimized import OptimizedDispatcher
from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.metrics.collector import MetricsCollector
from drt_opt.models.stop import Stop
from drt_opt.simulation.engine import Simulator
from drt_opt.simulation.recorder import RecordedSimulation, SimSnapshot, StepCallback
from drt_opt.simulation.snapshot import build_snapshot

DEFAULT_FRAME_INTERVAL_MIN = 1.0


def _make_dispatcher(name: str, config: dict, use_gpu: bool = False):
    if name == "baseline":
        return NearestVehicleDispatcher(config)
    if use_gpu and config.get("gpu", {}).get("enabled", True):
        return GpuOptimizedDispatcher(config)
    return OptimizedDispatcher(config)


def _pick_frame_at_time(frames: list[SimSnapshot], time_min: float) -> SimSnapshot | None:
    if not frames:
        return None
    best = frames[0]
    for frame in frames:
        if frame.time_min <= time_min:
            best = frame
        else:
            break
    return best


def resample_frames(
    frames: list[SimSnapshot],
    start_min: float,
    end_min: float,
    step_min: float,
) -> list[SimSnapshot]:
    """Align both algorithms to the same timeline for side-by-side playback."""
    if not frames:
        return []

    timeline: list[float] = []
    t = start_min
    while t <= end_min:
        timeline.append(t)
        t += step_min

    last = frames[-1]
    if timeline and timeline[-1] < last.time_min:
        timeline.append(last.time_min)

    resampled: list[SimSnapshot] = []
    for target in timeline:
        frame = _pick_frame_at_time(frames, target)
        if frame is None:
            continue
        resampled.append(
            SimSnapshot(
                time_min=target,
                time_label=frame.time_label if abs(frame.time_min - target) < step_min / 2 else _time_label(target),
                vehicles=frame.vehicles,
                recent_requests=frame.recent_requests,
                metrics=frame.metrics,
                pending_markers=frame.pending_markers,
                rejected_markers=frame.rejected_markers,
            )
        )
    return resampled or frames


def _time_label(minutes: float) -> str:
    h = int(minutes // 60) % 24
    m = int(minutes % 60)
    return f"{h:02d}:{m:02d}"


def run_recorded(
    config: dict,
    requests: list,
    algorithm: str,
    stops: list[Stop],
    travel: TravelMatrix,
    demand_map: dict[str, float],
    on_step: StepCallback | None = None,
    use_gpu: bool = False,
    frame_interval_min: float = DEFAULT_FRAME_INTERVAL_MIN,
) -> RecordedSimulation:
    dispatcher = _make_dispatcher(algorithm, config, use_gpu)
    sim = Simulator(config, stops, travel, demand_map, dispatcher)
    stops_by_id = {s.id: s for s in stops}
    frames: list[SimSnapshot] = []
    last_frame_time = float("-inf")

    def capture(state, metrics: MetricsCollector) -> None:
        nonlocal last_frame_time
        if state.current_time - last_frame_time < frame_interval_min and frames:
            return
        last_frame_time = state.current_time
        snap = build_snapshot(state, metrics, stops_by_id, travel, algorithm)
        frames.append(snap)
        if on_step:
            on_step(snap)

    collector = sim.run(deepcopy(requests), on_step=capture)

    if not frames:
        frames.append(build_snapshot_from_final(collector, stops_by_id, travel, algorithm))

    return RecordedSimulation(
        algorithm=algorithm,
        frames=frames,
        final_metrics=collector.metrics.to_dict(),
    )


def build_snapshot_from_final(
    collector: MetricsCollector,
    stops_by_id: dict[str, Stop],
    travel: TravelMatrix,
    algorithm: str,
) -> SimSnapshot:
    from drt_opt.simulation.state import SimulationState

    state = SimulationState(
        current_time=0,
        vehicles=collector.final_vehicles,
        requests=collector.final_requests,
    )
    return build_snapshot(state, collector, stops_by_id, travel, algorithm)


def run_comparison(
    seed: int = 42,
    config: dict | None = None,
    use_gpu: bool = False,
    max_frames: int = 600,
) -> dict:
    config = config or load_config()
    web_cfg = config.get("web", {})

    stops, travel, demand_map = load_processed_network(config)

    start = web_cfg.get("start_time_min", config["simulation"]["start_time_min"])
    end = web_cfg.get("end_time_min", min(config["simulation"]["end_time_min"], start + 240))
    rph = web_cfg.get("requests_per_hour", config["demand"]["requests_per_hour"])
    peak_mult = web_cfg.get("peak_multiplier", config["demand"]["peak_multiplier"])
    frame_interval = web_cfg.get("frame_interval_min", DEFAULT_FRAME_INTERVAL_MIN)

    requests = generate_od_requests(
        stops,
        start,
        end,
        requests_per_hour=rph,
        peak_hours=config["demand"]["peak_hours"],
        peak_multiplier=peak_mult,
        seed=seed,
    )

    baseline = run_recorded(
        config, requests, "baseline", stops, travel, demand_map,
        use_gpu=False, frame_interval_min=frame_interval,
    )
    optimized = run_recorded(
        config, requests, "optimized", stops, travel, demand_map,
        use_gpu=use_gpu, frame_interval_min=frame_interval,
    )

    baseline.frames = resample_frames(baseline.frames, start, end, frame_interval)[:max_frames]
    optimized.frames = resample_frames(optimized.frames, start, end, frame_interval)[:max_frames]

    return serialize_comparison(stops, config, seed, baseline, optimized, rph, peak_mult)


def serialize_comparison(
    stops,
    config,
    seed,
    baseline,
    optimized,
    requests_per_hour: float,
    peak_multiplier: float,
) -> dict:
    bbox = config.get("region", {}).get("bbox")
    b = baseline.final_metrics
    o = optimized.final_metrics
    wait_improve = (
        (b["avg_wait_time_min"] - o["avg_wait_time_min"]) / b["avg_wait_time_min"] * 100
        if b.get("avg_wait_time_min")
        else 0.0
    )
    success_improve = (
        (o["dispatch_success_rate"] - b["dispatch_success_rate"]) / b["dispatch_success_rate"] * 100
        if b.get("dispatch_success_rate")
        else 0.0
    )
    reject_delta = b.get("rejected_requests", 0) - o.get("rejected_requests", 0)

    return {
        "seed": seed,
        "region": config.get("region", {}).get("name", "yeongjong"),
        "bbox": bbox,
        "scenario": {
            "requests_per_hour": requests_per_hour,
            "peak_multiplier": peak_multiplier,
            "total_requests": b.get("total_requests", 0),
        },
        "stops": [
            {
                "id": s.id,
                "name": s.name,
                "lat": s.lat,
                "lon": s.lon,
                "demand": s.demand_weight,
            }
            for s in stops
        ],
        "baseline": {
            "label": "기존 (가까운 차량 + 쌓임)",
            "frames": [_frame_to_dict(f) for f in baseline.frames],
            "final_metrics": baseline.final_metrics,
        },
        "optimized": {
            "label": "최적화 (삽입·제약·수요 가중)",
            "frames": [_frame_to_dict(f) for f in optimized.frames],
            "final_metrics": optimized.final_metrics,
        },
        "comparison": {
            "wait_improve_pct": round(wait_improve, 1),
            "success_improve_pct": round(success_improve, 1),
            "reject_reduction": reject_delta,
            "baseline_wait_min": round(b.get("avg_wait_time_min", 0), 1),
            "optimized_wait_min": round(o.get("avg_wait_time_min", 0), 1),
            "baseline_success_pct": round(b.get("dispatch_success_rate", 0) * 100, 1),
            "optimized_success_pct": round(o.get("dispatch_success_rate", 0) * 100, 1),
        },
    }


def _frame_to_dict(frame: SimSnapshot) -> dict:
    return {
        "time_min": frame.time_min,
        "time_label": frame.time_label,
        "vehicles": frame.vehicles,
        "recent_requests": frame.recent_requests,
        "pending_markers": frame.pending_markers,
        "rejected_markers": frame.rejected_markers,
        "metrics": frame.metrics,
    }
