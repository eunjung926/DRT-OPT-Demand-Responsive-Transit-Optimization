from __future__ import annotations

from dataclasses import dataclass, asdict
from copy import deepcopy
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

from drt_opt.config import load_config, project_root
from drt_opt.data.loader import load_processed_network
from drt_opt.data.od_generator import generate_od_requests
from drt_opt.dispatch.baseline import NearestVehicleDispatcher
from drt_opt.dispatch.optimized import OptimizedDispatcher
from drt_opt.metrics.collector import SimulationMetrics
from drt_opt.simulation.engine import Simulator


@dataclass
class WeightSet:
    wait: float
    detour: float
    distance: float
    demand: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, float]) -> WeightSet:
        return cls(
            wait=float(d["wait"]),
            detour=float(d["detour"]),
            distance=float(d["distance"]),
            demand=float(d["demand"]),
        )


@dataclass
class TrialResult:
    trial_id: int
    weights: WeightSet
    avg_wait_time_min: float
    dispatch_success_rate: float
    rejected_requests: int
    total_requests: int
    total_distance_km: float
    vehicle_utilization: float
    objective: float
    seed: int


def compute_objective(metrics: SimulationMetrics, obj_cfg: dict) -> float:
    """Lower is better."""
    reject_rate = (
        metrics.rejected_requests / metrics.total_requests if metrics.total_requests else 1.0
    )
    success_penalty = 1.0 - metrics.dispatch_success_rate
    return (
        obj_cfg.get("wait", 1.0) * metrics.avg_wait_time_min
        + obj_cfg.get("success", 80.0) * success_penalty
        + obj_cfg.get("distance", 0.05) * metrics.total_distance_km
        + obj_cfg.get("reject", 50.0) * reject_rate
    )


def _apply_tuning_window(config: dict, tuning_cfg: dict) -> dict:
    cfg = deepcopy(config)
    cfg["simulation"]["start_time_min"] = tuning_cfg.get(
        "start_time_min", cfg["simulation"]["start_time_min"]
    )
    cfg["simulation"]["end_time_min"] = tuning_cfg.get(
        "end_time_min", min(cfg["simulation"]["end_time_min"], 720)
    )
    cfg["demand"]["requests_per_hour"] = tuning_cfg.get(
        "requests_per_hour", cfg["demand"]["requests_per_hour"]
    )
    return cfg


def run_sim_with_weights(
    config: dict,
    weights: WeightSet,
    requests: list,
    stops,
    travel,
    demand_map,
) -> SimulationMetrics:
    cfg = deepcopy(config)
    cfg["weights"] = weights.to_dict()
    dispatcher = OptimizedDispatcher(cfg)
    sim = Simulator(cfg, stops, travel, demand_map, dispatcher)
    return sim.run(requests).metrics


def evaluate_weights(
    config: dict,
    weights: WeightSet,
    seeds: list[int],
    tuning_cfg: dict | None = None,
) -> tuple[float, list[TrialResult]]:
    """Run simulator for each seed; return mean objective and per-seed results."""
    tuning_cfg = tuning_cfg or config.get("tuning", {})
    sim_config = _apply_tuning_window(config, tuning_cfg)
    obj_cfg = tuning_cfg.get("objective", {})
    stops, travel, demand_map = load_processed_network(sim_config)

    results: list[TrialResult] = []
    objectives: list[float] = []

    for seed in seeds:
        requests = generate_od_requests(
            stops,
            sim_config["simulation"]["start_time_min"],
            sim_config["simulation"]["end_time_min"],
            requests_per_hour=sim_config["demand"]["requests_per_hour"],
            peak_hours=sim_config["demand"]["peak_hours"],
            peak_multiplier=sim_config["demand"]["peak_multiplier"],
            seed=seed,
        )
        metrics = run_sim_with_weights(
            sim_config, weights, requests, stops, travel, demand_map
        )
        obj = compute_objective(metrics, obj_cfg)
        objectives.append(obj)
        results.append(
            TrialResult(
                trial_id=-1,
                weights=weights,
                avg_wait_time_min=metrics.avg_wait_time_min,
                dispatch_success_rate=metrics.dispatch_success_rate,
                rejected_requests=metrics.rejected_requests,
                total_requests=metrics.total_requests,
                total_distance_km=metrics.total_distance_km,
                vehicle_utilization=metrics.vehicle_utilization,
                objective=obj,
                seed=seed,
            )
        )

    return float(np.mean(objectives)), results


def generate_grid_candidates(tuning_cfg: dict) -> list[WeightSet]:
    grid = tuning_cfg.get("grid", {})
    keys = ["wait", "detour", "distance", "demand"]
    axes = [grid.get(k, [1.0]) for k in keys]
    return [WeightSet(*combo) for combo in product(*axes)]


def generate_random_candidates(n: int, tuning_cfg: dict, rng: np.random.Generator) -> list[WeightSet]:
    ranges = tuning_cfg.get("ranges", {})
    candidates: list[WeightSet] = []
    for _ in range(n):
        candidates.append(
            WeightSet(
                wait=float(rng.uniform(*ranges.get("wait", [0.5, 2.0]))),
                detour=float(rng.uniform(*ranges.get("detour", [0.2, 1.0]))),
                distance=float(rng.uniform(*ranges.get("distance", [0.1, 0.5]))),
                demand=float(rng.uniform(*ranges.get("demand", [0.1, 0.5]))),
            )
        )
    return candidates


def optimize_weights(
    config: dict | None = None,
    method: Literal["grid", "random"] | None = None,
    n_trials: int | None = None,
    seeds: list[int] | None = None,
) -> dict:
    config = config or load_config()
    tuning_cfg = config.get("tuning", {})
    method = method or tuning_cfg.get("method", "random")
    seeds = seeds or tuning_cfg.get("evaluation_seeds", [0, 1, 2, 3, 4])
    n_trials = n_trials or tuning_cfg.get("n_trials", 40)

    if method == "grid":
        candidates = generate_grid_candidates(tuning_cfg)
    else:
        rng = np.random.default_rng(42)
        candidates = generate_random_candidates(n_trials, tuning_cfg, rng)

    # Always include current config weights as baseline candidate
    base_w = WeightSet.from_dict(config["weights"])
    if not any(c.to_dict() == base_w.to_dict() for c in candidates):
        candidates.insert(0, base_w)

    all_rows: list[dict] = []
    best_mean_obj = float("inf")
    best_weights: WeightSet | None = None
    best_aggregate: dict | None = None

    for trial_id, weights in enumerate(candidates):
        mean_obj, seed_results = evaluate_weights(config, weights, seeds, tuning_cfg)
        for r in seed_results:
            row = {
                "trial_id": trial_id,
                **weights.to_dict(),
                "seed": r.seed,
                "avg_wait_time_min": r.avg_wait_time_min,
                "dispatch_success_rate": r.dispatch_success_rate,
                "rejected_requests": r.rejected_requests,
                "total_requests": r.total_requests,
                "total_distance_km": r.total_distance_km,
                "vehicle_utilization": r.vehicle_utilization,
                "objective": r.objective,
                "mean_objective": mean_obj,
            }
            all_rows.append(row)

        if mean_obj < best_mean_obj:
            best_mean_obj = mean_obj
            best_weights = weights
            best_aggregate = {
                "mean_objective": mean_obj,
                "avg_wait_time_min": float(np.mean([r.avg_wait_time_min for r in seed_results])),
                "dispatch_success_rate": float(
                    np.mean([r.dispatch_success_rate for r in seed_results])
                ),
                "total_distance_km": float(np.mean([r.total_distance_km for r in seed_results])),
            }

    assert best_weights is not None and best_aggregate is not None

    # Baseline comparison on same seeds
    baseline_rows = _evaluate_baseline(config, seeds, tuning_cfg)

    output_dir = project_root() / tuning_cfg.get("output_dir", "results/tuning")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_dir / "trials.csv", index=False)
    pd.DataFrame(baseline_rows).to_csv(output_dir / "baseline.csv", index=False)

    best_path = output_dir / "best_weights.yaml"
    with open(best_path, "w", encoding="utf-8") as f:
        yaml.dump(
            {
                "weights": best_weights.to_dict(),
                "aggregate_metrics": best_aggregate,
                "method": method,
                "seeds": seeds,
            },
            f,
            allow_unicode=True,
            default_flow_style=False,
        )

    summary = {
        "best_weights": best_weights.to_dict(),
        "best_aggregate": best_aggregate,
        "baseline_per_seed": baseline_rows,
        "method": method,
        "n_candidates": len(candidates),
        "n_seeds": len(seeds),
        "output_dir": str(output_dir),
        "best_weights_path": str(best_path),
    }
    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        import json

        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def _evaluate_baseline(config: dict, seeds: list[int], tuning_cfg: dict) -> list[dict]:
    sim_config = _apply_tuning_window(config, tuning_cfg)
    obj_cfg = tuning_cfg.get("objective", {})
    stops, travel, demand_map = load_processed_network(sim_config)
    rows = []

    for seed in seeds:
        requests = generate_od_requests(
            stops,
            sim_config["simulation"]["start_time_min"],
            sim_config["simulation"]["end_time_min"],
            requests_per_hour=sim_config["demand"]["requests_per_hour"],
            peak_hours=sim_config["demand"]["peak_hours"],
            peak_multiplier=sim_config["demand"]["peak_multiplier"],
            seed=seed,
        )
        sim = Simulator(sim_config, stops, travel, demand_map, NearestVehicleDispatcher(sim_config))
        m = sim.run(requests).metrics
        rows.append(
            {
                "seed": seed,
                "avg_wait_time_min": m.avg_wait_time_min,
                "dispatch_success_rate": m.dispatch_success_rate,
                "total_distance_km": m.total_distance_km,
                "objective": compute_objective(m, obj_cfg),
            }
        )
    return rows


def apply_best_weights_to_config(
    config_path: Path | None = None,
    best_weights_path: Path | None = None,
) -> dict:
    """Load best weights from tuning output and merge into config YAML."""
    root = project_root()
    config_path = config_path or root / "config" / "default.yaml"
    best_weights_path = best_weights_path or root / "results" / "tuning" / "best_weights.yaml"

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    with open(best_weights_path, encoding="utf-8") as f:
        best = yaml.safe_load(f)

    config["weights"] = best["weights"]
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    return config
