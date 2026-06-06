#!/usr/bin/env python3
"""Run baseline vs proposed comparison experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drt_opt.config import load_config, project_root
from drt_opt.data.loader import load_processed_network
from drt_opt.data.od_generator import generate_od_requests
from drt_opt.dispatch.baseline import NearestVehicleDispatcher
from drt_opt.dispatch.gpu_optimized import GpuOptimizedDispatcher
from drt_opt.dispatch.optimized import OptimizedDispatcher
from drt_opt.gpu.backend import get_device, is_gpu_available
from drt_opt.simulation.engine import Simulator
from drt_opt.viz.comparison import plot_dispatcher_comparison


def run_single(config: dict, requests, dispatcher_name: str, use_gpu: bool = True):
    stops, travel, demand_map = load_processed_network(config)
    if dispatcher_name == "baseline":
        dispatcher = NearestVehicleDispatcher(config)
    elif use_gpu and config.get("gpu", {}).get("enabled", True):
        dispatcher = GpuOptimizedDispatcher(config)
    else:
        dispatcher = OptimizedDispatcher(config)
    sim = Simulator(config, stops, travel, demand_map, dispatcher)
    collector = sim.run(requests)
    return collector.metrics


def compare_and_plot(results_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["avg_wait_time_min", "dispatch_success_rate", "vehicle_utilization", "total_distance_km"]
    plot_path = plot_dispatcher_comparison(results_df, output_dir / "comparison.png")

    summary = results_df.groupby("dispatcher")[metrics].agg(["mean", "std"])
    summary.to_csv(output_dir / "summary.csv")
    print(f"Results saved to {output_dir}")
    print(f"Comparison chart: {plot_path.resolve()}")
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Baseline vs proposed experiment")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=project_root() / "results")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--gpu", action="store_true", default=True)
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    use_gpu = args.gpu and not args.no_gpu
    device = get_device(config.get("gpu", {}).get("device", "auto"))
    print(f"Region: {config.get('region', {}).get('name', 'yeongjong')}")
    print(f"Device: {device} (gpu={'on' if use_gpu and is_gpu_available() else 'off'})")
    args.output.mkdir(parents=True, exist_ok=True)
    stops, _, _ = load_processed_network(config)

    rows = []
    for run in range(args.runs):
        seed = args.seed_start + run
        requests = generate_od_requests(
            stops,
            config["simulation"]["start_time_min"],
            config["simulation"]["end_time_min"],
            requests_per_hour=config["demand"]["requests_per_hour"],
            peak_hours=config["demand"]["peak_hours"],
            peak_multiplier=config["demand"]["peak_multiplier"],
            seed=seed,
        )

        for name in ("baseline", "optimized"):
            metrics = run_single(config, requests, name, use_gpu=use_gpu)
            row = metrics.to_dict()
            row["dispatcher"] = name
            row["seed"] = seed
            rows.append(row)
            print(f"run={run} {name}: wait={metrics.avg_wait_time_min:.2f} success={metrics.dispatch_success_rate:.2%}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(args.output / "all_runs.csv", index=False)
    compare_and_plot(results_df, args.output)

    baseline = results_df[results_df["dispatcher"] == "baseline"]
    optimized = results_df[results_df["dispatcher"] == "optimized"]
    wait_improvement = (
        (baseline["avg_wait_time_min"].mean() - optimized["avg_wait_time_min"].mean())
        / baseline["avg_wait_time_min"].mean()
        * 100
    )
    success_improvement = (
        (optimized["dispatch_success_rate"].mean() - baseline["dispatch_success_rate"].mean())
        / baseline["dispatch_success_rate"].mean()
        * 100
    )
    print(f"\nAvg wait time improvement: {wait_improvement:.1f}%")
    print(f"Dispatch success rate improvement: {success_improvement:.1f}%")


if __name__ == "__main__":
    main()
