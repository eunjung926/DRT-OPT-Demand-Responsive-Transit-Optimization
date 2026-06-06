from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from drt_opt.config import load_config, project_root
from drt_opt.data.download import download_yeongjong_data
from drt_opt.data.loader import load_processed_network, preprocess_network
from drt_opt.data.od_generator import generate_od_requests
from drt_opt.dispatch.baseline import NearestVehicleDispatcher
from drt_opt.dispatch.gpu_optimized import GpuOptimizedDispatcher
from drt_opt.dispatch.optimized import OptimizedDispatcher
from drt_opt.gpu.backend import get_device, is_gpu_available
from drt_opt.simulation.engine import Simulator
from drt_opt.tuning.weight_optimizer import apply_best_weights_to_config, optimize_weights
from drt_opt.viz.map import create_network_map, create_simulation_map, save_map

logging.basicConfig(level=logging.INFO)


def _make_dispatcher(name: str, config: dict, use_gpu: bool):
    if name == "baseline":
        return NearestVehicleDispatcher(config)
    if use_gpu and config.get("gpu", {}).get("enabled", True):
        return GpuOptimizedDispatcher(config)
    return OptimizedDispatcher(config)


def cmd_web(args: argparse.Namespace) -> None:
    import uvicorn

    config = load_config(args.config)
    web_cfg = config.get("web", {})
    host = args.host or web_cfg.get("host", "127.0.0.1")
    port = args.port or web_cfg.get("port", 8080)
    print(f"DRT Simulator UI: http://{host}:{port}")
    print("  왼쪽: Baseline (쌓임)  |  오른쪽: 최적화")
    uvicorn.run("drt_opt.web.app:app", host=host, port=port, reload=args.reload)


def cmd_download_data(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    meta = download_yeongjong_data(
        output_dir=Path(args.output) if args.output else None,
        config=config,
        use_osm=not args.no_osm,
        use_incheon_api=not args.no_api,
    )
    print(f"Downloaded {meta['stop_count']} stops -> {meta['stops_path']}")
    print(f"Sources: {', '.join(meta['sources'])}")
    if args.preprocess:
        stops, _, _ = preprocess_network(config, use_sample=False)
        print(f"Preprocessed {len(stops)} stops -> {config['data']['processed_dir']}")


def cmd_preprocess(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    use_sample = args.sample or not (project_root() / config["data"]["raw_dir"] / "stops.csv").exists()
    stops, travel, demand_map = preprocess_network(config, use_sample=use_sample)
    region = config.get("region", {}).get("name", "yeongjong")
    print(f"[{region}] Preprocessed {len(stops)} stops -> {config['data']['processed_dir']}")
    if use_sample:
        print("Used Yeongjong sample data (place CSV in data/raw/ for real Incheon data)")


def cmd_map(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    stops, _, demand_map = load_processed_network(config)
    bbox = config.get("region", {}).get("bbox")

    m = create_network_map(stops, demand_map, bbox, title="영종도 I-MOD 정류장")
    out = Path(args.output)
    path = save_map(m, out)
    print(f"Network map saved: {path.resolve()}")
    print("Open in browser to view OpenStreetMap overlay.")


def cmd_simulate(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    stops, travel, demand_map = load_processed_network(config)

    use_gpu = args.gpu and not args.no_gpu
    device = get_device(config.get("gpu", {}).get("device", "auto"))
    print(f"Device: {device} (gpu={'on' if use_gpu and is_gpu_available() else 'off'})")

    requests = generate_od_requests(
        stops,
        config["simulation"]["start_time_min"],
        config["simulation"]["end_time_min"],
        requests_per_hour=config["demand"]["requests_per_hour"],
        peak_hours=config["demand"]["peak_hours"],
        peak_multiplier=config["demand"]["peak_multiplier"],
        seed=args.seed,
    )

    dispatcher = _make_dispatcher(args.dispatcher, config, use_gpu)
    sim = Simulator(config, stops, travel, demand_map, dispatcher)
    collector = sim.run(requests)
    metrics = collector.metrics

    print(json.dumps(metrics.to_dict(), indent=2, ensure_ascii=False))

    if args.map:
        map_cfg = config.get("map", {})
        m = create_simulation_map(
            stops,
            collector.final_vehicles,
            collector.final_requests,
            route_history=collector.route_history,
            bbox=config.get("region", {}).get("bbox"),
            use_osrm=map_cfg.get("use_osrm_routes", True),
            osrm_server=map_cfg.get("osrm_server", "https://router.project-osrm.org"),
            demand_map=demand_map,
        )
        out_dir = Path(map_cfg.get("output_dir", "results/maps"))
        out_path = out_dir / f"sim_{args.dispatcher}_seed{args.seed}.html"
        save_map(m, out_path)
        print(f"Simulation map saved: {out_path.resolve()}")


def cmd_tune_weights(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(f"Optimizing weights (method={args.method or config.get('tuning', {}).get('method', 'random')})...")
    summary = optimize_weights(
        config=config,
        method=args.method,
        n_trials=args.trials,
        seeds=args.seeds,
    )
    bw = summary["best_weights"]
    ba = summary["best_aggregate"]
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"\nBest: wait={bw['wait']:.3f} detour={bw['detour']:.3f} "
        f"distance={bw['distance']:.3f} demand={bw['demand']:.3f}"
    )
    print(
        f"Metrics: wait={ba['avg_wait_time_min']:.2f}min  "
        f"success={ba['dispatch_success_rate']:.2%}  "
        f"objective={ba['mean_objective']:.3f}"
    )
    print(f"Saved to {summary['output_dir']}")

    if args.apply:
        apply_best_weights_to_config(
            config_path=args.config or project_root() / "config" / "default.yaml",
            best_weights_path=Path(summary["best_weights_path"]),
        )
        print("Applied best weights to config.")


def main() -> None:
    parser = argparse.ArgumentParser(description="DRT dynamic dispatch optimization (영종도)")
    parser.add_argument("--config", type=Path, default=None, help="Path to config YAML")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download-data", help="Download real Yeongjong bus stops (OSM + optional Incheon API)")
    p_download.add_argument("--output", type=Path, default=None, help="Output directory (default: data/raw)")
    p_download.add_argument("--no-osm", action="store_true", help="Skip OpenStreetMap download")
    p_download.add_argument("--no-api", action="store_true", help="Skip Incheon BIS API even if key is set")
    p_download.add_argument("--preprocess", action="store_true", help="Run preprocess after download")
    p_download.set_defaults(func=cmd_download_data)

    p_preprocess = sub.add_parser("preprocess", help="Load and preprocess Yeongjong network")
    p_preprocess.add_argument("--sample", action="store_true", help="Force sample data")
    p_preprocess.set_defaults(func=cmd_preprocess)

    p_map = sub.add_parser("map", help="Generate interactive network map (OpenStreetMap)")
    p_map.add_argument("--output", default="results/maps/yeongjong_network.html")
    p_map.set_defaults(func=cmd_map)

    p_sim = sub.add_parser("simulate", help="Run a single simulation")
    p_sim.add_argument("--dispatcher", choices=["baseline", "optimized"], default="optimized")
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--gpu", action="store_true", default=True, help="Use GPU acceleration")
    p_sim.add_argument("--no-gpu", action="store_true", help="Disable GPU")
    p_sim.add_argument("--map", action="store_true", help="Save interactive map after simulation")
    p_sim.set_defaults(func=cmd_simulate)

    p_web = sub.add_parser("web", help="Launch comparison simulator web UI")
    p_web.add_argument("--host", default=None)
    p_web.add_argument("--port", type=int, default=None)
    p_web.add_argument("--reload", action="store_true", help="Dev auto-reload")
    p_web.set_defaults(func=cmd_web)

    p_tune = sub.add_parser("tune-weights", help="Find optimal score weights via simulator")
    p_tune.add_argument("--method", choices=["grid", "random"], default=None)
    p_tune.add_argument("--trials", type=int, default=None)
    p_tune.add_argument("--seeds", type=int, nargs="+", default=None)
    p_tune.add_argument("--apply", action="store_true", help="Write best weights to config YAML")
    p_tune.set_defaults(func=cmd_tune_weights)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
