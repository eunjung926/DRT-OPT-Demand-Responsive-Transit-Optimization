from copy import deepcopy

from drt_opt.config import load_config
from drt_opt.tuning.weight_optimizer import (
    WeightSet,
    compute_objective,
    evaluate_weights,
    generate_random_candidates,
    optimize_weights,
)
from drt_opt.metrics.collector import SimulationMetrics


def test_compute_objective_prefers_low_wait():
    obj_cfg = {"wait": 1.0, "success": 80.0, "distance": 0.05, "reject": 50.0}
    good = SimulationMetrics(avg_wait_time_min=3.0, dispatch_success_rate=0.95, total_distance_km=100)
    bad = SimulationMetrics(avg_wait_time_min=15.0, dispatch_success_rate=0.95, total_distance_km=100)
    assert compute_objective(good, obj_cfg) < compute_objective(bad, obj_cfg)


def test_evaluate_weights_runs():
    config = load_config()
    config = deepcopy(config)
    config["tuning"] = {
        **config.get("tuning", {}),
        "start_time_min": 480,
        "end_time_min": 540,
        "requests_per_hour": 10,
        "objective": {"wait": 1.0, "success": 80.0, "distance": 0.05, "reject": 50.0},
    }
    weights = WeightSet(wait=1.0, detour=0.5, distance=0.3, demand=0.2)
    mean_obj, results = evaluate_weights(config, weights, seeds=[1], tuning_cfg=config["tuning"])
    assert mean_obj > 0
    assert len(results) == 1


def test_optimize_weights_random_small():
    config = load_config()
    config = deepcopy(config)
    config["tuning"] = {
        **config.get("tuning", {}),
        "start_time_min": 480,
        "end_time_min": 540,
        "requests_per_hour": 8,
        "evaluation_seeds": [1],
        "method": "random",
        "n_trials": 3,
        "ranges": {
            "wait": [0.5, 1.5],
            "detour": [0.3, 0.7],
            "distance": [0.2, 0.4],
            "demand": [0.1, 0.3],
        },
        "objective": {"wait": 1.0, "success": 80.0, "distance": 0.05, "reject": 50.0},
        "output_dir": "results/tuning_test",
    }
    summary = optimize_weights(config=config, method="random", n_trials=3, seeds=[1])
    assert "best_weights" in summary
    assert all(k in summary["best_weights"] for k in ("wait", "detour", "distance", "demand"))


def test_generate_random_candidates():
    config = load_config()
    import numpy as np

    rng = np.random.default_rng(0)
    cands = generate_random_candidates(5, config.get("tuning", {}), rng)
    assert len(cands) == 5
