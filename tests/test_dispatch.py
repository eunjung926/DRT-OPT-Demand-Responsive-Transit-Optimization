import pytest

from drt_opt.dispatch.constraints import (
    check_capacity,
    check_existing_passenger_delay,
    check_new_passenger_wait,
)
from drt_opt.dispatch.insertion import enumerate_insertions, simulate_route_times
from drt_opt.dispatch.optimized import demand_bonus
from drt_opt.dispatch.baseline import NearestVehicleDispatcher
from drt_opt.dispatch.optimized import OptimizedDispatcher
from drt_opt.config import load_config
from drt_opt.data.loader import create_sample_stops, preprocess_network
from drt_opt.data.od_generator import generate_od_requests
from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.models.request import Request
from drt_opt.models.route import RouteStop
from drt_opt.models.stop import Stop
from drt_opt.simulation.engine import Simulator


@pytest.fixture
def toy_network():
    stops = [
        Stop("A", "A", 37.60, 126.65, 0.25),
        Stop("B", "B", 37.61, 126.66, 0.25),
        Stop("C", "C", 37.62, 126.67, 0.25),
        Stop("D", "D", 37.63, 126.68, 0.25),
    ]
    travel = TravelMatrix.from_stops(stops, speed_kmh=30)
    demand_map = {s.id: s.demand_weight for s in stops}
    return stops, travel, demand_map


def test_enumeration_count():
    route = [
        RouteStop("A", action="pass_through"),
        RouteStop("B", action="pass_through"),
        RouteStop("C", action="pass_through"),
        RouteStop("D", action="pass_through"),
    ]
    request = Request("R1", "S", "T", 0.0)
    candidates = enumerate_insertions(route, request)
    assert len(candidates) == 15


def test_pickup_before_dropoff_in_all_candidates():
    route = [RouteStop("A"), RouteStop("B")]
    request = Request("R1", "S", "T", 0.0)
    for candidate in enumerate_insertions(route, request):
        pickup_idx = next(i for i, s in enumerate(candidate) if s.action == "pickup")
        dropoff_idx = next(i for i, s in enumerate(candidate) if s.action == "dropoff")
        assert pickup_idx < dropoff_idx


def test_simulate_route_times_increasing(toy_network):
    _, travel, _ = toy_network
    route = [
        RouteStop("A", action="pickup", request_id="R1"),
        RouteStop("C", action="dropoff", request_id="R1"),
    ]
    timed = simulate_route_times(route, travel, "B", start_time=100.0, dwell_time_min=0.5)
    assert timed[0].arrival_time >= 100.0
    assert timed[1].arrival_time > timed[0].depart_time


def test_capacity_rejects_overfull():
    route = [RouteStop(f"p{i}", action="pickup", request_id=f"r{i}") for i in range(13)]
    assert check_capacity(route, capacity=12) is False
    assert check_capacity(route[:12], capacity=12) is True


def test_existing_delay_constraint():
    old = [
        RouteStop("A", arrival_time=0, depart_time=1, action="pickup", request_id="E1"),
        RouteStop("D", arrival_time=20, depart_time=21, action="dropoff", request_id="E1"),
    ]
    new = [
        RouteStop("A", arrival_time=0, depart_time=1, action="pickup", request_id="E1"),
        RouteStop("X", arrival_time=10, depart_time=11, action="pass_through"),
        RouteStop("D", arrival_time=40, depart_time=41, action="dropoff", request_id="E1"),
    ]
    onboard = {"E1": Request("E1", "A", "D", 0.0)}
    assert check_existing_passenger_delay(old, new, onboard, max_delay_min=15) is False
    assert check_existing_passenger_delay(old, new, onboard, max_delay_min=25) is True


def test_new_passenger_wait_constraint():
    route = [
        RouteStop("S", arrival_time=30, action="pickup", request_id="R1"),
        RouteStop("D", arrival_time=40, action="dropoff", request_id="R1"),
    ]
    request = Request("R1", "S", "T", request_time=0.0)
    assert check_new_passenger_wait(route, request, max_wait_min=20) is False
    assert check_new_passenger_wait(route, request, max_wait_min=35) is True


def test_demand_bonus_prefers_high_demand_route():
    demand_map = {"A": 0.1, "B": 0.9}
    low = [RouteStop("A")]
    high = [RouteStop("B")]
    assert demand_bonus(high, demand_map) > demand_bonus(low, demand_map)


def test_preprocess_sample_data():
    config = load_config()
    stops, travel, demand_map = preprocess_network(config, use_sample=True)
    assert len(stops) >= 10
    assert all("영종" in s.name or s.district == "영종" for s in stops)
    assert travel.time(stops[0].id, stops[1].id) > 0
    assert abs(sum(demand_map.values()) - 1.0) < 0.01


def test_yeongjong_bbox_filter():
    from drt_opt.data.loader import filter_stops_by_region, Stop

    config = load_config()
    region = config["region"]
    inside = Stop("1", "영종 테스트", 37.49, 126.50, 0.1, "영종")
    outside = Stop("2", "검단 테스트", 37.60, 126.65, 0.1, "검단")
    # 이름에 '영종'이 있어도 bbox 밖이면 제외 (검암역 케이스)
    name_only = Stop("3", "영종_검암역", 37.57, 126.68, 0.1, "영종")
    filtered = filter_stops_by_region([inside, outside, name_only], region)
    assert len(filtered) == 1
    assert filtered[0].id == "1"


def test_network_map_creation():
    from drt_opt.viz.map import create_network_map

    config = load_config()
    stops, _, demand_map = preprocess_network(config, use_sample=True)
    m = create_network_map(stops, demand_map, config["region"].get("bbox"))
    assert m is not None


def test_od_generator_respects_time_window():
    stops = create_sample_stops(n_stops=10)
    requests = generate_od_requests(stops, 480, 540, requests_per_hour=60, seed=1)
    assert len(requests) > 0
    assert all(480 <= r.request_time < 540 for r in requests)
    assert all(r.origin_stop_id != r.dest_stop_id for r in requests)


def test_end_to_end_simulation(toy_network):
    config = load_config()
    config["vehicle"]["count"] = 2
    config["simulation"]["end_time_min"] = 600
    stops, travel, demand_map = toy_network

    requests = [
        Request("R1", "A", "C", 480.0),
        Request("R2", "B", "D", 490.0),
    ]

    dispatcher = OptimizedDispatcher(config)
    sim = Simulator(config, stops, travel, demand_map, dispatcher)
    collector = sim.run(requests)
    metrics = collector.metrics

    assert metrics.total_requests == 2
    assert metrics.assigned_requests >= 1


def test_optimized_vs_baseline_on_sample():
    config = load_config()
    config["simulation"]["end_time_min"] = 720
    stops, travel, demand_map = preprocess_network(config, use_sample=True)
    requests = generate_od_requests(stops, 480, 720, requests_per_hour=20, seed=99)

    baseline_sim = Simulator(config, stops, travel, demand_map, NearestVehicleDispatcher(config))
    optimized_sim = Simulator(config, stops, travel, demand_map, OptimizedDispatcher(config))

    b = baseline_sim.run(requests).metrics
    o = optimized_sim.run(requests).metrics

    assert b.total_requests == o.total_requests == len(requests)
