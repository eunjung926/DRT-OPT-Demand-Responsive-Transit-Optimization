from __future__ import annotations

from drt_opt.dispatch.insertion import get_dropoff_time, get_pickup_time, passenger_travel_time
from drt_opt.models.request import Request
from drt_opt.models.route import RouteStop


def check_capacity(route: list[RouteStop], capacity: int) -> bool:
    onboard = 0
    for stop in route:
        if stop.action == "pickup":
            onboard += 1
        elif stop.action == "dropoff":
            onboard -= 1
        if onboard > capacity:
            return False
        if onboard < 0:
            return False
    return True


def check_new_passenger_wait(
    route: list[RouteStop],
    request: Request,
    max_wait_min: float,
) -> bool:
    pickup = get_pickup_time(route, request.id)
    if pickup is None:
        return False
    wait = pickup - request.request_time
    return wait <= max_wait_min and wait >= 0


def check_existing_passenger_delay(
    old_route: list[RouteStop],
    new_route: list[RouteStop],
    onboard_requests: dict[str, Request],
    max_delay_min: float,
) -> bool:
    for req_id in onboard_requests:
        old_travel = passenger_travel_time(old_route, req_id)
        new_travel = passenger_travel_time(new_route, req_id)
        if old_travel is None or new_travel is None:
            continue
        if new_travel - old_travel > max_delay_min:
            return False
    return True


def validate_candidate(
    old_route: list[RouteStop],
    new_route: list[RouteStop],
    request: Request,
    onboard_requests: dict[str, Request],
    capacity: int,
    max_new_wait_min: float,
    max_existing_delay_min: float,
) -> bool:
    if not check_capacity(new_route, capacity):
        return False
    if not check_new_passenger_wait(new_route, request, max_new_wait_min):
        return False
    if not check_existing_passenger_delay(
        old_route, new_route, onboard_requests, max_existing_delay_min
    ):
        return False
    return True


def total_detour_time(
    old_route: list[RouteStop],
    new_route: list[RouteStop],
    onboard_requests: dict[str, Request],
) -> float:
    total = 0.0
    for req_id in onboard_requests:
        old_travel = passenger_travel_time(old_route, req_id)
        new_travel = passenger_travel_time(new_route, req_id)
        if old_travel is not None and new_travel is not None:
            total += max(0.0, new_travel - old_travel)
    return total
