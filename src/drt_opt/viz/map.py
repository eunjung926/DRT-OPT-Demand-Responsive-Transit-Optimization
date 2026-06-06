"""Map visualization with OpenStreetMap + optional OSRM road routing."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import folium
from folium.plugins import HeatMap

if TYPE_CHECKING:
    from drt_opt.models.request import Request
    from drt_opt.models.stop import Stop
    from drt_opt.models.vehicle import Vehicle

logger = logging.getLogger(__name__)


def _region_center(stops: list[Stop], bbox: dict | None = None) -> tuple[float, float]:
    if bbox:
        lat = (bbox["lat_min"] + bbox["lat_max"]) / 2
        lon = (bbox["lon_min"] + bbox["lon_max"]) / 2
        return lat, lon
    if stops:
        return sum(s.lat for s in stops) / len(stops), sum(s.lon for s in stops) / len(stops)
    return 37.492, 126.522


def create_network_map(
    stops: list[Stop],
    demand_map: dict[str, float] | None = None,
    bbox: dict | None = None,
    title: str = "영종도 DRT 정류장",
) -> folium.Map:
    """Create interactive OSM map with stop markers and demand heatmap."""
    center = _region_center(stops, bbox)
    m = folium.Map(location=center, zoom_start=13, tiles="OpenStreetMap")

    if bbox:
        folium.Rectangle(
            bounds=[
                [bbox["lat_min"], bbox["lon_min"]],
                [bbox["lat_max"], bbox["lon_max"]],
            ],
            color="#3388ff",
            fill=False,
            weight=2,
            popup="영종도 서비스 구역",
        ).add_to(m)

    demand_map = demand_map or {}
    heat_data = []
    max_d = max(demand_map.values()) if demand_map else 1.0

    for stop in stops:
        d = demand_map.get(stop.id, stop.demand_weight)
        heat_data.append([stop.lat, stop.lon, d / max_d if max_d > 0 else 0.1])
        radius = 5 + 15 * (d / max_d if max_d > 0 else 0.1)
        folium.CircleMarker(
            location=[stop.lat, stop.lon],
            radius=radius,
            popup=f"{stop.name}<br>수요: {d:.4f}",
            tooltip=stop.name,
            color="#e74c3c",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)

    if heat_data:
        HeatMap(heat_data, radius=18, blur=12, max_zoom=15).add_to(m)

    folium.LayerControl().add_to(m)
    title_html = f"""
    <div style="position:fixed;top:10px;left:50px;z-index:9999;
                background:white;padding:8px 12px;border-radius:4px;
                box-shadow:0 1px 4px rgba(0,0,0,.3);font-size:14px;">
        <b>{title}</b> — 정류장 {len(stops)}개
    </div>"""
    m.get_root().html.add_child(folium.Element(title_html))
    return m


def fetch_osrm_route(
    coords: list[tuple[float, float]],
    osrm_server: str = "https://router.project-osrm.org",
) -> list[list[float]] | None:
    """Fetch road-following route geometry from OSRM. Returns [[lat,lon], ...]."""
    if len(coords) < 2:
        return None

    try:
        import urllib.parse
        import urllib.request

        # OSRM expects lon,lat
        points = ";".join(f"{lon},{lat}" for lat, lon in coords)
        url = f"{osrm_server}/route/v1/driving/{points}?overview=full&geometries=geojson"
        req = urllib.request.Request(url, headers={"User-Agent": "drt-opt/0.1"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        if data.get("code") != "Ok":
            return None
        geometry = data["routes"][0]["geometry"]["coordinates"]
        return [[lat, lon] for lon, lat in geometry]
    except Exception as exc:
        logger.warning("OSRM routing failed: %s", exc)
        return None


def _stop_coords(stops_by_id: dict[str, Stop], stop_id: str) -> tuple[float, float] | None:
    s = stops_by_id.get(stop_id)
    if s is None:
        return None
    return s.lat, s.lon


def add_vehicle_route(
    m: folium.Map,
    route_stops: list,
    stops_by_id: dict[str, Stop],
    color: str = "#2980b9",
    label: str = "",
    use_osrm: bool = True,
    osrm_server: str = "https://router.project-osrm.org",
) -> None:
    """Draw a vehicle route on the map."""
    coords: list[tuple[float, float]] = []
    for rs in route_stops:
        c = _stop_coords(stops_by_id, rs.stop_id)
        if c:
            coords.append(c)

    if len(coords) < 2:
        return

    line_coords: list[list[float]]
    if use_osrm:
        osrm_line = fetch_osrm_route(coords, osrm_server)
        line_coords = osrm_line if osrm_line else [[lat, lon] for lat, lon in coords]
    else:
        line_coords = [[lat, lon] for lat, lon in coords]

    folium.PolyLine(
        line_coords,
        color=color,
        weight=4,
        opacity=0.8,
        popup=label,
    ).add_to(m)

    for rs in route_stops:
        c = _stop_coords(stops_by_id, rs.stop_id)
        if not c:
            continue
        icon_color = "green" if rs.action == "pickup" else "red" if rs.action == "dropoff" else "blue"
        folium.Marker(
            location=list(c),
            popup=f"{rs.action} @ {rs.stop_id}",
            icon=folium.Icon(color=icon_color, icon="info-sign"),
        ).add_to(m)


def create_simulation_map(
    stops: list[Stop],
    vehicles: list[Vehicle],
    requests: list[Request] | None = None,
    route_history: list[tuple[str, list]] | None = None,
    bbox: dict | None = None,
    use_osrm: bool = True,
    osrm_server: str = "https://router.project-osrm.org",
    demand_map: dict[str, float] | None = None,
) -> folium.Map:
    """Map showing stops, vehicle routes, and served request O-D pairs."""
    m = create_network_map(stops, demand_map, bbox, title="영종도 DRT 시뮬레이션")

    stops_by_id = {s.id: s for s in stops}
    colors = ["#2980b9", "#27ae60", "#8e44ad", "#e67e22", "#16a085", "#c0392b", "#2c3e50", "#d35400"]

    if route_history:
        for i, (vid, route) in enumerate(route_history[:30]):
            add_vehicle_route(
                m,
                route,
                stops_by_id,
                color=colors[i % len(colors)],
                label=f"차량 {vid}",
                use_osrm=use_osrm,
                osrm_server=osrm_server,
            )

    for i, vehicle in enumerate(vehicles):
        if vehicle.route:
            add_vehicle_route(
                m,
                vehicle.route,
                stops_by_id,
                color=colors[i % len(colors)],
                label=f"차량 {vehicle.id}",
                use_osrm=use_osrm,
                osrm_server=osrm_server,
            )
        c = _stop_coords(stops_by_id, vehicle.current_location)
        if c:
            folium.CircleMarker(
                location=list(c),
                radius=8,
                popup=f"차량 {vehicle.id} (탑승 {len(vehicle.onboard)}명)",
                color=colors[i % len(colors)],
                fill=True,
            ).add_to(m)

    if requests:
        served = [r for r in requests if r.status == "served"]
        for r in served[:50]:  # limit markers
            o = _stop_coords(stops_by_id, r.origin_stop_id)
            d = _stop_coords(stops_by_id, r.dest_stop_id)
            if o and d:
                folium.PolyLine(
                    [[o[0], o[1]], [d[0], d[1]]],
                    color="#95a5a6",
                    weight=1,
                    opacity=0.4,
                    dash_array="5,5",
                ).add_to(m)

    return m


def save_map(m: folium.Map, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(path))
    return path
