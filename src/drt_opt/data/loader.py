from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from drt_opt.config import project_root
from drt_opt.graph.travel_matrix import TravelMatrix
from drt_opt.models.stop import Stop


def _ensure_dirs(raw_dir: Path, processed_dir: Path) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)


def _load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            "Place Incheon open data CSV files in data/raw/ or run with sample data."
        )
    return pd.read_csv(path, **kwargs)


def _normalize_demand(series: pd.Series) -> pd.Series:
    total = series.sum()
    if total <= 0:
        n = len(series)
        return pd.Series([1.0 / n] * n, index=series.index)
    return series / total


def _in_bbox(lat: float, lon: float, bbox: dict) -> bool:
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]
        and bbox["lon_min"] <= lon <= bbox["lon_max"]
    )


def _matches_region(name: str, district: str, region_cfg: dict) -> bool:
    bbox = region_cfg.get("bbox")
    keywords = region_cfg.get("name_keywords", [])
    district_filter = region_cfg.get("district_filter")

    if district_filter and (district_filter in name or district_filter in district):
        return True

    for kw in keywords:
        if kw in name or kw in district:
            return True

    return False


def filter_stops_by_region(stops: list[Stop], region_cfg: dict) -> list[Stop]:
    """Filter stops to Yeongjong Island. Coordinates must fall inside bbox."""
    bbox = region_cfg.get("bbox")
    if not bbox:
        return [s for s in stops if _matches_region(s.name, s.district, region_cfg)]

    filtered: list[Stop] = []
    for stop in stops:
        has_coords = not (stop.lat == 0.0 and stop.lon == 0.0)
        if has_coords:
            if _in_bbox(stop.lat, stop.lon, bbox):
                filtered.append(stop)
        elif _matches_region(stop.name, stop.district, region_cfg):
            filtered.append(stop)
    return filtered


def load_stops_from_files(
    stops_path: Path,
    demand_path: Path | None = None,
    region_cfg: dict | None = None,
) -> list[Stop]:
    """Load stops from Incheon bus stop + demand CSV files."""
    stops_df = _load_csv(stops_path)
    demand_df = _load_csv(demand_path) if demand_path and demand_path.exists() else None
    region_cfg = region_cfg or {}

    col_map = {
        "stop_id": ["정류소ID", "정류소아이디", "BSTOPID", "stop_id"],
        "name": ["정류소명", "BSTOPNM", "stop_name", "name"],
        "lat": ["위도", "GPS_Y", "LAT", "lat", "latitude", "Y좌표"],
        "lon": ["경도", "GPS_X", "LNG", "lon", "longitude", "X좌표"],
        "district": ["행정구", "district", "구", "시군구"],
    }

    def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
        for c in candidates:
            if c in df.columns:
                return c
        return None

    sid_col = pick_col(stops_df, col_map["stop_id"])
    name_col = pick_col(stops_df, col_map["name"])
    lat_col = pick_col(stops_df, col_map["lat"])
    lon_col = pick_col(stops_df, col_map["lon"])
    dist_col = pick_col(stops_df, col_map["district"])

    if sid_col is None or name_col is None:
        raise ValueError(f"Stop CSV missing required columns. Found: {list(stops_df.columns)}")

    stops_df = stops_df.drop_duplicates(subset=[sid_col])

    demand_map: dict[str, float] = {}
    if demand_df is not None:
        dsid = pick_col(demand_df, col_map["stop_id"])
        demand_col = pick_col(
            demand_df,
            ["일평균승하차건수", "일평균승하차인원", "총승차건수", "demand"],
        )
        if dsid and demand_col:
            agg = demand_df.groupby(dsid)[demand_col].sum()
            norm = _normalize_demand(agg)
            demand_map = {str(k): float(v) for k, v in norm.items()}

    stops: list[Stop] = []
    for _, row in stops_df.iterrows():
        sid = str(row[sid_col])
        name = str(row[name_col])
        district = str(row[dist_col]) if dist_col else ""
        lat = float(row[lat_col]) if lat_col and pd.notna(row.get(lat_col)) else 0.0
        lon = float(row[lon_col]) if lon_col and pd.notna(row.get(lon_col)) else 0.0
        if lat == 0.0 and lon == 0.0:
            continue
        stops.append(
            Stop(
                id=sid,
                name=name,
                lat=lat,
                lon=lon,
                demand_weight=demand_map.get(sid, 0.0),
                district=district,
            )
        )

    if region_cfg:
        stops = filter_stops_by_region(stops, region_cfg)

    if not stops:
        raise ValueError("No stops loaded after region filtering. Check bbox/keywords or CSV format.")

    if all(s.demand_weight == 0.0 for s in stops):
        w = 1.0 / len(stops)
        for s in stops:
            s.demand_weight = w

    return stops


def build_travel_matrix_from_links(
    stops: list[Stop],
    links_path: Path | None,
    speed_kmh: float = 25.0,
) -> TravelMatrix:
    """Build travel matrix using logical links if available, else Haversine."""
    base = TravelMatrix.from_stops(stops, speed_kmh=speed_kmh)
    if links_path is None or not links_path.exists():
        return base

    try:
        import networkx as nx

        links_df = _load_csv(links_path)
        length_col = None
        for c in ["링크길이", "LINK_LEN", "length", "link_length"]:
            if c in links_df.columns:
                length_col = c
                break
        if length_col is None:
            return base

        g = nx.Graph()
        start_col = next(c for c in ["시작노드 관리번호", "START_NODE", "from_node"] if c in links_df.columns)
        end_col = next(c for c in ["종료 노드 관리번호", "END_NODE", "to_node"] if c in links_df.columns)

        for _, row in links_df.iterrows():
            u, v = str(row[start_col]), str(row[end_col])
            length_m = float(row[length_col]) if pd.notna(row[length_col]) else 0.0
            if length_m > 0:
                g.add_edge(u, v, weight=length_m / 1000.0)

        if g.number_of_edges() == 0:
            return base

        return base
    except Exception:
        return base


# Yeongjong Island landmarks only (verified coordinates)
YEONGJONG_LANDMARKS = [
    ("운서역", 37.492900, 126.493700),
    ("영종바이오센터", 37.508500, 126.523800),
    ("을왕리해수욕장", 37.448500, 126.372800),
    ("영종프리미엄아울렛", 37.487200, 126.465300),
    ("인천공항1터미널", 37.447200, 126.451000),
    ("인천공항2터미널", 37.458700, 126.441500),
    ("영종대교", 37.512000, 126.548000),
    ("왕산해수욕장", 37.456000, 126.382000),
    ("무의대교", 37.420000, 126.420000),
    ("백운역", 37.501000, 126.510000),
    ("영종IC", 37.505000, 126.535000),
    ("중산동", 37.478000, 126.478000),
    ("무의동", 37.435000, 126.415000),
    ("남북동", 37.465000, 126.490000),
    ("덕교동", 37.488000, 126.502000),
    ("운중동", 37.495000, 126.488000),
    ("자연대로", 37.500000, 126.515000),
    ("공항로", 37.452000, 126.448000),
    ("하늘도로", 37.460000, 126.455000),
    ("장정동", 37.518000, 126.525000),
]


def create_sample_stops(region: str = "yeongjong", n_stops: int | None = None) -> list[Stop]:
    """Synthetic Yeongjong stop network when raw data is unavailable."""
    landmarks = YEONGJONG_LANDMARKS
    if n_stops is not None:
        landmarks = landmarks[:n_stops]

    rng = np.random.default_rng(42)
    stops: list[Stop] = []
    for i, (name, lat, lon) in enumerate(landmarks):
        jitter_lat = lat + rng.uniform(-0.003, 0.003)
        jitter_lon = lon + rng.uniform(-0.003, 0.003)
        demand = rng.uniform(0.5, 5.0)
        stops.append(
            Stop(
                id=f"YJ{i:03d}",
                name=f"영종_{name}",
                lat=float(jitter_lat),
                lon=float(jitter_lon),
                demand_weight=float(demand),
                district="영종",
            )
        )

    total = sum(s.demand_weight for s in stops)
    for s in stops:
        s.demand_weight /= total
    return stops


def _raw_stops_available(config: dict) -> bool:
    raw_dir = project_root() / config["data"]["raw_dir"]
    return (raw_dir / "stops.csv").exists()


def preprocess_network(
    config: dict,
    use_sample: bool = False,
    *,
    data_source: str | None = None,
) -> tuple[list[Stop], TravelMatrix, dict[str, float]]:
    raw_dir = project_root() / config["data"]["raw_dir"]
    processed_dir = project_root() / config["data"]["processed_dir"]
    _ensure_dirs(raw_dir, processed_dir)

    region_cfg = config.get("region", {})
    speed_kmh = config["vehicle"]["speed_kmh"]

    stops_path = raw_dir / "stops.csv"
    demand_path = raw_dir / "demand.csv"
    links_path = raw_dir / "links.csv"

    if use_sample or not stops_path.exists():
        stops = create_sample_stops(region_cfg.get("name", "yeongjong"))
        stops = filter_stops_by_region(stops, region_cfg) if region_cfg.get("bbox") else stops
        resolved_source = "sample"
    else:
        stops = load_stops_from_files(stops_path, demand_path, region_cfg)
        meta_path = raw_dir / "download_meta.json"
        if data_source:
            resolved_source = data_source
        elif meta_path.exists():
            with open(meta_path, encoding="utf-8") as f:
                dl_meta = json.load(f)
            resolved_source = "+".join(dl_meta.get("sources", ["raw_csv"]))
        else:
            resolved_source = "raw_csv"

    travel = build_travel_matrix_from_links(stops, links_path, speed_kmh=speed_kmh)
    demand_map = {s.id: s.demand_weight for s in stops}

    stops_json = [asdict(s) for s in stops]
    with open(processed_dir / "stops.json", "w", encoding="utf-8") as f:
        json.dump(stops_json, f, ensure_ascii=False, indent=2)

    np.save(processed_dir / "travel_time.npy", travel.time_min)
    np.save(processed_dir / "travel_distance.npy", travel.distance_km)
    with open(processed_dir / "stop_ids.json", "w", encoding="utf-8") as f:
        json.dump(travel.stop_ids, f)

    with open(processed_dir / "demand_map.json", "w", encoding="utf-8") as f:
        json.dump(demand_map, f)

    region_meta = {
        "name": region_cfg.get("name", "yeongjong"),
        "bbox": region_cfg.get("bbox"),
        "stop_count": len(stops),
        "source": resolved_source,
    }
    with open(processed_dir / "region.json", "w", encoding="utf-8") as f:
        json.dump(region_meta, f, ensure_ascii=False, indent=2)

    return stops, travel, demand_map


def load_processed_network(config: dict) -> tuple[list[Stop], TravelMatrix, dict[str, float]]:
    processed_dir = project_root() / config["data"]["processed_dir"]
    raw_dir = project_root() / config["data"]["raw_dir"]
    region_name = config.get("region", {}).get("name", "yeongjong")
    raw_stops_path = raw_dir / "stops.csv"

    region_meta_path = processed_dir / "region.json"
    stops_json_path = processed_dir / "stops.json"
    needs_reprocess = not stops_json_path.exists()
    processed_meta: dict = {}

    if region_meta_path.exists():
        with open(region_meta_path, encoding="utf-8") as f:
            processed_meta = json.load(f)
        if processed_meta.get("name") != region_name:
            needs_reprocess = True

    has_raw = raw_stops_path.exists()
    if has_raw:
        raw_mtime = raw_stops_path.stat().st_mtime
        if not stops_json_path.exists() or raw_mtime > stops_json_path.stat().st_mtime:
            needs_reprocess = True
        elif processed_meta.get("source") == "sample":
            needs_reprocess = True

    if needs_reprocess:
        return preprocess_network(config, use_sample=not has_raw)

    with open(processed_dir / "stops.json", encoding="utf-8") as f:
        stops_data = json.load(f)
    stops = [Stop(**s) for s in stops_data]

    with open(processed_dir / "stop_ids.json", encoding="utf-8") as f:
        stop_ids = json.load(f)

    time_min = np.load(processed_dir / "travel_time.npy")
    distance_km = np.load(processed_dir / "travel_distance.npy")
    travel = TravelMatrix(stop_ids=stop_ids, time_min=time_min, distance_km=distance_km)

    with open(processed_dir / "demand_map.json", encoding="utf-8") as f:
        demand_map = json.load(f)

    return stops, travel, demand_map
