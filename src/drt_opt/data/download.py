"""Download real bus stop data for Yeongjong Island."""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from drt_opt.config import load_config, project_root

logger = logging.getLogger(__name__)

OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
]
INCHEON_AROUND_URL = (
    "http://apis.data.go.kr/6280000/busStationService/getBusStationAroundList"
)
INCHEON_NAME_URL = (
    "http://apis.data.go.kr/6280000/busStationService/getBusStationNmList"
)

STOP_COLUMNS = ["BSTOPID", "BSTOPNM", "LAT", "LNG", "district", "source"]
DEMAND_COLUMNS = ["BSTOPID", "일평균승하차건수", "source"]


def _load_api_key() -> str | None:
    key = os.environ.get("DATA_GO_KR_KEY")
    if key and key != "your_api_key_here":
        return key
    env_path = project_root() / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("DATA_GO_KR_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value and value != "your_api_key_here":
                    return value
    return None


def _http_get(url: str, data: bytes | None = None, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(
        url,
        data=data,
        headers={"User-Agent": "drt-opt/0.1 (DRT research project)"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _in_bbox(lat: float, lon: float, bbox: dict) -> bool:
    return (
        bbox["lat_min"] <= lat <= bbox["lat_max"]
        and bbox["lon_min"] <= lon <= bbox["lon_max"]
    )


def _write_stops_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STOP_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_demand_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=DEMAND_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _dedupe_stops(stops: dict[str, dict]) -> list[dict]:
    by_coord: dict[tuple[float, float], dict] = {}
    for stop in stops.values():
        lat = round(float(stop["LAT"]), 5)
        lon = round(float(stop["LNG"]), 5)
        key = (lat, lon)
        existing = by_coord.get(key)
        if existing is None or stop.get("source") == "incheon_api":
            by_coord[key] = stop
        elif existing.get("source") != "incheon_api" and stop.get("BSTOPNM"):
            if not existing.get("BSTOPNM") or existing["BSTOPNM"].startswith("?"):
                by_coord[key] = stop
    return sorted(by_coord.values(), key=lambda s: s["BSTOPID"])


def download_osm_stops(bbox: dict) -> list[dict]:
    """Fetch bus stops from OpenStreetMap Overpass (no API key required)."""
    query = (
        "[out:json][timeout:90];("
        f'node["highway"="bus_stop"]({bbox["lat_min"]},{bbox["lon_min"]},'
        f'{bbox["lat_max"]},{bbox["lon_max"]});'
        f'node["public_transport"="platform"]["bus"="yes"]'
        f'({bbox["lat_min"]},{bbox["lon_min"]},{bbox["lat_max"]},{bbox["lon_max"]});'
        ");out body;"
    )
    logger.info("Fetching OSM bus stops for Yeongjong bbox...")
    payload = urllib.parse.urlencode({"data": query}).encode("utf-8")
    last_error: Exception | None = None
    raw: bytes | None = None
    for url in OVERPASS_URLS:
        try:
            raw = _http_get(url, data=payload, timeout=120.0)
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            logger.warning("Overpass request failed (%s): %s", url, exc)
            time.sleep(1.0)
    if raw is None:
        raise RuntimeError(f"All Overpass endpoints failed: {last_error}") from last_error
    data = json.loads(raw.decode("utf-8"))

    stops: dict[str, dict] = {}
    for element in data.get("elements", []):
        if element.get("type") != "node":
            continue
        lat = element.get("lat")
        lon = element.get("lon")
        if lat is None or lon is None:
            continue
        if not _in_bbox(float(lat), float(lon), bbox):
            continue
        tags = element.get("tags", {})
        name = tags.get("name") or tags.get("name:ko") or tags.get("ref") or f"정류장_{element['id']}"
        sid = str(tags.get("ref") or f"osm_{element['id']}")
        stops[sid] = {
            "BSTOPID": sid,
            "BSTOPNM": name,
            "LAT": f"{float(lat):.7f}",
            "LNG": f"{float(lon):.7f}",
            "district": "영종",
            "source": "osm",
        }
    result = _dedupe_stops(stops)
    logger.info("OSM: %d bus stops in Yeongjong bbox", len(result))
    return result


def _parse_incheon_xml(raw: bytes) -> list[dict]:
    root = ET.fromstring(raw)
    items: list[dict] = []
    for item in root.iter("item"):
        row = {child.tag: (child.text or "").strip() for child in item}
        if row.get("BSTOPID"):
            items.append(row)
    return items


def _fetch_incheon_page(
    base_url: str,
    api_key: str,
    params: dict[str, str],
    page_no: int = 1,
    num_of_rows: int = 500,
) -> tuple[list[dict], int]:
    query = {
        "serviceKey": api_key,
        "pageNo": str(page_no),
        "numOfRows": str(num_of_rows),
        **params,
    }
    url = f"{base_url}?{urllib.parse.urlencode(query)}"
    raw = _http_get(url, timeout=30.0)
    items = _parse_incheon_xml(raw)
    root = ET.fromstring(raw)
    total_el = root.find(".//totalCount")
    total = int(total_el.text) if total_el is not None and total_el.text else len(items)
    return items, total


def download_incheon_api_stops(api_key: str, bbox: dict) -> list[dict]:
    """Fetch stops via Incheon BIS Open API (requires DATA_GO_KR_KEY)."""
    stops: dict[str, dict] = {}
    keywords = ["영종", "운서", "을왕", "무의", "중산", "공항", "백운", "장정"]

    for keyword in keywords:
        page = 1
        while True:
            try:
                items, total = _fetch_incheon_page(
                    INCHEON_NAME_URL,
                    api_key,
                    {"bstopNm": keyword},
                    page_no=page,
                )
            except urllib.error.HTTPError as exc:
                logger.warning("Incheon name search failed for %s: %s", keyword, exc)
                break
            for item in items:
                try:
                    lat = float(item.get("LAT", "0") or "0")
                    lon = float(item.get("LNG", "0") or "0")
                except ValueError:
                    continue
                if lat == 0.0 or lon == 0.0 or not _in_bbox(lat, lon, bbox):
                    continue
                sid = item["BSTOPID"]
                stops[sid] = {
                    "BSTOPID": sid,
                    "BSTOPNM": item.get("BSTOPNM", sid),
                    "LAT": f"{lat:.7f}",
                    "LNG": f"{lon:.7f}",
                    "district": "영종",
                    "source": "incheon_api",
                }
            if page * 500 >= total or not items:
                break
            page += 1
            time.sleep(0.15)

    lat = bbox["lat_min"]
    step = 0.004
    grid_points = 0
    while lat <= bbox["lat_max"]:
        lon = bbox["lon_min"]
        while lon <= bbox["lon_max"]:
            grid_points += 1
            try:
                items, _ = _fetch_incheon_page(
                    INCHEON_AROUND_URL,
                    api_key,
                    {"LAT": f"{lat:.6f}", "LNG": f"{lon:.6f}"},
                    num_of_rows=200,
                )
            except urllib.error.HTTPError as exc:
                logger.warning("Incheon around search failed at %.4f,%.4f: %s", lat, lon, exc)
                lon += step
                continue
            for item in items:
                try:
                    slat = float(item.get("LAT", "0") or "0")
                    slon = float(item.get("LNG", "0") or "0")
                except ValueError:
                    continue
                if slat == 0.0 or slon == 0.0 or not _in_bbox(slat, slon, bbox):
                    continue
                sid = item["BSTOPID"]
                stops[sid] = {
                    "BSTOPID": sid,
                    "BSTOPNM": item.get("BSTOPNM", sid),
                    "LAT": f"{slat:.7f}",
                    "LNG": f"{slon:.7f}",
                    "district": "영종",
                    "source": "incheon_api",
                }
            lon += step
            time.sleep(0.12)
        lat += step

    result = _dedupe_stops(stops)
    logger.info(
        "Incheon API: %d stops (%d grid queries, keywords=%d)",
        len(result),
        grid_points,
        len(keywords),
    )
    return result


def _merge_stops(osm_rows: list[dict], incheon_rows: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {row["BSTOPID"]: row for row in osm_rows}
    for row in incheon_rows:
        merged[row["BSTOPID"]] = row

    by_coord: dict[tuple[float, float], dict] = {}
    for row in merged.values():
        lat = round(float(row["LAT"]), 4)
        lon = round(float(row["LNG"]), 4)
        key = (lat, lon)
        existing = by_coord.get(key)
        if existing is None:
            by_coord[key] = row
        elif row.get("source") == "incheon_api":
            by_coord[key] = row
    return sorted(by_coord.values(), key=lambda s: s["BSTOPID"])


def build_uniform_demand(stops: list[dict]) -> list[dict]:
    if not stops:
        return []
    weight = 1.0 / len(stops)
    return [
        {
            "BSTOPID": s["BSTOPID"],
            "일평균승하차건수": f"{weight:.6f}",
            "source": "uniform",
        }
        for s in stops
    ]


def download_yeongjong_data(
    output_dir: Path | None = None,
    config: dict | None = None,
    *,
    use_osm: bool = True,
    use_incheon_api: bool = True,
) -> dict:
    """Download real stop data into data/raw/. Returns summary metadata."""
    config = config or load_config()
    output_dir = output_dir or (project_root() / config["data"]["raw_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    bbox = config.get("region", {}).get("bbox")
    if not bbox:
        raise ValueError("region.bbox is required in config")

    api_key = _load_api_key()
    sources: list[str] = []
    osm_rows: list[dict] = []
    incheon_rows: list[dict] = []

    if use_osm:
        osm_rows = download_osm_stops(bbox)
        sources.append("osm")

    if use_incheon_api and api_key:
        incheon_rows = download_incheon_api_stops(api_key, bbox)
        sources.append("incheon_api")
    elif use_incheon_api:
        logger.info("DATA_GO_KR_KEY not set — skipping Incheon BIS API (OSM only)")

    if incheon_rows and osm_rows:
        stops = _merge_stops(osm_rows, incheon_rows)
    elif incheon_rows:
        stops = incheon_rows
    elif osm_rows:
        stops = osm_rows
    else:
        raise RuntimeError("No stops downloaded. Check network or API key.")

    stops_path = output_dir / "stops.csv"
    demand_path = output_dir / "demand.csv"
    _write_stops_csv(stops_path, stops)
    _write_demand_csv(demand_path, build_uniform_demand(stops))

    meta = {
        "sources": sources,
        "stop_count": len(stops),
        "bbox": bbox,
        "stops_path": str(stops_path),
        "demand_path": str(demand_path),
        "demand_note": (
            "Uniform demand weights (set DATA_GO_KR_KEY and replace demand.csv "
            "from data.go.kr 정류장별 이용승객 for passenger counts)"
        ),
    }
    meta_path = output_dir / "download_meta.json"
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
