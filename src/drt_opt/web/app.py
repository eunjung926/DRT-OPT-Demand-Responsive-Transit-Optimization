from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from drt_opt.config import load_config
from drt_opt.data.loader import load_processed_network
from drt_opt.web.service import run_comparison

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="DRT-OPT Simulator", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/network")
async def get_network():
    config = load_config()
    stops, _, demand_map = load_processed_network(config)
    bbox = config.get("region", {}).get("bbox")
    return {
        "region": config.get("region", {}).get("name", "yeongjong"),
        "bbox": bbox,
        "stops": [
            {
                "id": s.id,
                "name": s.name,
                "lat": s.lat,
                "lon": s.lon,
                "demand": demand_map.get(s.id, s.demand_weight),
            }
            for s in stops
        ],
    }


@app.get("/api/compare")
async def compare(
    seed: int = Query(42, ge=0),
    use_gpu: bool = Query(False),
):
    config = load_config()
    web_cfg = config.get("web", {})
    max_frames = web_cfg.get("max_frames", 600)
    return run_comparison(seed=seed, config=config, use_gpu=use_gpu, max_frames=max_frames)
