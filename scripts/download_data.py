#!/usr/bin/env python3
"""Download real Yeongjong bus stop data."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from drt_opt.config import load_config, project_root
from drt_opt.data.download import download_yeongjong_data
from drt_opt.data.loader import preprocess_network

logging.basicConfig(level=logging.INFO)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download real Yeongjong bus stop data")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=project_root() / "data" / "raw")
    parser.add_argument("--no-osm", action="store_true")
    parser.add_argument("--no-api", action="store_true")
    parser.add_argument("--preprocess", action="store_true", default=True)
    parser.add_argument("--no-preprocess", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    meta = download_yeongjong_data(
        output_dir=args.output,
        config=config,
        use_osm=not args.no_osm,
        use_incheon_api=not args.no_api,
    )
    print(f"Downloaded {meta['stop_count']} stops -> {meta['stops_path']}")
    print(f"Sources: {', '.join(meta['sources'])}")

    if args.preprocess and not args.no_preprocess:
        stops, _, _ = preprocess_network(config, use_sample=False)
        print(f"Preprocessed {len(stops)} stops -> {config['data']['processed_dir']}")


if __name__ == "__main__":
    main()
