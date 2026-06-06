from pathlib import Path

import yaml


def load_config(path: str | Path | None = None) -> dict:
    if path is None:
        path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]
