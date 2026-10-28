"""Configuration: YAML file + deep-merged overrides. Plain dicts keep it easy to edit."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "config" / "default.yaml"
HORIZON_KEYS = ("short", "mid", "long")
STAGES = ("A1", "A2", "A3", "B1", "B2", "B3")
STAGE_CODES = {s: i + 1 for i, s in enumerate(STAGES)}  # 0 = UNKNOWN
STAGE_NAMES = {0: "UNKNOWN", **{i + 1: s for i, s in enumerate(STAGES)}}
STAGE_ZH = {
    "A1": "A1 修正打底（買進）",
    "A2": "A2 相隨上漲（續抱）",
    "A3": "A3 過熱（賣出）",
    "B1": "B1 修正下跌（賣出）",
    "B2": "B2 相隨下跌（持現金等待）",
    "B3": "B3 恐慌過度賣出（逆向買進）",
    "UNKNOWN": "階段不明",
}


def deep_merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict:
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if path and Path(path).resolve() != DEFAULT_PATH.resolve():
        with open(path, encoding="utf-8") as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})
    cfg = deep_merge(cfg, overrides)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    r = cfg["research"]
    if r["signal_step"] < 1:
        raise ValueError("signal_step must be >= 1")
    for key in HORIZON_KEYS:
        h = cfg["horizons"][key]
        if h["days"] % r["signal_step"]:
            raise ValueError(f"horizon {key} days must be a multiple of signal_step")
        fp = h["family_prior"]
        if abs(sum(fp.values()) - 1) > 1e-6 or set(fp) != {"T", "C", "F", "K"}:
            raise ValueError(f"family_prior for {key} must have T/C/F/K summing to 1")
    for name, prof in cfg["kostolany"]["exposure"].items():
        if set(prof) != set(STAGES) | {"UNKNOWN"}:
            raise ValueError(f"exposure profile {name} must define A1..B3 and UNKNOWN")
        if any(not 0 <= v <= 1 for v in prof.values()):
            raise ValueError("exposure must be within [0, 1]")
    for lam in r["lambdas"]:
        if not 0 <= lam <= 1:
            raise ValueError("lambda must be within [0,1]")
    if r["top_k"] < 1:
        raise ValueError("top_k must be >= 1")


def dump_yaml(cfg: dict) -> str:
    return yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False)
