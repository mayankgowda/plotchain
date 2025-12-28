# common.py
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ItemMeta:
    difficulty: str  # clean|moderate|edge
    edge_tag: str    # "" if none
    seed: int


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def stable_int_seed(master_seed: int, family: str, idx: int) -> int:
    """
    Deterministic seed that does NOT depend on Python's hash randomization.
    """
    s = f"{master_seed}::{family}::{idx}".encode("utf-8")
    h = hashlib.sha256(s).hexdigest()[:16]
    return int(h, 16) % (2**32 - 1)


def write_jsonl(path: Path, items: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def float_close(a: Optional[float], b: Optional[float], abs_tol: float = 1e-9, rel_tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    da = abs(a - b)
    if da <= abs_tol:
        return True
    denom = max(abs(b), 1e-12)
    return (da / denom) <= rel_tol


def make_difficulty_plan(
    n: int,
    clean_frac: float = 0.60,
    moderate_frac: float = 0.25,
    edge_frac: float = 0.15,
) -> List[str]:
    """
    Returns a deterministic list of difficulties of length n.
    """
    n_clean = int(round(n * clean_frac))
    n_mod = int(round(n * moderate_frac))
    n_edge = n - n_clean - n_mod
    return (["clean"] * n_clean) + (["moderate"] * n_mod) + (["edge"] * n_edge)


def axis_ticks_linear(vmin: float, vmax: float, n_ticks: int) -> float:
    span = max(vmax - vmin, 1e-12)
    raw = span / max(n_ticks - 1, 1)
    k = 10 ** int(np.floor(np.log10(raw)))
    candidates = np.array([1, 2, 5, 10]) * k
    return float(candidates[np.argmin(np.abs(candidates - raw))])


def save_figure(fig, out_path: Path, dpi: int = 160) -> None:
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
