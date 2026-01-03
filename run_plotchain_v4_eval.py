#!/usr/bin/env python3
"""
run_plotchain_v4_eval.py

PlotChain v4 multimodal eval runner with "process" scoring via checkpoint fields (cp_*)
and constraint-consistency checks (novel chain-of-reasoning evaluation without hidden CoT).

Modes:
  --mode eval  : call LLM, write raw jsonl, then score+write reports
  --mode score : read existing raw jsonl, score+write reports (no API calls)

Writes to out_dir:
  raw_<provider>_<model>.jsonl
  per_item.csv
  summary.csv
  overall.csv
  run.log

Usage (OpenAI example):
  python3 run_plotchain_v4_eval.py --mode eval --provider openai --model gpt-4.1 \
    --data_dir data/plotchain_v4 --jsonl data/plotchain_v4/plotchain_v4.jsonl \
    --out_dir results/openai_gpt-4.1_plotread --tol_policy plotread
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# ---------------------------
# Logging
# ---------------------------
def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("plotchain_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    fh = logging.FileHandler(out_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------------------
# Dataset helpers
# ---------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_image_path(it: Dict[str, Any], data_dir: Optional[Path], images_root: Optional[Path]) -> Path:
    # dataset uses "image_path" relative like "images/<type>/<id>.png"
    p = Path(it.get("image_path", it.get("image", "")))
    if not str(p):
        raise KeyError("Missing image_path in item")
    if p.is_absolute():
        return p
    if data_dir is not None:
        return (data_dir / p).resolve()
    if images_root is not None:
        return (images_root / p).resolve()
    # fallback: relative to cwd
    return p.resolve()


def get_type(it: Dict[str, Any]) -> str:
    return str(it.get("type", ""))


def get_id(it: Dict[str, Any]) -> str:
    return str(it.get("id", ""))


def get_question(it: Dict[str, Any]) -> str:
    for k in ("question", "prompt", "query", "text_question"):
        if k in it:
            return str(it[k])
    raise KeyError("Missing question")


def get_ground_truth(it: Dict[str, Any]) -> Dict[str, Any]:
    gt = it.get("ground_truth")
    if not isinstance(gt, dict):
        raise KeyError("Missing ground_truth")
    return gt


def expected_fields(it: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    gen = it.get("generation", {}) or {}
    finals = list(gen.get("final_fields", []))
    cps = list(gen.get("checkpoint_fields", []))

    # fallback if generator metadata missing
    if not finals:
        gt = get_ground_truth(it)
        finals = [k for k in gt.keys() if not str(k).startswith("cp_")]
    if not cps:
        gt = get_ground_truth(it)
        cps = [k for k in gt.keys() if str(k).startswith("cp_")]

    # Only include keys that exist in GT (tight schema)
    gt = get_ground_truth(it)
    finals = [k for k in finals if k in gt]
    cps = [k for k in cps if k in gt]
    return finals, cps


# ---------------------------
# Prompt + JSON extraction
# ---------------------------
UNITS_HINT = {
    # finals
    "percent_overshoot": "%",
    "settling_time_s": "s",
    "steady_state": "unitless",
    "dc_gain_db": "dB",
    "cutoff_hz": "Hz",
    "phase_deg_at_10fc": "deg",
    "resonance_hz": "Hz",
    "bandwidth_hz": "Hz",
    "frequency_hz": "Hz",
    "vpp_v": "V",
    "dominant_frequency_hz": "Hz",
    "secondary_frequency_hz": "Hz",
    "f1_hz": "Hz",
    "f2_hz": "Hz",
    "resistance_ohm": "Ohm",
    "turn_on_voltage_v_at_target_i": "V",
    "small_signal_gain": "unitless",
    "saturation_v": "V",
    "zeta": "unitless",
    "wn_rad_s": "rad/s",
    # checkpoints (common)
    "cp_peak_time_s": "s",
    "cp_peak_value": "unitless",
    "cp_band_lower": "unitless",
    "cp_band_upper": "unitless",
    "cp_mag_at_fc_db": "dB",
    "cp_slope_db_per_decade": "dB/dec",
    "cp_phase_deg_at_fc": "deg",
    "cp_f1_3db_hz": "Hz",
    "cp_f2_3db_hz": "Hz",
    "cp_q_factor": "unitless",
    "cp_period_s": "s",
    "cp_vmax_v": "V",
    "cp_vmin_v": "V",
    "cp_duty": "ratio",
    "cp_peak_ratio": "unitless",
    "switch_time_s": "s",
    "cp_duration_s": "s",
    "cp_slope_ohm": "Ohm",
    "target_current_a": "A",
    "cp_vin_at_saturation": "V",
    "cp_pole_real": "rad/s",
    "cp_pole_imag": "rad/s",
}

def build_schema(fields: List[str]) -> str:
    lines = []
    for f in fields:
        lines.append(f'  "{f}": <number or null>  // {UNITS_HINT.get(f,"")}')
    return "{\n" + ",\n".join(lines) + "\n}"


def build_prompt(it: Dict[str, Any]) -> str:
    finals, cps = expected_fields(it)
    fields = finals + cps
    schema = build_schema(fields)

    # We do NOT request hidden chain-of-thought.
    # We request numeric checkpoints (cp_*) as the "reasoning trace".
    return f"""
You are given an engineering plot image. Read the plot and answer the question.

Question:
{get_question(it)}

Return ONLY a single JSON object matching this schema (numbers or null; no strings; no units; no extra keys):
{schema}

Notes:
- Use cp_* fields as intermediate plot reads (checkpoints).
- If you cannot determine a value, output null for that key.
- All values must be JSON numbers (e.g., 1.6667). Do NOT write fractions like 1025/615 or any expressions.
- Do not include any text in your response; only return the JSON object.
""".strip()


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
# Matches numeric divisions like: 1025/615 or 1025 / 615 or -3.2/0.8
_FRACTION_RE = re.compile(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*/\s*(-?\d+(?:\.\d+)?)(?![\w.])")

def _sanitize_json_blob(blob: str) -> str:
    s = blob.strip()

    # 1) Remove trailing commas before } or ]
    s = re.sub(r",\s*([}\]])", r"\1", s)

    # 2) Replace simple numeric fractions with decimal numbers
    #    Do multiple passes in case there are several fractions.
    for _ in range(10):
        m = _FRACTION_RE.search(s)
        if not m:
            break
        a = float(m.group(1))
        b = float(m.group(2))
        # Avoid divide-by-zero; if it happens, leave as-is (parsing will fail, which is fine).
        if abs(b) < 1e-18:
            break
        val = a / b
        # Use a stable decimal representation
        rep = f"{val:.12g}"
        s = s[:m.start()] + rep + s[m.end():]

    return s

def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # Quick path: exact JSON
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = _JSON_CANDIDATE_RE.search(text)
    if not m:
        return None

    blob = m.group(0)
    blob = _sanitize_json_blob(blob)

    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except Exception:
            return None
    return None


# ---------------------------
# Tolerance policy (defensible)
# ---------------------------
def tolerances_plotread() -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    plotread: fair for eyeballing axis/ticks.
    Return map: (type, field) -> (abs_tol, rel_tol)
    """
    T: Dict[Tuple[str, str], Tuple[float, float]] = {}

    # step response
    T[("step_response", "percent_overshoot")] = (3.0, 0.08)
    T[("step_response", "settling_time_s")] = (0.40, 0.20)
    T[("step_response", "steady_state")] = (0.05, 0.05)
    T[("step_response", "cp_peak_time_s")] = (0.20, 0.15)
    T[("step_response", "cp_peak_value")] = (0.08, 0.08)
    T[("step_response", "cp_band_lower")] = (0.05, 0.05)
    T[("step_response", "cp_band_upper")] = (0.05, 0.05)

    # bode magnitude
    T[("bode_magnitude", "dc_gain_db")] = (0.75, 0.05)
    T[("bode_magnitude", "cutoff_hz")] = (6.0, 0.06)
    T[("bode_magnitude", "cp_mag_at_fc_db")] = (0.90, 0.06)
    T[("bode_magnitude", "cp_slope_db_per_decade")] = (5.0, 0.30)

    # bode phase
    T[("bode_phase", "cutoff_hz")] = (6.0, 0.06)
    T[("bode_phase", "phase_deg_at_10fc")] = (4.0, 0.08)
    T[("bode_phase", "cp_phase_deg_at_fc")] = (4.0, 0.10)

    # bandpass
    T[("bandpass_response", "resonance_hz")] = (8.0, 0.06)
    T[("bandpass_response", "bandwidth_hz")] = (10.0, 0.10)
    T[("bandpass_response", "cp_f1_3db_hz")] = (8.0, 0.08)
    T[("bandpass_response", "cp_f2_3db_hz")] = (8.0, 0.08)
    T[("bandpass_response", "cp_q_factor")] = (0.6, 0.20)

    # time waveform
    T[("time_waveform", "frequency_hz")] = (2.0, 0.03)
    T[("time_waveform", "vpp_v")] = (0.25, 0.10)
    T[("time_waveform", "cp_period_s")] = (0.02, 0.10)
    T[("time_waveform", "cp_vmax_v")] = (0.15, 0.10)
    T[("time_waveform", "cp_vmin_v")] = (0.15, 0.10)
    T[("time_waveform", "cp_duty")] = (0.06, 0.15)

    # fft
    T[("fft_spectrum", "dominant_frequency_hz")] = (6.0, 0.05)
    T[("fft_spectrum", "secondary_frequency_hz")] = (6.0, 0.05)
    T[("fft_spectrum", "cp_peak_ratio")] = (0.6, 0.25)

    # spectrogram
    T[("spectrogram", "f1_hz")] = (15.0, 0.06)
    T[("spectrogram", "f2_hz")] = (15.0, 0.06)
    T[("spectrogram", "switch_time_s")] = (0.08, 0.20)
    T[("spectrogram", "cp_duration_s")] = (0.05, 0.08)

    # iv_curve
    T[("iv_curve", "resistance_ohm")] = (25.0, 0.08)
    T[("iv_curve", "cp_slope_ohm")] = (25.0, 0.10)
    T[("iv_curve", "turn_on_voltage_v_at_target_i")] = (0.08, 0.15)
    T[("iv_curve", "target_current_a")] = (0.004, 0.25)

    # transfer
    T[("transfer_characteristic", "small_signal_gain")] = (0.25, 0.10)
    T[("transfer_characteristic", "saturation_v")] = (0.12, 0.10)
    T[("transfer_characteristic", "cp_vin_at_saturation")] = (0.12, 0.12)

    # pole-zero
    T[("pole_zero", "zeta")] = (0.08, 0.15)
    T[("pole_zero", "wn_rad_s")] = (1.0, 0.12)
    T[("pole_zero", "cp_pole_real")] = (1.0, 0.15)
    T[("pole_zero", "cp_pole_imag")] = (1.0, 0.15)

    return T


def tolerances_strict() -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    strict: tighter (useful as a second table in paper).
    """
    T = tolerances_plotread()
    # tighten by ~0.6x
    tightened = {}
    for k, (a, r) in T.items():
        tightened[k] = (0.6 * a, 0.6 * r)
    return tightened


def get_tolerances(policy: str) -> Dict[Tuple[str, str], Tuple[float, float]]:
    if policy == "plotread":
        return tolerances_plotread()
    if policy == "strict":
        return tolerances_strict()
    raise ValueError(policy)


# ---------------------------
# Consistency constraints (process scoring)
# ---------------------------
@dataclass
class ConsistencyResult:
    n_checks: int
    n_pass: int
    tags: str


def _close(a: Optional[float], b: Optional[float], abs_tol: float, rel_tol: float) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1e-12))


def consistency_checks(typ: str, pred: Dict[str, Any], abs_scale: float, rel_scale: float) -> ConsistencyResult:
    """
    Check internal coherence between predicted final + cp fields.
    These don't require ground truth; they evaluate reasoning trace consistency.
    """
    p = {k: _to_float(v) for k, v in (pred or {}).items()}
    checks: List[Tuple[str, bool]] = []

    def add(name: str, ok: bool):
        checks.append((name, ok))

    # --- step_response ---
    if typ == "step_response":
        steady = p.get("steady_state")
        peak = p.get("cp_peak_value")
        os = p.get("percent_overshoot")
        if steady is not None and peak is not None and os is not None and steady != 0:
            os_from = (peak - steady) / steady * 100.0
            add("os_from_peak", _close(os, os_from, 3.0*abs_scale, 0.10*rel_scale))
        lo = p.get("cp_band_lower")
        hi = p.get("cp_band_upper")
        if steady is not None and lo is not None:
            add("band_lower", _close(lo, 0.98*steady, 0.06*abs_scale, 0.08*rel_scale))
        if steady is not None and hi is not None:
            add("band_upper", _close(hi, 1.02*steady, 0.06*abs_scale, 0.08*rel_scale))

    # --- bode magnitude ---
    if typ == "bode_magnitude":
        dc = p.get("dc_gain_db")
        magfc = p.get("cp_mag_at_fc_db")
        if dc is not None and magfc is not None:
            add("mag_fc_is_dc_minus_3db", _close((dc - magfc), 3.0103, 0.6*abs_scale, 0.12*rel_scale))
        slope = p.get("cp_slope_db_per_decade")
        if slope is not None:
            add("slope_about_-20", _close(slope, -20.0, 6.0*abs_scale, 0.30*rel_scale))

    # --- bode phase ---
    if typ == "bode_phase":
        ph_fc = p.get("cp_phase_deg_at_fc")
        if ph_fc is not None:
            add("phase_fc_about_-45", _close(ph_fc, -45.0, 5.0*abs_scale, 0.15*rel_scale))

    # --- bandpass ---
    if typ == "bandpass_response":
        f1 = p.get("cp_f1_3db_hz")
        f2 = p.get("cp_f2_3db_hz")
        res = p.get("resonance_hz")
        bw = p.get("bandwidth_hz")
        q = p.get("cp_q_factor")
        if f1 is not None and f2 is not None and res is not None:
            add("res_sqrt_f1f2", _close(res, math.sqrt(max(f1*f2, 0.0)), 10.0*abs_scale, 0.10*rel_scale))
        if f1 is not None and f2 is not None and bw is not None:
            add("bw_f2_minus_f1", _close(bw, (f2 - f1), 12.0*abs_scale, 0.15*rel_scale))
        if res is not None and bw is not None and q is not None and bw != 0:
            add("q_res_over_bw", _close(q, (res / bw), 0.9*abs_scale, 0.25*rel_scale))

    # --- time waveform ---
    if typ == "time_waveform":
        f = p.get("frequency_hz")
        per = p.get("cp_period_s")
        vmax = p.get("cp_vmax_v")
        vmin = p.get("cp_vmin_v")
        vpp = p.get("vpp_v")
        if f is not None and per is not None and f != 0:
            add("period_inv_freq", _close(per, 1.0/f, 0.03*abs_scale, 0.15*rel_scale))
        if vmax is not None and vmin is not None and vpp is not None:
            add("vpp_vmax_minus_vmin", _close(vpp, vmax - vmin, 0.25*abs_scale, 0.15*rel_scale))
        duty = p.get("cp_duty")
        if duty is not None:
            add("duty_in_0_1", (0.0 <= duty <= 1.0))

    # --- fft ---
    if typ == "fft_spectrum":
        dom = p.get("dominant_frequency_hz")
        sec = p.get("secondary_frequency_hz")
        if dom is not None and sec is not None:
            add("distinct_peaks", abs(dom - sec) > 1e-9)

    # --- spectrogram ---
    if typ == "spectrogram":
        t0 = p.get("switch_time_s")
        dur = p.get("cp_duration_s")
        if dur is not None and t0 is not None:
            add("switch_within_duration", 0.0 <= t0 <= dur + 1e-9)

    # --- iv ---
    if typ == "iv_curve":
        R = p.get("resistance_ohm")
        slope = p.get("cp_slope_ohm")
        if R is not None and slope is not None:
            add("slope_matches_R", _close(R, slope, 40.0*abs_scale, 0.15*rel_scale))

    # --- transfer ---
    if typ == "transfer_characteristic":
        g = p.get("small_signal_gain")
        vs = p.get("saturation_v")
        vin_sat = p.get("cp_vin_at_saturation")
        if g is not None and vs is not None and vin_sat is not None:
            add("vsat_equals_gain_times_vin_sat", _close(vs, g*vin_sat, 0.20*abs_scale, 0.15*rel_scale))

    # --- pole_zero ---
    if typ == "pole_zero":
        zeta = p.get("zeta")
        wn = p.get("wn_rad_s")
        pr = p.get("cp_pole_real")
        pi = p.get("cp_pole_imag")
        if zeta is not None and wn is not None and pr is not None:
            add("pole_real", _close(pr, -zeta*wn, 1.2*abs_scale, 0.20*rel_scale))
        if zeta is not None and wn is not None and pi is not None:
            add("pole_imag", _close(pi, wn*math.sqrt(max(1.0 - zeta*zeta, 0.0)), 1.2*abs_scale, 0.20*rel_scale))

    n = len(checks)
    n_pass = sum(1 for _, ok in checks if ok)
    tags = ",".join([name for name, ok in checks if not ok])
    return ConsistencyResult(n_checks=n, n_pass=n_pass, tags=tags)


# ---------------------------
# Model adapters
# ---------------------------
def guess_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def openai_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI()

    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)
    data_url = f"data:{mime};base64,{b64}"

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": data_url},
                {"type": "input_text", "text": prompt},
            ],
        }],
        max_output_tokens=max_output_tokens,
    )
    return getattr(resp, "output_text", "") or ""


def call_model(provider: str, model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    if provider == "openai":
        return openai_call(model, image_path, prompt, max_output_tokens=max_output_tokens)

    # kept as explicit errors so the script is "complete" but you can add adapters later
    if provider in ("gemini", "anthropic"):
        raise RuntimeError(f"Provider '{provider}' not enabled in this file yet. Start with openai; add adapter if needed.")
    raise ValueError(f"Unknown provider: {provider}")


# ---------------------------
# Scoring vs ground truth
# ---------------------------
def score_one_field(typ: str, field: str, pred_val: Optional[float], gold_val: Optional[float],
                    tols: Dict[Tuple[str, str], Tuple[float, float]], abs_scale: float, rel_scale: float) -> Tuple[bool, Optional[float], Optional[float], float, float]:
    ae = None if (pred_val is None or gold_val is None) else float(abs(pred_val - gold_val))
    re = None if (pred_val is None or gold_val is None) else float(abs(pred_val - gold_val) / max(abs(gold_val), 1e-12))

    abs_tol, rel_tol = tols.get((typ, field), (0.0, 0.0))
    abs_tol *= abs_scale
    rel_tol *= rel_scale

    passed = False
    if ae is not None and re is not None:
        passed = (ae <= abs_tol) or (re <= rel_tol)
    return bool(passed), ae, re, abs_tol, rel_tol


# ---------------------------
# Raw caching + resume
# ---------------------------
def already_done_ids(raw_path: Path) -> set[str]:
    done = set()
    if not raw_path.exists():
        return done
    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                iid = str(obj.get("id", ""))
                if iid:
                    done.add(iid)
            except Exception:
                continue
    return done


# ---------------------------
# Report writing
# ---------------------------
def write_reports(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "per_item.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    # tag checkpoint fields
    df2 = df.copy()
    if "is_checkpoint" not in df2.columns:
        df2["is_checkpoint"] = df2["field"].astype(str).str.startswith("cp_") | df2["field"].astype(str).isin(["target_current_a", "switch_time_s"])
    df2["scope"] = np.where(df2["is_checkpoint"], "checkpoint", "final")

    summary = (
        df2.groupby(["provider", "model", "type", "scope", "field"])
           .agg(
               n=("pass", "size"),
               pass_rate=("pass", "mean"),
               mean_abs_err=("abs_err", "mean"),
               median_abs_err=("abs_err", "median"),
               mean_rel_err=("rel_err", "mean"),
               completeness=("is_null", lambda s: float(1.0 - np.mean(s))),
               mean_latency_s=("latency_s", "mean"),
           )
           .reset_index()
           .sort_values(["provider", "model", "type", "scope", "field"])
    )
    (out_dir / "summary.csv").write_text(summary.to_csv(index=False), encoding="utf-8")

    # overall with final vs cp + process metrics
    overall = (
        df2.groupby(["provider", "model"])
           .agg(
               n=("pass", "size"),
               overall_pass_rate=("pass", "mean"),
               completeness_rate=("is_null", lambda s: float(1.0 - np.mean(s))),
               mean_latency_s=("latency_s", "mean"),
           )
           .reset_index()
    )

    final_overall = (
        df2[df2["scope"] == "final"]
        .groupby(["provider", "model"])
        .agg(final_n=("pass", "size"), final_pass_rate=("pass", "mean"))
        .reset_index()
    )

    cp_overall = (
        df2[df2["scope"] == "checkpoint"]
        .groupby(["provider", "model"])
        .agg(checkpoint_n=("pass", "size"), checkpoint_pass_rate=("pass", "mean"))
        .reset_index()
    )

    # consistency is stored at per-item level as separate rows with field="__consistency__"
    # We'll compute it directly from df2 for simplicity:
    cons = (
        df2[df2["field"] == "__consistency__"]
        .groupby(["provider", "model"])
        .agg(consistency_n=("pass", "size"), consistency_pass_rate=("pass", "mean"))
        .reset_index()
    )

    merged = overall.merge(final_overall, on=["provider", "model"], how="left") \
                    .merge(cp_overall, on=["provider", "model"], how="left") \
                    .merge(cons, on=["provider", "model"], how="left")

    # process fidelity score
    merged["process_fidelity_score"] = 0.5 * merged["checkpoint_pass_rate"].fillna(0.0) + 0.5 * merged["consistency_pass_rate"].fillna(0.0)

    merged = merged.sort_values(["final_pass_rate", "process_fidelity_score", "overall_pass_rate"], ascending=[False, False, False])
    (out_dir / "overall.csv").write_text(merged.to_csv(index=False), encoding="utf-8")


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["eval", "score"], required=True)
    ap.add_argument("--provider", type=str, default="openai", help="openai (supported here)")
    ap.add_argument("--model", type=str, default="", help="Model name (e.g., gpt-4.1)")
    ap.add_argument("--jsonl", type=str, required=True, help="Path to plotchain_v4.jsonl")
    ap.add_argument("--data_dir", type=str, default="", help="Dataset dir containing images/ (recommended)")
    ap.add_argument("--images_root", type=str, default="", help="Optional root dir if not using data_dir")
    ap.add_argument("--out_dir", type=str, required=True, help="Output directory")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_output_tokens", type=int, default=350, help="Model output budget")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--tol_policy", choices=["plotread", "strict"], default="plotread")
    ap.add_argument("--abs_tol_scale", type=float, default=1.0)
    ap.add_argument("--rel_tol_scale", type=float, default=1.0)
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Sleep between API calls")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    logger = setup_logger(out_dir)
    logger.info(f"mode={args.mode} provider={args.provider} model={args.model} tol={args.tol_policy}")

    jsonl_path = Path(args.jsonl)
    data_dir = Path(args.data_dir).resolve() if args.data_dir else None
    images_root = Path(args.images_root).resolve() if args.images_root else None

    items = load_jsonl(jsonl_path)
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    provider = args.provider.strip().lower()

    # ----- EVAL MODE -----
    raw_path = out_dir / f"raw_{provider}_{(args.model or 'MODEL').replace('/','_')}.jsonl"

    if args.mode == "eval":
        if not args.model:
            raise SystemExit("--model is required in eval mode")

        if raw_path.exists() and not args.overwrite:
            done = already_done_ids(raw_path)
            logger.info(f"[eval] raw exists: {raw_path} (resume enabled). done={len(done)}")
        else:
            done = set()
            if args.overwrite and raw_path.exists():
                raw_path.unlink()
            logger.info(f"[eval] writing raw: {raw_path}")

        iterator = items
        if tqdm is not None:
            iterator = tqdm(items, desc=f"{provider}:{args.model}")

        with raw_path.open("a", encoding="utf-8") as f:
            for it in iterator:
                iid = get_id(it)
                if iid in done:
                    continue

                typ = get_type(it)
                img_path = get_image_path(it, data_dir=data_dir, images_root=images_root)
                prompt = build_prompt(it)

                t0 = time.time()
                err = None
                txt = ""
                parsed = None

                # basic retry for transient errors
                for attempt in range(3):
                    try:
                        txt = call_model(provider, args.model, img_path, prompt, max_output_tokens=args.max_output_tokens)
                        parsed = extract_first_json(txt)
                        break
                    except Exception as e:
                        err = f"{type(e).__name__}: {e}"
                        # backoff
                        time.sleep(1.0 * (attempt + 1))
                dt = time.time() - t0

                f.write(json.dumps({
                    "provider": provider,
                    "model": args.model,
                    "id": iid,
                    "type": typ,
                    "image_path": str(img_path),
                    "prompt": prompt,
                    "raw_text": txt,
                    "parsed_json": parsed,
                    "latency_s": dt,
                    "error": err,
                }, ensure_ascii=False) + "\n")
                f.flush()

                if args.sleep_s > 0:
                    time.sleep(args.sleep_s)

        logger.info("[eval] complete. Now scoring...")

    # ----- SCORE MODE (or after eval) -----
    # If user used eval, raw_path exists; if score-only, we still use the standard raw file name pattern.
    if not raw_path.exists():
        # allow score mode to work even if model name differs; user can pass --model or use a glob later
        raise SystemExit(f"Raw file not found: {raw_path}")

    tols = get_tolerances(args.tol_policy)
    abs_scale = float(args.abs_tol_scale)
    rel_scale = float(args.rel_tol_scale)

    # build lookup from item id -> item (for GT/fields)
    item_by_id = {get_id(it): it for it in items}

    rows: List[Dict[str, Any]] = []

    with raw_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            iid = str(r.get("id", ""))
            it = item_by_id.get(iid)
            if it is None:
                continue

            typ = get_type(it)
            gt = get_ground_truth(it)
            finals, cps = expected_fields(it)
            fields = finals + cps

            parsed = r.get("parsed_json")
            parsed = parsed if isinstance(parsed, dict) else None

            # field scoring
            for field in fields:
                gold = _to_float(gt.get(field))
                pred = _to_float(parsed.get(field)) if parsed else None
                is_null = pred is None

                passed, ae, re_, abs_tol, rel_tol = score_one_field(
                    typ, field, pred, gold, tols, abs_scale, rel_scale
                )

                rows.append({
                    "provider": str(r.get("provider", "")),
                    "model": str(r.get("model", "")),
                    "id": iid,
                    "type": typ,
                    "field": field,
                    "pred": pred,
                    "gold": gold,
                    "abs_err": ae,
                    "rel_err": re_,
                    "abs_tol": abs_tol,
                    "rel_tol": rel_tol,
                    "pass": bool(passed),
                    "is_null": bool(is_null),
                    "latency_s": float(r.get("latency_s", 0.0) or 0.0),
                    "error": r.get("error", None),
                })

            # consistency scoring (one synthetic row per item)
            cons = consistency_checks(typ, parsed or {}, abs_scale=abs_scale, rel_scale=rel_scale)
            cons_pass = (cons.n_checks > 0 and cons.n_pass == cons.n_checks)

            rows.append({
                "provider": str(r.get("provider", "")),
                "model": str(r.get("model", "")),
                "id": iid,
                "type": typ,
                "field": "__consistency__",
                "pred": float(cons.n_pass),
                "gold": float(cons.n_checks),
                "abs_err": float(cons.n_checks - cons.n_pass),
                "rel_err": None,
                "abs_tol": 0.0,
                "rel_tol": 0.0,
                "pass": bool(cons_pass),
                "is_null": False,
                "latency_s": float(r.get("latency_s", 0.0) or 0.0),
                "error": r.get("error", None),
                "tags": cons.tags,
            })

    df = pd.DataFrame(rows)
    write_reports(df, out_dir)
    logger.info(f"[score] wrote reports to: {out_dir}")


if __name__ == "__main__":
    main()
