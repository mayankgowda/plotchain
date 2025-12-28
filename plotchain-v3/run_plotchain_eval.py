#!/usr/bin/env python3
"""run_plotchain_eval.py

Run multimodal LLMs on PlotChain v3 (image + question) and score vs ground truth.

Writes (in out_dir):
  - per_item.csv
  - summary.csv
  - overall.csv
  - raw_<provider>_<model>.jsonl  (full responses for audit)

Notes:
- Requests STRUCTURED "reasoning checkpoints" (cp_*) as additional numeric fields.
- Disable checkpoint prompting with --final_only.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_image_path(it: Dict[str, Any], images_root: Optional[Path]) -> Path:
    for k in ("image_path", "image_file", "plot_path", "image"):
        if k in it:
            p = Path(it[k])
            return (images_root / p).resolve() if images_root and not p.is_absolute() else p.resolve()
    raise KeyError("Could not find image path field (expected one of image_path/image_file/plot_path/image).")


def guess_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"


def build_schema_hint(keys: List[str]) -> str:
    lines = []
    for k in keys:
        lines.append(f'  "{k}": <number>  // number only')
    return "{\n" + ",\n".join(lines) + "\n}"


def build_prompt(it: Dict[str, Any], final_only: bool) -> str:
    gen = it.get("generation", {}) or {}
    finals = list(gen.get("final_fields", []))
    cps = list(gen.get("checkpoint_fields", []))
    keys = finals[:] if final_only else (finals + cps)

    schema = build_schema_hint(keys)
    q = it["question"]

    return f"""You are given an engineering plot image. Answer the question by reading the plot.

Question:
{q}

Return ONLY a single JSON object matching this schema (numbers only; no strings; no units):
{schema}

If a value cannot be determined, still output the key with value null.
""".strip()


_JSON_CANDIDATE_RE = re.compile(r"\{.*\}", re.DOTALL)


def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = _JSON_CANDIDATE_RE.search(text)
    if not m:
        return None
    blob = m.group(0)

    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    for end in range(len(blob), 1, -1):
        try:
            obj = json.loads(blob[:end])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
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


def abs_err(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(abs(a - b))


def rel_err(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(abs(a - b) / max(abs(b), 1e-12))


def tolerances_plotread() -> Dict[Tuple[str, str], Tuple[float, float]]:
    return {
        ("step_response", "percent_overshoot"): (2.5, 0.07),
        ("step_response", "settling_time_s"):   (0.35, 0.20),
        ("step_response", "steady_state"):      (0.05, 0.02),
        ("step_response", "cp_peak_value"):     (0.08, 0.04),
        ("step_response", "cp_peak_time_s"):    (0.20, 0.20),
        ("step_response", "cp_band_upper"):     (0.05, 0.02),
        ("step_response", "cp_band_lower"):     (0.05, 0.02),

        ("bode_magnitude", "dc_gain_db"):       (0.5, 0.03),
        ("bode_magnitude", "cutoff_hz"):        (5.0, 0.05),
        ("bode_magnitude", "cp_mag_at_fc_db"):  (0.6, 0.06),
        ("bode_magnitude", "cp_slope_db_per_decade"): (3.0, 0.20),

        ("bode_phase", "cutoff_hz"):            (5.0, 0.05),
        ("bode_phase", "cp_phase_deg_at_fc"):   (12.0, 0.10),
        ("bode_phase", "phase_deg_at_10fc"):    (12.0, 0.10),

        ("bandpass_response", "resonance_hz"):  (5.0, 0.05),
        ("bandpass_response", "bandwidth_hz"):  (10.0, 0.10),
        ("bandpass_response", "cp_f1_3db_hz"):  (8.0, 0.08),
        ("bandpass_response", "cp_f2_3db_hz"):  (8.0, 0.08),
        ("bandpass_response", "cp_q_factor"):   (0.5, 0.20),

        ("fft_spectrum", "dominant_frequency_hz"):   (2.0, 0.03),
        ("fft_spectrum", "secondary_frequency_hz"):  (2.0, 0.03),
        ("fft_spectrum", "cp_peak_ratio"):           (0.12, 0.15),

        ("spectrogram", "f1_hz"):               (5.0, 0.05),
        ("spectrogram", "f2_hz"):               (5.0, 0.05),
        ("spectrogram", "switch_time_s"):       (0.15, 0.15),
        ("spectrogram", "cp_duration_s"):       (0.15, 0.15),

        ("time_waveform", "frequency_hz"):      (1.0, 0.02),
        ("time_waveform", "vpp_v"):             (0.20, 0.10),
        ("time_waveform", "cp_duty"):           (0.04, 0.10),
        ("time_waveform", "cp_period_s"):       (0.04, 0.10),
        ("time_waveform", "cp_vmax_v"):         (0.12, 0.08),
        ("time_waveform", "cp_vmin_v"):         (0.12, 0.08),

        ("iv_curve", "resistance_ohm"):         (10.0, 0.05),
        ("iv_curve", "turn_on_voltage_v_at_target_i"): (0.08, 0.15),
        ("iv_curve", "target_current_a"):       (0.0, 0.0),
        ("iv_curve", "cp_slope_ohm"):           (12.0, 0.08),
        ("iv_curve", "cp_Is"):                  (0.0, 0.0),
        ("iv_curve", "cp_nVt"):                 (0.0, 0.0),
        ("iv_curve", "cp_Rs"):                  (0.0, 0.0),

        ("transfer_characteristic", "small_signal_gain"): (0.20, 0.10),
        ("transfer_characteristic", "saturation_v"):       (0.10, 0.10),
        ("transfer_characteristic", "cp_vin_at_saturation"): (0.12, 0.15),
    }


def tolerances_strict() -> Dict[Tuple[str, str], Tuple[float, float]]:
    # "Precision read" — tight tolerances intended for clear plots.
    # We recommend reporting BOTH strict + plotread in the paper.
    return {
        ("step_response", "percent_overshoot"): (2.0, 0.05),
        ("step_response", "settling_time_s"):   (0.20, 0.10),
        ("step_response", "steady_state"):      (0.03, 0.02),
        ("step_response", "cp_peak_value"):     (0.05, 0.03),
        ("step_response", "cp_peak_time_s"):    (0.12, 0.15),
        ("step_response", "cp_band_upper"):     (0.03, 0.02),
        ("step_response", "cp_band_lower"):     (0.03, 0.02),

        ("bode_magnitude", "dc_gain_db"):       (0.25, 0.02),
        ("bode_magnitude", "cutoff_hz"):        (2.0, 0.03),
        ("bode_magnitude", "cp_mag_at_fc_db"):  (0.30, 0.03),
        ("bode_magnitude", "cp_slope_db_per_decade"): (1.5, 0.10),

        ("bode_phase", "cutoff_hz"):            (2.0, 0.03),
        ("bode_phase", "cp_phase_deg_at_fc"):   (6.0, 0.05),
        ("bode_phase", "phase_deg_at_10fc"):    (6.0, 0.05),

        ("bandpass_response", "resonance_hz"):  (2.0, 0.02),
        ("bandpass_response", "bandwidth_hz"):  (4.0, 0.04),
        ("bandpass_response", "cp_f1_3db_hz"):  (3.0, 0.03),
        ("bandpass_response", "cp_f2_3db_hz"):  (3.0, 0.03),
        ("bandpass_response", "cp_q_factor"):   (0.25, 0.10),

        ("fft_spectrum", "dominant_frequency_hz"):   (1.0, 0.02),
        ("fft_spectrum", "secondary_frequency_hz"):  (1.0, 0.02),
        ("fft_spectrum", "cp_peak_ratio"):           (0.08, 0.10),

        ("spectrogram", "f1_hz"):               (2.0, 0.02),
        ("spectrogram", "f2_hz"):               (2.0, 0.02),
        ("spectrogram", "switch_time_s"):       (0.08, 0.08),
        ("spectrogram", "cp_duration_s"):       (0.08, 0.08),

        ("time_waveform", "frequency_hz"):      (0.5, 0.01),
        ("time_waveform", "vpp_v"):             (0.12, 0.07),
        ("time_waveform", "cp_duty"):           (0.02, 0.05),
        ("time_waveform", "cp_period_s"):       (0.02, 0.05),
        ("time_waveform", "cp_vmax_v"):         (0.08, 0.05),
        ("time_waveform", "cp_vmin_v"):         (0.08, 0.05),

        ("iv_curve", "resistance_ohm"):         (5.0, 0.03),
        ("iv_curve", "turn_on_voltage_v_at_target_i"): (0.05, 0.10),
        ("iv_curve", "target_current_a"):       (0.0, 0.0),  # deterministic target in GT
        ("iv_curve", "cp_slope_ohm"):           (6.0, 0.04),
        ("iv_curve", "cp_Is"):                  (0.0, 0.0),
        ("iv_curve", "cp_nVt"):                 (0.0, 0.0),
        ("iv_curve", "cp_Rs"):                  (0.0, 0.0),

        ("transfer_characteristic", "small_signal_gain"): (0.12, 0.06),
        ("transfer_characteristic", "saturation_v"):       (0.06, 0.06),
        ("transfer_characteristic", "cp_vin_at_saturation"): (0.08, 0.12),
    }


def get_tolerances(policy: str, abs_scale: float = 1.0, rel_scale: float = 1.0) -> Dict[Tuple[str, str], Tuple[float, float]]:
    """Return (abs_tol, rel_tol) per (type, field). Optionally scale all tolerances."""
    policy = (policy or "plotread").strip().lower()
    if policy == "strict":
        base = tolerances_strict()
    elif policy == "plotread":
        base = tolerances_plotread()
    else:
        raise ValueError(f"Unknown --tol_policy '{policy}'. Use 'plotread' or 'strict'.")

    abs_scale = float(abs_scale)
    rel_scale = float(rel_scale)
    if abs_scale <= 0 or rel_scale <= 0:
        raise ValueError("abs/rel tolerance scales must be > 0")

    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    for k, (a, r) in base.items():
        out[k] = (float(a) * abs_scale, float(r) * rel_scale)
    return out


def failure_tags(pred: Optional[float], gold: Optional[float]) -> List[str]:
    tags = []
    if pred is None:
        tags.append("missing_value")
        return tags
    if gold is None:
        tags.append("missing_gold")
        return tags
    if gold != 0:
        ratio = pred / gold
        if abs(ratio - 1000) < 50 or abs(ratio - 0.001) < 0.0001:
            tags.append("likely_unit_scale_error_x1000")
        if abs(ratio - 2*math.pi) < 0.25 or abs(ratio - 1/(2*math.pi)) < 0.05:
            tags.append("hz_vs_rad_s")
    return tags


@dataclass(frozen=True)
class ModelSpec:
    provider: str  # "openai" | "gemini" | "anthropic"
    model: str


def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def gemini_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    image_bytes = image_path.read_bytes()
    mime = guess_mime(image_path)

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
    )
    return getattr(resp, "text", "") or ""


def anthropic_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)

    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)

    msg = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    out = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "text":
            out.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "\n".join(out).strip()


def call_model(spec: ModelSpec, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    if spec.provider == "openai":
        return openai_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    if spec.provider == "gemini":
        return gemini_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    if spec.provider == "anthropic":
        return anthropic_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    raise ValueError(f"Unknown provider: {spec.provider}")


def parse_models(args_models: List[str]) -> List[ModelSpec]:
    out = []
    for s in args_models:
        if ":" not in s:
            raise ValueError(f"Model must be provider:model, got: {s}")
        p, m = s.split(":", 1)
        out.append(ModelSpec(provider=p.strip().lower(), model=m.strip()))
    return out


def _build_item_index(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    idx: Dict[str, Dict[str, Any]] = {}
    for it in items:
        iid = str(it.get("id", "") or it.get("uid", "") or "")
        if iid:
            idx[iid] = it
    return idx


def _keys_for_item(it: Dict[str, Any], final_only: bool) -> List[str]:
    gen = it.get("generation", {}) or {}
    finals = list(gen.get("final_fields", []))
    cps = list(gen.get("checkpoint_fields", []))
    keys = finals[:] if final_only else (finals + cps)

    gt = dict(it.get("ground_truth", {}) or {})
    return [k for k in keys if k in gt]


def _score_one_item(
    *,
    provider: str,
    model: str,
    iid: str,
    typ: str,
    keys: List[str],
    gt: Dict[str, Any],
    parsed: Optional[Dict[str, Any]],
    latency_s: Optional[float],
    error: Optional[str],
    tols: Dict[Tuple[str, str], Tuple[float, float]],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k in keys:
        gold = _to_float(gt.get(k))
        pred = _to_float(parsed.get(k)) if isinstance(parsed, dict) else None

        ae = abs_err(pred, gold)
        re_ = rel_err(pred, gold)
        abs_tol, rel_tol = tols.get((typ, k), (0.0, 0.0))

        passed = False
        if ae is not None and re_ is not None:
            passed = (ae <= abs_tol) or (re_ <= rel_tol)

        rows.append({
            "provider": provider,
            "model": model,
            "id": iid,
            "type": typ,
            "field": k,
            "pred": pred,
            "gold": gold,
            "abs_err": ae,
            "rel_err": re_,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
            "pass": bool(passed),
            "latency_s": float(latency_s) if latency_s is not None else None,
            "error": error,
            "tags": ",".join(failure_tags(pred, gold)),
        })
    return rows


def _run_and_write_raw(
    *,
    spec: ModelSpec,
    items: List[Dict[str, Any]],
    images_root: Optional[Path],
    raw_path: Path,
    max_output_tokens: int,
    final_only: bool,
    sleep_s: float = 0.0,
) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("w", encoding="utf-8") as raw_f:
        for it in tqdm(items, desc=f"{spec.provider}:{spec.model}"):
            iid = str(it.get("id", "") or "")
            typ = str(it.get("type", "") or "")
            img_path = get_image_path(it, images_root)
            prompt = build_prompt(it, final_only=final_only)

            t0 = time.time()
            try:
                txt = call_model(spec, img_path, prompt, max_output_tokens=max_output_tokens)
                err = None
            except Exception as e:
                txt = ""
                err = f"{type(e).__name__}: {e}"
            dt = time.time() - t0

            parsed = extract_first_json(txt) if txt else None

            raw_f.write(json.dumps({
                "provider": spec.provider,
                "model": spec.model,
                "id": iid,
                "type": typ,
                "image_path": str(img_path),
                "prompt": prompt,
                "raw_text": txt,
                "parsed_json": parsed,
                "latency_s": dt,
                "error": err,
            }, ensure_ascii=False) + "\n")

            if sleep_s and sleep_s > 0:
                time.sleep(float(sleep_s))


def _score_from_raw_files(
    *,
    raw_files: List[Path],
    item_by_id: Dict[str, Dict[str, Any]],
    allowed_ids: Optional[set],
    tols: Dict[Tuple[str, str], Tuple[float, float]],
    final_only: bool,
) -> List[Dict[str, Any]]:
    all_rows: List[Dict[str, Any]] = []
    for raw_path in raw_files:
        with raw_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                iid = str(rec.get("id", "") or "")
                if not iid:
                    continue
                if allowed_ids is not None and iid not in allowed_ids:
                    continue

                it = item_by_id.get(iid)
                if it is None:
                    continue

                provider = str(rec.get("provider", "") or "")
                model = str(rec.get("model", "") or "")
                typ = str(rec.get("type", it.get("type", "")) or it.get("type", ""))

                gt = dict(it.get("ground_truth", {}) or {})

                parsed = rec.get("parsed_json", None)
                if parsed is None and rec.get("raw_text"):
                    parsed = extract_first_json(str(rec.get("raw_text")))

                keys = _keys_for_item(it, final_only=final_only)

                rows = _score_one_item(
                    provider=provider,
                    model=model,
                    iid=iid,
                    typ=typ,
                    keys=keys,
                    gt=gt,
                    parsed=parsed if isinstance(parsed, dict) else None,
                    latency_s=rec.get("latency_s", None),
                    error=rec.get("error", None),
                    tols=tols,
                )
                all_rows.extend(rows)
    return all_rows


def _write_reports(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_item_csv = out_dir / "per_item.csv"
    df.to_csv(per_item_csv, index=False)

    summary = (
        df.groupby(["provider", "model", "type", "field"])
          .agg(
              n=("pass", "size"),
              pass_rate=("pass", "mean"),
              mean_abs_err=("abs_err", "mean"),
              median_abs_err=("abs_err", "median"),
              mean_rel_err=("rel_err", "mean"),
              p95_abs_err=("abs_err", lambda s: float(np.nanpercentile(s.dropna(), 95)) if s.dropna().size else np.nan),
              mean_latency_s=("latency_s", "mean"),
          )
          .reset_index()
          .sort_values(["provider", "model", "type", "field"])
    )
    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    overall = (
        df.groupby(["provider", "model"])
          .agg(
              n=("pass", "size"),
              overall_pass_rate=("pass", "mean"),
              mean_abs_err=("abs_err", "mean"),
              mean_rel_err=("rel_err", "mean"),
              mean_latency_s=("latency_s", "mean"),
          )
          .reset_index()
          .sort_values(["overall_pass_rate", "mean_abs_err"], ascending=[False, True])
    )
    overall_csv = out_dir / "overall.csv"
    overall.to_csv(overall_csv, index=False)

    print(f"\nWrote:\n- {per_item_csv}\n- {summary_csv}\n- {overall_csv}\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, required=True, help="Path to plotchain_v3.jsonl")
    ap.add_argument("--images_root", type=str, default="", help="Optional root directory for relative image paths")

    ap.add_argument("--mode", type=str, default="eval", choices=["eval", "score"],
                    help="eval: call models (if needed) + write raw + score. score: score from existing raw_*.jsonl only.")
    ap.add_argument("--models", type=str, nargs="*",
                    help="List like: openai:gpt-4.1 gemini:gemini-2.5-flash anthropic:claude-sonnet-4-5 (required for --mode eval)")
    ap.add_argument("--out_dir", type=str, default="results_v3", help="Output directory for raw + reports")
    ap.add_argument("--raw_glob", type=str, default="raw_*.jsonl", help="(score mode) glob inside out_dir for raw files")

    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of items (also filters scoring)")
    ap.add_argument("--max_output_tokens", type=int, default=500, help="Max tokens for model output")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing raw files (eval mode)")

    ap.add_argument("--final_only", action="store_true", help="Prompt and score only final_fields (no checkpoints)")
    ap.add_argument("--tol_policy", type=str, default="plotread", choices=["plotread", "strict"],
                    help="Tolerance policy to use for scoring.")
    ap.add_argument("--abs_tol_scale", type=float, default=1.0, help="Scale all absolute tolerances (e.g., 1.2 loosens).")
    ap.add_argument("--rel_tol_scale", type=float, default=1.0, help="Scale all relative tolerances (e.g., 1.2 loosens).")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Optional sleep between API calls (eval mode).")

    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    images_root = Path(args.images_root).resolve() if args.images_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(jsonl_path)
    if args.limit and args.limit > 0:
        items = items[:args.limit]
    allowed_ids = {str(it.get("id", "")) for it in items if str(it.get("id", ""))} if (args.limit and args.limit > 0) else None

    item_by_id = _build_item_index(items)
    tols = get_tolerances(args.tol_policy, abs_scale=args.abs_tol_scale, rel_scale=args.rel_tol_scale)

    raw_files: List[Path] = []

    if args.mode == "eval":
        if not args.models:
            raise SystemExit("--models is required in --mode eval")

        specs = parse_models(args.models)

        for spec in specs:
            raw_path = out_dir / f"raw_{spec.provider}_{spec.model.replace('/', '_')}.jsonl"
            if raw_path.exists() and not args.overwrite:
                print(f"[reuse] raw exists: {raw_path}")
            else:
                print(f"[eval] writing raw: {raw_path}")
                _run_and_write_raw(
                    spec=spec,
                    items=items,
                    images_root=images_root,
                    raw_path=raw_path,
                    max_output_tokens=args.max_output_tokens,
                    final_only=args.final_only,
                    sleep_s=args.sleep_s,
                )
            raw_files.append(raw_path)

    else:
        raw_files = sorted(out_dir.glob(args.raw_glob))
        if not raw_files:
            raise SystemExit(f"No raw files found in {out_dir} matching glob '{args.raw_glob}'")

    rows = _score_from_raw_files(
        raw_files=raw_files,
        item_by_id=item_by_id,
        allowed_ids=allowed_ids,
        tols=tols,
        final_only=args.final_only,
    )

    if not rows:
        raise SystemExit("No results produced (check that raw files match the dataset ids).")

    df = pd.DataFrame(rows)
    _write_reports(df, out_dir)


if __name__ == "__main__":
    main()
