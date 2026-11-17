"""gta6-twitch-ccu-forecast-01 entry: an hour-by-hour quantile forecaster for a launch day's
Twitch category viewers, from the case's first-hour print and its pool of past launches.

Method (v1): every comparable launch in the case contributes its release-day curve as log
ratios to its own hour-1 print, lr_h = ln(v_h) - ln(v_1). For each hour the forecast's five
levels are the pool's empirical quantiles of lr_h, shrunk toward a robust normal fit of the
same pool (median and IQR-based sd) and lightly smoothed across neighbouring hours; the
case's hour-1 print scales them back to viewers. Deterministic: the output is a pure
function of the case file's bytes.

Run by the grader as `python3 -P -s /bot/main.py` (Python 3.13, standard library only);
outside it, MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR stand in for /input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Shrinkage of the empirical quantiles toward the normal fit: weight n / (n + SHRINK_K).
SHRINK_K = 8.0


def round2(x: float):
    """Round to 2 decimals, half up (as the starter does), returning an int when whole."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if float(value).is_integer() else value


def parse_ts(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def quantile(sorted_vals, tau):
    """Linear-interpolation (type 7) quantile of an ascending list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def comparable_log_ratios(comp):
    """{hour: ln(v_h / v_1)} for the comparable's release day, or None if it has no usable day."""
    ds = parse_ts(comp.get("day_start"))
    if ds is None:
        return None
    pts = {}
    for p in comp.get("hourly") or []:
        t = parse_ts(p.get("from"))
        v = p.get("viewers")
        if t is not None and isinstance(v, (int, float)) and v > 0:
            pts[t] = float(v)
    v1 = pts.get(ds)
    if not v1:
        return None
    lv1 = math.log(v1)
    out = {}
    for h in range(2, 25):
        v = pts.get(ds + timedelta(hours=h - 1))
        if v:
            out[h] = math.log(v) - lv1
    return out


def pool_log_ratios(case):
    """{hour: sorted [log ratios]} pooled over the case's comparable launches."""
    pooled = {h: [] for h in range(2, 25)}
    for comp in case.get("comparables") or []:
        lr = comparable_log_ratios(comp)
        if not lr:
            continue
        for h, val in lr.items():
            pooled[h].append(val)
    for h in pooled:
        pooled[h].sort()
    return pooled


def hour_quantile_curves(pooled):
    """{hour: [lr at each level]}: empirical quantiles shrunk to a normal fit, smoothed over hours."""
    raw = {}
    for h in range(2, 25):
        vals = pooled.get(h) or []
        if not vals:
            raw[h] = [0.0] * 5
            continue
        med = quantile(vals, 0.5)
        iqr = quantile(vals, 0.75) - quantile(vals, 0.25)
        sd = max(0.05, iqr / 1.349)
        w = len(vals) / (len(vals) + SHRINK_K)
        raw[h] = [
            w * quantile(vals, tau) + (1.0 - w) * (med + z * sd)
            for tau, z in zip(LEVELS, Z)
        ]
    # Triangular smoothing across neighbouring hours, per level.
    smooth = {}
    for h in range(2, 25):
        row = []
        for i in range(5):
            acc, wsum = 0.0, 0.0
            for dh, wgt in ((-1, 1.0), (0, 2.0), (1, 1.0)):
                hh = h + dh
                if 2 <= hh <= 24:
                    acc += wgt * raw[hh][i]
                    wsum += wgt
            row.append(acc / wsum)
        smooth[h] = row
    return smooth


def hour1_print(case):
    for k in case.get("known") or []:
        if k.get("hour") == 1 and isinstance(k.get("viewers"), (int, float)):
            return max(1.0, float(k["viewers"]))
    hist = case.get("history") or []
    for p in reversed(hist):
        v = p.get("viewers")
        if isinstance(v, (int, float)) and v > 0:
            return max(1.0, float(v))
    return 1000.0


def forecast_case(case):
    p1 = hour1_print(case)
    curves = hour_quantile_curves(pool_log_ratios(case))
    rows = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        qs = sorted(min(1e9, p1 * math.exp(lr)) for lr in curves.get(h, [0.0] * 5))
        rows.append({"hour": h, "q": [round2(v) for v in qs]})
    rows.sort(key=lambda r: r["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
