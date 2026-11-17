"""gta6-twitch-ccu-forecast-01 entry: an hour-by-hour quantile forecaster for a launch day's
Twitch category viewers, from the case's first-hour print and its pool of past launches.

Method (v0): every comparable launch in the case contributes its release-day curve as log
ratios to its own hour-1 print, lr_h = ln(v_h) - ln(v_1). The median forecast is the case's
hour-1 print times the pool's median ratio at that hour; the five levels fan out around it
with fixed normal z-scores and sigma = 0.35 in log space. Deterministic: the output is a
pure function of the case file's bytes.

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
SIGMA = 0.35


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


def median(sorted_vals):
    n = len(sorted_vals)
    mid = n // 2
    if n % 2:
        return sorted_vals[mid]
    return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])


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


def hour1_print(case):
    for k in case.get("known") or []:
        if k.get("hour") == 1 and isinstance(k.get("viewers"), (int, float)):
            return max(1.0, float(k["viewers"]))
    # Fallbacks: last history value, then a nominal idle-category level.
    hist = case.get("history") or []
    for p in reversed(hist):
        v = p.get("viewers")
        if isinstance(v, (int, float)) and v > 0:
            return max(1.0, float(v))
    return 1000.0


def forecast_case(case):
    p1 = hour1_print(case)
    pooled = pool_log_ratios(case)
    rows = []
    for row in case.get("forecast") or []:
        h = row["hour"]
        vals = pooled.get(h) or []
        med = median(vals) if vals else 0.0
        centre = p1 * math.exp(med)
        q = [round2(centre * math.exp(z * SIGMA)) for z in Z]
        rows.append({"hour": h, "q": q})
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
