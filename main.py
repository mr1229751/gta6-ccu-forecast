"""gta6-twitch-ccu-forecast-01 entry: an hour-by-hour quantile forecaster for a launch day's
Twitch category viewers, from the case's first-hour print and its pool of past launches.

Method (v2): every comparable launch contributes its release-day curve as log ratios to its
own hour-1 print, lr_h = ln(v_h) - ln(v_1). The pool is weighted by similarity to the case
on three launch-state features, all computed the same way for the case and for each
comparable from its own series: ln(v1) (scale), ln(v1 / pre12) (hour 1 against the mean of
the 12 hours before the day) and ln(v1 / base) (hour 1 against the idle baseline, the
median of the hours 48..12 before the day). Launches whose hour 1 already holds a launched
audience resemble each other; launches whose day has not started at hour 1 do not forecast a
rolling-midnight console launch. For each hour the forecast's five levels are the weighted
empirical quantiles of lr_h, shrunk toward a robust normal fit of the same weighted pool
and lightly smoothed across neighbouring hours; the case's hour-1 print scales them back
to viewers. Deterministic: the output is a pure function of the case file's bytes.

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
# Shrinkage of the empirical quantiles toward the normal fit: weight n_eff / (n_eff + SHRINK_K).
SHRINK_K = 8.0
# Similarity kernel: per-feature scales in log space, and the flat-pool blend share.
# (Scales and blend tuned by leave-one-out over the comparable pool; see README.)
KERNEL_SCALES = [1.0, 1.2, 1.5]
FLAT_BLEND = 0.15
# Post-calibration of the log-ratio levels, per hour bucket, fitted by coordinate descent
# on leave-one-out forecasts over the comparable pool and shrunk by half toward zero: the
# raw pool quantiles leave the early hours' upper tail slightly narrow.
CALIB = {
    (2, 5): [-0.05, 0.0, 0.025, 0.05, 0.175],
    (6, 11): [-0.025, 0.0, 0.0, 0.05, 0.2],
    (12, 17): [-0.075, 0.05, 0.025, 0.025, 0.025],
    (18, 24): [0.025, 0.05, 0.025, 0.0, 0.0],
}


def calib_delta(h, i):
    for (a, b), row in CALIB.items():
        if a <= h <= b:
            return row[i]
    return 0.0


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


def median_of(vals):
    s = sorted(vals)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def hourly_series(points):
    """{hour-start datetime: viewers} from points carrying a timestamp and viewers.

    `points` items may use `from` (whole-hour rows) or `at` (raw reads); reads inside one
    hour are averaged, so minute reads and hourly points land on the same footing.
    """
    buckets = {}
    for p in points or []:
        t = parse_ts(p.get("from") or p.get("at"))
        v = p.get("viewers")
        if t is None or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        hour = t.replace(minute=0, second=0)
        buckets.setdefault(hour, []).append(float(v))
    return {t: sum(vs) / len(vs) for t, vs in buckets.items()}


def launch_features(series, ds, v1):
    """[ln v1, ln(v1/pre12), ln(v1/base)] for a launch; None where a window has no data.

    pre12 = mean of the 12 whole hours before day_start; base = median of the whole hours
    from 48 to 12 hours before day_start. Windows widen to whatever pre-day hours exist.
    """
    if not v1 or v1 <= 0:
        return None
    pre = {t: v for t, v in series.items() if t < ds and v > 0}
    if not pre:
        return [math.log(v1), None, None]
    pre12_vals = [v for t, v in pre.items() if t >= ds - timedelta(hours=12)]
    if not pre12_vals:
        last = sorted(pre)[-12:]
        pre12_vals = [pre[t] for t in last]
    base_vals = [v for t, v in pre.items() if ds - timedelta(hours=48) <= t < ds - timedelta(hours=12)]
    if not base_vals:
        base_vals = [v for t, v in pre.items() if t < ds - timedelta(hours=12)] or list(pre.values())
    pre12 = sum(pre12_vals) / len(pre12_vals)
    base = median_of(base_vals)
    return [
        math.log(v1),
        math.log(v1 / pre12) if pre12 > 0 else None,
        math.log(v1 / base) if base > 0 else None,
    ]


def comparable_record(comp):
    """(log-ratio curve, features) for one comparable launch, or None if it has no usable day."""
    ds = parse_ts(comp.get("day_start"))
    if ds is None:
        return None
    series = hourly_series(comp.get("hourly"))
    v1 = series.get(ds)
    if not v1:
        return None
    lv1 = math.log(v1)
    lr = {}
    for h in range(2, 25):
        v = series.get(ds + timedelta(hours=h - 1))
        if v:
            lr[h] = math.log(v) - lv1
    if not lr:
        return None
    return lr, launch_features(series, ds, v1)


def similarity_weights(case_feats, comp_feats_list):
    """Gaussian kernel weights over the features both sides carry, blended with the flat pool."""
    n = len(comp_feats_list)
    if n == 0:
        return []
    raw = []
    for feats in comp_feats_list:
        d2, used = 0.0, 0
        if case_feats is not None and feats is not None:
            for j, scale in enumerate(KERNEL_SCALES):
                a, b = case_feats[j], feats[j]
                if a is not None and b is not None:
                    d2 += ((a - b) / scale) ** 2
                    used += 1
        raw.append(math.exp(-0.5 * d2) if used else 1.0)
    total = sum(raw)
    if total <= 0:
        return [1.0 / n] * n
    return [(1.0 - FLAT_BLEND) * (w / total) + FLAT_BLEND / n for w in raw]


def weighted_quantile(sorted_pairs, tau):
    """Quantile of (value, weight) pairs sorted by value, on midpoint plotting positions."""
    total = sum(w for _, w in sorted_pairs)
    if total <= 0:
        return sorted_pairs[len(sorted_pairs) // 2][0]
    pts = []
    cum = 0.0
    for val, w in sorted_pairs:
        cum += w
        pts.append(((cum - 0.5 * w) / total, val))
    if tau <= pts[0][0]:
        return pts[0][1]
    if tau >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        if tau <= pts[i][0]:
            (p0, v0), (p1, v1) = pts[i - 1], pts[i]
            if p1 == p0:
                return v1
            return v0 + (v1 - v0) * (tau - p0) / (p1 - p0)
    return pts[-1][1]


def hour_quantile_curves(records, weights):
    """{hour: [lr at each level]}: weighted empirical quantiles shrunk to a normal fit, smoothed."""
    raw = {}
    for h in range(2, 25):
        pairs = sorted(
            (rec[0][h], w) for rec, w in zip(records, weights) if h in rec[0]
        )
        if not pairs:
            raw[h] = [0.0] * 5
            continue
        n_eff = (sum(w for _, w in pairs) ** 2) / max(1e-12, sum(w * w for _, w in pairs))
        med = weighted_quantile(pairs, 0.5)
        iqr = weighted_quantile(pairs, 0.75) - weighted_quantile(pairs, 0.25)
        sd = max(0.05, iqr / 1.349)
        wgt = n_eff / (n_eff + SHRINK_K)
        raw[h] = [
            wgt * weighted_quantile(pairs, tau) + (1.0 - wgt) * (med + z * sd)
            for tau, z in zip(LEVELS, Z)
        ]
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
        v = k.get("viewers")
        if k.get("hour") == 1 and isinstance(v, (int, float)) and math.isfinite(v):
            return max(1.0, float(v))
    hist = case.get("history") or []
    for p in reversed(hist):
        v = p.get("viewers")
        if isinstance(v, (int, float)) and math.isfinite(v) and v > 0:
            return max(1.0, float(v))
    return 1000.0


def case_features(case, p1):
    ds = parse_ts(case.get("day_start"))
    if ds is None:
        return None
    series = hourly_series(case.get("history_hourly") or [])
    if len(series) < 6:
        merged = hourly_series(case.get("history") or [])
        merged.update(series)
        series = merged
    return launch_features(series, ds, p1)


def forecast_case(case):
    p1 = hour1_print(case)
    records, feats_list = [], []
    for comp in case.get("comparables") or []:
        rec = comparable_record(comp)
        if rec:
            records.append(rec)
            feats_list.append(rec[1])
    weights = similarity_weights(case_features(case, p1), feats_list)
    curves = hour_quantile_curves(records, weights)
    wanted = []
    for row in case.get("forecast") or []:
        h = row.get("hour")
        if isinstance(h, int) and 2 <= h <= 24 and h not in wanted:
            wanted.append(h)
    if len(wanted) != 23:  # a malformed forecast list still owes the full 23 rows
        wanted = list(range(2, 25))
    rows = []
    for h in wanted:
        qs = sorted(
            min(1e9, p1 * math.exp(lr + calib_delta(h, i)))
            for i, lr in enumerate(curves.get(h, [0.0] * 5))
        )
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
