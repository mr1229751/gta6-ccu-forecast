# gta6-ccu-forecast

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mr1229751/gta6-ccu-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/mr1229751/gta6-ccu-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

An hour-by-hour quantile forecaster for a game launch day's Twitch audience. Given one case
file (schema `makefun/twitch-ccu-case/v1`) — everything recorded up to the end of hour 1 of
the release day, that hour's viewer count included, plus a pool of past launches' hourly
curves — it writes, for each of hours 2..24, five quantiles (5%, 25%, 50%, 75%, 95%) of that
hour's concurrent viewers in the game's Twitch category. It is built for make.fun's
"Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty, scored by pinball loss
on the log scale.

## Method

Every comparable launch in the case contributes its release-day curve as log ratios to its
own hour-1 print, `lr_h = ln(v_h) − ln(v_1)`. The pool is **weighted by similarity** to the
case on three launch-state features, computed identically for the case and for each
comparable from its own series:

- `ln v1` — the scale of hour 1;
- `ln(v1 / pre12)` — hour 1 against the mean of the 12 hours before the day (has the launch
  already happened by hour 1, or does it start later in the UTC day?);
- `ln(v1 / base)` — hour 1 against the idle baseline (median of the hours 48..12 before).

A Gaussian kernel over those features (blended 15% with the flat pool) concentrates the
forecast on launches whose first hour means the same thing as this case's first hour — the
single biggest error source in a flat pool, where a rolling-midnight console launch and a
late-afternoon global unlock otherwise average into a shape neither one follows. For each
hour, the five levels are the **weighted empirical quantiles** of `lr_h`, shrunk toward a
robust normal fit of the same weighted pool (median, IQR-based sd), lightly smoothed across
neighbouring hours, and adjusted by a small per-bucket calibration fitted by coordinate
descent on leave-one-out forecasts and shrunk by half. The case's hour-1 print scales the
curves back to viewers.

The program is a pure function of the case file: no randomness, no clock, no network —
two runs on the same case are byte-identical. Missing or malformed inputs (no hour-1 print,
no comparables, empty history, null fields) degrade to a persistence-shaped fallback rather
than an error, so every case yields a valid 23-row forecast.

## Run it

Python 3.13, standard library only. The grader runs the program once per case with only
that case's file in the input directory:

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
# reads input/<id>.in, writes out/<id>.out
```

Without the environment variables it reads `/input` and writes `/output`. A case takes
~0.25 s (the cap is 8 s).

## Local evaluation

The kit publishes no practice answers (grading is server-side), so the model was tuned by
leave-one-out over the 37 real launches in the practice cases' comparable pool: each launch
in turn becomes a pseudo-case — its own release day, its own hour-1 print, the other 36 as
its pool — and is scored with the contest's exact pinball loss. Mean loss over the 37
pseudo-cases (the house persistence baseline scores 0.930 on the same set; on the subset of
27 launches with hour-1 ≥ 20k viewers, the scale of all four real practice cases, it
scores 0.321):

| version | mean loss | big-hour-1 subset |
|---|---|---|
| v0 pooled median shape, fixed σ = 0.35 | 0.882 | 0.323 |
| v1 per-hour empirical quantiles | 0.716 | 0.364 |
| v2 similarity-weighted pool | 0.411 | 0.252 |
| v3 + shrunk post-calibration (this entry) | **0.407** | **0.251** |

Two plausible priors were tested and rejected by the same leave-one-out losses: a wider
upper tail in the unlock/peak hours, and a rolling-launch peak of 3.5–5.5× hour 1 (the
pool's launched-by-hour-1 launches median ≈ 1.9×, peaking at 19:00–22:00 UTC).

---

<sub>Entered on [make.fun](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
