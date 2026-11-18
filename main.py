"""gta6-twitch-ccu-forecast-01: size-conditioned launch-day shape forecast.

Model: for each hour h = 2..24, the forecast quantiles are the kernel-weighted
quantiles of the comparables' log-ratios log(v_h / v_1), where each comparable is
weighted by how close its hour-1 size is to the target's hour-1 print (a Gaussian
kernel on log(p1), bandwidth 1.0).  q_tau(h) = p1 * exp(weighted-quantile_tau).

Why this beats the persistence baseline (leave-one-out on the 38 comparables,
mean pinball on log scale; baseline = 0.9297):
  - persistence alone:                    0.9297  (the house baseline to beat)
  - median shape + persistence spread:    0.8337  (-10.3%)
  - empirical quantiles, all comparables: 0.7139  (-23.2%)
  - Gaussian kernel on log(p1), bw = 1.0: 0.4100  (-55.9%), wins 33/37 targets

The size conditioning matters because a launch's shape depends strongly on its
hour-1 size: the biggest launches (hour 1 already huge) stay near flat (median
peak ratio ~1.3x), while small launches grow ~10x as regions unlock.  A plain
median shape fits neither group; conditioning on p1 picks the right peers.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only, no network.  Reads every <id>.in in the
input directory (one case at a time), writes <id>.out: {"hours": [{"hour": 2,
"q": [q05, q25, q50, q75, q95]}, ...]} — exactly 23 rows, hours 2..24, each q
five finite numbers > 0 and <= 1e9, non-decreasing, at most 65,536 bytes.

Outside the grader, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR stand in for
/input and /output.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Bandwidth of the Gaussian kernel on log(hour-1 viewers), in natural-log units.
# Chosen by leave-one-out on the comparables (bw 0.75-1.5 all give ~55% better
# than persistence; 1.0 is the best and sits on a broad plateau, not a sharp peak).
KERNEL_BW = 1.0
# The persistence baseline's spread, used only as a fallback when a case has no
# usable comparables for an hour.
FALLBACK_SIGMA = 0.15
FALLBACK_Z = [-1.645, -0.674, 0, 0.674, 1.645]


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    """Parse an ISO-8601 UTC timestamp ('...Z') into an aware datetime."""
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_hours(comp: dict) -> dict:
    """A comparable's release-day hourly viewers, keyed by hour 1..24.

    `hourly` holds points from 14 days before the release to 3 days after; a
    point at `from` covers the hour [from, from + 1h).  Only points with a
    positive viewer count land in the map; an hour SullyGnome did not sample
    is simply absent (never interpolated).
    """
    day_start = parse_ts(comp["day_start"])
    out = {}
    for pt in comp.get("hourly", []):
        try:
            t = parse_ts(pt["from"])
        except (KeyError, TypeError, ValueError):
            continue
        h = int((t - day_start).total_seconds() // 3600) + 1
        if 1 <= h <= 24:
            v = pt.get("viewers")
            if isinstance(v, (int, float)) and v > 0:
                out[h] = float(v)
    return out


def weighted_quantile(pairs, tau: float):
    """The tau-quantile of weighted values.

    pairs is a list of (value, weight); the quantile is the value where the
    cumulative weight first reaches tau of the total, linearly interpolated
    between the bracketing values.  Deterministic: pairs are sorted by value.
    """
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    target = tau * total
    cum = 0.0
    prev_v, prev_c = pairs[0][0], 0.0
    for v, w in pairs:
        cum += w
        if cum >= target:
            if cum == w:
                return v
            span = cum - prev_c
            frac = (target - prev_c) / span if span > 0 else 0.0
            return prev_v + frac * (v - prev_v)
        prev_v, prev_c = v, cum
    return pairs[-1][0]


def hour_quantiles(p1: float, h: int, curves) -> list:
    """The five quantiles of hour h, from the size-conditioned comparables.

    curves is a list of release-day hour maps (one per comparable).  Each
    comparable that has both hour 1 and hour h contributes its log-ratio
    log(v_h / v_1) with a Gaussian-kernel weight on how close its hour-1 size
    is to p1.  The quantiles of that weighted log-ratio distribution, shifted
    by log(p1), are the forecast.  Falls back to the persistence spread when
    no comparable has the hour.
    """
    log_p1 = math.log(p1)
    pairs = []
    for hours in curves:
        v1 = hours.get(1)
        vh = hours.get(h)
        if v1 is None or vh is None:
            continue
        d = math.log(v1) - log_p1
        w = math.exp(-0.5 * (d / KERNEL_BW) ** 2)
        pairs.append((math.log(vh / v1), w))
    if not pairs:
        return [round2(p1 * math.exp(z * FALLBACK_SIGMA * math.sqrt(h - 1))) for z in FALLBACK_Z]
    qs = []
    for tau in LEVELS:
        qlog = weighted_quantile(pairs, tau)
        qs.append(round2(p1 * math.exp(qlog)))
    # Guard the contract: finite, > 0, <= 1e9, non-decreasing (rounding can
    # only move a value by a cent, but a degenerate pair set could invert two
    # adjacent quantiles; the running max keeps the row valid).
    fixed = []
    prev = 0.0
    for q in qs:
        v = float(q)
        if not math.isfinite(v) or v <= 0:
            v = 1.0
        if v > 1e9:
            v = 1e9
        if v < prev:
            v = prev
        fixed.append(round2(v))
        prev = float(fixed[-1])
    return fixed


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    curves = [release_day_hours(c) for c in case.get("comparables", [])]
    curves = [c for c in curves if 1 in c]
    return {"hours": [{"hour": row["hour"], "q": hour_quantiles(p1, row["hour"], curves)}
                    for row in case["forecast"]]}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
