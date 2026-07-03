"""Time-true mode: the human's clock anchored to your real clock, not an
artificial multiplier — its day is your day, and time doesn't stop just
because you closed the terminal.

An anchor is a single fixed point: (real_seconds, sim_minutes) captured at
some moment. Given any later real time, the sim time it corresponds to is a
pure function of that anchor — no accumulated drift, and the math is
identical whether "later" is one tick away or three real days away (a
relaunch after being closed). That symmetry is what makes offline catch-up
free: it's not a separate system, just the same formula with a bigger gap.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass

REAL_SECONDS_PER_SIM_MINUTE = 60.0   # 1:1 wall-clock pacing


@dataclass
class Anchor:
    real_seconds: float   # time.time() at the anchor point
    sim_minutes: float    # the human's sim clock at that same moment

    def to_dict(self) -> dict:
        return {"real_seconds": self.real_seconds, "sim_minutes": self.sim_minutes}

    @classmethod
    def from_dict(cls, d: dict) -> "Anchor":
        return cls(real_seconds=float(d["real_seconds"]),
                  sim_minutes=float(d["sim_minutes"]))


def target_sim_minutes(anchor: Anchor, now: float | None = None) -> float:
    """What the sim clock should read right now, given the anchor. Time
    never runs backward: a clock that's somehow already ahead of the
    target (e.g. the anchor is stale in the other direction) just holds."""
    now = _time.time() if now is None else now
    elapsed_real = max(0.0, now - anchor.real_seconds)
    return anchor.sim_minutes + elapsed_real / REAL_SECONDS_PER_SIM_MINUTE


def rebase(sim_minutes: float, now: float | None = None) -> Anchor:
    """A fresh anchor tying the given sim clock to the current real moment —
    call this whenever the sim clock is deliberately moved (real mode
    turned on, a catch-up just finished) so future targets are computed
    from where things actually stand now."""
    return Anchor(real_seconds=_time.time() if now is None else now,
                 sim_minutes=sim_minutes)


def real_hour_of_day(now: float | None = None) -> float:
    lt = _time.localtime(_time.time() if now is None else now)
    return lt.tm_hour + lt.tm_min / 60.0


def align_hour_of_day(sim_minutes: float, target_hour: float,
                      minutes_per_day: float = 1440.0) -> float:
    """Shift a sim clock to a given hour-of-day while preserving the day
    count — used once, the moment real-time mode is first turned on, so a
    human who's been living at some artificial speed doesn't wake up at a
    jarring hour relative to your actual clock."""
    day = int(sim_minutes // minutes_per_day)
    return day * minutes_per_day + target_hour * 60.0


# --- optional real-weather mirroring (Open-Meteo WMO weather codes) -------

_WMO_SUNNY = frozenset({0, 1, 2})
_WMO_CLOUDY = frozenset({3, 45, 48})
_WMO_RAINY = frozenset({51, 53, 55, 56, 57, 61, 63, 65, 66, 67,
                        71, 73, 75, 77, 80, 81, 82, 85, 86})
_WMO_STORMY = frozenset({95, 96, 99})


def wmo_to_weather(code: int) -> str:
    """Map an Open-Meteo WMO weather code onto this sim's four weathers
    (there's no "snow" here, so it lands on the closest analog: rainy).
    Unrecognized codes default to 'cloudy' — a safe, unremarkable middle
    ground — rather than raising."""
    if code in _WMO_STORMY:
        return "stormy"
    if code in _WMO_RAINY:
        return "rainy"
    if code in _WMO_SUNNY:
        return "sunny"
    return "cloudy"
