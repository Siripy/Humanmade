"""The physical body: homeostasis, metabolism, circadian rhythm, mortality.

Rates are grounded in real human physiology, scaled to simulation minutes:
  - Water:   death after ~3 days without (hydration 100 -> 0 in ~4320 sim-min)
  - Food:    hunger cycles ~5h between meals; starvation damages health slowly
             (real starvation takes ~3 weeks; we scale that to health drain)
  - Sleep:   ~16h of wakefulness builds full sleep pressure (adenosine model),
             modulated by circadian phase (Borbély's two-process model)
  - Breath:  12-16 breaths/min at rest, rising with exertion and stress
  - Waste:   bladder fills with hydration intake, bowel with meals
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, asdict

MINUTES_PER_DAY = 24 * 60


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


@dataclass
class Body:
    # Resources: 100 = satisfied, 0 = critical
    energy: float = 80.0        # sleep/rest reserve
    hydration: float = 75.0     # body water
    satiety: float = 70.0       # stomach/blood sugar
    hygiene: float = 90.0       # cleanliness
    fun: float = 60.0           # mental stimulation
    social: float = 50.0        # connection (loneliness when low)
    health: float = 100.0       # overall integrity; 0 = death

    # Pressures: 0 = none, 100 = bursting
    bladder: float = 20.0
    bowel: float = 10.0

    # State
    asleep: bool = False
    awake_minutes: float = 0.0
    sim_minutes: float = 8 * 60.0   # sim clock, minutes since day 0 (starts 08:00)
    alive: bool = True
    cause_of_death: str | None = None
    accidents: int = 0              # times bladder/bowel overflowed

    # Vitals (derived each tick, kept for display)
    heart_rate: float = 68.0
    breaths_per_min: float = 14.0
    spo2: float = 98.0
    body_temp_c: float = 36.7

    # ------------------------------------------------------------------ time

    @property
    def day(self) -> int:
        return int(self.sim_minutes // MINUTES_PER_DAY)

    @property
    def clock(self) -> str:
        m = int(self.sim_minutes % MINUTES_PER_DAY)
        return f"{m // 60:02d}:{m % 60:02d}"

    @property
    def hour(self) -> float:
        return (self.sim_minutes % MINUTES_PER_DAY) / 60.0

    def circadian_sleep_drive(self) -> float:
        """0..1 — how strongly the circadian clock pushes toward sleep.

        Peaks around 03:00, trough around 15:00 (post-lunch dip added).
        """
        phase = math.cos((self.hour - 3.0) / 24.0 * 2 * math.pi)  # 1 at 03:00
        dip = 0.15 * math.exp(-((self.hour - 14.5) ** 2) / 2.0)   # afternoon dip
        return clamp((phase + 1) / 2 + dip, 0.0, 1.0)

    # ------------------------------------------------------------------ tick

    MAX_STEP_MINUTES = 5.0

    def tick(self, minutes: float, activity: str = "idle") -> list[str]:
        """Advance physiology by `minutes` of sim time. Returns event strings.

        Integrates in small sub-steps: a slow LLM call plus a high sim speed can
        produce one huge time jump, and a single Euler step across hours would
        skip overflow/death thresholds. Repeated warnings are deduplicated.
        """
        events: list[str] = []
        remaining = minutes
        while remaining > 1e-9 and self.alive:
            step = min(self.MAX_STEP_MINUTES, remaining)
            remaining -= step
            for e in self._step(step, activity):
                if e not in events:
                    events.append(e)
        return events

    def _step(self, minutes: float, activity: str) -> list[str]:
        if not self.alive:
            return []
        events: list[str] = []
        exertion = {"exercise": 3.0, "work": 1.3, "shower": 1.1}.get(activity, 1.0)

        self.sim_minutes += minutes

        if self.asleep:
            self.energy = clamp(self.energy + 0.21 * minutes)   # 8h ~ full recharge
            self.awake_minutes = 0.0
            self.satiety = clamp(self.satiety - 0.035 * minutes)  # slower BMR asleep
            self.hydration = clamp(self.hydration - 0.012 * minutes)
            self.fun = clamp(self.fun + 0.01 * minutes)  # dreams count a little
        else:
            self.awake_minutes += minutes
            # Sleep pressure: ~16h awake drains a full charge
            self.energy = clamp(self.energy - (100 / 960) * minutes
                                * (0.7 + 0.6 * self.circadian_sleep_drive()))
            self.satiety = clamp(self.satiety - 0.055 * minutes * exertion)
            self.hydration = clamp(self.hydration - 0.023 * minutes * exertion)
            self.fun = clamp(self.fun - (0.06 if activity == "idle" else 0.02) * minutes)
            self.social = clamp(self.social - 0.035 * minutes)
            self.hygiene = clamp(self.hygiene - 0.045 * minutes * exertion)

        # Waste production tracks intake/metabolism
        if self.hydration > 5:
            self.bladder = clamp(self.bladder + 0.09 * minutes * (0.4 if self.asleep else 1.0))
        self.bowel = clamp(self.bowel + 0.018 * minutes)

        events += self._overflow_check()
        events += self._health_update(minutes)
        self._update_vitals(exertion)
        return events

    # ---------------------------------------------------------------- damage

    def _overflow_check(self) -> list[str]:
        events = []
        if self.bladder >= 100:
            self.bladder = 15.0
            self.hygiene = clamp(self.hygiene - 35)
            self.accidents += 1
            if self.asleep:
                events.append("wet the bed while sleeping — woke up mortified")
                self.asleep = False
            else:
                events.append("couldn't hold it any longer and had an accident")
        if self.bowel >= 100:
            self.bowel = 10.0
            self.hygiene = clamp(self.hygiene - 45)
            self.accidents += 1
            events.append("had a humiliating bowel accident")
        return events

    def _health_update(self, minutes: float) -> list[str]:
        events = []
        drain = 0.0
        if self.hydration <= 0:
            drain += (100 / (1.5 * MINUTES_PER_DAY))  # ~1.5 further days to death
            events.append("severely dehydrated — organs are straining")
        if self.satiety <= 0:
            drain += (100 / (18 * MINUTES_PER_DAY))   # starvation is slow
        if self.awake_minutes > 2 * MINUTES_PER_DAY:  # >48h awake
            drain += (100 / (7 * MINUTES_PER_DAY))
            events.append("hallucinating from sleep deprivation")
        if self.hygiene <= 5:
            drain += (100 / (30 * MINUTES_PER_DAY))   # infection risk

        if drain > 0:
            self.health = clamp(self.health - drain * minutes)
        elif self.hydration > 30 and self.satiety > 25 and self.energy > 20:
            self.health = clamp(self.health + 0.01 * minutes)  # slow recovery

        if self.health <= 0:
            self.alive = False
            self.asleep = False
            if self.hydration <= 0:
                self.cause_of_death = "dehydration"
            elif self.satiety <= 0:
                self.cause_of_death = "starvation"
            else:
                self.cause_of_death = "organ failure from neglect"
            events.append(f"DIED of {self.cause_of_death}")
        return events

    def _update_vitals(self, exertion: float) -> None:
        stress = (100 - self.health) / 100 * 0.3 + max(0, (self.bladder - 80)) / 100
        base_hr = 52.0 if self.asleep else 66.0
        self.heart_rate = base_hr + 30 * (exertion - 1) + 25 * stress + random.uniform(-2, 2)
        self.breaths_per_min = (10 if self.asleep else 14) + 8 * (exertion - 1) + 6 * stress
        self.spo2 = clamp(98.5 - (100 - self.health) * 0.06 + random.uniform(-0.3, 0.3), 80, 100)
        self.body_temp_c = 36.7 + 0.4 * (exertion - 1) - (0.5 if self.health < 30 else 0)

    # ---------------------------------------------------------------- affect

    def mood(self) -> tuple[float, str]:
        """Valence -1..1 plus a label, derived from need satisfaction (PAD-lite)."""
        needs = [self.energy, self.hydration, self.satiety, self.hygiene,
                 self.fun, self.social, 100 - self.bladder, 100 - self.bowel]
        valence = (sum(needs) / len(needs) - 50) / 50
        valence -= (100 - self.health) / 150
        valence = max(-1.0, min(1.0, valence))
        for threshold, label in [(0.45, "content"), (0.2, "okay"), (-0.1, "restless"),
                                 (-0.35, "miserable")]:
            if valence >= threshold:
                return valence, label
        return valence, "in agony"

    def urgent_needs(self) -> list[str]:
        """Ordered by survival priority (Maslow's physiological base)."""
        urgent = []
        if self.hydration < 20: urgent.append("desperately thirsty")
        if self.bladder > 80:   urgent.append("bladder is bursting")
        if self.bowel > 80:     urgent.append("urgently needs the toilet")
        if self.satiety < 20:   urgent.append("very hungry")
        if self.energy < 15:    urgent.append("exhausted, can barely stay awake")
        if self.hygiene < 25:   urgent.append("filthy, skin itching")
        if self.fun < 15:       urgent.append("mind-numbingly bored")
        if self.social < 15:    urgent.append("achingly lonely")
        if self.health < 40:    urgent.append("feels seriously ill")
        return urgent

    # --------------------------------------------------------------- actions

    def eat(self, kcal_quality: float = 55.0) -> None:
        self.satiety = clamp(self.satiety + kcal_quality)
        self.bowel = clamp(self.bowel + 12)
        self.hydration = clamp(self.hydration + 5)

    def drink(self) -> None:
        self.hydration = clamp(self.hydration + 30)
        self.bladder = clamp(self.bladder + 9)

    def use_toilet(self) -> None:
        self.bladder = 0.0
        self.bowel = max(0.0, self.bowel - 90)

    def shower(self) -> None:
        self.hygiene = 100.0

    def fall_asleep(self) -> None:
        self.asleep = True

    def wake_up(self) -> None:
        self.asleep = False
        self.awake_minutes = 0.0

    # ----------------------------------------------------------- persistence

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Body":
        b = cls()
        for k, v in d.items():
            if hasattr(b, k):
                setattr(b, k, v)
        return b

    # ---------------------------------------------------------------- report

    def status_lines(self) -> list[str]:
        def bar(v: float) -> str:
            filled = int(round(v / 10))
            return "█" * filled + "░" * (10 - filled) + f" {v:5.1f}"

        state = "asleep" if self.asleep else "awake"
        _, mood_label = self.mood()
        return [
            f"Day {self.day}, {self.clock} — {state}, feeling {mood_label}",
            f"  health    {bar(self.health)}   energy   {bar(self.energy)}",
            f"  hydration {bar(self.hydration)}   satiety  {bar(self.satiety)}",
            f"  bladder   {bar(self.bladder)}   bowel    {bar(self.bowel)}",
            f"  hygiene   {bar(self.hygiene)}   fun      {bar(self.fun)}",
            f"  social    {bar(self.social)}",
            f"  vitals: {self.heart_rate:.0f} bpm · {self.breaths_per_min:.0f} breaths/min"
            f" · SpO2 {self.spo2:.0f}% · {self.body_temp_c:.1f}°C",
        ]
