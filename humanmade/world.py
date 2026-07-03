"""A small apartment: the environment the human lives in.

Water is on tap, but food is finite and costs money. The human earns credits
by working; only its companion (you) can place the grocery order (/restock),
which spends the human's credits. Between "I'm broke", "we're out of food",
and "can you order groceries?", it always has real, need-driven reasons to
start conversations.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, asdict

PORTION_PRICE = 1.5     # credits per portion of food
WORK_WAGE = 10.0        # credits per hour-long work session
MEDICINE_PRICE = 12.0   # credits per dose

WEATHERS = ("sunny", "cloudy", "rainy", "stormy")
_WEATHER_WEIGHTS = (4, 4, 3, 1)
_WEATHER_VALENCE = {"sunny": 0.06, "cloudy": 0.0, "rainy": -0.05, "stormy": -0.12}
_WEATHER_CHILL = {"sunny": 1.0, "cloudy": 1.1, "rainy": 1.6, "stormy": 2.0}


@dataclass
class World:
    food_portions: int = 6          # meals in the fridge
    money: float = 30.0             # the human's credits
    room_temp_c: float = 21.0
    air_quality: float = 100.0      # future: windows, smoke, etc.
    weather: str = "sunny"
    weather_minutes_left: float = 360.0

    def advance(self, minutes: float) -> list[str]:
        """Move the outside world along. Returns notable events."""
        events: list[str] = []
        self.weather_minutes_left -= minutes
        if self.weather_minutes_left <= 0:
            new = random.choices(WEATHERS, weights=_WEATHER_WEIGHTS)[0]
            self.weather_minutes_left = random.uniform(240, 720)
            if new != self.weather:
                self.weather = new
                events.append(f"outside, the weather turns {new}")
        return events

    def weather_valence(self) -> float:
        """Small mood shift from the sky (Denissen et al. 2008)."""
        return _WEATHER_VALENCE[self.weather]

    def chill_factor(self) -> float:
        """Foul weather raises the odds of catching something."""
        return _WEATHER_CHILL[self.weather]

    def describe(self) -> str:
        if self.food_portions <= 0:
            fridge = "The fridge is EMPTY — no food left."
        elif self.food_portions <= 2:
            fridge = f"The fridge is nearly empty ({self.food_portions} portions left)."
        else:
            fridge = f"The fridge holds {self.food_portions} portions of food."
        money = (f"You have {self.money:.0f} credits. Food costs "
                 f"{PORTION_PRICE:g}/portion and an hour of work earns "
                 f"{WORK_WAGE:g}. Only your companion can place the grocery "
                 "order, but it spends YOUR credits — if the fridge is low, "
                 "ask them to restock (and work if you can't afford it).")
        sky = {"sunny": "Sun slants through the window.",
               "cloudy": "The sky outside is flat and grey.",
               "rainy": "Rain taps steadily at the window.",
               "stormy": "A storm rattles the windowpane."}[self.weather]
        return (f"{fridge} {money} Water runs from the tap. {sky} The room is "
                f"{self.room_temp_c:.0f}°C. You have a bed, toilet, shower, "
                "some books and music, and dumbbells.")

    def can_afford(self, portions: int) -> bool:
        return self.money >= portions * PORTION_PRICE

    def buy_medicine(self) -> bool:
        if self.money < MEDICINE_PRICE:
            return False
        self.money -= MEDICINE_PRICE
        return True

    def take_meal(self) -> bool:
        if self.food_portions <= 0:
            return False
        self.food_portions -= 1
        return True

    def earn(self, hours: float = 1.0) -> float:
        wage = WORK_WAGE * hours
        self.money += wage
        return wage

    def restock(self, portions: int = 8) -> int:
        """Buy up to `portions`, limited by the human's credits.
        Returns how many portions were actually bought."""
        affordable = min(portions, int(self.money // PORTION_PRICE))
        self.food_portions += affordable
        self.money -= affordable * PORTION_PRICE
        return affordable

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        w = cls()
        for k, v in d.items():
            if hasattr(w, k):
                setattr(w, k, v)
        return w
