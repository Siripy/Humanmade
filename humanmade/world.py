"""A small apartment: the environment the human lives in.

Water is on tap, but food is finite — the human depends on its companion
(you) to restock groceries, which gives it a real reason to start
conversations instead of just answering them.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class World:
    food_portions: int = 6          # meals in the fridge
    room_temp_c: float = 21.0
    air_quality: float = 100.0      # future: windows, smoke, etc.

    def describe(self) -> str:
        if self.food_portions <= 0:
            fridge = "The fridge is EMPTY — no food left. Only your companion can restock it."
        elif self.food_portions <= 2:
            fridge = f"The fridge is nearly empty ({self.food_portions} portions left). You should ask your companion to buy groceries."
        else:
            fridge = f"The fridge holds {self.food_portions} portions of food."
        return (f"{fridge} Water runs from the tap. The room is {self.room_temp_c:.0f}°C. "
                "You have a bed, toilet, shower, some books and music, and dumbbells.")

    def take_meal(self) -> bool:
        if self.food_portions <= 0:
            return False
        self.food_portions -= 1
        return True

    def restock(self, portions: int = 8) -> int:
        self.food_portions += portions
        return self.food_portions

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "World":
        w = cls()
        for k, v in d.items():
            if hasattr(w, k):
                setattr(w, k, v)
        return w
