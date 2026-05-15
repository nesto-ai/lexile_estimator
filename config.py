"""Configuration for the TO Lexile-like calculator."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LevelRange:
    level: str
    low: float
    high: float

    @property
    def midpoint(self) -> float:
        return (self.low + self.high) / 2.0

    @property
    def width(self) -> float:
        return self.high - self.low


TO_CONTENT_RANGES: dict[str, LevelRange] = {
    "T": LevelRange("T", 600.0, 700.0),
    "H": LevelRange("H", 700.0, 800.0),
    "E": LevelRange("E", 800.0, 1000.0),
    "O": LevelRange("O", 1000.0, 1100.0),
    "P": LevelRange("P", 1100.0, 1250.0),
    "N": LevelRange("N", 1250.0, 1600.0),
}

TO_OFFICIAL_RANGES: dict[str, LevelRange] = {
    "T": LevelRange("T", 700.0, 800.0),
    "H": LevelRange("H", 800.0, 900.0),
    "E": LevelRange("E", 900.0, 1100.0),
    "O": LevelRange("O", 1100.0, 1300.0),
    "P": LevelRange("P", 1300.0, 1600.0),
    "N": LevelRange("N", 1300.0, 1600.0),
}

# Empirical offsets from TO level-doc validation. These are content-dev scale
# corrections for already-known levels; P/N did not have local calibration data.
TO_LEVEL_OFFSETS: dict[str, float] = {
    "T": 11.0,
    "H": -102.5,
    "E": 0.0,
    "O": -173.0,
    "P": 0.0,
    "N": 0.0,
}

# A conservative pure-text fallback when the TO level is unknown. It is derived
# from the observed global-model shift on the TO level-doc set and should be
# treated as a candidate-ranking aid, not as an official Lexile transformation.
TO_GLOBAL_OFFSET = -220.0

LEVEL_ORDER = ("T", "H", "E", "O", "P", "N")

MODEL_VERSION = "to-lexile-calc-v0.1"

