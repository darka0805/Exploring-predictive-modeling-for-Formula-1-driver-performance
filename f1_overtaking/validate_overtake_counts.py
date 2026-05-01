from __future__ import annotations

import warnings

import fastf1
import pandas as pd

from config import (
    ALL_2022_RACES, CACHE_DIR
)
from data_loading.data_loader import load_race_pairs
from data_preprocessing_and_labeling.labeling import _detect_overtake_in_window

warnings.filterwarnings("ignore")
fastf1.Cache.enable_cache(str(CACHE_DIR))

PIRELLI_OFFICIAL_TOTAL = 785


def count_overtakes_for_race(year: int, event_name: str) -> dict:
    """Load one race, count unique overtake events across all pair-laps.

    An overtake is counted once per (attacker, defender, lap) tuple
    where the time-gap crosses zero for at least OVERTAKE_MIN_CONSEC
    consecutive samples.
    """
    pairs = load_race_pairs(year, event_name, skip_lap1=True)
    if not pairs:
        return {"event": event_name, "year": year,
                "pairs": 0, "overtakes": 0, "laps_with_overtake": 0}

    overtake_events: set[tuple] = set()
    laps_with_ot: set[int] = set()

    for p in pairs:
        df = p["df"]
        gap = df["DistanceToDriverAhead"].values

        if _detect_overtake_in_window(gap) == 1:
            key = (p["attacker"], p["defender"], p["lap"])
            if key not in overtake_events:
                overtake_events.add(key)
                laps_with_ot.add(p["lap"])

    return {
        "event": event_name,
        "year": year,
        "pairs": len(pairs),
        "overtakes": len(overtake_events),
        "laps_with_overtake": len(laps_with_ot),
    }


def validate_all_2022_races() -> pd.DataFrame:
    """Run overtake counting across all 22 main races of 2022."""
    print("=" * 70)
    print("OVERTAKE COUNT VALIDATION — 2022 season vs Pirelli official (785)")
    print("=" * 70)

    rows: list[dict] = []
    for event in ALL_2022_RACES:
        print(f"\n--- {event} ---")
        info = count_overtakes_for_race(2022, event)
        rows.append(info)
        print(f"  Pairs: {info['pairs']}  |  Overtakes detected: {info['overtakes']}")

    df = pd.DataFrame(rows)
    total = int(df["overtakes"].sum())

    print(f"\n{'─' * 50}")
    print(f"  Total detected overtakes : {total}")
    print(f"  Pirelli official total   : {PIRELLI_OFFICIAL_TOTAL}")
    print(f"  Difference               : {total - PIRELLI_OFFICIAL_TOTAL} "
          f"({(total - PIRELLI_OFFICIAL_TOTAL) / PIRELLI_OFFICIAL_TOTAL * 100:+.1f}%)")
    print(f"{'─' * 50}")

    
    return df


if __name__ == "__main__":
    validate_all_2022_races()
