import numpy as np
import pandas as pd

from config import QUALIFYING_107_PCT


def exclude_sprint_races(races: pd.DataFrame) -> pd.DataFrame:
    """Drop sprint weekends; keep rows with EventFormat == 'conventional'.

    Falls through unchanged if the EventFormat column is absent.
    """
    if "EventFormat" not in races.columns:
        return races
    return races[races["EventFormat"] == "conventional"].reset_index(drop=True)


def is_wet_race(race_laps: pd.DataFrame) -> bool:
    """A race is wet if 3+ drivers used INTERMEDIATE/WET tyres at any point."""
    wet_compounds = {"INTERMEDIATE", "WET"}
    if "Compound" not in race_laps.columns:
        return False
    drivers_on_wets = race_laps[race_laps["Compound"].isin(wet_compounds)]["Driver"].nunique()
    return drivers_on_wets >= 3


def exclude_wet_races(all_laps: pd.DataFrame, race_results: dict, wet_race_ids: set):
    """Drop wet race_ids from both laps and results in place-style (returns new objects)."""
    if not wet_race_ids:
        return all_laps, race_results
    laps = all_laps[~all_laps["race_id"].isin(wet_race_ids)].copy()
    results = {rid: r for rid, r in race_results.items() if rid not in wet_race_ids}
    return laps, results


def apply_107_percent_rule(all_laps: pd.DataFrame) -> pd.DataFrame:
    """Drop qualifying laps slower than 1.07x the session best per race.

    Non-qualifying laps are passed through unchanged.
    """
    needed = {"session", "LapTime_s"}
    if not needed.issubset(all_laps.columns):
        print("107% rule skipped: required columns missing")
        return all_laps

    q_laps = all_laps[all_laps["session"] == "Q"].copy()
    other_laps = all_laps[all_laps["session"] != "Q"].copy()

    filtered = []
    for _, grp in q_laps.groupby("race_id"):
        best = grp["LapTime_s"].min()
        if pd.notna(best):
            mask = grp["LapTime_s"].isna() | (grp["LapTime_s"] <= best * QUALIFYING_107_PCT)
            filtered.append(grp[mask])
        else:
            filtered.append(grp)

    if filtered:
        q_filtered = pd.concat(filtered, ignore_index=True)
        removed = len(q_laps) - len(q_filtered)
        if len(q_laps) > 0:
            pct = removed / len(q_laps) * 100
            print(f"107% rule: removed {removed} Q laps ({pct:.1f}%)")
    else:
        q_filtered = q_laps

    return pd.concat([other_laps, q_filtered], ignore_index=True)


def flag_dnf(results: pd.DataFrame) -> pd.Series:
    """Boolean Series — True for drivers who did NOT finish."""
    status = results["Status"].astype(str)
    finished = (
        status.str.contains("Finished", na=False)
        | status.str.match(r"^\+\d", na=False)
        | (status == "Lapped")
    )
    return ~finished
