import numpy as np
import pandas as pd
import fastf1

from config import YEARS, DATA_DIR, CACHE_FILE
from .filters import (
    is_wet_race, exclude_wet_races, apply_107_percent_rule, exclude_sprint_races,
)
from .cache import save_cache, load_cache


def load_event_schedules(years=YEARS, conventional_only: bool = True):
    """Return {year: DataFrame of races} for the given years.

    When `conventional_only=True` we run `exclude_sprint_races()` to drop
    sprint weekends.
    """
    out = {}
    for year in years:
        fastf1.Cache.enable_cache(str(DATA_DIR / str(year) / "cache"))
        schedule = fastf1.get_event_schedule(year)
        races = schedule[schedule["RoundNumber"] > 0]
        if conventional_only:
            races = exclude_sprint_races(races)
        out[year] = races.sort_values("RoundNumber").reset_index(drop=True)
        print(f"{year}: {len(out[year])} races kept")
    return out


def load_session_data(year: int, event_name: str, session_name: str):
    """Load lap-level data from FastF1 (no telemetry)."""
    fastf1.Cache.enable_cache(str(DATA_DIR / str(year) / "cache"))
    session = fastf1.get_session(year, event_name, session_name)
    session.load(telemetry=False, weather=False, messages=False)
    laps = session.laps.copy()
    track_status = session.track_status.copy() if hasattr(session, "track_status") else pd.DataFrame()
    return laps, track_status, session


def _td_to_seconds(td):
    if pd.isna(td):
        return np.nan
    return td.total_seconds()


def prepare_laps_df(laps: pd.DataFrame, year: int, round_num: int, session_name: str, event_name: str) -> pd.DataFrame:
    """Convert FastF1 laps DataFrame into a clean working format with `_s` time columns."""
    cols_needed = [
        "Driver", "DriverNumber", "Team", "LapNumber", "LapTime",
        "Sector1Time", "Sector2Time", "Sector3Time",
        "Compound", "TyreLife", "Stint", "FreshTyre",
        "PitInTime", "PitOutTime", "TrackStatus",
        "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
        "IsPersonalBest", "IsAccurate",
        "LapStartTime", "Time",
    ]
    cols_available = [c for c in cols_needed if c in laps.columns]
    df = laps[cols_available].copy()

    for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time",
                "PitInTime", "PitOutTime", "LapStartTime", "Time"]:
        if col in df.columns:
            df[col + "_s"] = df[col].apply(_td_to_seconds)

    df["year"] = year
    df["round_num"] = round_num
    df["race_id"] = year * 100 + round_num
    df["session"] = session_name
    df["event_name"] = event_name
    df["session_id"] = {"FP1": 1, "FP2": 2, "FP3": 3, "Q": 4, "R": 5}.get(session_name, 0)
    return df


def load_all_laps_and_results(years=YEARS, cache_file=CACHE_FILE, use_cache: bool = True):
    """End-to-end loader: schedules → per-session laps → results → wet/107% filters.

    Returns:
        all_laps:        DataFrame, dry races, 107% Q rule applied
        race_results:    {race_id: DataFrame} of classified results
        track_statuses:  {(race_id, session): track_status DataFrame}
        wet_race_ids:    set of race_ids excluded as wet
    """
    if use_cache and cache_file.exists():
        print(f"Loading from cache: {cache_file}")
        all_laps_list, race_results, track_statuses, wet_race_ids = load_cache(cache_file)
    else:
        all_laps_list, race_results, track_statuses, wet_race_ids = _load_from_fastf1(years)
        save_cache(cache_file, all_laps_list, race_results, track_statuses, wet_race_ids)
        print(f"Cached to {cache_file}")

    all_laps = pd.concat(all_laps_list, ignore_index=True)
    print(f"Total laps loaded: {len(all_laps):,}")
    print(f"Years present: {sorted(all_laps['year'].unique())}")
    print(f"Races with results: {len(race_results)}")
    print(f"Wet race_ids: {sorted(wet_race_ids) if wet_race_ids else 'none'}")

    all_laps, race_results = exclude_wet_races(all_laps, race_results, wet_race_ids)
    print(f"After wet exclusion: {len(race_results)} dry races, {len(all_laps):,} laps")

    all_laps = apply_107_percent_rule(all_laps)
    return all_laps, race_results, track_statuses, wet_race_ids


def _load_from_fastf1(years):
    """Inner loop that hits FastF1 for every (year, race, session)."""
    schedules = load_event_schedules(years, conventional_only=True)

    all_laps_list = []
    race_results = {}
    track_statuses = {}
    wet_race_ids = set()

    for year in years:
        races = schedules[year]
        fastf1.Cache.enable_cache(str(DATA_DIR / str(year) / "cache"))
        print(f"\n{'='*60}\n  LOADING {year}\n{'='*60}")

        for _, race in races.iterrows():
            event_name = race["EventName"]
            round_num = int(race["RoundNumber"])
            race_id = year * 100 + round_num
            print(f"\n--- {year} R{round_num}: {event_name} (race_id={race_id}) ---")

            for sess in ["FP1", "FP2", "FP3", "Q"]:
                try:
                    laps, ts, _ = load_session_data(year, event_name, sess)
                    df = prepare_laps_df(laps, year, round_num, sess, event_name)
                    all_laps_list.append(df)
                    track_statuses[(race_id, sess)] = ts
                    print(f"  {sess}: {len(df)} laps from {df['Driver'].nunique()} drivers")
                except Exception as e:
                    print(f"  {sess}: FAILED - {e}")

            try:
                race_session = fastf1.get_session(year, event_name, "R")
                race_session.load(telemetry=False, weather=False, messages=False)
                results = race_session.results[[
                    "Abbreviation", "DriverNumber", "TeamName",
                    "GridPosition", "Position", "ClassifiedPosition", "Status"
                ]].copy()
                results["year"] = year
                results["round_num"] = round_num
                results["race_id"] = race_id
                results["event_name"] = event_name
                race_results[race_id] = results

                if is_wet_race(race_session.laps):
                    wet_race_ids.add(race_id)
                    print(f"  Race: {len(results)} drivers — *** WET RACE (EXCLUDED) ***")
                else:
                    print(f"  Race: {len(results)} drivers classified (DRY)")
            except Exception as e:
                print(f"  Race results: FAILED - {e}")

    return all_laps_list, race_results, track_statuses, wet_race_ids
