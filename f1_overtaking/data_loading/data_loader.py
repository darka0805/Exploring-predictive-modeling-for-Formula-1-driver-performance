import gc
import pickle

import fastf1
import pandas as pd

from config import CACHE_DIR, DATA_DIR
from feature_engineering.features import build_pair_features

fastf1.Cache.enable_cache(str(CACHE_DIR))



def _compound_weight(compound: str) -> int:
    return {"SOFT": 3, "MEDIUM": 2, "HARD": 1,
            "INTERMEDIATE": 0, "WET": -1}.get(str(compound).upper(), 2)


def _safe_tyre_life(row) -> float:
    val = row.get("TyreLife", 99)
    return 99.0 if pd.isna(val) else float(val)


def _pitting_this_lap(row) -> bool:
    return (pd.notna(row.get("PitInTime", pd.NaT)) or
            pd.notna(row.get("PitOutTime", pd.NaT)))


def _cumulative_pit_done(laps_df, driver: str, up_to_lap: int) -> bool:
    drv = laps_df[(laps_df["Driver"] == driver) &
                  (laps_df["LapNumber"] <= up_to_lap)]
    return bool(drv["PitInTime"].notna().any())



def _build_pairs_for_lap(laps, session, year: int, event_name: str,
                         lap_num: int, sc_laps: set,
                         skip_lap1: bool = True) -> list[dict]:
    if skip_lap1 and lap_num <= 1:
        return []
    if lap_num in sc_laps:
        return []

    lap_data = (laps[laps["LapNumber"] == lap_num]
                .dropna(subset=["Position"])
                .sort_values("Position"))
    if len(lap_data) < 2:
        return []

    drv_list = lap_data["Driver"].tolist()
    pairs = []

    for i in range(1, min(len(drv_list), 15)):
        defender = drv_list[i - 1]
        attacker = drv_list[i]
        try:
            att_row = lap_data[lap_data["Driver"] == attacker].iloc[0]
            def_row = lap_data[lap_data["Driver"] == defender].iloc[0]

            if abs(float(att_row.get("LapNumber", lap_num)) -
                   float(def_row.get("LapNumber", lap_num))) > 0:
                continue
            if _pitting_this_lap(att_row) or _pitting_this_lap(def_row):
                continue

            att_tl = _safe_tyre_life(att_row)
            def_tl = _safe_tyre_life(def_row)
            if att_tl <= 2 or def_tl <= 2:
                continue

            context = {
                "att_FreshTyre": int(att_tl < 5),
                "def_FreshTyre": int(def_tl < 5),
                "att_compound": _compound_weight(att_row.get("Compound", "MEDIUM")),
                "def_compound": _compound_weight(def_row.get("Compound", "MEDIUM")),
                "compound_advantage": (_compound_weight(att_row.get("Compound", "MEDIUM")) -
                                       _compound_weight(def_row.get("Compound", "MEDIUM"))),
                "att_MandatoryPitDone": int(_cumulative_pit_done(laps, attacker, lap_num)),
                "def_MandatoryPitDone": int(_cumulative_pit_done(laps, defender, lap_num)),
                "lap_number": lap_num,
                "att_TyreLife": att_tl,
                "def_TyreLife": def_tl,
            }

            att_tel = att_row.get_telemetry().add_distance()
            def_tel = def_row.get_telemetry().add_distance()
            if len(att_tel) < 50 or len(def_tel) < 50:
                continue

            pair_df = build_pair_features(att_tel, def_tel, context)
            if pair_df is None or len(pair_df) < 60:
                continue

            pairs.append({
                "year": year, "event": event_name,
                "lap": lap_num,
                "attacker": attacker, "defender": defender,
                "df": pair_df, "context": context,
            })
        except Exception:
            continue
    return pairs



def load_race_pairs(year: int, event_name: str,
                     skip_lap1: bool = True) -> list[dict]:
    """Load one race and return a list of pair dicts."""
    cache_path = DATA_DIR / f"pairs_{year}_{event_name.replace(' ', '_')}.pkl"
    if cache_path.exists():
        print(f"  [cache] {year} {event_name}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    print(f"  Loading {year} {event_name} …")
    try:
        session = fastf1.get_session(year, event_name, "R")
        session.load(telemetry=True, weather=False, messages=False)
    except Exception as e:
        print(f"    FAILED: {e}")
        return []

    laps = session.laps
    lap_numbers = sorted(laps["LapNumber"].unique().astype(int))
    try:
        sc_mask = laps["TrackStatus"].str.contains(r"[46]", na=False, regex=True)
        sc_laps = set(laps.loc[sc_mask, "LapNumber"].astype(int).unique())
    except Exception:
        sc_laps = set()

    all_pairs: list[dict] = []
    for lap_num in lap_numbers:
        all_pairs.extend(
            _build_pairs_for_lap(laps, session, year, event_name, lap_num,
                                 sc_laps, skip_lap1=skip_lap1))

    print(f"    → {len(all_pairs)} pairs")

    with open(cache_path, "wb") as f:
        pickle.dump(all_pairs, f, protocol=pickle.HIGHEST_PROTOCOL)

    del session, laps
    gc.collect()
    return all_pairs


def load_multiple_races(years: list[int],
                        tracks: list[str],
                        skip_lap1: bool = True) -> list[dict]:
    """Load pairs for every (year, track) combination."""
    all_pairs: list[dict] = []
    for year in years:
        for track in tracks:
            all_pairs.extend(load_race_pairs(year, track,
                                             skip_lap1=skip_lap1))
    print(f"Total pairs loaded: {len(all_pairs)}")
    return all_pairs
