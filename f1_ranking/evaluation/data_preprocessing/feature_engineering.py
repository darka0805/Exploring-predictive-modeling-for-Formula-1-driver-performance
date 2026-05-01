import numpy as np
import pandas as pd

from .filters import flag_dnf


GROUP_COLS = ["race_id", "year", "round_num", "Driver", "event_name"]

BASE_METRICS = [
    ("LapTime_s", "min", "best_lap_s"),
    ("LapTime_s", "mean", "avg_lap_s"),
    ("LapTime_s", "std", "std_lap_s"),
    ("LapTime_s", "count", "n_laps"),
    ("Sector1Time_s", "min", "best_s1"),
    ("Sector2Time_s", "min", "best_s2"),
    ("Sector3Time_s", "min", "best_s3"),
    ("SpeedI1", "max", "max_speed_i1"),
    ("SpeedI2", "max", "max_speed_i2"),
    ("SpeedFL", "max", "max_speed_fl"),
    ("SpeedST", "max", "max_speed_st"),
    ("TyreLife", "mean", "avg_tyre_life"),
]

ALLOWED_COMPOUNDS = {"HARD", "MEDIUM", "SOFT", "INTERMEDIATE", "WET"}
SESSIONS_FOR_GAPS = ["Q", "FP1", "FP2", "FP3"]

NON_FEATURE_COLS = [
    "Driver", "event_name", "finish_position", "race_id", "year",
    "round_num", "team", "primary_compound", "track_complexity", "track_type",
]



def aggregate_laps(all_laps: pd.DataFrame) -> pd.DataFrame:
    """One row per (race_id, Driver) with lap-time / speed / tyre summaries."""
    lap_cols = set(all_laps.columns)
    agg_kwargs = {}
    missing = []
    for src, func, out in BASE_METRICS:
        if src in lap_cols:
            agg_kwargs[out] = (src, func)
        else:
            missing.append(out)

    if not agg_kwargs:
        raise ValueError("No numeric lap metrics available for aggregation.")

    agg = all_laps.groupby(GROUP_COLS).agg(**agg_kwargs).reset_index()

    for out in missing:
        if out not in agg.columns:
            if out == "n_laps":
                counts = all_laps.groupby(GROUP_COLS).size().reset_index(name="n_laps")
                agg = agg.merge(counts, on=GROUP_COLS, how="left")
            else:
                agg[out] = np.nan
    return agg



def add_team_and_tyre(agg: pd.DataFrame, all_laps: pd.DataFrame) -> pd.DataFrame:
    """Attach `team` (mode of Team) and `primary_compound` (mode of allowed Compound)."""
    if "Team" in all_laps.columns:
        team_agg = (
            all_laps.dropna(subset=["Team"])
            .groupby(["race_id", "Driver", "event_name"])["Team"]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
            .reset_index(name="team")
        )
        agg = agg.merge(team_agg, on=["race_id", "Driver", "event_name"], how="left")
    else:
        agg["team"] = np.nan

    if "Compound" in all_laps.columns:
        valid = all_laps.assign(
            Compound_valid=all_laps["Compound"].where(all_laps["Compound"].isin(ALLOWED_COMPOUNDS))
        )
        tyre_pref = (
            valid.dropna(subset=["Compound_valid"])
            .groupby(["race_id", "Driver", "event_name"])["Compound_valid"]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else x.iloc[0])
            .reset_index(name="primary_compound")
        )
        agg = agg.merge(tyre_pref, on=["race_id", "Driver", "event_name"], how="left")
    else:
        agg["primary_compound"] = np.nan

    return agg



def add_session_gaps(agg: pd.DataFrame, all_laps: pd.DataFrame) -> pd.DataFrame:
    """For each (race, driver) add `<sess>_best_s` and `<sess>_gap_pct`.

    `<sess>_gap_pct` = 100 * (driver_best - session_best) / session_best
    """
    for sess in SESSIONS_FOR_GAPS:
        sess_laps = all_laps[all_laps["session"] == sess]
        if len(sess_laps) == 0 or "LapTime_s" not in sess_laps.columns:
            continue
        prefix = sess.lower()

        sess_best = sess_laps.groupby("race_id")["LapTime_s"].min().reset_index(name="sess_best")
        driver_best = (
            sess_laps.groupby(["race_id", "Driver"])["LapTime_s"]
            .min()
            .reset_index(name=f"{prefix}_best_s")
        )
        driver_best = driver_best.merge(sess_best, on="race_id", how="left")
        driver_best[f"{prefix}_gap_pct"] = (
            (driver_best[f"{prefix}_best_s"] - driver_best["sess_best"])
            / driver_best["sess_best"] * 100
        )
        agg = agg.merge(
            driver_best[["race_id", "Driver", f"{prefix}_best_s", f"{prefix}_gap_pct"]],
            on=["race_id", "Driver"],
            how="left",
        )
    return agg



def merge_with_results(agg: pd.DataFrame, race_results: dict) -> pd.DataFrame:
    """Inner-join aggregated features with classified race results.

    Adds: finish_position (target), grid_position, team_result, is_dnf.
    Drops drivers with no classified position.
    """
    rows = []
    for _, res in race_results.items():
        r = res.copy()
        r["Driver"] = r["Abbreviation"]
        r["finish_position"] = pd.to_numeric(r["Position"], errors="coerce")
        r["grid_position"] = pd.to_numeric(r["GridPosition"], errors="coerce")
        r["team_result"] = r["TeamName"]
        r["is_dnf"] = flag_dnf(r)
        rows.append(r[["Driver", "race_id", "finish_position", "grid_position",
                       "event_name", "team_result", "is_dnf"]])
    results_df = pd.concat(rows, ignore_index=True)

    dataset = agg.merge(results_df, on=["race_id", "Driver", "event_name"], how="inner")
    dataset = dataset[dataset["finish_position"].notna()]
    dataset = dataset.sort_values(["race_id", "finish_position"]).reset_index(drop=True)
    return dataset



def encode_team_and_tyre(dataset: pd.DataFrame) -> pd.DataFrame:
    """OHE `team` and `primary_compound`. Falls back to results-team if missing."""
    if "team_result" in dataset.columns:
        dataset["team"] = dataset["team"].fillna(dataset["team_result"])
        dataset = dataset.drop(columns=["team_result"])

    team_ohe = pd.get_dummies(dataset["team"], prefix="team", dummy_na=True, dtype=np.int8)
    tyre_ohe = pd.get_dummies(dataset["primary_compound"], prefix="tyre", dummy_na=True, dtype=np.int8)
    dataset = pd.concat([dataset, team_ohe, tyre_ohe], axis=1)

    for col in ["team_nan", "tyre_nan"]:
        if col in dataset.columns:
            dataset = dataset.drop(columns=[col])
    return dataset


def add_recent_form(dataset: pd.DataFrame) -> pd.DataFrame:
    """Add per-driver rolling history features.

    Columns added:
        recent3_avg_finish    mean finishing pos over last 3 races
        recent3_avg_grid      mean grid pos over last 3 races
        recent5_avg_finish    mean finishing pos over last 5 races
        recent3_pos_delta     mean (grid - finish) over last 3 races
                              i.e. positions gained on race day

    Only past races (race_id < current) are used → no leakage.
    """
    rows = []
    races_sorted = sorted(dataset["race_id"].unique())

    for rid in races_sorted:
        current = dataset[dataset["race_id"] == rid]
        past = dataset[dataset["race_id"] < rid]
        if current.empty:
            continue

        for _, row in current.iterrows():
            driver = row["Driver"]
            rec = {"race_id": rid, "Driver": driver}
            drv_past = past[past["Driver"] == driver].sort_values("race_id", ascending=False)

            if len(drv_past) >= 1:
                last3 = drv_past.head(3)
                last5 = drv_past.head(5)
                rec["recent3_avg_finish"] = last3["finish_position"].mean()
                rec["recent3_avg_grid"] = last3["grid_position"].mean()
                rec["recent5_avg_finish"] = last5["finish_position"].mean()
                rec["recent3_pos_delta"] = (last3["grid_position"] - last3["finish_position"]).mean()
            else:
                rec["recent3_avg_finish"] = np.nan
                rec["recent3_avg_grid"] = np.nan
                rec["recent5_avg_finish"] = np.nan
                rec["recent3_pos_delta"] = np.nan
            rows.append(rec)

    hist_df = pd.DataFrame(rows)
    return dataset.merge(hist_df, on=["race_id", "Driver"], how="left")



def add_fp2_avg_lap(dataset: pd.DataFrame, all_laps: pd.DataFrame) -> pd.DataFrame:
    """Adds `fp2_avg_lap_s`: mean lap time during FP2 for each (race, driver).

    FP2 long-run pace is a strong proxy for race pace, so this is a
    high-value single feature.
    """
    fp2_laps = all_laps[all_laps["session"] == "FP2"]
    if len(fp2_laps) == 0 or "LapTime_s" not in fp2_laps.columns:
        dataset["fp2_avg_lap_s"] = np.nan
        print("WARNING: no FP2 lap data available")
        return dataset

    fp2_avg = (
        fp2_laps.groupby(["race_id", "Driver"])["LapTime_s"]
        .mean()
        .reset_index(name="fp2_avg_lap_s")
    )
    out = dataset.merge(fp2_avg, on=["race_id", "Driver"], how="left")
    nan_rate = out["fp2_avg_lap_s"].isna().mean() * 100
    print(f"FP2 avg lap time added. NaN rate: {nan_rate:.1f}%")
    return out

def compute_feature_cols(df: pd.DataFrame) -> list[str]:
    """Numeric columns that should be used as model inputs."""
    return [
        c for c in df.columns
        if c not in NON_FEATURE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]



def build_dataset(all_laps: pd.DataFrame, race_results: dict):
    """End-to-end feature engineering pipeline.

    Returns:
        dataset:      DataFrame with one row per (race_id, Driver) and all features.
        feature_cols: list of column names to feed into models.
    """
    agg = aggregate_laps(all_laps)
    agg = add_team_and_tyre(agg, all_laps)
    agg = add_session_gaps(agg, all_laps)

    dataset = merge_with_results(agg, race_results)
    dataset = encode_team_and_tyre(dataset)
    dataset = add_recent_form(dataset)
    dataset = add_fp2_avg_lap(dataset, all_laps)

    feature_cols = compute_feature_cols(dataset)
    print(f"Dataset shape: {dataset.shape}")
    print(f"Features: {len(feature_cols)}")
    print(f"Races: {dataset['race_id'].nunique()}, "
          f"years={sorted(dataset['year'].unique())}")
    return dataset, feature_cols
