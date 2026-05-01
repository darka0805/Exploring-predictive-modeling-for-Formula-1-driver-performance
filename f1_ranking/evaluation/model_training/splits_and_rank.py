import numpy as np
import pandas as pd


def walk_forward_split(df: pd.DataFrame,
                       feature_cols: list[str],
                       min_train_races: int = 4,
                       race_col: str = "race_id"):
    """Generate one fold per race; train uses every prior race only.

    Each fold dict carries:
        test_race_id, test_year, test_round, test_event, n_train_races,
        X_train, y_train, X_test, y_test, meta_test, train_groups, test_groups
    """
    races = sorted(df[race_col].unique())
    folds = []

    for test_race in races:
        train_races = [r for r in races if r < test_race]
        if len(train_races) < min_train_races:
            continue

        train_mask = df[race_col].isin(train_races)
        test_mask = df[race_col] == test_race

        folds.append({
            "test_race_id": int(test_race),
            "test_year": int(df.loc[test_mask, "year"].iloc[0]),
            "test_round": int(df.loc[test_mask, "round_num"].iloc[0]),
            "test_event": df.loc[test_mask, "event_name"].iloc[0],
            "n_train_races": len(train_races),
            "X_train": df.loc[train_mask, feature_cols],
            "y_train": df.loc[train_mask, "finish_position"],
            "X_test": df.loc[test_mask, feature_cols],
            "y_test": df.loc[test_mask, "finish_position"],
            "meta_test": df.loc[test_mask, [
                "Driver", "race_id", "year", "round_num", "event_name",
                "grid_position", "finish_position",
            ]].copy(),
            "train_groups": df.loc[train_mask].groupby(race_col).size().values,
            "test_groups": df.loc[test_mask].groupby(race_col).size().values,
        })

    return folds


def rank_within_race(predictions: pd.DataFrame, score_col: str = "pred_score") -> pd.Series:
    """Convert raw model scores into integer ranks 1..N within each race.
    """
    return (
        predictions.groupby("race_id")[score_col]
        .rank(method="min", ascending=True)
        .astype(float)
    )


def impute_train_test(X_train_df: pd.DataFrame, X_test_df: pd.DataFrame):
    """Mean-impute NaNs using training-set means only (no leakage)."""
    X_train = X_train_df.to_numpy(dtype=np.float32, copy=True)
    X_test = X_test_df.to_numpy(dtype=np.float32, copy=True)
    mu = np.nanmean(X_train, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    X_train = np.where(np.isnan(X_train), mu, X_train)
    X_test = np.where(np.isnan(X_test), mu, X_test)
    return X_train, X_test
