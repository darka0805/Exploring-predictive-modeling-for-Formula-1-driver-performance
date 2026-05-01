import pandas as pd


def grid_baseline_predictions(folds: list[dict]) -> pd.DataFrame:
    """Return a predictions DataFrame where pred_position == grid_position.
    """
    rows = []
    for fold in folds:
        meta = fold["meta_test"].copy()
        meta["pred_score"] = meta["grid_position"].astype(float)
        meta["pred_position"] = meta["grid_position"].astype(float)
        rows.append(meta)
    return pd.concat(rows, ignore_index=True)
