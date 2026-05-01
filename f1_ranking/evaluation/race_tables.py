from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_race_predictions(predictions: pd.DataFrame):
    """Print per-driver Pred / Grid / Actual / Error for every race."""
    for rid in sorted(predictions["race_id"].unique()):
        race = predictions[predictions["race_id"] == rid].sort_values("pred_position")
        year = int(race["year"].iloc[0])
        rnd = int(race["round_num"].iloc[0])
        event = race["event_name"].iloc[0]

        print("\n" + "=" * 60)
        print(f"{year} R{rnd}: {event}")
        print("=" * 60)
        print(f'{"Pred":>4s} {"Driver":<6s} {"Grid":>4s} {"Actual":>6s} {"Error":>6s}')
        print("-" * 30)
        for _, row in race.iterrows():
            err = int(row["pred_position"] - row["finish_position"])
            err_str = f"{err:+d}" if err != 0 else "  OK"
            print(f"P{int(row['pred_position']):>2d}  {row['Driver']:<6s} "
                  f"G{int(row['grid_position']):>2d}  "
                  f"P{int(row['finish_position']):>2d}    {err_str}")


def print_race_prediction_table(pred_df: pd.DataFrame,
                                year: int = 2025,
                                event_keyword: str = "Abu Dhabi",
                                save_dir: Path | None = None,
                                limit: int | None = None):
    """Detailed table for one race, plus optional saved plots/CSV.

    Returns the race-only DataFrame (sorted by pred_position) or None if no match.
    """
    mask = (pred_df["year"] == year) & (
        pred_df["event_name"].str.contains(event_keyword, case=False, na=False)
        | pred_df["event_name"].str.contains(r"yas|marina", case=False, na=False, regex=True)
    )
    if mask.any():
        race_id = int(pred_df.loc[mask, "race_id"].max())
    else:
        yr = pred_df[pred_df["year"] == year]
        if yr.empty:
            print(f"No rows for year {year} in predictions.")
            return None
        race_id = int(yr["race_id"].max())
        print(f"No name match for {event_keyword!r}; using last {year} race (race_id={race_id}).")

    race = (pred_df[pred_df["race_id"] == race_id]
            .copy()
            .sort_values("pred_position")
            .reset_index(drop=True))
    rnd = int(race["round_num"].iloc[0])
    event = race["event_name"].iloc[0]

    print("\n" + "=" * 60)
    print(f"{year} R{rnd}: {event}")
    print("=" * 60)
    print(f'{"Pred":<5s} {"Driver":<6s} {"Grid":<5s} {"Actual":<6s} {"Error":>6s}')
    print("-" * 30)

    view = race if limit is None else race.head(limit)
    for _, row in view.iterrows():
        pred = int(round(row["pred_position"]))
        grid = int(round(row["grid_position"]))
        actual = int(round(row["finish_position"]))
        err = pred - actual
        err_str = "OK" if err == 0 else f"{err:+d}"
        print(f"P{pred:>2d}  {str(row['Driver']):<6s} G{grid:>2d}  P{actual:>2d}    {err_str:>4s}")

    if save_dir is not None:
        _save_race_artifacts(race, save_dir, event_slug=event_keyword.lower().replace(" ", "_"), year=year)

    return race


def _save_race_artifacts(race: pd.DataFrame, save_dir: Path, event_slug: str, year: int):
    """CSV + dumbbell plot + scatter of predicted vs actual for a single race."""
    save_dir.mkdir(parents=True, exist_ok=True)
    csv_path = save_dir / f"{event_slug}_{year}_predictions_table.csv"
    race.to_csv(csv_path, index=False)

    order = race.sort_values("pred_position").reset_index(drop=True)
    y = np.arange(len(order))

    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(order))))
    ax.hlines(y, order["pred_position"], order["finish_position"],
              color="#9E9E9E", alpha=0.6, linewidth=2)
    ax.scatter(order["pred_position"], y, color="#1E88E5", s=70, label="Predicted")
    ax.scatter(order["finish_position"], y, color="#E53935", s=70, label="Actual")
    ax.set_yticks(y)
    ax.set_yticklabels(order["Driver"])
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("Finishing Position (lower is better)")
    ax.set_title(f'{order["event_name"].iloc[0]} {year}: Predicted vs Actual Positions')
    ax.grid(axis="x", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_dir / f"{event_slug}_{year}_pred_vs_actual.png", dpi=160, bbox_inches="tight")
    plt.show()

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(race["finish_position"], race["pred_position"],
               alpha=0.75, s=80, color="#2196F3", edgecolors="k", linewidths=0.3, label="Drivers")
    ax.plot([0, 21], [0, 21], "k--", alpha=0.5, label="Perfect prediction")
    ax.fill_between([0, 21], [0 - 2, 21 - 2], [0 + 2, 21 + 2],
                    alpha=0.1, color="green", label="+/-2 positions")
    for _, row in race.iterrows():
        ax.annotate(str(row["Driver"]), (row["finish_position"], row["pred_position"]),
                    textcoords="offset points", xytext=(4, 4), fontsize=8, alpha=0.85)
    rnd = int(race["round_num"].iloc[0])
    evt = race["event_name"].iloc[0]
    ax.set_xlabel("Actual Finishing Position")
    ax.set_ylabel("Predicted Finishing Position")
    ax.set_title(f"Predicted vs Actual — {evt} ({year} R{rnd})")
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_dir / f"{event_slug}_{year}_pred_vs_actual_scatter.png", dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved race artifacts -> {save_dir}")
