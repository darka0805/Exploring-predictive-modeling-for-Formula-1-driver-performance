from __future__ import annotations

from pathlib import Path

import pandas as pd

from experiments.e7_config import NEG_POS_RATIO


def markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        vals = [str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_report(report_path: Path,
                 class_stats: pd.DataFrame,
                 sweep_summary: pd.DataFrame,
                 top_trials: pd.DataFrame,
                 stacking_df: pd.DataFrame,
                 final_df: pd.DataFrame,
                 full_rerun_df: pd.DataFrame,
                 plots: dict[str, str]):
    lines: list[str] = []
    lines.append("# 2025 Holdout Experiments (2022-2024 Train, 2025 Test)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("- Track: Abu Dhabi Grand Prix")
    lines.append("- Train years: 2022, 2023, 2024")
    lines.append("- Test year: 2025")
    lines.append(f"- Train balancing: successful:unsuccessful = 1:{NEG_POS_RATIO}")
    lines.append("- Model families used: XGBoost, LightGBM, RandomForest, CNN, BiGRU")
    lines.append("")

    lines.append("## Class Distribution")
    lines.append("")
    lines.append(markdown_table(class_stats))
    lines.append("")

    lines.append("## Sweep Coverage")
    lines.append("")
    lines.append(markdown_table(sweep_summary))
    lines.append("")

    lines.append("## Top Validation Trials (per model)")
    lines.append("")
    lines.append(markdown_table(top_trials))
    lines.append("")

    lines.append("## Final 2025 Test Results (Base Models)")
    lines.append("")
    lines.append(markdown_table(final_df))
    lines.append("")

    lines.append("## Stacking Ensembles")
    lines.append("")
    if len(stacking_df) == 0:
        lines.append("No stacking results generated.")
    else:
        lines.append(markdown_table(stacking_df))
    lines.append("")

    lines.append("## Best Models Rerun On Full 2022-2024 Raw Train Data")
    lines.append("")
    if len(full_rerun_df) == 0:
        lines.append("No full-data reruns generated.")
    else:
        lines.append(markdown_table(full_rerun_df))
    lines.append("")

    lines.append("## Generated Plots")
    lines.append("")
    for name, rel in plots.items():
        lines.append(f"- {name}: {rel}")

    report_path.write_text("\n".join(lines), encoding="utf-8")
