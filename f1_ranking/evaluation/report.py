from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from config import ARTIFACT_ROOT, YEARS


def _fmt(v) -> str:
    """Format numeric value to 4 decimals or 'nan' if missing."""
    return "nan" if pd.isna(v) else f"{v:.4f}"


def write_validation_report(
    dataset: pd.DataFrame,
    folds: list,
    base_eval: pd.DataFrame,
    pca_eval: pd.DataFrame | None,
    pca_components: list[int] | None,
    saved_model_files: list[str],
    report_path: Path | None = None,
) -> Path:
    """Generate a markdown validation report summarizing model performance and experiment setup.
    
    Includes dataset split info, baseline metrics, optional PCA ablation results,
    and paths to saved artifacts (SHAP outputs, model files, race views).
    """
    report_path = report_path or ARTIFACT_ROOT / "BASEMODEL_VALIDATION_REPORT_2022_2025.md"
    dataset_years = sorted(int(y) for y in dataset["year"].dropna().unique())

    base_summary = {
        "spearman": float(base_eval["spearman_model"].mean()),
        "mae": float(base_eval["mae_model"].mean()),
        "ndcg": float(base_eval["ndcg_model"].mean()),
    }

    lines = [
        "# Basemodel Validation Report (2022-2025)",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Years: {YEARS}",
        f"Dataset contains: {dataset_years}",
        f"Walk-forward splits: {len(folds)}",
        "",
        "## ExtraTrees Performance",
        f"Spearman: {_fmt(base_summary['spearman'])}",
        f"MAE: {_fmt(base_summary['mae'])}",
        f"NDCG: {_fmt(base_summary['ndcg'])}",
        "",
    ]

    if pca_eval is not None and pca_components:
        pca_summary = {
            "spearman": float(pca_eval["spearman_model"].mean()),
            "mae": float(pca_eval["mae_model"].mean()),
            "ndcg": float(pca_eval["ndcg_model"].mean()),
        }
        avg_n = np.mean(pca_components)
        min_n = min(pca_components)
        max_n = max(pca_components)
        lines += [
            "## PCA Ablation (95% variance)",
            f"Components: {avg_n:.1f} ± ({min_n}–{max_n})",
            f"Spearman: {_fmt(pca_summary['spearman'])}",
            f"MAE: {_fmt(pca_summary['mae'])}",
            f"NDCG: {_fmt(pca_summary['ndcg'])}",
            "",
        ]

    lines += [
        "## Outputs",
        f"SHAP: {ARTIFACT_ROOT / 'shap'}",
    ]

    if saved_model_files:
        lines.append("Models:")
        lines.extend(f"  {p}" for p in saved_model_files)
    
    lines += [
        f"Race views: {ARTIFACT_ROOT / 'race_views'}",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"✓ Report: {report_path}")
    return report_path
