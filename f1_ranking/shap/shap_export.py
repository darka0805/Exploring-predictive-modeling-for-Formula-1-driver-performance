from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ARTIFACT_ROOT


def _ensure_shap():
    try:
        import shap
    except Exception:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "shap"])
        import shap
    return shap


def _safe_name(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_")


def _coerce_shap_array(shap_values):
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        arr = arr.mean(axis=0)
    return arr


def export_tree_shap(model,
                     model_name: str,
                     x_sample_df: pd.DataFrame,
                     shap_dir: Path | None = None) -> pd.DataFrame | None:
    """TreeSHAP for any tree model (ExtraTrees, XGBoost). Saves CSV + PNGs.

    Returns mean |SHAP| per feature, sorted descending. None if SHAP fails.
    """
    shap = _ensure_shap()
    if shap_dir is None:
        shap_dir = ARTIFACT_ROOT / "shap"
    shap_dir.mkdir(parents=True, exist_ok=True)
    tag = _safe_name(model_name)

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_sample_df)
        shap_arr = _coerce_shap_array(shap_values)
        if shap_arr.ndim != 2 or shap_arr.shape[1] != x_sample_df.shape[1]:
            raise ValueError(f"Unexpected SHAP shape {shap_arr.shape}")

        mean_abs = np.abs(shap_arr).mean(axis=0)
        imp = pd.DataFrame({
            "model": model_name,
            "feature": x_sample_df.columns,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        imp.to_csv(shap_dir / f"{tag}_mean_abs_shap.csv", index=False)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_arr, x_sample_df,
            plot_type="bar",
            max_display=min(25, x_sample_df.shape[1]),
            show=False,
        )
        plt.title(f"{model_name}: SHAP Mean |Value|")
        plt.tight_layout()
        plt.savefig(shap_dir / f"{tag}_shap_bar.png", dpi=160, bbox_inches="tight")
        plt.show()

        np.savez_compressed(
            shap_dir / f"{tag}_shap_sample_matrix.npz",
            shap_values=shap_arr.astype(np.float64),
            feature_names=x_sample_df.columns.astype(str).to_numpy(),
            x_sample=x_sample_df.to_numpy(dtype=np.float64),
        )

        long_rows = []
        for si in range(shap_arr.shape[0]):
            for fi, fn in enumerate(x_sample_df.columns):
                long_rows.append({
                    "sample_idx": si,
                    "feature": str(fn),
                    "shap": float(shap_arr[si, fi]),
                    "feature_value": float(x_sample_df.iloc[si, fi]),
                })
        pd.DataFrame(long_rows).to_csv(
            shap_dir / f"{tag}_shap_per_sample_long.csv", index=False
        )

        per_feat_dir = shap_dir / "per_feature" / tag
        per_feat_dir.mkdir(parents=True, exist_ok=True)
        for fi, fname in enumerate(x_sample_df.columns):
            fn_safe = _safe_name(str(fname)) or f"f{fi}"
            try:
                plt.figure(figsize=(7, 5))
                shap.dependence_plot(fi, shap_arr, x_sample_df, show=False)
                plt.tight_layout()
                plt.savefig(per_feat_dir / f"{fi:03d}_{fn_safe}.png",
                            dpi=120, bbox_inches="tight")
                plt.close()
            except Exception as ex:
                print(f"SHAP dependence skipped for {fname}: {ex}")

        print(f"SHAP exported for {model_name} -> {shap_dir}")
        return imp
    except Exception as e:
        print(f"SHAP failed for {model_name}: {e}")
        return None


def export_all_shap(models: dict,
                    dataset: pd.DataFrame,
                    feature_cols: list[str],
                    sample_n: int = 400,
                    seed: int = 42,
                    shap_dir: Path | None = None) -> pd.DataFrame:
    """Run TreeSHAP for every tree model in `models` and concatenate the tables.

    `models` is `{"ExtraTrees": fitted_et, "XGBoost": fitted_xgb}`.
    """
    if shap_dir is None:
        shap_dir = ARTIFACT_ROOT / "shap"

    x_full = dataset[feature_cols].apply(pd.to_numeric, errors="coerce")
    x_full = x_full.fillna(x_full.mean(numeric_only=True)).fillna(0.0)
    n = min(sample_n, len(x_full))
    x_sample = x_full.sample(n, random_state=seed) if n < len(x_full) else x_full.copy()

    tables = []
    for name, model in models.items():
        if model is None:
            continue
        imp = export_tree_shap(model, name, x_sample, shap_dir=shap_dir)
        if imp is not None:
            tables.append(imp)

    if tables:
        out = pd.concat(tables, ignore_index=True)
        out.to_csv(shap_dir / "shap_all_models_feature_importance.csv", index=False)
        return out
    return pd.DataFrame()
