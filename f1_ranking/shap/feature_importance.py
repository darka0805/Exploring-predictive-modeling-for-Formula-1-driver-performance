from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


_MODEL_COLORS = {
    "ExtraTrees": "#1976D2",
    "XGBoost":    "#2E7D32",
    "TabNet":     "#FF5722",
}


def _plot_one(model, model_name: str, feature_cols: list[str],
              top_n: int = 25, save_path: Path | None = None) -> pd.DataFrame | None:
    """Bar plot for a single model. Returns sorted importance DataFrame or None."""
    if model is None or not hasattr(model, "feature_importances_"):
        print(f"[{model_name}] no feature_importances_ on model — skipping")
        return None

    imp = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    n = min(top_n, len(imp))
    top = imp.head(n)

    color = _MODEL_COLORS.get(model_name, "#555555")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(n), top["importance"].values, color=color, alpha=0.9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(top["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance")
    ax.set_title(f"Top {n} Feature Importances — {model_name}")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[{model_name}] saved → {save_path}")
    plt.show()

    print(f"\n[{model_name}] Top 15 features:")
    print(imp.head(15).to_string(index=False))
    return imp


def plot_extratrees_importance(model, feature_cols, top_n=25, save_path=None):
    """Convenience wrapper for ExtraTrees."""
    return _plot_one(model, "ExtraTrees", feature_cols, top_n=top_n, save_path=save_path)


def plot_xgboost_importance(model, feature_cols, top_n=25, save_path=None):
    """Convenience wrapper for XGBoost."""
    return _plot_one(model, "XGBoost", feature_cols, top_n=top_n, save_path=save_path)


def plot_tabnet_importance(model, feature_cols, top_n=25, save_path=None):
    """Convenience wrapper for TabNet."""
    return _plot_one(model, "TabNet", feature_cols, top_n=top_n, save_path=save_path)


def plot_all_feature_importances(models: dict,
                                 feature_cols: list[str],
                                 save_dir: Path | None = None,
                                 top_n: int = 25) -> dict[str, pd.DataFrame]:
    """Bar plots for every model in `models`.

    `models` is a dict like {"ExtraTrees": et_model, "XGBoost": xgb_model, "TabNet": tn_model}.
    Saves one PNG per model under `save_dir/<model_lowercase>_feature_importance.png`.
    Returns {model_name: importance_df}.
    """
    out: dict[str, pd.DataFrame] = {}
    for name, model in models.items():
        save_path = None
        if save_dir is not None:
            save_path = save_dir / f"{name.lower()}_feature_importance.png"
        imp = _plot_one(model, name, feature_cols, top_n=top_n, save_path=save_path)
        if imp is not None:
            out[name] = imp
    return out
