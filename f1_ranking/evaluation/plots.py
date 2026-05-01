from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import ARTIFACT_ROOT


def plot_pred_vs_actual_scatter(predictions: pd.DataFrame,
                                save_path: Path | None = None,
                                title: str = "Predicted vs Actual Finishing Position"):
    """Scatter of predicted vs actual finishing position with ±2 band."""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(predictions["finish_position"], predictions["pred_position"],
               alpha=0.4, s=30, color="#2196F3", label="Model prediction")
    ax.plot([0, 21], [0, 21], "k--", alpha=0.5, label="Perfect prediction")
    ax.fill_between([0, 21], [0 - 2, 21 - 2], [0 + 2, 21 + 2],
                    alpha=0.1, color="green", label="+/-2 positions")
    ax.set_xlabel("Actual Finishing Position")
    ax.set_ylabel("Predicted Finishing Position")
    ax.set_title(title)
    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(0.5, 20.5)
    ax.set_aspect("equal")
    ax.legend()
    ax.invert_xaxis()
    ax.invert_yaxis()
    plt.tight_layout()

    if save_path is None:
        save_path = ARTIFACT_ROOT / "pred_vs_actual_2022_2025.png"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved scatter -> {save_path}")
    plt.show()
    return fig


def plot_tabnet_training_curves(tabnet_model, save_path: Path | None = None):
    """Plot TabNet train/val curves from `model.history`. No-op if unavailable."""
    if tabnet_model is None or not hasattr(tabnet_model, "history"):
        print("TabNet model has no `.history` attribute — skipping curve plot.")
        return None

    hist = tabnet_model.history
    if not isinstance(hist, dict) or len(hist) == 0:
        print("TabNet `.history` is empty — skipping curve plot.")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    train_key = next((k for k in hist if "loss" in k.lower() or ("train" in k.lower() and "mae" in k.lower())), None)
    val_key = next((k for k in hist if "val" in k.lower() and ("mae" in k.lower() or "loss" in k.lower())), None)

    if train_key and len(hist[train_key]) > 0:
        axes[0].plot(hist[train_key], color="#2196F3", alpha=0.85, label=train_key)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("TabNet: Training Curve")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    if val_key and len(hist[val_key]) > 0:
        axes[1].plot(hist[val_key], color="#FF9800", alpha=0.85, label=val_key)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("MAE")
    axes[1].set_title("TabNet: Validation Curve")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
