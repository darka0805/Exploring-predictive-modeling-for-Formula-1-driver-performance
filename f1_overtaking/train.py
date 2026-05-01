import pickle
from pathlib import Path

import numpy as np
import torch

from config import SEED, MODEL_DIR
from evaluation.evaluate import compute_metrics, print_metrics, plot_pr_roc, plot_confusion, save_metrics


def save_model_weights(model, tag: str, save_dir: Path | None = None) -> Path:
    """Persist model weights to saved_models/.

    Tree models (XGBoost, LGBM, RF) are pickled.
    DL models (CNN, BiGRU, End2Race) are saved via torch.save.
    Returns the path where weights were written.
    """
    save_dir = save_dir or MODEL_DIR
    save_dir.mkdir(exist_ok=True)
    safe_tag = tag.replace(" ", "_").replace("/", "_")

    if hasattr(model, "model") and isinstance(model.model, torch.nn.Module):
        path = save_dir / f"{safe_tag}.pt"
        torch.save(model.model.state_dict(), path)
    elif hasattr(model, "model"):
        path = save_dir / f"{safe_tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model.model, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        path = save_dir / f"{safe_tag}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Weights saved → {path}")
    return path


def train_and_evaluate(model_cls,
                       X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       experiment_name: str,
                       model_extra: dict | None = None,
                       save_plots: bool = True,
                       persist_metrics: bool = True,
                       save_weights: bool = False) -> dict:
    """Train one model on the provided split and return test metrics."""
    np.random.seed(SEED)

    model = model_cls(extra_params=model_extra)
    label = f"{experiment_name}/{model.name}"
    print(f"\n{'=' * 60}")
    print(f"Training: {label}")
    print(f"  Train: {X_train.shape[0]} samples "
          f"({y_train.sum()} pos, {(y_train == 0).sum()} neg)")
    print(f"  Test : {X_test.shape[0]} samples "
          f"({y_test.sum()} pos, {(y_test == 0).sum()} neg)")
    print(f"{'=' * 60}")

    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    y_prob = model.predict_proba(X_test)
    metrics = compute_metrics(y_test, y_prob)
    print_metrics(metrics, label)

    if save_plots and not np.isnan(metrics.get("roc_auc", float("nan"))):
        plot_pr_roc(y_test, y_prob, label)
        y_pred = (y_prob >= metrics["threshold"]).astype(int)
        plot_confusion(y_test, y_pred, label)

    if persist_metrics:
        save_metrics(metrics, experiment_name, model.name)

    if save_weights:
        save_model_weights(model, label)

    return {"metrics": metrics, "model": model, "y_prob": y_prob}
