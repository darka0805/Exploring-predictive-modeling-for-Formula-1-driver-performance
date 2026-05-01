from __future__ import annotations

from pathlib import Path

import numpy as np

from config import DATA_DIR, TEMPORAL_FEATURES
from data_preprocessing_and_labeling.dataset_builder import load_dataset
from feature_engineering.end2race_preprocessing import End2RacePreprocessor


def preprocess_arrays(X_train: np.ndarray,
                      X_test: np.ndarray | None = None,
                      feature_order: list[str] | None = None
                      ) -> dict:
    """Fit on X_train, transform train (and optionally test).

    Returns dict with three streams per split:
      train: (dist, speed, other)
      test : (dist, speed, other)  if X_test provided
    """
    feature_order = feature_order or list(TEMPORAL_FEATURES)
    pre = End2RacePreprocessor(feature_order=feature_order)
    pre.fit(X_train)

    out = {"preprocessor": pre,
           "train": pre.transform(X_train)}
    if X_test is not None:
        out["test"] = pre.transform(X_test)
    return out


def preprocess_dataset_file(train_path: Path,
                            test_path: Path | None = None) -> dict:
    """Load pickled episode datasets and run End2Race preprocessing."""
    train_ds = load_dataset(train_path)
    test_ds = load_dataset(test_path) if test_path is not None else None

    out = preprocess_arrays(
        train_ds["X"],
        X_test=test_ds["X"] if test_ds is not None else None,
    )
    out["y_train"] = train_ds["y"]
    out["meta_train"] = train_ds["meta"]
    if test_ds is not None:
        out["y_test"] = test_ds["y"]
        out["meta_test"] = test_ds["meta"]
    return out


def describe_streams(streams: tuple[np.ndarray, np.ndarray, np.ndarray],
                     label: str = "") -> None:
    dist, speed, other = streams
    print(f"\n[{label}]")
    print(f"  dist  : shape={dist.shape}  dtype={dist.dtype}")
    print(f"  speed : shape={speed.shape}  dtype={speed.dtype}  "
          f"range=[{speed.min():.3f}, {speed.max():.3f}]")
    print(f"  other : shape={other.shape}  dtype={other.dtype}  "
          f"mean={other.mean():.3f}  std={other.std():.3f}")


if __name__ == "__main__":
    train_path = DATA_DIR / "dataset_AbuDhabi_2022_2023.pkl"
    test_path = DATA_DIR / "dataset_2024_Abu_Dhabi_Grand_Prix.pkl"
    if not train_path.exists():
        raise SystemExit(f"Missing dataset: {train_path}. "
                         f"Run dataset_builder.build_all_datasets() first.")

    out = preprocess_dataset_file(
        train_path, test_path if test_path.exists() else None)
    describe_streams(out["train"], "train")
    if "test" in out:
        describe_streams(out["test"], "test")
