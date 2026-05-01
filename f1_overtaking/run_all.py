#!/usr/bin/env python3
"""End-to-end entry point: build all datasets and run the full experiment matrix."""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from config import SEED

np.random.seed(SEED)


def main():
    from data_preprocessing_and_labeling.dataset_builder import build_all_datasets
    from experiments.experiments_e1_e6 import run_all_experiments
    from experiments.experiments_e7_2025 import run_2025_holdout_experiments
    from experiments.experiments_validation import run_all_validation_experiments

    print("=" * 70)
    print("F1 OVERTAKE PREDICTION - FULL PIPELINE")
    print("=" * 70)

    build_all_datasets()

    df = run_all_experiments()

    run_2025_holdout_experiments()

    run_all_validation_experiments()

    print("\n\nDone. See results/ for metrics, plots, and CSV.")
    return df


if __name__ == "__main__":
    main()
