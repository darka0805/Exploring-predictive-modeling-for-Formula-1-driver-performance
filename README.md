# Exploring predictive modeling for Formula 1 driver performance

This repository contains the code for the Bachelor Thesis: "Exploring predictive modeling for Formula 1 driver performance"

## Abstract

Formula 1 is a data-driven sport, yet current research relies on outdated datasets and high-frequency telemetry that fail to capture the modern ground-effect era or enable overtake prediction from standard, low-frequency real-world data.

This thesis develops a predictive modeling framework on publicly available 2022–2025 Formula 1 data with two contributions: (i) forecasting final driver finishing positions from pre-race data using ensemble and deep tabular models (TabNet, LightGBM, XGBoost, ExtraTrees) on 86,758 laps across 62 dry-asphalt tracks, validated by walk-forward temporal cross-validation; and (ii) an overtaking analytical tool trained on 312 overtakes labeled via a three-phase pipeline from Abu Dhabi, Monza, and Bahrain (2022–2024), forecasting the probability of an overtake within a 2-second window with XGBoost, LightGBM, Random Forest, 1D-CNN, and BiGRU.

ExtraTrees achieved a Spearman correlation of 0.9268 and MAE of 1.533 for finishing positions, while XGBoost reached PR-AUC 0.720 and F1 0.711 on a cross-circuit temporal split (train: 2022–2023 Abu Dhabi/Bahrain/Monza; test: 2024 Abu Dhabi), demonstrating robust performance despite low-frequency telemetry constraints.

## Pretrained Weights

Pretrained model weights are available on Hugging Face:

- Finishing position forecasting: [kuzyshyn/f1_ranking_pred_2022-2025](https://huggingface.co/kuzyshyn/f1_ranking_pred_2022-2025/tree/main)
- Overtake prediction: [kuzyshyn/overtake_prediction_f1](https://huggingface.co/kuzyshyn/overtake_prediction_f1/tree/main)

## Repository Structure

The project is organized into two main components (each with detailed `README.md` instructions):

### 1. Finishing Position Forecasting ([f1_ranking/](f1_ranking/))

Codebase for training and evaluating models that forecast final driver finishing positions from pre-race data.

- **Dataset**: 86,758 laps across 62 dry-asphalt tracks (2022–2025 seasons).
- **Models**: ExtraTrees, XGBoost, LightGBM, TabNet.
- **Validation**: Walk-forward temporal cross-validation.
- **Analyses**: PCA ablation, SHAP feature attribution, evaluation reporting.
- Visit folder's README for more details.

### 2. Overtake Prediction ([f1_overtaking/](f1_overtaking/))

Codebase for the overtake-prediction pipeline: per-frame telemetry forecasting of an overtake within a 2-second window.

- **Dataset**: Telemetry from Abu Dhabi, Monza, and Bahrain (2022–2024), 312 labeled overtakes via a three-phase labeling pipeline.
- **Features**: 40-feature pipeline (distance-grid interpolation, slipstream coefficient, Kalman line deviation, rolling stats, battle persistence, driver context, track geometry).
- **Models**: XGBoost, LightGBM, Random Forest, 1D-CNN, BiGRU, plus the End2Race base model.
- **Experiments**: Single-year, multi-year, leave-one-year-out, cross-track transfer, and future-year holdout (E1–E7) with ablation studies.
- Visit folder's README for more details.

## Setup & Installation

Each subsystem manages its own dependencies and entry points. Please refer to their respective READMEs to get started:

- [Finishing Position Forecasting Setup](f1_ranking/)
- [Overtake Prediction Setup](f1_overtaking/)

## Author Information

Daryna Kuzyshyn
Ukrainian Catholic University, Faculty of Applied Sciences

## About

Thesis implementation of *Exploring predictive modeling for Formula 1 driver performance*: a framework for forecasting driver finishing positions and predicting overtaking maneuvers from publicly available 2022–2025 Formula 1 data.
