# f1_ranking — Finishing Position Forecasting

Codebase for forecasting final driver finishing positions from pre-race Formula 1 data (2022–2025), using ensemble and deep tabular architectures validated by walk-forward temporal cross-validation.

## Setup

Python 3.10+ recommended.

```bash
cd f1_ranking
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pretrained Weights

Download from Hugging Face: [kuzyshyn/f1_ranking_pred_2022-2025](https://huggingface.co/kuzyshyn/f1_ranking_pred_2022-2025/tree/main)

```bash
# optional: install the Hugging Face CLI and pull the weights
pip install huggingface_hub
huggingface-cli download kuzyshyn/f1_ranking_pred_2022-2025 --local-dir ./weights
```

## Usage

Run the full pipeline (data load → feature engineering → walk-forward CV → ExtraTrees / XGBoost / TabNet training → SHAP + PCA ablation → reports):

```bash
python -c "from pipeline import run_full_pipeline; run_full_pipeline()"
```

Configurable flags inside `run_full_pipeline`:
- `use_cache` — reuse cached FastF1 telemetry/results.
- `tune_models` — run hyperparameter search.
- `run_shap` — export SHAP feature attributions.
- `run_pca_ablation` — evaluate ExtraTrees over PCA-reduced features.
- `save_full_models` — persist final fitted models.
- `write_report` — write the validation report.

Outputs are written to `results/`.

## Data

Public 2022–2025 Formula 1 data via [FastF1](https://docs.fastf1.dev/). On first run, FastF1 caches sessions to disk; subsequent runs are fast.
