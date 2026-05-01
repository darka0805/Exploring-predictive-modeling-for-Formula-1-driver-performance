# f1_overtaking — Overtake Prediction

Codebase for the overtake-prediction pipeline: forecasts the probability of an overtake within a 2-second window from low-frequency Formula 1 telemetry (Abu Dhabi, Monza, Bahrain; 2022–2024).


## Setup

Python 3.10+ recommended.

```bash
cd f1_overtaking
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Pretrained Weights

Download from Hugging Face: [kuzyshyn/overtake_prediction_f1](https://huggingface.co/kuzyshyn/overtake_prediction_f1/tree/main)

```bash
pip install huggingface_hub
huggingface-cli download kuzyshyn/overtake_prediction_f1 --local-dir ./weights
```

## Usage

Run the full pipeline (build datasets → run all experiments E1–E7 → ablation studies V1–V5):

```bash
python run_all.py
```

Run a single experiment matrix:

```bash
python -c "from experiments.experiments_e1_e6 import run_all_experiments; run_all_experiments()"
python -c "from experiments.experiments_e7_2025 import run_e7; run_e7()"
```

Validate the Pirelli 2022 benchmark of 785 overtakes:

```bash
python validate_overtake_counts.py
```

Outputs (metrics JSONL, PR/ROC curves, confusion matrices, per-experiment CSVs) are written to `results/`.

## Data

Public Formula 1 telemetry via [FastF1](https://docs.fastf1.dev/). The first run caches raw sessions to `cache/`; subsequent runs reuse the cache.
