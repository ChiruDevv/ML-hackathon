#!/usr/bin/env bash
# Reproduce every result in results/ and the model in models/ from the raw data.
# Expects data/raw/train and data/raw/student_test (see README). ~30 min on 2 CPU cores.
set -euo pipefail
cd "$(dirname "$0")"
python scripts/02_build_dataset.py                 # labels, candidates, features (cache)
python scripts/01_eda.py                           # EDA tables, plots, Folium maps
python scripts/03_train_compare.py                 # 8 models, 5-fold grouped CV + student-test
python scripts/04_bayes_opt.py --trials 30         # Optuna TPE on XGBoost -> models/
python scripts/05_hmm_eval.py                      # decoder tuning + classifier vs HMM + maps
python scripts/06_plot_models.py                   # comparison / optimisation figures
python scripts/07_static_route_figures.py          # PNG route comparisons for the report
