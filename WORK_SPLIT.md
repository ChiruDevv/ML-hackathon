# Team 10 – Work split and commit plan

| Member | SRN | Area | Evaluation-day role |
|---|---|---|---|
| **Harsh Mannan** | PES2UG24CS183 | Data loading, projection, EDA, Folium maps, evaluator UI | **Laptop 2 – UI** |
| **Chiranthan Shankar** (`chirudevv`) | PES2UG24CS137 | Road network, reference labelling, candidate generation, feature engineering | **Laptop 1 – API / model server** |
| **Meenakshi P** | PES2UG24CS811 | Metrics, 8-model comparison, Bayesian optimisation, comparison plots | Explains Sections 3–4 of the report |
| **Hansika R** | PES2UG24CS178 | HMM/Viterbi, route reconstruction, end-to-end pipeline, FastAPI backend | Explains Section 5 and the API code |

## Model training

Each member trains two of the eight models on their own laptop:

| Member | Models trained | Algorithm families |
|---|---|---|
| Harsh Mannan | Logistic Regression, Gaussian Naive Bayes | linear, probabilistic (generative) |
| Chiranthan Shankar | KNN (distance-weighted), Decision Tree | instance-based, tree |
| Meenakshi P | Random Forest, XGBoost, and the Bayesian tuning of XGBoost (the final model) | bagging ensemble, boosting |
| Hansika R | LightGBM, MLP neural network | boosting, neural network |

```bash
pip install -r requirements.txt
python scripts/02_build_dataset.py      # builds the feature cache (about 20 s); everyone runs this first
# then ONLY your own models:
python scripts/03_train_compare.py --models LogisticRegression,GaussianNB   # Harsh
python scripts/03_train_compare.py --models KNN,DecisionTree               # Chiranthan
python scripts/03_train_compare.py --models RandomForest,XGBoost           # Meenakshi
python scripts/03_train_compare.py --models LightGBM,MLP                   # Hansika
```
Every run adds its rows to `results/models/comparison.csv`, `cv_folds.csv`, `per_trip_student_test.csv` and `histories.json`, and writes `confusion_<model>.csv`. It does not overwrite the other members' models. Pull before you run and push right after, so the shared CSVs don't conflict. Seeds are fixed (42), so your numbers should match the ones in the README. Meenakshi runs `04_bayes_opt.py` and `06_plot_models.py` once all 8 models are in.

## Laptops on evaluation day

Only Harsh and Chiranthan operate the two laptops. Hansika wrote `api/main.py`, so she walks Chiranthan through starting it and reading its logs before the evaluation.

## File ownership

Each person commits **only their own files**, from **their own GitHub account**.

### Harsh Mannan
```
src/pathsnap/__init__.py
src/pathsnap/io.py            src/pathsnap/geo.py           src/pathsnap/viz.py
scripts/01_eda.py
ui/index.html  ui/app.js  ui/style.css  ui/vendor/*
results/eda/inventory.csv  results/eda/network_connectivity.csv
results/eda/partition_summary.csv  results/eda/data_quality_notes.md
results/figures/eda/*      results/maps/eda_*.html
scripts/07_static_route_figures.py   results/figures/maps/*   results/figures/ui/*
README.md (first version)
```

### Chiranthan Shankar
```
requirements.txt  requirements-lock.txt  .gitignore  data/raw/.gitkeep
configs/config.json           src/pathsnap/config.py
src/pathsnap/network.py       src/pathsnap/labels.py
src/pathsnap/features.py      src/pathsnap/dataset.py
scripts/02_build_dataset.py
results/eda/label_alignment_train.csv  results/eda/label_alignment_student_test.csv
results/eda/candidate_miss_grid.csv
```

### Meenakshi P
```
src/pathsnap/metrics.py       src/pathsnap/models.py        src/pathsnap/evaluation.py
scripts/03_train_compare.py   scripts/04_bayes_opt.py       scripts/06_plot_models.py
results/models/* rows for RandomForest + XGBoost (each member pushes their own rows)
results/bayes_opt/*           results/figures/models/*
```

### Hansika R
```
src/pathsnap/hmm.py           src/pathsnap/pipeline.py      src/pathsnap/scoring.py
scripts/05_hmm_eval.py        scripts/api_smoke_test.py     run_all.sh
api/main.py
models/xgb_model.json  models/xgb_untuned.json  models/pathsnap_bundle.json
results/hmm/*                 results/figures/hmm/*         results/maps/student_test_*.html
```

`docs/VIVA_NOTES.md` has one section per person, and each person commits their own section.

## Commit order

Commit in this order. Later code imports earlier code.

1. **Harsh** creates the repository `ML-HACKATHON` (public) and adds the other three as collaborators. He then pushes `README.md`, `src/pathsnap/__init__.py`, `io.py` and `geo.py`.
2. **Chiranthan** pushes the environment/config files, then `network.py`, `labels.py`, `features.py` and `dataset.py`, then `02_build_dataset.py` with its CSVs.
3. **Harsh** pushes `viz.py`, `01_eda.py` and the EDA results and maps. After step 7 he also pushes `07_static_route_figures.py` and its figures.
4. **Meenakshi** pushes `metrics.py`, `models.py`, `evaluation.py` and `03_train_compare.py`. Then **each member** trains their own two models and pushes their rows of `results/models/` (order: Harsh, Chiranthan, Meenakshi, Hansika).
5. **Hansika** pushes `hmm.py`, `pipeline.py` and `scoring.py`.
6. **Meenakshi** pushes `04_bayes_opt.py`, `06_plot_models.py`, `results/bayes_opt/` and the model figures.
7. **Hansika** pushes `05_hmm_eval.py`, `models/`, the HMM results and maps, `api/main.py`, `api_smoke_test.py` and `run_all.sh`.
8. **Harsh** pushes `ui/`.
9. **Everyone** pushes their own section of `docs/VIVA_NOTES.md` and README fixes.

Commands for each step (run from your own clone, with your own git identity):
```bash
git config user.name  "<your name>"
git config user.email "<the email on your GitHub account>"
git pull
# copy your files into place, then add ONLY your files, for example:
git add src/pathsnap/hmm.py src/pathsnap/pipeline.py src/pathsnap/scoring.py
git commit -m "Add HMM/Viterbi decoder, route reconstruction and scoring"
git push
```
Write commit messages that describe what you added. Before you push, run your own script once on your laptop so you know it works.
