# PathSnap – Distributed Map-Matching & Route Reconstruction

**UE24CS352A Machine Learning Hackathon · PES University (RR Campus) · Team 10**

| Member | SRN |
|---|---|
| Harsh Mannan | PES2UG24CS183 |
| Chiranthan Shankar | PES2UG24CS137 |
| Meenakshi P | PES2UG24CS811 |
| Hansika R | PES2UG24CS178 |

PathSnap matches noisy GPS fixes to a road network and rebuilds the route the vehicle drove. A learned candidate scorer (XGBoost, Bayesian-tuned) proposes the road for each fix. An HMM/Viterbi decoder then turns those per-fix guesses into one connected route. The system runs as a FastAPI server (Laptop 1) and a browser UI (Laptop 2).

Who did what: [`WORK_SPLIT.md`](WORK_SPLIT.md). Viva preparation: [`docs/VIVA_NOTES.md`](docs/VIVA_NOTES.md).

---

## Repository layout

```
configs/config.json          candidate, label and HMM settings (single source of truth)
src/pathsnap/
  io.py  geo.py              raw readers + cleaning, UTM projection, geometry
  network.py                 directed road graph, KD-tree index, bounded Dijkstra
  labels.py                  per-fix reference labels (monotonic route alignment)
  features.py                candidate generation + 24 features
  dataset.py                 cached dataset loading
  models.py metrics.py evaluation.py   model zoo, metric definitions, evaluation
  hmm.py                     emission/transition model, Viterbi, route reconstruction
  pipeline.py scoring.py     end-to-end inference; server-side scoring
  viz.py                     Folium maps
scripts/
  01_eda.py                  dataset inventory, quality checks, plots, Folium maps
  02_build_dataset.py        labels + candidates + features for train / student-test
  03_train_compare.py        8 models, 5-fold trip-grouped CV, student-test evaluation
  04_bayes_opt.py            Optuna TPE optimisation of XGBoost, final model export
  05_hmm_eval.py             decoder tuning, classifier-only vs HMM, per-trip maps
  06_plot_models.py          comparison and optimisation figures
  07_static_route_figures.py PNG route comparisons for the report
  api_smoke_test.py          calls the running API for every trip
api/main.py                  FastAPI backend (Laptop 1)
ui/                          evaluator UI: index.html, app.js, style.css, vendored Leaflet (Laptop 2)
models/                      xgb_model.json (weights) + pathsnap_bundle.json (inference config)
results/                     every CSV, log, figure and map quoted below
docs/VIVA_NOTES.md           technical notes per member
```

## Installation

```bash
git clone https://github.com/<owner>/ML-HACKATHON.git
cd ML-HACKATHON
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # exact tested versions: requirements-lock.txt
```

The organisers distribute the raw data, so it is not committed. Put it here:
```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/raw/student_test/<trip_id>/...
data/raw/eval/<trip_id>/...        # the 10 evaluator trips, on Laptop 1 only
```
Any folder can be used instead through the env vars `PATHSNAP_TRAIN_DIR`, `PATHSNAP_STUDENT_TEST_DIR` and `PATHSNAP_EVAL_DIR`, or through `--data-dir` for the API.

## Reproducing the results

```bash
bash run_all.sh       # about 30-40 min on 2 CPU cores; or run the scripts in that order
```
All randomness is seeded (`seed = 42`). The evaluator partition is never read by any training or tuning script.

## Deployment (two laptops)

| Laptop | Operator | Runs |
|---|---|---|
| **Laptop 1 – model + API** | Chiranthan Shankar | `api/main.py`: classifier, HMM, road network, server-side scoring |
| **Laptop 2 – UI** | Harsh Mannan | `ui/`: lists the 10 trips, sends the selected id, draws the route and metrics |

Both laptops must be on the same network (the same Wi-Fi or a phone hotspot).

**Laptop 1 (Chiranthan)**
```bash
python api/main.py --data-dir data/raw/eval --host 0.0.0.0 --port 8000
# find this laptop's IP: Windows `ipconfig`, macOS/Linux `ip a` or `ifconfig`
# allow port 8000 through the firewall if Windows asks
```
The terminal prints a startup block (model, decoder settings, trips found). Then for every request it shows the client IP, trip id, stage timings and the server-side accuracy. Keep it visible for the examiners. A copy is written to `logs/api.log`.

**Laptop 2 (Harsh)**
```bash
cd ui
python -m http.server 5500
# open http://localhost:5500/?api=http://<LAPTOP1-IP>:8000
```
`&trip=<id>` opens a trip straight away. Pick one of the ten trips. The UI calls `POST /predict` and draws the raw GPS fixes, the matched positions and the predicted route. It also shows the accuracy and route metrics computed on Laptop 1. Leaflet is bundled in `ui/vendor`, so the UI works without a CDN; only the background map tiles need internet.

**Checking the API from a terminal** (screenshots for the report):
```bash
curl http://<LAPTOP1-IP>:8000/trips
curl -X POST http://<LAPTOP1-IP>:8000/predict -H "Content-Type: application/json" -d "{\"trip_id\": \"<id>\"}"
python scripts/api_smoke_test.py --api http://<LAPTOP1-IP>:8000
```

**Data-leakage safeguards**
- `pipeline.predict_trip` reads only `.track`, `.nodes` and `.arcs`.
- `.route` is read by `scoring.py` only after the prediction exists.
- The API returns aggregate metrics, never the reference arcs.
- Nothing is hard-coded per trip. Every trip goes through the same pipeline.

## Method summary

| Stage | What we do | Code |
|---|---|---|
| Projection | Each trip goes to its own UTM zone, so all distances are in metres. Trips span 30° S – 62° N on several continents. | `geo.py` |
| Labels | A monotonic DP aligns the fixes to the reference arc sequence, which gives one reference arc per fix. Fixes more than 100 m from their aligned arc are flagged unreliable and excluded from training and scoring. | `labels.py` |
| Candidates | All directed arcs within 100 m of a fix, keeping the 10 closest. Chosen on TRAIN: candidate miss rate 1.18% on train and 0.16% on student-test. | `features.py`, `results/eda/candidate_miss_grid.csv` |
| Features | 24 per (fix, candidate) pair: distance and its rank or ratio; Gaussian posterior; heading agreement over ±2 and ±5 fixes (separates the two directions of a street); speed, turn rate, time gaps; projection fraction, arc length, node degrees; distance of the neighbouring fixes (±3) to the same arc. | `features.py` |
| Classifier | Binary "is this the reference arc"; 8 models compared; XGBoost deployed | `models.py`, `03_train_compare.py` |
| Decoder | HMM: states = candidate arcs. Emission = classifier probability. Transition = exp(−\|d_route − d_gc\|/β), with d_route from a bounded Dijkstra. Viterbi decoding, then shortest-path gap filling. | `hmm.py` |

## Results

Partitions:
- **CV** = 5-fold cross-validation grouped by trip, on the 80 training trips (every 3rd fix).
- **Student-test** = the 10 student-test trips (every fix with a reliable label, 19,143 fixes).
- The evaluator partition was never used.

Metric definitions are in `src/pathsnap/metrics.py`:
- **Top-1** is per-fix accuracy of the best-scoring candidate. It is the primary metric.
- **Macro F1** is computed over candidate rows, using the same top-1 decision rule.
- Because exactly one candidate per fix is predicted positive, candidate-level precision equals Top-1, so it is not repeated in the table.

### Model comparison (`results/models/comparison.csv`)

| Model | Family | Trained by | CV Macro F1 (±sd) | CV Top-1 | Student-test Top-1 | Student-test Macro F1 | Log loss | Route continuity | Train s | Inference ms / 1k fixes |
|---|---|---|---|---|---|---|---|---|---|---|
| LogisticRegression | Linear / probabilistic | Harsh | 0.9389 ± 0.0162 | 88.77% | 88.77% | 0.9364 | 0.0814 | 94.83% | 1.1 | 1.0 |
| GaussianNB | Probabilistic (generative) | Harsh | 0.9253 ± 0.0155 | 86.41% | 87.56% | 0.9295 | 0.5391 | 93.30% | 0.3 | 2.3 |
| KNN | Instance-based | Chiranthan | 0.9352 ± 0.0182 | 88.12% | 88.62% | 0.9356 | 0.1101 | 93.48% | 0.0 | 1184.1 |
| DecisionTree | Tree-based | Chiranthan | 0.9397 ± 0.0187 | 88.92% | 89.16% | 0.9386 | 0.0791 | 94.80% | 9.5 | 0.6 |
| RandomForest | Ensemble (bagging) | Meenakshi | 0.9439 ± 0.0164 | 89.64% | 90.21% | 0.9446 | 0.0625 | 95.67% | 39.9 | 22.3 |
| XGBoost **(selected)** | Boosting | Meenakshi | 0.9432 ± 0.0176 | 89.52% | 91.50% | 0.9519 | 0.0589 | 95.93% | 6.6 | 13.2 |
| LightGBM | Boosting | Hansika | 0.9393 ± 0.0174 | 88.85% | 90.50% | 0.9462 | 0.0728 | 94.54% | 5.2 | 30.5 |
| MLP | Neural network | Hansika | 0.9405 ± 0.0165 | 89.06% | 90.59% | 0.9467 | 0.0642 | 95.59% | 15.4 | 2.0 |

KNN's train time is about 0 s because KNN only stores the data; its cost is at prediction time. It uses a 60k-row sample. Times were measured on a 2-core machine while other jobs were running, so compare them by order of magnitude only.

**Selection: XGBoost.** It had the best student-test Top-1, Macro F1, log loss and continuity. Its CV Macro F1 is within 0.0007 of Random Forest, far inside one fold standard deviation. It also trains about 6× faster than Random Forest and exports to a small portable JSON file.

Overfitting evidence is in `results/figures/models/training_curves.png`:
- **XGBoost:** validation loss bottoms out near iteration 50 and then rises slowly. Mild overfitting.
- **LightGBM** (lr = 0.1): validation loss becomes unstable after about 80 iterations.
- **MLP:** validation loss follows training loss, so no overfitting.

### Bayesian hyperparameter optimisation (`results/bayes_opt/`)

- **Optimiser:** Optuna TPE (seed 42, 8 random start-up trials) with a median pruner.
- **Trials:** 30 (4 pruned), 22 min on 2 cores.
- **Objective:** mean 5-fold trip-grouped CV Top-1.
- **Trial 0** is the untuned configuration.
- **Search space:**
  - n_estimators 100–800
  - max_depth 3–10
  - learning_rate 0.01–0.3 (log)
  - min_child_weight 1–50 (log)
  - subsample 0.5–1
  - colsample_bytree 0.4–1
  - gamma 0–5
  - reg_lambda 1e-3–10 (log)
  - reg_alpha 1e-3–10 (log)
- **Best:** n_estimators = 450, max_depth = 7, learning_rate = 0.0119, min_child_weight = 2.36, subsample = 0.624, colsample_bytree = 0.959, gamma = 3.09, reg_lambda = 0.0036, reg_alpha = 0.642.

| | CV Top-1 | CV Macro F1 | CV log loss | Student-test Top-1 | Student-test Macro F1 | Student-test log loss | Inference ms / 1k fixes |
|---|---|---|---|---|---|---|---|
| XGBoost untuned | 89.52% | 0.9432 | 0.0669 | **91.50%** | **0.9519** | **0.0589** | 11.4 |
| XGBoost tuned | **89.85%** | **0.9451** | **0.0645** | 90.40% | 0.9457 | 0.0613 | 26.7 |

**Did tuning help?** On the 80-trip CV it did: every CV metric improved and fold variance dropped. On the 10 student-test trips it did not. It was lower on 9 of the 10 trips and equal on the tenth, mostly by under 1 point, and by 4.4 and 2.6 points on trips 00000007 and 00000077.

Possible reasons:
- The CV gain is small (+0.33 pt).
- Student-test is only 10 trips.
- The tuned model is more strongly regularised (gamma ≈ 3, low learning rate), which may not suit some of the test networks.

We made the final choice on the deployed system (classifier + HMM) on student-test. That also favours the untuned model (94.28% vs 93.37%, `results/hmm/final_model_selection.csv`), so **the deployed model is the untuned XGBoost**. The tuned model is kept in `models/xgb_model.json` for comparison.

### HMM / Viterbi route reconstruction (`results/hmm/`)

Decoder parameters were grid-searched on **out-of-fold** probabilities of 25 training trips: α ∈ {0.5, 1, 2}, β ∈ {2.5, 5, 10, 20, 40} m. Chosen: α = 1, β = 5 m.

| Decoder | Top-1 (fix-weighted) | Point continuity | Route continuity before fill | Route precision | Route recall | Route F1 |
|---|---|---|---|---|---|---|
| classifier-only (untuned) | 91.50% | 95.93% | 83.43% | 78.80% | 99.77% | 87.69% |
| classifier-only (tuned) | 90.40% | 95.79% | 83.38% | 74.44% | 99.73% | 84.30% |
| distance-HMM | 92.11% | 97.59% | 90.48% | 96.29% | 99.63% | 97.91% |
| classifier+HMM (untuned) – deployed | **94.28%** | **97.91%** | **91.36%** | **97.42%** | **99.71%** | **98.54%** |
| classifier+HMM (tuned) | 93.37% | 97.84% | 91.00% | 97.18% | 99.64% | 98.39% |

- **Accuracy.** HMM decoding raises Top-1 from 91.50% to 94.28%. It improves 9 of the 10 trips, by up to +6.0 points on trip 00000077, a dense network with many parallel streets.
- **Routes.** Length-weighted route precision rises from 78.8% to 97.4%: the classifier alone produces scattered wrong arcs, which the HMM removes.
- **Continuity.** Consecutive predicted arcs are connected 91.4% of the time before gap filling, against 83.4% for the classifier alone. After shortest-path filling the routes are 99.9% continuous (3 unfilled breaks across 10 trips).
- **Where it did not help.** On trip 00000007 the HMM is 0.6 points below the classifier alone: a simple route where the classifier was already at 97.4%. The distance-only HMM (no classifier) is weaker than the classifier on trips 00000007, 00000009 and 00000076, which shows that the learned emissions matter.

### Folium maps and figures

- `results/maps/eda_*.html`: GPS, network, reference route.
- `results/maps/student_test_<trip>.html`: GPS, network, reference, classifier-only arcs, HMM route, with a layer control and legend.
- Static versions for the report: `results/figures/maps/route_comparison_*.png`.
- All other plots: `results/figures/{eda,models,hmm}/`.
- UI screenshots from a local test run (API and UI on one machine): `results/figures/ui/`. Retake them on the two laptops for the report.

### API check on the 10 student-test trips (`scripts/api_smoke_test.py`)

Every trip returned a route. Server-side accuracy was 88.0–99.6% and the request time was 0.1–1.7 s per trip on a 2-core machine.

## Limitations

- Reference routes are not ground truth. Two training trips (00000054, 00000033) disagree with their GPS by more than 100 m for long stretches. Our per-fix labels come from an alignment and inherit those errors.
- Fixes on arcs shorter than the distance between fixes can be skipped. The reconstruction step restores them from the network.
- The decoder runs Dijkstra in Python. That is fine for 1 Hz trips of a few thousand fixes, but long trips on dense networks take a few seconds.

## AI-assistance disclosure

As the event's Assessment and Academic Integrity Policy requires, we disclose that an AI coding assistant was used while developing this repository: drafting code, scripts and documentation. Every experiment was actually executed, and every number, table, figure and map in `results/` is produced by the code in this repository; they can be regenerated with `run_all.sh`. Each member re-ran their own part, and each member can explain it (see `docs/VIVA_NOTES.md`).
