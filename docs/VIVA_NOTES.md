# Viva preparation notes – Team 10

Read your own section first, then skim the others. The examiners can ask any of you about any part. Run your own scripts before evaluation day so you know what they print.

## The whole system in one minute

1. Each trip has its own GPS track (`.track`), road network (`.nodes` + `.arcs` as directed straight segments) and a reference arc sequence (`.route`).
2. Each trip is projected to its own UTM zone, so all geometry is in metres.
3. **Labels.** The route says which arcs were driven, not which arc each GPS fix belongs to. A monotonic DP alignment assigns every fix to a route position. This gives one reference arc per fix.
4. **Candidates.** For each fix we take all arcs within 100 m and keep the 10 closest. Each (fix, candidate) pair becomes one row with 24 features. The target is "is this the reference arc?"
5. **Classifier.** It gives P(candidate is correct). We compared 8 models with 5-fold trip-grouped CV and the student-test trips. XGBoost was selected and tuned with Optuna TPE; the untuned XGBoost is deployed because tuning did not carry over to student-test.
6. **HMM.** Hidden state = candidate arc of each fix. Emission = classifier probability. Transition = how well the road-network distance matches the GPS distance. Viterbi finds the best sequence.
7. **Route reconstruction.** Repeated arcs are collapsed and gaps between unconnected arcs are filled with shortest paths. The result is a connected route.
8. **Deployment.** The FastAPI server on Laptop 1 runs the whole pipeline and scores the result against the reference route on the server. The UI on Laptop 2 only lists the trips, sends the trip id and draws what comes back.

---

## Harsh Mannan – data, EDA, Folium, UI (Laptop 2)

**Files:** `io.py`, `geo.py`, `viz.py`, `scripts/01_eda.py`, `ui/`. **Models you trained:** Logistic Regression, Gaussian NB.

- **Dataset structure.** There are 80 train and 10 student-test trips. Each trip has 4 headerless whitespace files. Node id = row index in `.nodes`; arc id = row index in `.arcs`. GPS rows do not correspond to node rows.
- **Cleaning** (`io.clean_track`). We drop NaN or impossible coordinates, sort by time and drop duplicate timestamps. The data turned out clean: 0 NaNs, timestamps always increasing. We still keep the code because the evaluator trips could be different.
- **Why UTM.** Degrees are not metres. A degree of longitude shrinks with latitude, and the trips go from 30° S to 62° N on several continents. We pick the UTM zone from the track's median position. Inside one zone the scale error is under 0.1%, far below GPS noise (5–20 m).
- **EDA findings** (`results/eda/data_quality_notes.md`):
  - About 97% of steps are exactly 1 s.
  - 23 trips have gaps over 60 s. Two training trips have gaps of 14 h and 42 h.
  - About 95% of arcs have a reverse twin (two-way streets), so heading is needed to pick the direction.
  - Networks are mostly one large strongly connected component (about 96% of nodes) plus small pieces.
  - Two training trips (00000054, 00000033) have a median GPS-to-route distance over 100 m. The reference itself is unreliable there.
- **Folium maps.** Layers: raw GPS (blue), network within 150 m (grey), reference route (green), classifier-only arcs (orange), HMM route (red). There is a LayerControl, a legend, Fullscreen, and the map zooms to the track's bounds.
- **UI** (`ui/app.js`). Plain HTML/JS with Leaflet vendored locally, so it works without a CDN; only the basemap tiles need internet. It calls `GET /trips` and `POST /predict`. It draws the GPS points, the snapped points and the predicted route, and shows the server-side metrics. The UI never sees the reference route.
- **Likely questions.**
  - *Why not Euclidean on lat/lon?* Distances would be wrong, and wrong by a different amount at each latitude.
  - *What does a row in `.arcs` mean?* A directed straight segment between two nodes.
  - *How does the UI get its accuracy?* The API computes it on the server.
- **Your models.**
  - **Logistic Regression** (linear) uses scaled features. Its score is a weighted sum of features, so it cannot model interactions such as "heading matters only when the vehicle is moving". It plateaus below the trees.
  - **Gaussian NB** assumes features are independent given the class. Many of our features are strongly correlated (dist, dist rank, gauss_post, window distance), so it double-counts evidence. Its probabilities are badly calibrated, which is why it has by far the worst log loss (≈0.54).

## Chiranthan Shankar – network, labels, candidates, features (Laptop 1)

**Files:** `network.py`, `labels.py`, `features.py`, `dataset.py`, `config.py`, `configs/config.json`, `scripts/02_build_dataset.py`. **Models you trained:** KNN, Decision Tree.

- **Spatial index.** Every arc is sampled every 10 m into a KD-tree. We query radius + 5 m, which guarantees that no arc within the radius is missed. Then we compute the exact point-to-segment distance.
- **Monotonic alignment** (`labels.py`). We choose route positions k_0 ≤ k_1 ≤ … to minimise the sum of distances. The DP uses a prefix minimum, so it runs in O(N·R). Monotonicity stops a fix from jumping back to a part of the route driven earlier, for example when the vehicle passes the same street twice.
- **Unreliable labels.** Fixes more than 100 m from their aligned arc are excluded from training and scoring. That is 9.9% of train fixes, and 95% of those come from just 2 trips (00000054: 16,455 fixes, 00000033: 2,978). We do not delete trips; the reliable fixes of those trips are still used.
- **Candidate settings.** We use r = 100 m and k = 10, chosen on TRAIN only (`results/eda/candidate_miss_grid.csv`). The candidate miss rate is 1.2% on train and 0.16% on student-test. Going larger gave little gain and slowed the HMM.
- **Features (24).**
  - Distance block: dist, rank, dist − min, ratio, Gaussian posterior.
  - Direction block: cos and |angle| between GPS heading (±2 fixes and ±5 fixes) and arc bearing. This separates the two directions of a two-way street.
  - Context: speed, turn rate, time gaps.
  - Arc geometry: projection fraction t, end-point flag, log length, node degrees, has-reverse.
  - Temporal window: mean and max distance of fixes i−3..i+3 to this arc, its rank, and the gap to the best one. This is the strongest feature after distance.
- **Why it transfers across countries.** Every feature is in metres, seconds or radians, relative to the local geometry.
- **Laptop 1 duties.** `python api/main.py --data-dir data/raw/eval --host 0.0.0.0 --port 8000`. Find your IP (`ipconfig` / `ip a`) and give it to Harsh. Keep the terminal visible: the examiners must see the request logs.
- **Your models.**
  - **KNN** (k = 25, distance-weighted, scaled features). It needs a 60k-row training sample because prediction cost grows with training size. It is by far the slowest at inference (about 0.75–1.2 s per 1000 fixes in our runs) and its probabilities are coarse.
  - **Decision Tree** (depth 12, min_leaf 50). It is fast but a single tree is unstable. Bagging and boosting both beat it.

## Meenakshi P – metrics, model comparison, Bayesian optimisation

**Files:** `metrics.py`, `models.py`, `evaluation.py`, `scripts/03_train_compare.py`, `04_bayes_opt.py`, `06_plot_models.py`. **Models you trained:** Random Forest, XGBoost (+ tuning).

- **Group-aware CV.** There are 5 folds split by trip (16 trips each, seed 42; `results/models/fold_assignment.csv`). Fixes of one trip are strongly correlated, so a random row split would leak information.
- **Training stride 3.** Consecutive 1 Hz fixes are nearly identical. Taking every 3rd fix cuts the training time to a third with almost no loss of information. Student-test is scored on every fix.
- **Metrics.**
  - Top-1 = per-fix accuracy of the argmax candidate. This is the primary metric because it is what the route depends on.
  - Macro F1 = binary F1 over candidate rows (top-1 decision rule), averaged over the 2 classes.
  - Log loss shows how good the probabilities are. This matters because the HMM uses them.
  - Route continuity, candidate miss rate, and train and inference time.
- **Selection.** XGBoost had the best student-test Top-1 (91.5%), Macro F1 and log loss, and the best continuity. Its CV Macro F1 was within 0.001 of Random Forest, well inside one fold standard deviation. It also trains about 6× faster than RF and gives a small portable JSON model.
- **Overfitting evidence** (`results/figures/models/training_curves.png`). For XGBoost the validation loss bottoms out around iteration 50 and then creeps up while the training loss keeps falling: mild overfitting. LightGBM at lr = 0.1 becomes unstable after about 80 iterations. The MLP's validation loss tracks its training loss, so it is not overfitting. That is why tuning searched learning rate, depth and regularisation.
- **Bayesian optimisation.** Optuna TPE (seed 42, 8 random start-up trials) with a median pruner. The objective is mean 5-fold grouped CV Top-1. The search covers 9 hyperparameters (ranges in the README). Trial 0 is the untuned configuration. All trials are stored in `results/bayes_opt/optuna_study.db` and `trials.csv`.
- **Tuning outcome (be ready for this question).**
  - CV improved: Top-1 89.52% → 89.85%, Macro F1 0.9432 → 0.9451, log loss down, fold std down.
  - Student-test got worse: Top-1 91.50% → 90.40%. It was lower on 9 of 10 trips, by 4.4 points on trip 00000007.
  - The full classifier+HMM system also prefers the untuned model (94.28% vs 93.37%), so the **untuned XGBoost is deployed**.
  - Possible reasons: the CV gain is tiny, and student-test is only 10 trips. The tuned model is heavily regularised (gamma ≈ 3, lr ≈ 0.012), so it scores well on average across folds but transfers worse to some networks.
  - It is fine that tuning did not help. Reporting it honestly is what the guidelines ask for.
- **Why trial 0 shows 89.62% but the untuned refit shows 89.52%.** Trial 0 used reg_alpha = 0.001 instead of 0, because the log-uniform search range cannot contain 0. That tiny difference changes the trees a little.
- **Your models.**
  - **Random Forest**: bagging lowers the variance of deep trees, and it had the best CV score. But it is slower and its model file is large.
  - **XGBoost**: sequential error correction, strong on tabular features of different scales.

## Hansika R – HMM/Viterbi, pipeline, API

**Files:** `hmm.py`, `pipeline.py`, `scoring.py`, `scripts/05_hmm_eval.py`, `api/main.py`, `scripts/api_smoke_test.py`. **Models you trained:** LightGBM, MLP.

- **Hidden states.** The candidate slots of each fix. Each slot keeps its arc id, projection fraction t and distance, so the state-to-arc mapping is preserved.
- **Emission.** α · log p, where p is the classifier probability renormalised over that fix's candidates. The baseline instead uses a Gaussian on distance (σ = 15 m), as in Newson & Krumm.
- **Transition.** log T = −|d_route − d_gc| / β.
  - d_gc = straight-line distance between the two fixes.
  - d_route = forward distance along the same arc, or remaining part of arc a + shortest path end(a)→start(b) + t_b · length(b).
  - Paths come from a bounded Dijkstra, cached per source node. The bound grows with the gap: max(3·d_gc + 200, 40·dt, 500) m, capped at 20 km.
- **Disconnected segments / no path.** Instead of −∞ the transition gets a floor of −30 in log space, so Viterbi never breaks. That transition is effectively a "teleport", used only if nothing else is possible.
- **Gaps over 600 s.** The chain restarts: the next fix's states get the best previous score, with no transition cost.
- **Fixes with no candidates** are skipped by the decoder and get no arc.
- **Viterbi.** δ_i(b) = max_a [δ_{i−1}(a) + log T(a,b)] + E_i(b), with back-pointers. It is vectorised over the K×K matrix per step.
- **Route reconstruction.** Repeated arcs are collapsed. If consecutive arcs are not connected, the shortest path between them is inserted.
- **Decoder tuning.** Grid over α ∈ {0.5, 1, 2} and β ∈ {2.5, 5, 10, 20, 40} m, run on **out-of-fold** probabilities of 25 training trips. In-sample probabilities would be over-confident. Chosen: α = 1, β = 5 m.
- **Results (student-test).**
  - Top-1: 91.50% for the classifier alone, 92.11% for the distance-only HMM, 94.28% for classifier+HMM.
  - Route F1: 87.7% → 98.5%.
  - Continuity before gap filling: 83.4% → 91.4%; after filling 99.9%.
  - The HMM helps on 9 of 10 trips, most on 00000077 (+6.0 points, a dense grid). On 00000007 it is −0.6 points, where the classifier was already at 97.4% on a simple route.
  - The distance-only HMM is worse than the classifier alone on 3 trips, which shows that the learned emission matters.
- **API.** `/health`, `/trips`, `POST /predict`. Prediction loads only the track and network. The `.route` file is read only afterwards, by `scoring.py`, and only aggregate numbers are returned. Every request is logged to the console and to `logs/api.log`.
- **Your models.**
  - **LightGBM**: leaf-wise boosting. At the untuned lr = 0.1 with 300 trees it overfits and its validation loss oscillates (see the training curves).
  - **MLP (64, 32)**: trained with `partial_fit` so we have real per-epoch training and validation loss. It is smooth but slightly below the boosted trees.
