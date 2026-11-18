"""Train and compare candidate-scoring models.

Owner: Meenakshi P (PES2UG24CS811)

* 5-fold group-aware CV on the 80 training trips (folds split by trip, so no
  trip is ever in both the fitting and the validation part of a fold).
* Every model is refit on all 80 training trips and evaluated on the 10
  student-test trips.
* Training rows: every 3rd GPS fix (consecutive 1 Hz fixes are near
  duplicates); evaluation on student-test uses every fix.

Outputs in results/models/:
  cv_folds.csv, comparison.csv, per_trip_student_test.csv, histories.json,
  fold_assignment.csv, confusion_<model>.csv
Run:  python scripts/03_train_compare.py [--models LightGBM,XGBoost]
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from pathsnap.config import RESULTS, ROOT, load_config
from pathsnap.dataset import load_partition, stack
from pathsnap.evaluation import evaluate, score_trips
from pathsnap.metrics import row_decisions, top1_slots
from pathsnap.models import MODEL_SPECS, MLPWrapper, build

OUT = RESULTS / "models"
ART = ROOT / "artifacts" / "models"


def fold_assignment(n_trips, k, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_trips)
    f = np.empty(n_trips, int)
    for i, t in enumerate(perm):
        f[t] = i % k
    return f


def subsample(X, y, cap, seed):
    if cap is None or len(y) <= cap:
        return X, y
    idx = np.random.default_rng(seed).choice(len(y), cap, replace=False)
    return X[idx], y[idx]


def fit_model(name, X, y, Xv=None, yv=None, record=False):
    m = build(name)
    hist = None
    if name == "MLP":
        m.fit(X, y, Xv, yv)
        hist = m.history
    elif name in ("LightGBM", "XGBoost") and record and Xv is not None:
        if name == "LightGBM":
            m.fit(X, y, eval_set=[(X, y), (Xv, yv)], eval_metric="binary_logloss")
            r = m.evals_result_
            hist = {"train_loss": r["training"]["binary_logloss"], "val_loss": r["valid_1"]["binary_logloss"]}
        else:
            m.fit(X, y, eval_set=[(X, y), (Xv, yv)], verbose=False)
            r = m.evals_result()
            hist = {"train_loss": r["validation_0"]["logloss"], "val_loss": r["validation_1"]["logloss"]}
    else:
        m.fit(X, y)
    return m, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(MODEL_SPECS))
    args = ap.parse_args()
    cfg = load_config()
    seed, stride, K = cfg["seed"], cfg["training"]["point_stride"], cfg["training"]["cv_folds"]
    OUT.mkdir(parents=True, exist_ok=True); ART.mkdir(parents=True, exist_ok=True)

    train, test = load_partition("train"), load_partition("student_test")
    folds = fold_assignment(len(train), K, seed)
    pd.DataFrame({"trip": [t.trip for t in train], "fold": folds}).to_csv(OUT / "fold_assignment.csv", index=False)

    hist_path = OUT / "histories.json"
    histories = json.loads(hist_path.read_text()) if hist_path.exists() else {}
    cv_rows, comp_rows, trip_rows = [], [], []
    for name in args.models.split(","):
        fam, _, conf, cap = MODEL_SPECS[name]
        print(f"\n=== {name} ({fam})", flush=True)
        for f in range(K):
            tr = [t for t, ff in zip(train, folds) if ff != f]
            va = [t for t, ff in zip(train, folds) if ff == f]
            X, y, *_ = stack(tr, stride=stride)
            X, y = subsample(X, y, cap, seed)
            Xv, yv, *_ = stack(va, stride=stride)
            t0 = time.perf_counter()
            m, h = fit_model(name, X, y, Xv, yv, record=(f == 0))
            fit_s = time.perf_counter() - t0
            if h is not None:
                histories[name] = h
            probas, _ = score_trips(m, va)
            met, _ = evaluate(va, probas, stride=stride, with_continuity=False)
            met.update(model=name, fold=f, fit_seconds=fit_s, train_rows=len(y))
            cv_rows.append(met)
            print(f"  fold {f}: top1={met['top1']:.4f} macroF1={met['macro_f1']:.4f} "
                  f"logloss={met['log_loss']:.4f} fit={fit_s:.1f}s", flush=True)

        X, y, *_ = stack(train, stride=stride)
        X, y = subsample(X, y, cap, seed)
        t0 = time.perf_counter()
        m, _ = fit_model(name, X, y)
        fit_s = time.perf_counter() - t0
        probas, inf_s = score_trips(m, test)
        met, per_trip = evaluate(test, probas, stride=1, with_continuity=True)
        n_obs_all = sum(t.n for t in test)
        cvd = pd.DataFrame([r for r in cv_rows if r["model"] == name])
        row = dict(model=name, family=fam, config=conf,
                   cv_macro_f1_mean=cvd.macro_f1.mean(), cv_macro_f1_std=cvd.macro_f1.std(),
                   cv_top1_mean=cvd.top1.mean(), cv_top1_std=cvd.top1.std(),
                   cv_log_loss_mean=cvd.log_loss.mean(),
                   test_top1=met["top1"], test_macro_f1=met["macro_f1"], test_precision=met["precision"],
                   test_recall=met["recall"], test_log_loss=met["log_loss"],
                   test_route_continuity=met["route_continuity"],
                   test_candidate_miss_rate=met["candidate_miss_rate"],
                   train_seconds=fit_s, inference_ms_per_1k_obs=1000 * inf_s / n_obs_all * 1000,
                   train_rows=len(y))
        comp_rows.append(row)
        for r in per_trip:
            r["model"] = name
        trip_rows += per_trip
        # confusion matrix on student-test (row-level, top-1 decision rule)
        ys, ds = [], []
        for td, p in zip(test, probas):
            best, _ = top1_slots(p, td.rp, td.rs, td.n, td.C.shape[1])
            keep = td.reliable[td.rp]
            ys.append(td.y[keep]); ds.append(row_decisions(best, td.rp, td.rs)[keep])
        cm = confusion_matrix(np.concatenate(ys), np.concatenate(ds), labels=[0, 1])
        pd.DataFrame(cm, index=["true_0", "true_1"], columns=["pred_0", "pred_1"]).to_csv(OUT / f"confusion_{name}.csv")
        joblib.dump(m, ART / f"{name}.joblib")
        print(f"  TEST top1={met['top1']:.4f} macroF1={met['macro_f1']:.4f} logloss={met['log_loss']:.4f} "
              f"continuity={met['route_continuity']:.4f} train={fit_s:.1f}s", flush=True)

        # write incrementally so partial runs are kept
        def merge(path, new, keys):
            new = pd.DataFrame(new)
            if path.exists():
                old = pd.read_csv(path, dtype={"trip": str})
                old = old[~old.model.isin(new.model.unique())]
                new = pd.concat([old, new], ignore_index=True)
            new.to_csv(path, index=False)
        merge(OUT / "cv_folds.csv", [r for r in cv_rows if r["model"] == name], None)
        merge(OUT / "comparison.csv", [row], None)
        merge(OUT / "per_trip_student_test.csv", [r for r in trip_rows if r["model"] == name], None)
        hist_path.write_text(json.dumps(histories))


if __name__ == "__main__":
    main()
