"""Bayesian hyperparameter optimisation (Optuna TPE) of the selected model (XGBoost).

Owner: Meenakshi P (PES2UG24CS811)

* Search: Tree-structured Parzen Estimator (a sequential model-based /
  Bayesian optimiser), seeded, with median pruning after each CV fold.
* Objective: mean observation-level Top-1 accuracy over the same 5
  trip-grouped CV folds used in 03_train_compare.py (80 training trips only).
* The tuned model is refit on all training trips and compared with the
  untuned configuration on the student-test trips.
* Also writes out-of-fold probabilities for every training fix, used to tune
  the HMM decoder without leaking in-sample confidence.

Outputs: results/bayes_opt/{trials.csv, best_params.json, tuned_vs_untuned.csv,
         optuna_study.db}, models/{xgb_model.json, pathsnap_bundle.json}, artifacts/oof_train.npz
Run:  python scripts/04_bayes_opt.py --trials 30
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBClassifier

from pathsnap.config import MODELS, RESULTS, ROOT, load_config
from pathsnap.dataset import load_partition, stack
from pathsnap.evaluation import evaluate, score_trips
from pathsnap.features import FEATURE_NAMES
from pathsnap.pipeline import save_bundle_meta

OUT = RESULTS / "bayes_opt"
UNTUNED = dict(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8)


def make(params, seed):
    return XGBClassifier(**params, tree_method="hist", n_jobs=2, eval_metric="logloss", random_state=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--refit-only", action="store_true",
                    help="skip the search; reuse results/bayes_opt/best_params.json")
    args = ap.parse_args()
    cfg = load_config()
    seed, stride, K = cfg["seed"], cfg["training"]["point_stride"], cfg["training"]["cv_folds"]
    OUT.mkdir(parents=True, exist_ok=True); MODELS.mkdir(parents=True, exist_ok=True)
    train, test = load_partition("train"), load_partition("student_test")
    folds = pd.read_csv(RESULTS / "models" / "fold_assignment.csv", dtype={"trip": str})
    fmap = dict(zip(folds.trip, folds.fold))
    fold_of = np.array([fmap[t.trip] for t in train])
    data = []
    for f in range(K):
        tr = [t for t, ff in zip(train, fold_of) if ff != f]
        va = [t for t, ff in zip(train, fold_of) if ff == f]
        X, y, *_ = stack(tr, stride=stride)
        data.append((X, y, va))

    def cv(params, trial=None):
        scores, f1s, lls = [], [], []
        for f, (X, y, va) in enumerate(data):
            m = make(params, seed).fit(X, y)
            pr, _ = score_trips(m, va)
            met, _ = evaluate(va, pr, stride=stride, with_continuity=False)
            scores.append(met["top1"]); f1s.append(met["macro_f1"]); lls.append(met["log_loss"])
            if trial is not None:
                trial.report(float(np.mean(scores)), f)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return float(np.mean(scores)), float(np.mean(f1s)), float(np.mean(lls)), float(np.std(f1s))

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 50.0, log=True),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        )
        t0 = time.perf_counter()
        top1, f1, ll, f1sd = cv(params, trial)
        trial.set_user_attr("cv_macro_f1", f1); trial.set_user_attr("cv_log_loss", ll)
        trial.set_user_attr("cv_macro_f1_std", f1sd); trial.set_user_attr("seconds", time.perf_counter() - t0)
        print(f"trial {trial.number}: top1={top1:.4f} f1={f1:.4f} ll={ll:.4f} {params}", flush=True)
        return top1

    if args.refit_only:
        best = json.load(open(OUT / "best_params.json"))["best_params"]
    else:
        best = run_search(args, objective, seed)
    refit_and_export(best, cv, data, train, test, fold_of, cfg, seed, stride, K)


def run_search(args, objective, seed):
    db = OUT / "optuna_study.db"
    if db.exists():
        db.unlink()
    study = optuna.create_study(direction="maximize", study_name="pathsnap_xgb",
                                storage=f"sqlite:///{db}",
                                sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=1))
    # trial 0 = the untuned configuration from 03_train_compare.py (XGBoost defaults: lambda=1, alpha=0)
    study.enqueue_trial(dict(n_estimators=300, max_depth=6, learning_rate=0.1, min_child_weight=1.0,
                             subsample=0.8, colsample_bytree=0.8, gamma=0.0, reg_lambda=1.0, reg_alpha=1e-3))
    t0 = time.perf_counter()
    study.optimize(objective, n_trials=args.trials)
    search_s = time.perf_counter() - t0
    study.trials_dataframe().to_csv(OUT / "trials.csv", index=False)
    best = dict(study.best_params)
    json.dump(dict(best_params=best, best_cv_top1=study.best_value, n_trials=len(study.trials),
                   n_pruned=sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials),
                   search_seconds=search_s, objective="mean 5-fold trip-grouped CV observation Top-1 accuracy",
                   sampler="TPE (seed=42, 8 random start-up trials)", pruner="MedianPruner"),
              open(OUT / "best_params.json", "w"), indent=2)
    return best


def refit_and_export(best, cv, data, train, test, fold_of, cfg, seed, stride, K):

    # ---------------------------------------------- tuned vs untuned
    rows = []
    Xall, yall, *_ = stack(train, stride=stride)
    for label, params in [("untuned", UNTUNED), ("tuned", best)]:
        top1, f1, ll, f1sd = cv(params)
        t0 = time.perf_counter(); m = make(params, seed).fit(Xall, yall); fit_s = time.perf_counter() - t0
        pr, inf_s = score_trips(m, test)
        met, _ = evaluate(test, pr, stride=1)
        rows.append(dict(model=f"XGBoost ({label})", params=json.dumps(params), cv_top1=top1, cv_macro_f1=f1,
                         cv_macro_f1_std=f1sd, cv_log_loss=ll, test_top1=met["top1"],
                         test_macro_f1=met["macro_f1"], test_log_loss=met["log_loss"],
                         test_route_continuity=met["route_continuity"], train_seconds=fit_s,
                         inference_ms_per_1k_obs=1e6 * inf_s / sum(t.n for t in test)))
        m.save_model(MODELS / ("xgb_model.json" if label == "tuned" else "xgb_untuned.json"))
    pd.DataFrame(rows).to_csv(OUT / "tuned_vs_untuned.csv", index=False)
    print(pd.DataFrame(rows).T)

    # model weights in XGBoost's portable JSON format + a JSON bundle with everything
    # else the API needs (feature order, candidate settings, decoder settings)
    # (05_hmm_eval.py later picks tuned vs untuned using the full classifier+HMM pipeline)
    save_bundle_meta(dict(model_file="xgb_model.json", model_name="XGBoost (Bayesian-tuned)", params=best, tuned_params=best,
                          untuned_model_file="xgb_untuned.json", untuned_params=UNTUNED,
                          feature_names=FEATURE_NAMES, candidates=cfg["candidates"], hmm=cfg["hmm"],
                          labels=cfg["labels"], trained_on="80 training trips (every 3rd fix)",
                          seed=seed))

    # ---------------------------------------------- out-of-fold probabilities (all fixes)
    # for both configurations, so the HMM decoder can be tuned for each without leakage
    for label, params in [("tuned", best), ("untuned", UNTUNED)]:
        oof = {}
        for f in range(K):
            X, y, _ = data[f]
            m = make(params, seed).fit(X, y)
            va = [t for t, ff in zip(train, fold_of) if ff == f]
            pr, _ = score_trips(m, va)
            for td, p in zip(va, pr):
                oof[td.trip] = p
        np.savez_compressed(ROOT / "artifacts" / f"oof_train_{label}.npz", **oof)
    print("saved bundle and OOF predictions")


if __name__ == "__main__":
    main()
