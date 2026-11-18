"""Plots for model comparison and Bayesian optimisation (from saved results only).

Owner: Meenakshi P (PES2UG24CS811)

Every figure is drawn from the CSV/JSON files written by 03 and 04, so the
numbers in the report can be traced back to the experiment logs.
Run:  python scripts/06_plot_models.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import RESULTS

M = RESULTS / "models"
B = RESULTS / "bayes_opt"
FIG = RESULTS / "figures" / "models"
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    comp = pd.read_csv(M / "comparison.csv")
    order = comp.sort_values("cv_macro_f1_mean").model.tolist()
    comp = comp.set_index("model").loc[order]
    cvf = pd.read_csv(M / "cv_folds.csv")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(order, comp.cv_macro_f1_mean, xerr=comp.cv_macro_f1_std, color="#4c72b0", capsize=3)
    for m in order:
        v = cvf[cvf.model == m].macro_f1
        ax.scatter(v, [m] * len(v), color="k", s=8, zorder=3)
    lo = comp.cv_macro_f1_mean.min() - 3 * comp.cv_macro_f1_std.max()
    ax.set(xlim=(max(0, lo), 1), xlabel="Macro F1 (mean +- std over 5 trip-grouped folds; dots = folds)",
           title="Cross-validation Macro F1 (80 training trips)")
    fig.tight_layout(); fig.savefig(FIG / "cv_macro_f1.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(order)); w = 0.38
    ax.bar(x - w / 2, comp.test_top1 * 100, w, label="Top-1 accuracy", color="#55a868")
    ax.bar(x + w / 2, comp.test_macro_f1 * 100, w, label="Macro F1", color="#c44e52")
    for i, (a, b) in enumerate(zip(comp.test_top1, comp.test_macro_f1)):
        ax.text(i - w / 2, a * 100 + 0.2, f"{a * 100:.1f}", ha="center", fontsize=8)
        ax.text(i + w / 2, b * 100 + 0.2, f"{b * 100:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x, order, rotation=25, ha="right")
    ax.set(ylim=(min(comp.test_top1.min(), comp.test_macro_f1.min()) * 100 - 3, 100), ylabel="%",
           title="Student-test (10 trips): observation Top-1 accuracy and candidate Macro F1")
    ax.legend(loc="upper left", ncol=2); fig.tight_layout(); fig.savefig(FIG / "student_test_top1_f1.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].barh(order, comp.test_log_loss, color="#8172b2"); ax[0].set_xscale("log")
    ax[0].set(title="Student-test log loss (lower is better, log scale)")
    ax[1].barh(order, comp.inference_ms_per_1k_obs, color="#937860"); ax[1].set_xscale("log")
    ax[1].set(title="Inference time: ms per 1000 GPS fixes (log scale)")
    fig.tight_layout(); fig.savefig(FIG / "log_loss_inference_time.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for m in order:
        ax.scatter(comp.loc[m, "cv_top1_mean"] * 100, comp.loc[m, "test_top1"] * 100, s=40)
        ax.annotate(m, (comp.loc[m, "cv_top1_mean"] * 100, comp.loc[m, "test_top1"] * 100), fontsize=8,
                    xytext=(4, 2), textcoords="offset points")
    ax.set(xlabel="CV Top-1 %", ylabel="Student-test Top-1 %", title="CV vs student-test Top-1 (generalisation)")
    fig.tight_layout(); fig.savefig(FIG / "cv_vs_test_top1.png"); plt.close(fig)

    # confusion matrices
    n = len(order); cols = 4; rows = int(np.ceil(n / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for ax, m in zip(axs.ravel(), order):
        cm = pd.read_csv(M / f"confusion_{m}.csv", index_col=0).values
        ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}\n({cm[i, j] / cm[i].sum() * 100:.1f}%)", ha="center", va="center", fontsize=8)
        ax.set_xticks([0, 1], ["pred not-ref", "pred ref"]); ax.set_yticks([0, 1], ["not-ref", "ref arc"])
        ax.set_title(m, fontsize=10)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.suptitle("Candidate-level confusion matrices on student-test (top-1 decision rule)")
    fig.tight_layout(); fig.savefig(FIG / "confusion_matrices.png"); plt.close(fig)

    # training histories (only models that really produce them)
    hp = M / "histories.json"
    if hp.exists():
        h = json.loads(hp.read_text())
        fig, axs = plt.subplots(1, len(h), figsize=(5 * len(h), 3.8))
        axs = np.atleast_1d(axs)
        for ax, (m, d) in zip(axs, h.items()):
            ax.plot(d["train_loss"], label="train (folds 2-5)")
            ax.plot(d["val_loss"], label="validation (fold 1)")
            ax.set(title=m, xlabel="epoch" if m == "MLP" else "boosting iteration", ylabel="log loss")
            ax.legend()
        fig.suptitle("Training vs validation loss, CV fold 1 (models with iterative training only)")
        fig.tight_layout(); fig.savefig(FIG / "training_curves.png"); plt.close(fig)

    # Bayesian optimisation
    tp = B / "trials.csv"
    if tp.exists():
        t = pd.read_csv(tp)
        done = t[t.state == "COMPLETE"]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(done.number, done.value * 100, label="completed trial", s=18)
        pr = t[t.state == "PRUNED"]
        ax.scatter(pr.number, [done.value.min() * 100] * len(pr), marker="x", c="grey", label="pruned trial")
        ax.plot(done.number, done.value.cummax() * 100, c="r", label="best so far")
        ax.set(xlabel="trial", ylabel="CV Top-1 %", title="Optuna TPE optimisation history (XGBoost)")
        ax.legend(); fig.tight_layout(); fig.savefig(FIG / "bayes_opt_history.png"); plt.close(fig)
        try:
            import optuna
            st = optuna.load_study(study_name="pathsnap_xgb", storage=f"sqlite:///{B / 'optuna_study.db'}")
            imp = optuna.importance.get_param_importances(st)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.barh(list(imp)[::-1], list(imp.values())[::-1], color="#4c72b0")
            ax.set(title="Hyperparameter importance (fANOVA on Optuna trials)", xlabel="importance")
            fig.tight_layout(); fig.savefig(FIG / "bayes_opt_importance.png"); plt.close(fig)
        except Exception as e:  # importance needs enough completed trials
            print("importance plot skipped:", e)
    tv = B / "tuned_vs_untuned.csv"
    if tv.exists():
        t = pd.read_csv(tv).set_index("model")
        cols = ["cv_macro_f1", "test_top1", "test_macro_f1"]
        ax = (t[cols] * 100).T.plot.bar(figsize=(8, 4), color=["#999999", "#4c72b0"], rot=0)
        ax.set(ylim=((t[cols].values.min()) * 100 - 2, 100), ylabel="%", title="Untuned vs Bayesian-tuned XGBoost")
        plt.tight_layout(); plt.savefig(FIG / "tuned_vs_untuned.png"); plt.close()
    print("figures written to", FIG)


if __name__ == "__main__":
    main()
