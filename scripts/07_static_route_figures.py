"""Static (PNG) route comparison figures for the PDF report.

Owner: Harsh Mannan (PES2UG24CS183)

Folium maps are interactive HTML; this script draws the same layers with
matplotlib so they can be pasted into the report. It runs the deployed
pipeline (models/) on student-test trips.
Run:  python scripts/07_static_route_figures.py [--trips 00000077 00000056]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathsnap.config import RESULTS, load_config
from pathsnap.io import load_reference_route, load_trip
from pathsnap.pipeline import load_bundle, predict_trip
from pathsnap.scoring import score_prediction

FIG = RESULTS / "figures" / "maps"


def draw_arcs(ax, net, arcs, **kw):
    arcs = np.asarray(arcs, int)
    xs = np.c_[net.ax[arcs], net.bx[arcs], np.full(len(arcs), np.nan)].ravel()
    ys = np.c_[net.ay[arcs], net.by[arcs], np.full(len(arcs), np.nan)].ravel()
    ax.plot(xs, ys, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trips", nargs="*", default=["00000077", "00000056", "00000009"])
    a = ap.parse_args()
    cfg = load_config(); FIG.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    sdir = cfg["data"]["student_test_dir"]
    for tid in a.trips:
        trip = load_trip(sdir, tid)
        pred = predict_trip(trip, bundle)
        ref = load_reference_route(sdir, tid)
        sc = score_prediction(pred, ref)
        net, pxy = pred["ctx"].net, pred["ctx"].pxy
        _, near, _, _ = net.arcs_near(pxy[::5], 150)
        clf = pred["clf_point_arcs"]
        fig, axs = plt.subplots(1, 2, figsize=(15, 7.5))
        for ax, (title, pred_arcs, col) in zip(axs, [
                (f"Classifier only - Top-1 {sc['classifier_only_accuracy'] * 100:.1f}%", np.unique(clf[clf >= 0]), "#ff7f0e"),
                (f"Classifier + HMM route - Top-1 {sc['observation_accuracy'] * 100:.1f}%, route F1 {sc['route_f1'] * 100:.1f}%",
                 pred["route_arcs"], "#d62728")]):
            draw_arcs(ax, net, np.unique(near), color="#cccccc", lw=0.6, label="road network")
            draw_arcs(ax, net, ref, color="#2ca02c", lw=5, alpha=0.45, label="reference route")
            draw_arcs(ax, net, pred_arcs, color=col, lw=1.8, label="prediction", zorder=4)
            ax.scatter(pxy[:, 0], pxy[:, 1], s=1, c="#1f77b4", alpha=0.5, label="raw GPS", zorder=3)
            ax.set_title(title); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
            ax.legend(loc="best", fontsize=8)
        fig.suptitle(f"Student-test trip {tid} (UTM metres)")
        fig.tight_layout(); fig.savefig(FIG / f"route_comparison_{tid}.png", dpi=130); plt.close(fig)
        print("saved", tid, sc["observation_accuracy"], sc["classifier_only_accuracy"])


if __name__ == "__main__":
    main()
