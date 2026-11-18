"""HMM decoder tuning and classifier-only vs classifier+HMM comparison.

Owner: Hansika R (PES2UG24CS178)

1. Tune decoder parameters (alpha, beta) by grid search on OUT-OF-FOLD
   classifier probabilities of 25 training trips (training data only),
   separately for the untuned and the Bayesian-tuned XGBoost.
2. Evaluate on the 10 student-test trips:
     - classifier-only (argmax per fix), for both XGBoost configurations
     - distance-only HMM (Gaussian emission, Newson & Krumm baseline)
     - classifier + HMM, for both XGBoost configurations
3. Select the deployed classifier (tuned vs untuned) by student-test
   classifier+HMM Top-1 - the metric of the system that is actually deployed
   (development-stage comparison on student-test is allowed; the evaluator
   partition is never touched). Write the choice into models/pathsnap_bundle.json.
4. Folium maps of every student-test trip and summary plots.

Run:  python scripts/05_hmm_eval.py            (add --plots-only to redraw figures)
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathsnap.config import CONFIG_PATH, MODELS, RESULTS, ROOT, load_config
from pathsnap.dataset import load_partition
from pathsnap.features import make_context
from pathsnap.hmm import emission_logp, reconstruct_route, route_continuity, viterbi
from pathsnap.io import load_reference_route, load_trip
from pathsnap.metrics import continuity, route_set_metrics
from pathsnap.pipeline import BUNDLE_FILE, save_bundle_meta
from xgboost import XGBClassifier
from pathsnap.viz import trip_map

OUT = RESULTS / "hmm"
FIG = RESULTS / "figures" / "hmm"
MAPS = RESULTS / "maps"
ALPHAS = [0.5, 1.0, 2.0]
BETAS = [2.5, 5.0, 10.0, 20.0, 40.0]


def point_cont(net, arcs):
    return continuity(np.where(arcs >= 0, net.arcs[np.maximum(arcs, 0), 0], -1),
                      np.where(arcs >= 0, net.arcs[np.maximum(arcs, 0), 1], -2), arcs)


def decode(ctx, td, P, hc, mode="classifier", alpha=1.0, beta=20.0):
    E = emission_logp(P, td.D, td.C, mode, alpha=alpha)
    slot = viterbi(ctx.net, ctx.pxy, ctx.t, td.C, td.T, E, beta=beta,
                   back_tol=hc["backtrack_tolerance_m"], min_log_prob=hc["min_log_prob"],
                   gap_break_s=hc["gap_break_s"])
    rows = np.arange(len(slot))
    return np.where(slot >= 0, td.C[rows, np.maximum(slot, 0)], -1)


def dense_P(td, p):
    P = np.zeros(td.C.shape); P[td.rp, td.rs] = p
    return P


def tune_decoder(sub, ctxs, oof, hc, with_distance):
    rows = []
    for td in sub:
        P = dense_P(td, oof[td.trip]); ctx = ctxs[td.trip]; rel = td.reliable
        clf = argmax_arcs(td, P)
        base = dict(trip=td.trip, n=int(rel.sum()))
        rows.append(dict(base, decoder="classifier-only", alpha=np.nan, beta=np.nan,
                         correct=int((clf == td.lab)[rel].sum())))
        for beta in BETAS:
            if with_distance:
                arcs = decode(ctx, td, P, hc, "distance", beta=beta)
                rows.append(dict(base, decoder="distance-HMM", alpha=np.nan, beta=beta,
                                 correct=int((arcs == td.lab)[rel].sum())))
            for alpha in ALPHAS:
                arcs = decode(ctx, td, P, hc, "classifier", alpha=alpha, beta=beta)
                rows.append(dict(base, decoder="classifier-HMM", alpha=alpha, beta=beta,
                                 correct=int((arcs == td.lab)[rel].sum())))
    return pd.DataFrame(rows)


def argmax_arcs(td, P):
    return np.where((td.C >= 0).any(1), td.C[np.arange(td.n), np.where(td.C >= 0, P, -1).argmax(1)], -1)


def main():
    if "--plots-only" in sys.argv:   # redraw figures from saved CSVs
        res = pd.read_csv(OUT / "hmm_student_test_per_trip.csv", dtype={"trip": str})
        summ = pd.read_csv(OUT / "hmm_student_test_summary.csv", index_col=0)
        final = json.load(open(OUT / "decoder_params.json"))["selected"]
        return make_plots(res, summ, final)
    cfg = load_config(); hc0 = dict(cfg["hmm"])
    for d in (OUT, FIG, MAPS):
        d.mkdir(parents=True, exist_ok=True)
    tdir, sdir = cfg["data"]["train_dir"], cfg["data"]["student_test_dir"]
    meta = json.loads((MODELS / BUNDLE_FILE).read_text())
    files = {"tuned": "xgb_model.json", "untuned": meta.get("untuned_model_file", "xgb_untuned.json")}
    params = {"tuned": meta.get("tuned_params", meta["params"]), "untuned": meta.get("untuned_params")}

    # ------------------------------------------------ 1. decoder grid on OOF train
    train = load_partition("train")
    rng = np.random.default_rng(cfg["seed"])
    sub = [train[i] for i in sorted(rng.choice(len(train), 25, replace=False))]
    ctxs = {td.trip: make_context(load_trip(tdir, td.trip)) for td in sub}
    grids = []
    for label in ("tuned", "untuned"):
        oof = np.load(ROOT / "artifacts" / f"oof_train_{label}.npz")
        g = tune_decoder(sub, ctxs, oof, hc0, with_distance=(label == "tuned"))
        g["model"] = np.where(g.decoder == "distance-HMM", "-", label)
        grids.append(g)
        print("decoder grid done for", label, flush=True)
    g = pd.concat(grids)
    gs = g.groupby(["model", "decoder", "alpha", "beta"], dropna=False)[["correct", "n"]].sum().reset_index()
    gs["top1"] = gs.correct / gs.n
    gs.to_csv(OUT / "decoder_grid_oof_train.csv", index=False)
    print(gs.to_string())
    dec = {}
    for label in ("tuned", "untuned"):
        b = gs[(gs.model == label) & (gs.decoder == "classifier-HMM")].sort_values("top1", ascending=False).iloc[0]
        dec[label] = dict(hc0, alpha=float(b.alpha), beta_m=float(b.beta))
    beta_dist = float(gs[gs.decoder == "distance-HMM"].sort_values("top1", ascending=False).iloc[0].beta)
    print("best decoders:", dec, "distance beta:", beta_dist)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, label in zip(axs, ("untuned", "tuned")):
        piv = gs[(gs.model == label) & (gs.decoder == "classifier-HMM")].pivot(
            index="alpha", columns="beta", values="top1") * 100
        im = ax.imshow(piv.values, cmap="viridis")
        ax.set_xticks(range(len(piv.columns)), [f"{x:g}" for x in piv.columns])
        ax.set_yticks(range(len(piv.index)), [f"{x:g}" for x in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        ax.set(xlabel="beta (m)", ylabel="alpha (emission weight)", title=f"XGBoost {label}")
    fig.suptitle("Classifier+HMM Top-1 % on out-of-fold probabilities (25 training trips)")
    fig.tight_layout(); fig.savefig(FIG / "decoder_grid.png"); plt.close(fig)

    # ------------------------------------------------ 2. student-test comparison
    models = {}
    for label, f in files.items():
        m = XGBClassifier(); m.load_model(MODELS / f); models[label] = m
    test = load_partition("student_test")
    store, rows = {}, []
    for td in test:
        trip = load_trip(sdir, td.trip)
        ctx = make_context(trip)
        ref = load_reference_route(sdir, td.trip)
        rel = td.reliable
        out = {}
        for label, m in models.items():
            t0 = time.perf_counter(); P = dense_P(td, m.predict_proba(td.X)[:, 1]); clf_s = time.perf_counter() - t0
            out[f"classifier-only ({label})"] = (argmax_arcs(td, P), np.nan, clf_s)
            t0 = time.perf_counter()
            arcs = decode(ctx, td, P, dec[label], "classifier", dec[label]["alpha"], dec[label]["beta_m"])
            out[f"classifier+HMM ({label})"] = (arcs, time.perf_counter() - t0, clf_s)
        t0 = time.perf_counter()
        out["distance-HMM"] = (decode(ctx, td, P, hc0, "distance", beta=beta_dist), time.perf_counter() - t0, 0.0)
        for name, (arcs, dec_s, clf_s) in out.items():
            route, filled, breaks = reconstruct_route(ctx.net, arcs)
            raw_seq = [a for i, a in enumerate(arcs) if a >= 0 and (i == 0 or a != arcs[i - 1])]
            r = dict(trip=td.trip, decoder=name, n_scored=int(rel.sum()),
                     top1=float((arcs == td.lab)[rel].mean()),
                     point_continuity=point_cont(ctx.net, arcs),
                     raw_route_continuity=route_continuity(ctx.net, raw_seq),
                     gaps_needing_fill=filled + breaks, unfilled_breaks=breaks,
                     final_route_continuity=route_continuity(ctx.net, route),
                     decode_seconds=dec_s, classifier_seconds=clf_s)
            r.update(route_set_metrics(route, ref, ctx.net.length))
            rows.append(r)
        store[td.trip] = (trip, ctx, ref, out)
        print("evaluated", td.trip, flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "hmm_student_test_per_trip.csv", index=False)
    summ = res.groupby("decoder").apply(lambda d: pd.Series(dict(
        top1_weighted=float((d.top1 * d.n_scored).sum() / d.n_scored.sum()),
        top1_trip_mean=d.top1.mean(), point_continuity=d.point_continuity.mean(),
        raw_route_continuity=d.raw_route_continuity.mean(),
        final_route_continuity=d.final_route_continuity.mean(),
        route_precision=d.route_precision.mean(), route_recall=d.route_recall.mean(),
        route_f1=d.route_f1.mean(), unfilled_breaks=d.unfilled_breaks.sum(),
        decode_seconds_total=d.decode_seconds.sum())), include_groups=False)
    order = ["classifier-only (untuned)", "classifier-only (tuned)", "distance-HMM",
             "classifier+HMM (untuned)", "classifier+HMM (tuned)"]
    summ = summ.loc[order]
    summ.to_csv(OUT / "hmm_student_test_summary.csv")
    print(summ.T.to_string())

    # ------------------------------------------------ 3. select deployed classifier
    s_t = summ.loc["classifier+HMM (tuned)", "top1_weighted"]
    s_u = summ.loc["classifier+HMM (untuned)", "top1_weighted"]
    final = "tuned" if s_t >= s_u else "untuned"
    sel = pd.DataFrame([dict(candidate=l, cv_note="see results/bayes_opt/tuned_vs_untuned.csv",
                             student_test_hmm_top1=summ.loc[f"classifier+HMM ({l})", "top1_weighted"],
                             student_test_clf_top1=summ.loc[f"classifier-only ({l})", "top1_weighted"],
                             alpha=dec[l]["alpha"], beta_m=dec[l]["beta_m"], selected=(l == final))
                        for l in ("untuned", "tuned")])
    sel.to_csv(OUT / "final_model_selection.csv", index=False)
    print(sel.to_string())
    meta.update(model_file=files[final], selected_configuration=final,
                model_name=f"XGBoost ({'Bayesian-tuned' if final == 'tuned' else 'untuned baseline'}) + HMM",
                params=params[final], tuned_params=params["tuned"], hmm=dec[final])
    meta.pop("deployed_params", None)
    save_bundle_meta(meta)
    full = json.loads(CONFIG_PATH.read_text())
    full["hmm"]["alpha"], full["hmm"]["beta_m"] = dec[final]["alpha"], dec[final]["beta_m"]
    CONFIG_PATH.write_text(json.dumps(full, indent=2) + "\n")
    json.dump(dict(selected=final, decoders=dec, distance_hmm_beta=beta_dist),
              open(OUT / "decoder_params.json", "w"), indent=2)

    # ------------------------------------------------ 4. maps + plots for the deployed system
    cn, hn = f"classifier-only ({final})", f"classifier+HMM ({final})"
    for tid, (trip, ctx, ref, out) in store.items():
        clf = out[cn][0]; hmm = out[hn][0]
        hmm_route, _, _ = reconstruct_route(ctx.net, hmm)
        m = trip_map(trip.track, ctx.net, ctx.pxy, {
            "Reference route": (ref, "reference"),
            "Classifier-only matched arcs": (np.unique(clf[clf >= 0]), "classifier"),
            "Predicted route (classifier + HMM)": (hmm_route, "hmm")},
            title=f"Student-test trip {tid}: classifier-only vs classifier+HMM")
        m.save(str(MAPS / f"student_test_{tid}.html"))
    make_plots(res, summ, final)

def make_plots(res, summ, final):
    cn, hn = f"classifier-only ({final})", f"classifier+HMM ({final})"
    cols = [cn, "distance-HMM", hn]
    for metric, fname, ylab, title in [
            ("top1", "hmm_vs_classifier_top1.png", "observation Top-1 accuracy %",
             "Student-test: classifier-only vs HMM decoding"),
            ("raw_route_continuity", "hmm_vs_classifier_continuity.png", "% consecutive arcs connected",
             "Route continuity before gap filling (student-test)")]:
        piv = res.pivot(index="trip", columns="decoder", values=metric)[cols] * 100
        ax = piv.plot.bar(figsize=(11, 4.2), color=["#ff7f0e", "#9467bd", "#d62728"], width=0.8, rot=0)
        ax.set(ylabel=ylab, title=title, xlabel="student-test trip", ylim=(max(0, piv.values.min() - 5), 100.5))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=9, frameon=False)
        plt.tight_layout(); plt.savefig(FIG / fname); plt.close()
    ax = (summ[["top1_weighted", "raw_route_continuity", "route_f1"]] * 100).plot.barh(figsize=(9, 4.5))
    ax.set(xlabel="%", title="Student-test summary by decoder", xlim=(80, 100))
    ax.legend(["Top-1 (fix-weighted)", "route continuity (before fill)", "route F1"], loc="lower left", fontsize=8)
    plt.tight_layout(); plt.savefig(FIG / "decoder_summary.png"); plt.close()


if __name__ == "__main__":
    main()
