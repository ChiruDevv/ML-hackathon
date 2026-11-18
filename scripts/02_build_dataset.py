"""Build the candidate-level dataset for the train and student-test partitions.

Owner: Chiranthan Shankar (PES2UG24CS137)

Outputs
  artifacts/cache/<partition>/<trip>.npz       feature rows per trip
  results/eda/candidate_miss_grid.csv           miss rate vs (radius, k) on TRAIN
  results/eda/label_alignment_<partition>.csv   alignment quality per trip
Run:  python scripts/02_build_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pathsnap.config import CACHE, RESULTS, load_config
from pathsnap.features import build_features, generate_candidates, make_context
from pathsnap.io import list_trips, load_reference_route, load_trip
from pathsnap.labels import align_track_to_route

RADII = [30, 50, 75, 100, 150]
KS = [4, 6, 8, 10, 15]


def process(part, pdir, tid, cfg):
    trip = load_trip(pdir, tid)
    route = load_reference_route(pdir, tid)
    ctx = make_context(trip, cfg["candidates"]["index_sample_step_m"])
    lab, k, ldist = align_track_to_route(ctx.pxy, route, ctx.net)
    reliable = ldist <= cfg["labels"]["reliable_max_dist_m"]

    grid = []
    if part == "train":
        Cb, Db, _ = generate_candidates(ctx, radius=max(RADII), k=max(KS), fallback_radius=0)
        for r in RADII:
            for kk in KS:
                ok = (Cb[:, :kk] == lab[:, None]) & (Db[:, :kk] <= r)
                grid.append(dict(trip=tid, radius=r, k=kk, n=int(reliable.sum()),
                                 miss=int((~ok.any(1))[reliable].sum()),
                                 mean_cands=float(((Db[:, :kk] <= r)).sum(1).mean())))

    R, K = cfg["candidates"]["radius_m"], cfg["candidates"]["k"]
    C, D, T = generate_candidates(ctx, radius=R, k=K)
    X, rp, rs = build_features(ctx, C, D, T)
    y = (C[rp, rs] == lab[rp]).astype(np.int8)
    out = CACHE / part
    out.mkdir(parents=True, exist_ok=True)
    Cu = np.where(C >= 0, ctx.net.arcs[np.maximum(C, 0), 0], -1)
    Cv = np.where(C >= 0, ctx.net.arcs[np.maximum(C, 0), 1], -1)
    np.savez_compressed(out / f"{tid}.npz", X=X, rp=rp, rs=rs, y=y, C=C, D=D, T=T, Cu=Cu, Cv=Cv, lab=lab,
                        lab_k=k, lab_dist=ldist, reliable=reliable, t=ctx.t)
    hit = (C == lab[:, None]).any(1)
    summ = dict(trip=tid, partition=part, n_obs=len(lab), n_route_arcs=len(route),
                route_arcs_with_fix=int(len(np.unique(k))), rows=len(y),
                mean_cands=float((C >= 0).sum(1).mean()),
                label_dist_median=float(np.median(ldist)), label_dist_p95=float(np.percentile(ldist, 95)),
                label_dist_max=float(ldist.max()), unreliable_labels=int((~reliable).sum()),
                candidate_miss_rate=float((~hit[reliable]).mean()) if reliable.any() else np.nan,
                **{f"clean_{k2}": v for k2, v in trip.meta.items()})
    return summ, grid


def main():
    cfg = load_config()
    for part, key in [("train", "train_dir"), ("student_test", "student_test_dir")]:
        pdir = cfg["data"][key]
        ids = list_trips(pdir)
        print(f"{part}: {len(ids)} trips")
        res = Parallel(n_jobs=2, verbose=5)(delayed(process)(part, pdir, t, cfg) for t in ids)
        summ = pd.DataFrame([r[0] for r in res])
        (RESULTS / "eda").mkdir(parents=True, exist_ok=True)
        summ.to_csv(RESULTS / "eda" / f"label_alignment_{part}.csv", index=False)
        print(summ.describe().T[["mean", "50%", "max"]])
        if part == "train":
            g = pd.DataFrame([x for r in res for x in r[1]])
            g = g.groupby(["radius", "k"]).agg(n=("n", "sum"), miss=("miss", "sum"),
                                               mean_cands=("mean_cands", "mean")).reset_index()
            g["miss_rate"] = g["miss"] / g["n"]
            g.to_csv(RESULTS / "eda" / "candidate_miss_grid.csv", index=False)
            print(g.pivot(index="radius", columns="k", values="miss_rate").round(4))


if __name__ == "__main__":
    main()
