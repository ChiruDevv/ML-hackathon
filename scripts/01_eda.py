"""Exploratory data analysis of the raw dataset (train + student-test).

Owner: Harsh Mannan (PES2UG24CS183)

Outputs in results/eda/:
  inventory.csv                 per-trip file sizes, counts, sampling, extent, gaps
  network_connectivity.csv      nodes, arcs, components, isolated nodes per trip
  partition_summary.csv         aggregated per partition
  data_quality_notes.md         short record of issues / assumptions
  figures/*.png                 plots (distributions, trajectories, networks)
  maps/eda_trip_<id>.html       Folium maps (GPS + network + reference route)
Run:  python scripts/01_eda.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from pathsnap.config import RESULTS, load_config
from pathsnap.features import make_context
from pathsnap.geo import haversine_m
from pathsnap.io import EXTS, list_trips, load_reference_route, load_trip
from pathsnap.viz import trip_map

OUT = RESULTS / "eda"
FIG = RESULTS / "figures" / "eda"
MAPS = RESULTS / "maps"


def raw_inventory(part, pdir, tid):
    raw = np.loadtxt(Path(pdir) / tid / f"{tid}.track", ndmin=2)
    trip = load_trip(pdir, tid)
    route = load_reference_route(pdir, tid)
    tr = trip.track
    dt = np.diff(tr[:, 2])
    step = haversine_m(tr[:-1, 0], tr[:-1, 1], tr[1:, 0], tr[1:, 1])
    sizes = {f"size_{e}_kb": (Path(pdir) / tid / f"{tid}.{e}").stat().st_size / 1024
             if (Path(pdir) / tid / f"{tid}.{e}").exists() else np.nan for e in EXTS}
    n = len(trip.nodes)
    A = trip.arcs
    g = coo_matrix((np.ones(len(A)), (A[:, 0], A[:, 1])), shape=(n, n))
    n_wcc, lab_w = connected_components(g, directed=True, connection="weak")
    n_scc, lab_s = connected_components(g, directed=True, connection="strong")
    deg = np.bincount(A[:, 0], minlength=n) + np.bincount(A[:, 1], minlength=n)
    bad_arc = int(((A < 0) | (A >= n)).any(1).sum())
    self_loops = int((A[:, 0] == A[:, 1]).sum())
    route_valid = None if route is None else bool(((route >= 0) & (route < len(A))).all())
    route_conn = None if route is None else float((A[route[:-1], 1] == A[route[1:], 0]).mean())
    ext_x = haversine_m(tr[:, 0].min(), tr[:, 1].mean(), tr[:, 0].max(), tr[:, 1].mean()) / 1000
    ext_y = haversine_m(tr[:, 0].mean(), tr[:, 1].min(), tr[:, 0].mean(), tr[:, 1].max()) / 1000
    inv = dict(partition=part, trip=tid, n_obs=len(tr), raw_rows=len(raw),
               nan_values=int(np.isnan(raw).sum()), duration_min=(tr[-1, 2] - tr[0, 2]) / 60,
               dt_median=float(np.median(dt)), dt_share_1s=float(np.mean(np.isclose(dt, 1.0))),
               dt_max=float(dt.max()), n_gaps_gt5s=int((dt > 5).sum()), n_gaps_gt60s=int((dt > 60).sum()),
               timestamps_monotonic=bool((np.diff(raw[:, 2]) > 0).all()),
               step_median_m=float(np.median(step)), step_max_m=float(step.max()),
               speed_p99_kmh=float(np.percentile(step / np.maximum(dt, 1e-6), 99) * 3.6),
               length_km=float(step.sum() / 1000), lon_min=tr[:, 0].min(), lon_max=tr[:, 0].max(),
               lat_min=tr[:, 1].min(), lat_max=tr[:, 1].max(), extent_ew_km=ext_x, extent_ns_km=ext_y,
               lon_c=float(np.median(tr[:, 0])), lat_c=float(np.median(tr[:, 1])),
               n_route_arcs=None if route is None else len(route), route_ids_valid=route_valid,
               route_consecutive_connected=route_conn, **sizes)
    net = dict(partition=part, trip=tid, n_nodes=n, n_arcs=len(A), invalid_arcs=bad_arc,
               self_loops=self_loops, isolated_nodes=int((deg == 0).sum()),
               weak_components=int(n_wcc), largest_wcc_share=float(np.bincount(lab_w).max() / n),
               strong_components=int(n_scc), largest_scc_share=float(np.bincount(lab_s).max() / n),
               mean_degree=float(deg.mean()),
               two_way_share=float(np.isin(A[:, 0] * n + A[:, 1], A[:, 1] * n + A[:, 0]).mean()))
    return inv, net


def main():
    cfg = load_config()
    for d in (OUT, FIG, MAPS):
        d.mkdir(parents=True, exist_ok=True)
    parts = [("train", cfg["data"]["train_dir"]), ("student_test", cfg["data"]["student_test_dir"])]
    inv, net = [], []
    for part, pdir in parts:
        for tid in list_trips(pdir):
            a, b = raw_inventory(part, pdir, tid)
            inv.append(a); net.append(b)
    inv, net = pd.DataFrame(inv), pd.DataFrame(net)
    inv.to_csv(OUT / "inventory.csv", index=False)
    net.to_csv(OUT / "network_connectivity.csv", index=False)
    summ = inv.groupby("partition").agg(trips=("trip", "count"), obs_total=("n_obs", "sum"),
                                        obs_median=("n_obs", "median"), obs_min=("n_obs", "min"),
                                        obs_max=("n_obs", "max"), nan_values=("nan_values", "sum"),
                                        dt_share_1s=("dt_share_1s", "mean"), max_gap_s=("dt_max", "max"),
                                        trips_with_gap_gt60s=("n_gaps_gt60s", lambda s: int((s > 0).sum())),
                                        length_km_median=("length_km", "median"))
    summ = summ.join(net.groupby("partition").agg(nodes_median=("n_nodes", "median"),
                                                  arcs_median=("n_arcs", "median"),
                                                  isolated_nodes_total=("isolated_nodes", "sum"),
                                                  weak_components_median=("weak_components", "median"),
                                                  largest_scc_share_median=("largest_scc_share", "median")))
    summ.to_csv(OUT / "partition_summary.csv")
    print(summ.T)

    # ------------------------------------------------------------------ plots
    plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for part, c in [("train", "#1f77b4"), ("student_test", "#ff7f0e")]:
        s = inv[inv.partition == part]
        ax[0].hist(s.n_obs, bins=30, alpha=0.7, label=part, color=c)
        ax[1].hist(s.length_km, bins=30, alpha=0.7, label=part, color=c)
        ax[2].hist(s.dt_share_1s * 100, bins=30, alpha=0.7, label=part, color=c)
    ax[0].set(title="GPS fixes per trip", xlabel="fixes", ylabel="trips")
    ax[1].set(title="Trip length", xlabel="km")
    ax[2].set(title="Share of 1-second sampling steps", xlabel="% of steps")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(FIG / "trip_distributions.png"); plt.close(fig)

    dts = []
    for part, pdir in parts:
        for tid in list_trips(pdir):
            t = load_trip(pdir, tid).track[:, 2]
            dts.append(np.diff(t))
    dts = np.concatenate(dts)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(np.clip(dts, 0.5, 1e4), bins=np.logspace(-0.3, 4, 60), color="#1f77b4")
    ax.set(xscale="log", yscale="log", title="Sampling interval between consecutive fixes (all trips)",
           xlabel="seconds (log, clipped at 10^4)", ylabel="count (log)")
    fig.tight_layout(); fig.savefig(FIG / "sampling_intervals.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for part, c, mk in [("train", "#1f77b4", "o"), ("student_test", "#ff7f0e", "^")]:
        s = inv[inv.partition == part]
        ax.scatter(s.lon_c, s.lat_c, c=c, marker=mk, label=part, s=30)
    ax.set(xlim=(-180, 180), ylim=(-60, 75), title="Trip locations (median GPS position)",
           xlabel="longitude", ylabel="latitude"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "trip_locations.png"); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].scatter(net.n_nodes, net.n_arcs, c=np.where(net.partition == "train", "#1f77b4", "#ff7f0e"), s=18)
    ax[0].set(xscale="log", yscale="log", title="Road-network size per trip", xlabel="nodes", ylabel="arcs")
    ax[1].hist(net.largest_scc_share * 100, bins=30, color="#2ca02c")
    ax[1].set(title="Largest strongly connected component", xlabel="% of nodes", ylabel="trips")
    fig.tight_layout(); fig.savefig(FIG / "network_size_connectivity.png"); plt.close(fig)

    # trajectories in metres for 8 training trips
    ids = list_trips(parts[0][1])[:8]
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for ax, tid in zip(axs.ravel(), ids):
        trip = load_trip(parts[0][1], tid)
        ctx = make_context(trip)
        route = load_reference_route(parts[0][1], tid)
        _, aid, _, _ = ctx.net.arcs_near(ctx.pxy[::5], 300)
        for a in np.unique(aid):
            ax.plot([ctx.net.ax[a], ctx.net.bx[a]], [ctx.net.ay[a], ctx.net.by[a]], c="#cccccc", lw=0.5)
        for a in route:
            ax.plot([ctx.net.ax[a], ctx.net.bx[a]], [ctx.net.ay[a], ctx.net.by[a]], c="#2ca02c", lw=2.5, alpha=0.6)
        ax.scatter(ctx.pxy[:, 0], ctx.pxy[:, 1], s=1.5, c="#1f77b4")
        ax.set_title(f"train {tid}  ({len(trip.track)} fixes)", fontsize=10)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("GPS fixes (blue), reference route (green) and nearby network (grey), UTM metres")
    fig.tight_layout(); fig.savefig(FIG / "trajectories_train.png"); plt.close(fig)

    # alignment quality and candidate miss grid (from 02_build_dataset)
    al = OUT / "label_alignment_train.csv"
    if al.exists():
        a = pd.read_csv(al, dtype={"trip": str})
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(range(len(a)), a.sort_values("label_dist_p95").label_dist_p95, color="#9467bd")
        ax.axhline(100, ls="--", c="k", lw=1); ax.set_yscale("log")
        ax.set(title="95th pct distance GPS fix -> aligned reference arc (train)",
               xlabel="trip (sorted)", ylabel="metres (log)")
        fig.tight_layout(); fig.savefig(FIG / "label_alignment_distance.png"); plt.close(fig)
    gr = OUT / "candidate_miss_grid.csv"
    if gr.exists():
        g = pd.read_csv(gr).pivot(index="radius", columns="k", values="miss_rate") * 100
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(g.values, cmap="viridis_r")
        ax.set_xticks(range(len(g.columns)), g.columns); ax.set_yticks(range(len(g.index)), g.index)
        for i in range(g.shape[0]):
            for j in range(g.shape[1]):
                ax.text(j, i, f"{g.values[i, j]:.2f}", ha="center", va="center", color="w", fontsize=9)
        ax.set(xlabel="k (max candidates)", ylabel="radius (m)", title="Candidate miss rate % (train)")
        fig.colorbar(im); fig.tight_layout(); fig.savefig(FIG / "candidate_miss_grid.png"); plt.close(fig)

    # ------------------------------------------------------------- folium maps
    for part, pdir, tid in [("train", parts[0][1], "00000001"), ("student_test", parts[1][1], "00000091")]:
        trip = load_trip(pdir, tid)
        ctx = make_context(trip)
        route = load_reference_route(pdir, tid)
        m = trip_map(trip.track, ctx.net, ctx.pxy, {"Reference route": (route, "reference")},
                     title=f"EDA - {part} trip {tid}: raw GPS, road network, reference route")
        m.save(str(MAPS / f"eda_{part}_{tid}.html"))

    # ------------------------------------------------------- data quality notes
    tr = inv[inv.partition == "train"]
    big = tr.sort_values("dt_max", ascending=False).head(3)
    lines = [
        "# Data-quality notes (generated by scripts/01_eda.py)", "",
        f"* Partitions: {inv.groupby('partition').trip.count().to_dict()} trips; each trip has its own "
        "`.track`, `.nodes`, `.arcs`, `.route` (all present).",
        f"* NaN values in raw tracks: {int(inv.nan_values.sum())}; non-monotonic timestamps: "
        f"{int((~inv.timestamps_monotonic).sum())} trips.",
        f"* Sampling: median interval 1 s, {100 * np.mean(np.isclose(dts, 1)):.1f}% of all steps are exactly 1 s. "
        f"{int((inv.n_gaps_gt60s > 0).sum())} trips have at least one gap > 60 s; largest gaps (train): "
        + ", ".join(f"{r.trip}: {r.dt_max:.0f} s" for r in big.itertuples()) + ".",
        "* Trips are spread over several continents (see trip_locations.png), so every trip is projected "
        "to its own UTM zone; degree-based distances are never used.",
        f"* Road networks: {int(net.n_nodes.min())}-{int(net.n_nodes.max())} nodes per trip, "
        f"{100 * net.two_way_share.median():.0f}% (median) of arcs have a reverse twin (two-way streets) -> "
        "direction must be inferred from GPS heading.",
        f"* Isolated nodes total: {int(net.isolated_nodes.sum())}; median weak components per trip: "
        f"{net.weak_components.median():.0f}; invalid arcs: {int(net.invalid_arcs.sum())}.",
        f"* Reference routes: all arc ids valid = {bool(inv.route_ids_valid.all())}; consecutive route arcs "
        f"connected (median trip) = {100 * inv.route_consecutive_connected.median():.1f}%.",
        "* The route gives the arc *sequence*, not a label per fix; labels per fix come from a monotonic "
        "alignment (src/pathsnap/labels.py). Fixes whose aligned distance exceeds 100 m are flagged "
        "unreliable and excluded from training and scoring (see label_alignment_*.csv).",
    ]
    if al.exists():
        a = pd.read_csv(al, dtype={"trip": str})
        worst = a.sort_values("label_dist_median", ascending=False).head(3)
        lines.append("* Trips with poor GPS/route agreement (median aligned distance): "
                     + ", ".join(f"{r.trip}: {r.label_dist_median:.0f} m" for r in worst.itertuples())
                     + " - treated as reference-route limitations, not deleted.")
    (OUT / "data_quality_notes.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
