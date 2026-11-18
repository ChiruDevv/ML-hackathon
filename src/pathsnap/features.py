"""Candidate generation and feature engineering.

Owner: Chiranthan Shankar (PES2UG24CS137)

Problem framing
---------------
For every GPS fix i we generate a small candidate set C_i of nearby directed
arcs (all arcs within ``radius`` metres, keep the ``k`` closest). Each
(fix, candidate) pair becomes one row; the binary target is
``candidate == reference arc of fix i``. A classifier scores the rows, the
best-scoring candidate is the classifier-only prediction, and the scores are
the emission probabilities of the HMM.

All features are expressed in metres / radians / seconds, so they transfer
between trips in different countries and with different road networks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import LocalProjector, angle_diff, bearing_xy
from .io import Trip
from .network import RoadNetwork

FEATURE_NAMES = [
    "dist", "dist_rank", "dist_minus_min", "dist_ratio", "gauss_post",
    "cos_heading", "abs_heading_diff", "cos_heading_long", "heading_disp",
    "speed", "turn_rate", "dt_prev", "dt_next",
    "t_proj", "at_endpoint", "log_arc_len", "deg_from", "deg_to", "has_reverse",
    "n_cands", "win_dist_mean", "win_dist_max", "win_rank", "win_minus_min",
]


@dataclass
class TripContext:
    trip: Trip
    proj: LocalProjector
    net: RoadNetwork
    pxy: np.ndarray          # (N, 2) metres
    t: np.ndarray            # (N,) seconds


def make_context(trip: Trip, sample_step: float = 10.0) -> TripContext:
    proj = LocalProjector.for_track(trip.track[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj, sample_step=sample_step)
    pxy = proj.to_xy(trip.track[:, :2])
    return TripContext(trip, proj, net, pxy, trip.track[:, 2].astype(float))


# ---------------------------------------------------------------- kinematics
def kinematics(pxy: np.ndarray, t: np.ndarray, w: int = 2, w_long: int = 5):
    N = len(pxy)
    i = np.arange(N)

    def window_heading(w):
        a, b = np.clip(i - w, 0, N - 1), np.clip(i + w, 0, N - 1)
        d = pxy[b] - pxy[a]
        return bearing_xy(d[:, 0], d[:, 1]), np.hypot(d[:, 0], d[:, 1]), np.maximum(t[b] - t[a], 1e-6)

    h, disp, dtw = window_heading(w)
    h_long, _, _ = window_heading(w_long)
    speed = disp / dtw
    dt_prev = np.r_[0.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 0.0]
    # turn rate between successive short headings
    h_prev = h[np.clip(i - w, 0, N - 1)]
    h_next = h[np.clip(i + w, 0, N - 1)]
    turn = np.abs(angle_diff(h_next, h_prev))
    return dict(heading=h, heading_long=h_long, disp=disp, speed=speed,
                dt_prev=dt_prev, dt_next=dt_next, turn=turn)


# ------------------------------------------------------------- candidates
def generate_candidates(ctx: TripContext, radius: float = 75.0, k: int = 8,
                        fallback_radius: float = 250.0):
    """Return padded arrays (N, k): arc ids (-1 = empty), distance, t."""
    N = len(ctx.pxy)
    pid, aid, d, tt = ctx.net.arcs_near(ctx.pxy, radius)
    have = np.bincount(pid, minlength=N) > 0
    if (~have).any() and fallback_radius > radius:
        miss = np.where(~have)[0]
        p2, a2, d2, t2 = ctx.net.arcs_near(ctx.pxy[miss], fallback_radius)
        pid = np.r_[pid, miss[p2]]; aid = np.r_[aid, a2]; d = np.r_[d, d2]; tt = np.r_[tt, t2]
    order = np.lexsort((d, pid))
    pid, aid, d, tt = pid[order], aid[order], d[order], tt[order]
    first = np.r_[0, np.flatnonzero(np.diff(pid)) + 1]
    rank = np.arange(len(pid)) - np.repeat(first, np.diff(np.r_[first, len(pid)]))
    m = rank < k
    pid, aid, d, tt, rank = pid[m], aid[m], d[m], tt[m], rank[m]
    C = np.full((N, k), -1, np.int64)
    D = np.full((N, k), np.nan)
    T = np.full((N, k), np.nan)
    C[pid, rank] = aid; D[pid, rank] = d; T[pid, rank] = tt
    return C, D, T


# ---------------------------------------------------------------- features
def build_features(ctx: TripContext, C: np.ndarray, D: np.ndarray, T: np.ndarray,
                   sigma: float = 15.0, win: int = 3):
    """Feature matrix for all valid (fix, candidate) rows.

    Returns X (n_rows, F) float32, row_point (n_rows,), row_slot (n_rows,).
    """
    net = ctx.net
    N, K = C.shape
    kin = kinematics(ctx.pxy, ctx.t)
    valid = C >= 0
    row_point, row_slot = np.nonzero(valid)
    arc = C[row_point, row_slot]
    d = D[row_point, row_slot]
    n_c = valid.sum(1)
    dmin = np.nanmin(np.where(valid, D, np.inf), axis=1)
    dmin = np.where(np.isfinite(dmin), dmin, 0.0)

    g = np.where(valid, np.exp(-0.5 * (np.nan_to_num(D, nan=1e9) / sigma) ** 2), 0.0)
    g = g / np.maximum(g.sum(1, keepdims=True), 1e-300)

    hd = angle_diff(kin["heading"][row_point], net.bearing[arc])
    hd_long = angle_diff(kin["heading_long"][row_point], net.bearing[arc])

    # temporal window: distance of neighbouring fixes to the same arc
    offs = np.arange(-win, win + 1)
    nb = np.clip(row_point[:, None] + offs[None, :], 0, N - 1)
    wd, *_ = net.dist_to_arcs(ctx.pxy[nb], np.broadcast_to(arc[:, None], nb.shape))
    wmean, wmax = wd.mean(1), wd.max(1)
    W = np.full((N, K), np.inf)
    W[row_point, row_slot] = wmean
    wrank = np.argsort(np.argsort(W, axis=1), axis=1)[row_point, row_slot]
    wmin = W.min(1)[row_point]

    u, v = net.arcs[arc, 0], net.arcs[arc, 1]
    tp = T[row_point, row_slot]
    X = np.column_stack([
        d, row_slot, d - dmin[row_point], (d + 1.0) / (dmin[row_point] + 1.0),
        g[row_point, row_slot],
        np.cos(hd), np.abs(hd), np.cos(hd_long), kin["disp"][row_point],
        kin["speed"][row_point], kin["turn"][row_point],
        np.clip(kin["dt_prev"][row_point], 0, 600), np.clip(kin["dt_next"][row_point], 0, 600),
        tp, ((tp <= 1e-6) | (tp >= 1 - 1e-6)).astype(float), np.log1p(net.length[arc]),
        net.out_degree[u] + net.in_degree[u], net.out_degree[v] + net.in_degree[v],
        (net.reverse_arc[arc] >= 0).astype(float),
        n_c[row_point], wmean, wmax, wrank, wmean - wmin,
    ]).astype(np.float32)
    return X, row_point, row_slot


def trip_rows(ctx: TripContext, radius=75.0, k=8):
    C, D, T = generate_candidates(ctx, radius=radius, k=k)
    X, rp, rs = build_features(ctx, C, D, T)
    return C, D, T, X, rp, rs
