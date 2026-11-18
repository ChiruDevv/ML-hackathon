"""HMM / Viterbi route reconstruction.

Owner: Hansika R (PES2UG24CS178)

Hidden states
    At time step i the hidden state is "the vehicle is on candidate arc c of
    fix i" (a candidate slot; each slot keeps its arc id, projection
    fraction and distance, so the candidate-to-arc correspondence is
    preserved).

Emission  log P(z_i | c)
    ``alpha * log p_clf(c | i)`` where p_clf is the classifier probability
    renormalised over the candidates of fix i. A distance-only baseline uses
    a Gaussian on the perpendicular distance instead (Newson & Krumm 2009).

Transition  log P(c_i | c_{i-1})
    Compare the network distance between the two projected positions with
    the straight-line distance between the two GPS fixes:
        log T = -| d_route - d_gc | / beta
    d_route uses the same arc (forward movement along it, small backward
    jitter tolerated) or the bounded Dijkstra shortest path end(a) -> start(b).
    Pairs with no path inside the search limit get ``min_log_prob`` instead
    of -inf, so a disconnected network or a bad fix never breaks the chain.

Gaps
    The search limit grows with the time gap. If two fixes are more than
    ``gap_break_s`` apart the chain is restarted (uniform transition) because
    the travelled path cannot be inferred reliably.
"""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork


def emission_logp(P: np.ndarray, D: np.ndarray, C: np.ndarray, mode="classifier", alpha=1.0, sigma=15.0):
    valid = C >= 0
    if mode == "classifier":
        Pn = np.where(valid, np.clip(P, 1e-9, 1), 0.0)
        Pn = Pn / np.maximum(Pn.sum(1, keepdims=True), 1e-12)
        E = alpha * np.log(np.maximum(Pn, 1e-12))
    else:
        E = -0.5 * (np.nan_to_num(D, nan=1e6) / sigma) ** 2
        E = E - np.logaddexp.reduce(np.where(valid, E, -np.inf), axis=1, keepdims=True)
    return np.where(valid, E, -np.inf)


def viterbi(net: RoadNetwork, pxy, t, C, T, E, beta=20.0, back_tol=15.0, min_log_prob=-30.0,
            gap_break_s=600.0):
    """Return best slot per fix (-1 for fixes with no candidates)."""
    N, K = C.shape
    has = (C >= 0).any(1)
    idx = np.flatnonzero(has)
    best = np.full(N, -1, int)
    if len(idx) == 0:
        return best
    L = net.length
    delta = E[idx[0]].copy()
    bps = []
    for j in range(1, len(idx)):
        i0, i1 = idx[j - 1], idx[j]
        dt = t[i1] - t[i0]
        dgc = float(np.hypot(*(pxy[i1] - pxy[i0])))
        if dt > gap_break_s:
            prev = int(np.argmax(delta))
            bps.append(np.full(K, prev))
            delta = E[i1] + delta[prev]
            continue
        limit = min(max(3.0 * dgc + 200.0, 40.0 * dt, 500.0), 20000.0)
        logT = np.full((K, K), -np.inf)
        for a in range(K):
            A = C[i0, a]
            if A < 0 or not np.isfinite(delta[a]):
                continue
            ta = T[i0, a]
            sp = net.node_dists(int(net.arcs[A, 1]), limit)
            rest = (1.0 - ta) * L[A]
            for b in range(K):
                B = C[i1, b]
                if B < 0:
                    continue
                tb = T[i1, b]
                if B == A:
                    fw = (tb - ta) * L[A]
                    dr = fw if fw >= -back_tol else None
                    dr = max(dr, 0.0) if dr is not None else None
                    if dr is None:   # going backwards along an arc: loop around
                        d0 = sp.get(int(net.arcs[A, 0]))
                        dr = None if d0 is None else rest + d0 + tb * L[A]
                else:
                    d0 = sp.get(int(net.arcs[B, 0]))
                    dr = None if d0 is None else rest + d0 + tb * L[B]
                logT[a, b] = min_log_prob if dr is None else max(-abs(dr - dgc) / beta, min_log_prob)
        cand = delta[:, None] + logT
        bp = np.argmax(cand, axis=0)
        delta = cand[bp, np.arange(K)] + E[i1]
        if not np.isfinite(delta).any():   # should not happen thanks to the floor
            delta = E[i1].copy()
        bps.append(bp)
    s = int(np.argmax(delta))
    best[idx[-1]] = s
    for j in range(len(idx) - 1, 0, -1):
        s = int(bps[j - 1][s])
        best[idx[j - 1]] = s
    return best


def reconstruct_route(net: RoadNetwork, point_arcs: np.ndarray, max_fill_m=5000.0):
    """Collapse per-fix arcs into a connected arc sequence.

    Consecutive duplicates are merged; if two consecutive arcs are not
    connected, the shortest path between them is inserted (if one exists
    within ``max_fill_m``), otherwise the route keeps a visible break.
    Returns (route_arcs, n_filled_gaps, n_unfilled_breaks).
    """
    seq = [int(a) for a in point_arcs if a >= 0]
    seq = [a for i, a in enumerate(seq) if i == 0 or a != seq[i - 1]]
    if not seq:
        return [], 0, 0
    out, filled, breaks = [seq[0]], 0, 0
    for b in seq[1:]:
        a = out[-1]
        if net.arcs[a, 1] != net.arcs[b, 0]:
            path = net.node_path(int(net.arcs[a, 1]), int(net.arcs[b, 0]), max_fill_m)
            if path is not None:
                out.extend(path); filled += 1
            else:
                breaks += 1
        out.append(b)
    return out, filled, breaks


def route_continuity(net: RoadNetwork, route_arcs) -> float:
    r = np.asarray(route_arcs, int)
    if len(r) < 2:
        return 1.0
    return float((net.arcs[r[:-1], 1] == net.arcs[r[1:], 0]).mean())
