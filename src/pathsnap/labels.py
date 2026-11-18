"""Observation-level reference labels from the reference route.

Owner: Chiranthan Shankar (PES2UG24CS137)

The ``.route`` file gives the *sequence* of arcs travelled, not which arc each
GPS fix belongs to (there are usually more fixes than arcs, and short arcs may
have no fix at all). We therefore align the track to the route with a
monotonic dynamic programme:

    k_0 <= k_1 <= ... <= k_{N-1}          (positions in the route sequence)
    minimise  sum_i  dist(p_i, route[k_i])

The monotonic constraint means a fix can never be matched to a route arc that
was travelled *before* the arc of the previous fix, which removes ambiguities
caused by loops or the vehicle passing the same street twice. The DP runs in
O(N*R) using a prefix-minimum.

The resulting label of fix i is ``route[k_i]``. Labels are reference
information, not verified truth: where the aligned distance is large (GPS
outliers, route errors) the label is flagged as unreliable (see EDA).
"""
from __future__ import annotations

import numpy as np

from .network import RoadNetwork


def align_track_to_route(pxy: np.ndarray, route: np.ndarray, net: RoadNetwork):
    route = np.asarray(route, int)
    N, R = len(pxy), len(route)
    ax, ay, bx, by = net.ax[route], net.ay[route], net.bx[route], net.by[route]
    from .geo import point_segment

    def cost_row(i):
        d, *_ = point_segment(pxy[i, 0], pxy[i, 1], ax, ay, bx, by)
        return d

    idx = np.arange(R)
    bp = np.empty((N, R), dtype=np.int32)
    prev = cost_row(0)
    bp[0] = idx
    for i in range(1, N):
        pm = np.minimum.accumulate(prev)
        arg = np.maximum.accumulate(np.where(prev <= pm, idx, 0))
        bp[i] = arg
        prev = cost_row(i) + pm
    k = np.empty(N, dtype=np.int64)
    k[-1] = int(np.argmin(prev))
    for i in range(N - 1, 0, -1):
        k[i - 1] = bp[i, k[i]]
    lab = route[k]
    d, *_ = net.dist_to_arcs(pxy, lab)
    return lab, k, d
