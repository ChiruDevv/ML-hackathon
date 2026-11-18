"""Road-network representation, spatial index and bounded shortest paths.

Owner: Chiranthan Shankar (PES2UG24CS137)

Arcs are *directed* straight segments node_from -> node_to. Two-way streets
appear as two arcs with opposite direction and identical geometry, so the
direction of travel is an important part of the matching problem.
"""
from __future__ import annotations

import heapq

import numpy as np
from scipy.spatial import cKDTree

from .geo import LocalProjector, bearing_xy, point_segment


class RoadNetwork:
    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjector,
                 sample_step: float = 10.0):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.xy = proj.to_xy(nodes_lonlat)
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)
        u, v = self.arcs[:, 0], self.arcs[:, 1]
        self.ax, self.ay = self.xy[u, 0], self.xy[u, 1]
        self.bx, self.by = self.xy[v, 0], self.xy[v, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_xy(self.bx - self.ax, self.by - self.ay)
        self.out_degree = np.bincount(u, minlength=self.n_nodes)
        self.in_degree = np.bincount(v, minlength=self.n_nodes)
        # reverse-arc lookup (v,u) for every (u,v)
        key = u * self.n_nodes + v
        rkey = v * self.n_nodes + u
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        hit = key[order][pos] == rkey
        self.reverse_arc = np.where(hit, order[pos], -1)
        # CSR adjacency of outgoing arcs per node
        o = np.argsort(u, kind="stable")
        self._out_arcs = o
        self._out_ptr = np.r_[0, np.cumsum(self.out_degree)]
        self._sp_cache: dict[int, tuple[float, dict]] = {}
        self._build_index(sample_step)

    # ------------------------------------------------------------------ index
    def _build_index(self, step: float):
        n_s = np.maximum(1, np.ceil(self.length / step).astype(int)) + 1
        arc_of = np.repeat(np.arange(self.n_arcs), n_s)
        starts = np.r_[0, np.cumsum(n_s)[:-1]]
        frac = (np.arange(arc_of.size) - np.repeat(starts, n_s)) / np.repeat(n_s - 1, n_s)
        sx = self.ax[arc_of] + frac * (self.bx - self.ax)[arc_of]
        sy = self.ay[arc_of] + frac * (self.by - self.ay)[arc_of]
        self._s_arc = arc_of
        self._tree = cKDTree(np.column_stack([sx, sy]))
        self._step = step

    def arcs_near(self, pxy: np.ndarray, radius: float):
        """All (point_idx, arc_idx, dist, t) pairs with segment distance <= radius."""
        lists = self._tree.query_ball_point(pxy, r=radius + self._step / 2 + 1e-6)
        lens = np.fromiter((len(l) for l in lists), int, len(lists))
        if lens.sum() == 0:
            e = np.empty(0, int)
            return e, e, np.empty(0), np.empty(0)
        pid = np.repeat(np.arange(len(lists)), lens)
        sid = np.fromiter((s for l in lists for s in l), int, lens.sum())
        aid = self._s_arc[sid]
        key = np.unique(pid * self.n_arcs + aid)
        pid, aid = key // self.n_arcs, key % self.n_arcs
        d, t, _, _ = point_segment(pxy[pid, 0], pxy[pid, 1], self.ax[aid], self.ay[aid],
                                   self.bx[aid], self.by[aid])
        m = d <= radius
        return pid[m], aid[m], d[m], t[m]

    def dist_to_arcs(self, pxy: np.ndarray, arc_idx: np.ndarray):
        return point_segment(pxy[..., 0], pxy[..., 1], self.ax[arc_idx], self.ay[arc_idx],
                             self.bx[arc_idx], self.by[arc_idx])

    # ---------------------------------------------------------- connectivity
    def out_arcs(self, node: int) -> np.ndarray:
        return self._out_arcs[self._out_ptr[node]:self._out_ptr[node + 1]]

    def connected(self, a: int, b: int) -> bool:
        """a == b, or b starts where a ends (directly connected)."""
        return a == b or self.arcs[a, 1] == self.arcs[b, 0]

    def node_dists(self, src: int, limit: float) -> dict:
        """Bounded Dijkstra from a node over directed arcs (cached)."""
        c = self._sp_cache.get(src)
        if c is not None and c[0] >= limit:
            return c[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        done = set()
        while heap:
            d, n = heapq.heappop(heap)
            if n in done:
                continue
            done.add(n)
            for a in self.out_arcs(n):
                m = int(self.arcs[a, 1])
                nd = d + self.length[a]
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    heapq.heappush(heap, (nd, m))
        self._sp_cache[src] = (limit, dist)
        return dist

    def node_path(self, src: int, dst: int, limit: float) -> list[int] | None:
        """Shortest arc path src -> dst (list of arc ids), or None."""
        if src == dst:
            return []
        prev = {src: -1}
        dist = {src: 0.0}
        heap = [(0.0, src)]
        while heap:
            d, n = heapq.heappop(heap)
            if n == dst:
                break
            if d > dist.get(n, np.inf):
                continue
            for a in self.out_arcs(n):
                m = int(self.arcs[a, 1])
                nd = d + self.length[a]
                if nd <= limit and nd < dist.get(m, np.inf):
                    dist[m] = nd
                    prev[m] = int(a)
                    heapq.heappush(heap, (nd, m))
        if dst not in prev:
            return None
        path, n = [], dst
        while n != src:
            a = prev[n]
            path.append(a)
            n = int(self.arcs[a, 0])
        return path[::-1]

    def arc_lonlat(self, arc_ids) -> list:
        arc_ids = np.asarray(arc_ids, int)
        u, v = self.arcs[arc_ids, 0], self.arcs[arc_ids, 1]
        return [[[float(self.nodes_lonlat[a, 1]), float(self.nodes_lonlat[a, 0])],
                 [float(self.nodes_lonlat[b, 1]), float(self.nodes_lonlat[b, 0])]]
                for a, b in zip(u, v)]
