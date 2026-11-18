"""Evaluate a candidate scorer on a list of trips.

Owner: Meenakshi P (PES2UG24CS811)
"""
from __future__ import annotations

import time

import numpy as np

from .dataset import TripData
from .metrics import candidate_metrics, continuity, row_decisions, top1_slots


def score_trips(model, trips: list[TripData]):
    """predict_proba for every row of every trip; returns list of arrays and seconds."""
    t0 = time.perf_counter()
    X = np.concatenate([td.X for td in trips])
    p = model.predict_proba(X)[:, 1]
    dt = time.perf_counter() - t0
    out, s = [], 0
    for td in trips:
        out.append(p[s:s + len(td.y)]); s += len(td.y)
    return out, dt


def evaluate(trips: list[TripData], probas: list[np.ndarray], stride: int = 1, with_continuity=True):
    ys, ps, ds = [], [], []
    correct = total = miss = 0
    conts, per_trip = [], []
    for td, p in zip(trips, probas):
        best, _ = top1_slots(p, td.rp, td.rs, td.n, td.C.shape[1])
        dec = row_decisions(best, td.rp, td.rs)
        pts = np.zeros(td.n, bool); pts[::stride] = True
        pts &= td.reliable
        rows = pts[td.rp]
        ys.append(td.y[rows]); ps.append(p[rows]); ds.append(dec[rows])
        pred_arc = np.where(best >= 0, td.C[np.arange(td.n), np.maximum(best, 0)], -1)
        ok = (pred_arc == td.lab)[pts]
        c_miss = ~(td.C == td.lab[:, None]).any(1)[pts]
        correct += ok.sum(); total += pts.sum(); miss += c_miss.sum()
        cont = np.nan
        if with_continuity:
            pu = np.where(best >= 0, td.Cu[np.arange(td.n), np.maximum(best, 0)], -1)
            pv = np.where(best >= 0, td.Cv[np.arange(td.n), np.maximum(best, 0)], -2)
            cont = continuity(pu, pv, pred_arc)
            conts.append(cont)
        per_trip.append(dict(trip=td.trip, top1=float(ok.mean()), n=int(pts.sum()), continuity=cont))
    m = candidate_metrics(np.concatenate(ys), np.concatenate(ps), np.concatenate(ds))
    m.update(top1=float(correct / total), candidate_miss_rate=float(miss / total), n_obs=int(total))
    if with_continuity:
        m["route_continuity"] = float(np.mean(conts))
    return m, per_trip
