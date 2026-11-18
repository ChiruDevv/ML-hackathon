"""End-to-end inference: raw trip -> matched arcs per fix -> connected route.

Owner: Hansika R (PES2UG24CS178)

This module never reads the reference route. Scoring against the reference
lives in :mod:`pathsnap.scoring` and is only called after prediction.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from .config import MODELS

from .features import FEATURE_NAMES, build_features, generate_candidates, make_context
from .hmm import emission_logp, reconstruct_route, route_continuity, viterbi
from .io import Trip
from .metrics import continuity


BUNDLE_FILE = "pathsnap_bundle.json"


def save_bundle_meta(meta: dict, models_dir=None):
    d = Path(models_dir or MODELS)
    (d / BUNDLE_FILE).write_text(json.dumps(meta, indent=2) + "\n")


def load_bundle(models_dir=None) -> dict:
    """Load model weights + inference configuration from ``models/``."""
    d = Path(models_dir or MODELS)
    b = json.loads((d / BUNDLE_FILE).read_text())
    assert b["feature_names"] == FEATURE_NAMES, "feature definition changed since training"
    m = XGBClassifier()
    m.load_model(d / b["model_file"])
    b["model"] = m
    return b


def predict_trip(trip: Trip, bundle: dict, use_hmm: bool = True) -> dict:
    timings = {}
    t0 = time.perf_counter()
    cc, hc = bundle["candidates"], bundle["hmm"]
    ctx = make_context(trip, cc["index_sample_step_m"])
    timings["network_s"] = time.perf_counter() - t0

    t1 = time.perf_counter()
    C, D, T = generate_candidates(ctx, radius=cc["radius_m"], k=cc["k"])
    X, rp, rs = build_features(ctx, C, D, T)
    timings["features_s"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    p = bundle["model"].predict_proba(X)[:, 1]
    P = np.zeros(C.shape); P[rp, rs] = p
    timings["classifier_s"] = time.perf_counter() - t2

    S = np.where(C >= 0, P, -np.inf)
    clf_slot = np.where((C >= 0).any(1), S.argmax(1), -1)
    t3 = time.perf_counter()
    if use_hmm:
        E = emission_logp(P, D, C, "classifier", alpha=hc["alpha"])
        slot = viterbi(ctx.net, ctx.pxy, ctx.t, C, T, E, beta=hc["beta_m"],
                       back_tol=hc["backtrack_tolerance_m"], min_log_prob=hc["min_log_prob"],
                       gap_break_s=hc["gap_break_s"])
    else:
        slot = clf_slot
    timings["decoder_s"] = time.perf_counter() - t3

    rows = np.arange(len(C))
    arc = np.where(slot >= 0, C[rows, np.maximum(slot, 0)], -1)
    clf_arc = np.where(clf_slot >= 0, C[rows, np.maximum(clf_slot, 0)], -1)
    t4 = time.perf_counter()
    route, filled, breaks = reconstruct_route(ctx.net, arc)
    timings["route_s"] = time.perf_counter() - t4
    timings["total_s"] = time.perf_counter() - t0

    net = ctx.net
    pt_cont = lambda a: continuity(np.where(a >= 0, net.arcs[np.maximum(a, 0), 0], -1),
                                   np.where(a >= 0, net.arcs[np.maximum(a, 0), 1], -2), a)
    conf = np.where(slot >= 0, P[rows, np.maximum(slot, 0)], 0.0)
    return dict(ctx=ctx, C=C, P=P, point_arcs=arc, clf_point_arcs=clf_arc, confidence=conf,
                route_arcs=route, filled_gaps=filled, route_breaks=breaks,
                point_continuity=pt_cont(arc), clf_point_continuity=pt_cont(clf_arc),
                route_continuity=route_continuity(net, route), timings=timings)
