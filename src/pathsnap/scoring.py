"""Server-side scoring against the reference route.

Owner: Hansika R (PES2UG24CS178)

Called by the API *after* the prediction has been produced. Only aggregate
numbers leave this module; the reference arcs are never sent to the UI.
"""
from __future__ import annotations

import numpy as np

from .labels import align_track_to_route
from .metrics import route_set_metrics


def score_prediction(pred: dict, route_ref: np.ndarray, reliable_max_dist=100.0) -> dict:
    ctx = pred["ctx"]
    lab, _, ldist = align_track_to_route(ctx.pxy, route_ref, ctx.net)
    rel = ldist <= reliable_max_dist
    acc = float((pred["point_arcs"] == lab)[rel].mean()) if rel.any() else float("nan")
    acc_clf = float((pred["clf_point_arcs"] == lab)[rel].mean()) if rel.any() else float("nan")
    miss = float((~(pred["C"] == lab[:, None]).any(1))[rel].mean()) if rel.any() else float("nan")
    out = dict(observation_accuracy=acc, classifier_only_accuracy=acc_clf, candidate_miss_rate=miss,
               scored_observations=int(rel.sum()), excluded_unreliable=int((~rel).sum()))
    out.update(route_set_metrics(pred["route_arcs"], route_ref, ctx.net.length))
    return out
