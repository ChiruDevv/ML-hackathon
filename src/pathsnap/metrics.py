"""Evaluation metrics shared by model comparison, HMM evaluation and the API.

Owner: Meenakshi P (PES2UG24CS811)

Definitions used throughout the project
---------------------------------------
* Observation-level Top-1 accuracy: for each GPS fix take the candidate with
  the highest score; correct if it equals the reference arc of that fix.
  Fixes whose reference arc is missing from the candidate set count as wrong.
* Candidate Miss Rate: share of fixes whose reference arc is not a candidate.
* Macro F1 (candidate level): every (fix, candidate) row is a binary decision.
  The predicted positive of a fix is its top-ranked candidate (the decision
  rule the system actually uses); F1 is averaged over the two classes.
* Log loss: binary cross-entropy of the predicted candidate probabilities.
* Route continuity: share of consecutive fix pairs whose predicted arcs are
  the same arc or directly connected (end node of a == start node of b).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score


def top1_slots(scores: np.ndarray, rp: np.ndarray, rs: np.ndarray, n_points: int, k: int):
    """Best slot per point (-1 if the point has no candidate)."""
    S = np.full((n_points, k), -np.inf)
    S[rp, rs] = scores
    best = S.argmax(1)
    best[~np.isfinite(S.max(1))] = -1
    return best, S


def row_decisions(best: np.ndarray, rp: np.ndarray, rs: np.ndarray):
    return (best[rp] == rs).astype(np.int8)


def continuity(pred_from: np.ndarray, pred_to: np.ndarray, pred_arc: np.ndarray) -> float:
    if len(pred_arc) < 2:
        return 1.0
    a, b = slice(None, -1), slice(1, None)
    ok = (pred_arc[a] == pred_arc[b]) | (pred_to[a] == pred_from[b])
    ok &= (pred_arc[a] >= 0) & (pred_arc[b] >= 0)
    return float(ok.mean())


def candidate_metrics(y: np.ndarray, proba: np.ndarray, decision: np.ndarray) -> dict:
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    return dict(
        macro_f1=float(f1_score(y, decision, average="macro")),
        precision=float(precision_score(y, decision, zero_division=0)),
        recall=float(recall_score(y, decision, zero_division=0)),
        log_loss=float(log_loss(y, p, labels=[0, 1])),
    )


def route_set_metrics(pred_arcs, ref_arcs, arc_length) -> dict:
    """Sequence-level metrics: length-weighted precision / recall / F1 between
    the set of arcs on the predicted route and the reference route."""
    P, R = set(map(int, pred_arcs)), set(map(int, ref_arcs))
    inter = P & R
    L = lambda s: float(sum(arc_length[a] for a in s)) if s else 0.0
    prec = L(inter) / L(P) if P else 0.0
    rec = L(inter) / L(R) if R else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(route_precision=prec, route_recall=rec, route_f1=f1,
                route_arc_jaccard=len(inter) / max(len(P | R), 1))
