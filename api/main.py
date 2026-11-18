"""PathSnap FastAPI backend (runs on Laptop 1).

Owner: Hansika R (PES2UG24CS178)

Endpoints
  GET  /health          model + data status
  GET  /trips           the evaluation trips found in the data directory
  POST /predict         {"trip_id": "..."} -> predicted route + server-side metrics

The reference route is read only inside :func:`pathsnap.scoring.score_prediction`
after the prediction is finished; it is never returned to the client.

Run (Laptop 1):
  python api/main.py --data-dir data/raw/eval --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pathsnap.config import MODELS, load_config
from pathsnap.io import list_trips, load_reference_route, load_trip
from pathsnap.pipeline import load_bundle, predict_trip
from pathsnap.scoring import score_prediction
from pathsnap.viz import route_lines

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout),
                              logging.FileHandler(LOG_DIR / "api.log", encoding="utf-8")])
log = logging.getLogger("pathsnap.api")

CFG = load_config()
DATA_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", CFG["data"]["eval_dir"]))
MODEL_DIR = Path(os.environ.get("PATHSNAP_MODEL_DIR", MODELS))
BUNDLE = load_bundle(MODEL_DIR)
TRIP_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")



class PredictRequest(BaseModel):
    trip_id: str
    use_hmm: bool = True


@asynccontextmanager
async def lifespan(_app):
    log.info("=" * 70)
    log.info("PathSnap API starting")
    log.info("model      : %s  (%s)", BUNDLE["model_name"], MODEL_DIR / BUNDLE["model_file"])
    log.info("decoder    : HMM alpha=%.2f beta=%.1f m", BUNDLE["hmm"]["alpha"], BUNDLE["hmm"]["beta_m"])
    log.info("data dir   : %s", DATA_DIR)
    log.info("trips found: %s", list_trips(DATA_DIR))
    log.info("=" * 70)
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return dict(status="ok", model=BUNDLE["model_name"], data_dir=str(DATA_DIR),
                n_trips=len(list_trips(DATA_DIR)), decoder=BUNDLE["hmm"])


@app.get("/trips")
def trips(request: Request):
    out = []
    for tid in list_trips(DATA_DIR):
        tr = np.loadtxt(DATA_DIR / tid / f"{tid}.track", ndmin=2)
        out.append(dict(trip_id=tid, n_obs=int(len(tr)), duration_s=float(tr[-1, 2] - tr[0, 2])))
    log.info("GET /trips from %s -> %d trips", request.client.host if request.client else "?", len(out))
    return dict(trips=out)


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    client = request.client.host if request.client else "?"
    if not TRIP_ID.match(req.trip_id) or req.trip_id not in list_trips(DATA_DIR):
        log.warning("POST /predict from %s: unknown trip %r", client, req.trip_id)
        raise HTTPException(404, f"trip {req.trip_id!r} not found")
    log.info("-" * 70)
    log.info("POST /predict from %s  trip=%s  hmm=%s", client, req.trip_id, req.use_hmm)
    t0 = time.perf_counter()
    trip = load_trip(DATA_DIR, req.trip_id)                  # GPS + network only
    pred = predict_trip(trip, BUNDLE, use_hmm=req.use_hmm)   # prediction never sees the reference
    tm = pred["timings"]
    log.info("  pipeline : %d fixes, %d candidate rows | network %.2fs features %.2fs classifier %.2fs "
             "viterbi %.2fs route %.2fs", trip.n_obs, int((pred["C"] >= 0).sum()), tm["network_s"],
             tm["features_s"], tm["classifier_s"], tm["decoder_s"], tm["route_s"])
    ref = load_reference_route(DATA_DIR, req.trip_id)        # server-side scoring only
    metrics = score_prediction(pred, ref, CFG["labels"]["reliable_max_dist_m"]) if ref is not None else {}
    metrics.update(point_continuity=pred["point_continuity"], route_continuity=pred["route_continuity"],
                   filled_gaps=pred["filled_gaps"], route_breaks=pred["route_breaks"],
                   n_route_arcs=len(pred["route_arcs"]))
    total = time.perf_counter() - t0
    if ref is not None:
        log.info("  accuracy : observation=%.4f (classifier-only %.4f) | route P/R/F1=%.3f/%.3f/%.3f | "
                 "continuity=%.4f", metrics["observation_accuracy"], metrics["classifier_only_accuracy"],
                 metrics["route_precision"], metrics["route_recall"], metrics["route_f1"],
                 metrics["route_continuity"])
    log.info("  done in %.2fs", total)

    net = pred["ctx"].net
    arcs = pred["point_arcs"]
    pxy = pred["ctx"].pxy
    d, t, qx, qy = net.dist_to_arcs(pxy, np.maximum(arcs, 0))
    snapped = pred["ctx"].proj.to_lonlat(np.column_stack([qx, qy]))
    return dict(
        trip_id=req.trip_id, model=BUNDLE["model_name"], n_obs=trip.n_obs,
        gps=[[round(float(a), 6), round(float(b), 6)] for b, a in trip.track[:, :2]],
        snapped=[[round(float(a), 6), round(float(b), 6)] if arcs[i] >= 0 else None
                 for i, (b, a) in enumerate(snapped)],
        confidence=[round(float(c), 3) for c in pred["confidence"]],
        route=route_lines(net, pred["route_arcs"]),
        metrics={k: (round(v, 5) if isinstance(v, float) else v) for k, v in metrics.items()},
        timings={**{k: round(v, 3) for k, v in tm.items()}, "request_s": round(total, 3)},
    )


if __name__ == "__main__":
    import uvicorn
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default=None, help="folder with the evaluation trips")
    a = ap.parse_args()
    if a.data_dir:
        os.environ["PATHSNAP_EVAL_DIR"] = str(Path(a.data_dir).resolve())
    uvicorn.run("main:app", app_dir=str(Path(__file__).parent), host=a.host, port=a.port, log_level="info")
