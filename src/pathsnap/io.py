"""Raw-data readers for the PathSnap dataset.

Owner: Harsh Mannan (PES2UG24CS183)

Each trip lives in its own folder ``<partition>/<trip_id>/`` and contains

* ``<id>.track`` - lon, lat, timestamp[s]   (whitespace separated, no header)
* ``<id>.nodes`` - lon, lat                  (row index = node id)
* ``<id>.arcs``  - from_node, to_node        (row index = arc id)
* ``<id>.route`` - ground-truth arc ids in travel order (reference only)

The reference route is deliberately loaded by a *separate* function
(:func:`load_reference_route`) so that the prediction pipeline never touches
it. Only the server-side scoring code and offline training/evaluation call it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

EXTS = ("track", "nodes", "arcs", "route")


@dataclass
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) lon, lat, t
    nodes: np.ndarray            # (M, 2) lon, lat
    arcs: np.ndarray             # (A, 2) int from, to
    meta: dict = field(default_factory=dict)

    @property
    def n_obs(self) -> int:
        return len(self.track)


def list_trips(partition_dir: str | os.PathLike) -> list[str]:
    """Trip ids (folder names) inside a partition directory, sorted."""
    p = Path(partition_dir)
    if not p.is_dir():
        return []
    return sorted(d.name for d in p.iterdir()
                  if d.is_dir() and (d / f"{d.name}.track").exists())


def _path(partition_dir, trip_id, ext) -> Path:
    return Path(partition_dir) / trip_id / f"{trip_id}.{ext}"


def _read(path: Path, dtype=float, ndmin=2) -> np.ndarray:
    # pandas-free fast reader; files have no header and are whitespace separated
    arr = np.loadtxt(path, dtype=dtype, ndmin=ndmin)
    return arr


def clean_track(track: np.ndarray) -> tuple[np.ndarray, dict]:
    """Basic cleaning of a raw GPS track.

    * drop rows containing NaN / inf
    * drop rows with impossible coordinates (|lat|>90, |lon|>180)
    * sort by timestamp and drop exact duplicate timestamps (keep first)

    Returns the cleaned track and a dict describing what was removed.
    """
    info = {"raw_rows": int(len(track))}
    finite = np.isfinite(track).all(axis=1)
    valid = finite & (np.abs(track[:, 1]) <= 90) & (np.abs(track[:, 0]) <= 180)
    info["dropped_invalid"] = int((~valid).sum())
    track = track[valid]
    order = np.argsort(track[:, 2], kind="stable")
    info["was_unsorted"] = bool((order != np.arange(len(order))).any())
    track = track[order]
    if len(track) > 1:
        keep = np.r_[True, np.diff(track[:, 2]) > 0]
    else:
        keep = np.ones(len(track), bool)
    info["dropped_duplicate_time"] = int((~keep).sum())
    track = track[keep]
    info["clean_rows"] = int(len(track))
    return track, info


def load_trip(partition_dir: str | os.PathLike, trip_id: str, clean: bool = True) -> Trip:
    """Load GPS track and road network of one trip (no reference route)."""
    track = _read(_path(partition_dir, trip_id, "track"))
    nodes = _read(_path(partition_dir, trip_id, "nodes"))
    arcs = _read(_path(partition_dir, trip_id, "arcs"), dtype=np.int64)
    meta = {}
    if clean:
        track, meta = clean_track(track)
    return Trip(trip_id=trip_id, track=track, nodes=nodes, arcs=arcs, meta=meta)


def load_reference_route(partition_dir: str | os.PathLike, trip_id: str) -> np.ndarray | None:
    """Reference arc sequence. Used ONLY for labelling / server-side scoring."""
    p = _path(partition_dir, trip_id, "route")
    if not p.exists():
        return None
    return _read(p, dtype=np.int64, ndmin=1)
