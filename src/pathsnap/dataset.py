"""Load cached candidate rows for a partition.

Owner: Chiranthan Shankar (PES2UG24CS137)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CACHE


@dataclass
class TripData:
    trip: str
    X: np.ndarray
    y: np.ndarray
    rp: np.ndarray
    rs: np.ndarray
    C: np.ndarray
    Cu: np.ndarray
    Cv: np.ndarray
    D: np.ndarray
    T: np.ndarray
    lab: np.ndarray
    reliable: np.ndarray
    t: np.ndarray

    @property
    def n(self):
        return len(self.lab)


def load_partition(part: str) -> list[TripData]:
    out = []
    for f in sorted((CACHE / part).glob("*.npz")):
        z = np.load(f)
        out.append(TripData(f.stem, z["X"], z["y"], z["rp"], z["rs"], z["C"], z["Cu"], z["Cv"],
                            z["D"], z["T"], z["lab"], z["reliable"], z["t"]))
    return out


def stack(trips: list[TripData], stride: int = 1, reliable_only: bool = True):
    """Concatenate rows; keep rows of every ``stride``-th fix with a reliable label.

    Returns X, y, group (trip index), point index, slot.
    """
    Xs, ys, gs, ps, ss = [], [], [], [], []
    for gi, td in enumerate(trips):
        keep = (td.rp % stride == 0)
        if reliable_only:
            keep &= td.reliable[td.rp]
        Xs.append(td.X[keep]); ys.append(td.y[keep]); gs.append(np.full(keep.sum(), gi))
        ps.append(td.rp[keep]); ss.append(td.rs[keep])
    return (np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs),
            np.concatenate(ps), np.concatenate(ss))
