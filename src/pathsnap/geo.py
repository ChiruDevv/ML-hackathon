"""Coordinate handling.

Owner: Harsh Mannan (PES2UG24CS183)

Longitude/latitude are angular, so Euclidean distance in degrees is wrong
(1 deg of longitude is ~111 km at the equator but ~55 km at 60 N, and the
dataset has trips from 30 S to 62 N). Every trip is projected to its own
UTM zone (chosen from the track centroid). UTM is conformal and its scale
error inside one zone is < 0.1 %, which is negligible next to GPS noise
(~5-20 m) for trips that span at most a few tens of kilometres.
"""
from __future__ import annotations

import numpy as np
from pyproj import Transformer


def utm_epsg(lon: float, lat: float) -> int:
    zone = int(np.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


class LocalProjector:
    """lon/lat <-> metres in the UTM zone of a reference point."""

    def __init__(self, lon0: float, lat0: float):
        self.epsg = utm_epsg(lon0, lat0)
        self._fwd = Transformer.from_crs("EPSG:4326", f"EPSG:{self.epsg}", always_xy=True)
        self._inv = Transformer.from_crs(f"EPSG:{self.epsg}", "EPSG:4326", always_xy=True)

    @classmethod
    def for_track(cls, track_lonlat: np.ndarray) -> "LocalProjector":
        lon0, lat0 = np.median(track_lonlat[:, 0]), np.median(track_lonlat[:, 1])
        return cls(float(lon0), float(lat0))

    def to_xy(self, lonlat: np.ndarray) -> np.ndarray:
        x, y = self._fwd.transform(lonlat[:, 0], lonlat[:, 1])
        return np.column_stack([x, y])

    def to_lonlat(self, xy: np.ndarray) -> np.ndarray:
        lon, lat = self._inv.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([lon, lat])


def haversine_m(lon1, lat1, lon2, lat2):
    """Great-circle distance in metres (used for sanity checks / EDA)."""
    r = 6371008.8
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def bearing_xy(dx, dy):
    """Grid bearing in radians, 0 = north, clockwise positive."""
    return np.arctan2(dx, dy)


def angle_diff(a, b):
    """Signed smallest difference a-b in (-pi, pi]."""
    d = (a - b + np.pi) % (2 * np.pi) - np.pi
    return d


def point_segment(px, py, ax, ay, bx, by):
    """Vectorised point-to-segment distance.

    Returns (distance, t, qx, qy) where t in [0,1] is the projection fraction
    along a->b and (qx,qy) is the closest point on the segment.
    """
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    L2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(L2 > 0, (wx * vx + wy * vy) / L2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    qx, qy = ax + t * vx, ay + t * vy
    d = np.hypot(px - qx, py - qy)
    return d, t, qx, qy
