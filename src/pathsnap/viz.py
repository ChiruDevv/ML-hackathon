"""Folium map generation.

Owner: Harsh Mannan (PES2UG24CS183)
"""
from __future__ import annotations

import folium
import numpy as np
from folium.plugins import Fullscreen

from .network import RoadNetwork

COLORS = {"gps": "#1f77b4", "network": "#9e9e9e", "reference": "#2ca02c",
          "classifier": "#ff7f0e", "hmm": "#d62728"}


def _legend(items):
    rows = "".join(f'<div><span style="display:inline-block;width:18px;height:4px;background:{c};'
                   f'margin-right:6px;vertical-align:middle"></span>{n}</div>' for n, c in items)
    html = (f'<div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:white;'
            f'padding:8px 10px;border:1px solid #888;border-radius:6px;font:13px sans-serif">'
            f'<b>Legend</b>{rows}</div>')
    return folium.Element(html)


def arcs_multiline(net: RoadNetwork, arc_ids):
    return net.arc_lonlat(arc_ids)


def route_lines(net: RoadNetwork, arc_ids):
    """Merge a connected arc sequence into as few polylines as possible."""
    lines, cur = [], []
    for a in arc_ids:
        u, v = net.arcs[a]
        pu = [float(net.nodes_lonlat[u, 1]), float(net.nodes_lonlat[u, 0])]
        pv = [float(net.nodes_lonlat[v, 1]), float(net.nodes_lonlat[v, 0])]
        if cur and cur[-1] == pu:
            cur.append(pv)
        else:
            if cur:
                lines.append(cur)
            cur = [pu, pv]
    if cur:
        lines.append(cur)
    return lines


def trip_map(track_lonlat: np.ndarray, net: RoadNetwork | None = None, pxy=None,
             routes: dict | None = None, title: str = "", network_radius=150.0, gps_every=1):
    """Folium map with GPS fixes, nearby road network and any number of routes.

    routes: {layer_name: (arc_ids, colour_key)}
    """
    lat, lon = track_lonlat[:, 1], track_lonlat[:, 0]
    m = folium.Map(location=[float(lat.mean()), float(lon.mean())], zoom_start=13,
                   tiles="OpenStreetMap", control_scale=True)
    Fullscreen().add_to(m)
    legend = []
    if net is not None and pxy is not None:
        _, aid, _, _ = net.arcs_near(pxy[::5], network_radius)
        aid = np.unique(aid)
        fg = folium.FeatureGroup(name=f"Road network (arcs within {int(network_radius)} m)", show=True)
        folium.PolyLine(arcs_multiline(net, aid), color=COLORS["network"], weight=1.5, opacity=0.6).add_to(fg)
        fg.add_to(m); legend.append(("Road network", COLORS["network"]))
    for name, (arcs, ck) in (routes or {}).items():
        fg = folium.FeatureGroup(name=name, show=True)
        w = 6 if ck == "reference" else 3.5
        folium.PolyLine(route_lines(net, arcs), color=COLORS[ck], weight=w,
                        opacity=0.55 if ck == "reference" else 0.9, tooltip=name).add_to(fg)
        fg.add_to(m); legend.append((name, COLORS[ck]))
    fg = folium.FeatureGroup(name="Raw GPS fixes", show=True)
    for i in range(0, len(lat), gps_every):
        folium.CircleMarker([float(lat[i]), float(lon[i])], radius=2, color=COLORS["gps"],
                            fill=True, fill_opacity=0.8, weight=0).add_to(fg)
    fg.add_to(m); legend.append(("Raw GPS fixes", COLORS["gps"]))
    folium.Marker([float(lat[0]), float(lon[0])], tooltip="Start",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker([float(lat[-1]), float(lon[-1])], tooltip="End",
                  icon=folium.Icon(color="red", icon="stop")).add_to(m)
    m.fit_bounds([[float(lat.min()), float(lon.min())], [float(lat.max()), float(lon.max())]])
    folium.LayerControl(collapsed=False).add_to(m)
    if title:
        m.get_root().html.add_child(folium.Element(
            f'<div style="position:fixed;top:10px;left:60px;z-index:9999;background:white;padding:6px 10px;'
            f'border:1px solid #888;border-radius:6px;font:14px sans-serif"><b>{title}</b></div>'))
    m.get_root().html.add_child(_legend(legend))
    return m
