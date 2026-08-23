"""Loading, cleaning and projection of GridKit spatial power-grid networks.

Conventions match the rest of the repo: scripts insert the repo root on
sys.path and import this module as `task_47.data_loading`. All tunable
parameters come from `config/task_47/config.yaml` via `common.utils.load_config`.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx

BASE_DIR = Path(__file__).parent.parent.parent


def lonlat_to_xy_km(lon, lat, lon0=None, lat0=None):
    """Equirectangular projection of lon/lat (deg) to planar x/y (km).

    Centered on the network centroid; R = 6371 km. Suitable for first-pass
    distance-based fitting over a continent (documented limitation: not a
    conformal projection).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if lon0 is None:
        lon0 = np.nanmean(lon)
    if lat0 is None:
        lat0 = np.nanmean(lat)

    R = 6371.0  # Earth radius in km

    lon_rad = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    lon0_rad = np.deg2rad(lon0)
    lat0_rad = np.deg2rad(lat0)

    x = R * (lon_rad - lon0_rad) * np.cos(lat0_rad)
    y = R * (lat_rad - lat0_rad)

    return np.column_stack([x, y])


def load_gridkit(root, vertices_file, links_file, largest_component=True,
                 collapse_multiedges=True):
    """Load a GridKit network and return coordinates + integer edge list.

    Returns a dict with ``points_xy_km`` (N, 2), ``edges`` (E, 2), the cleaned
    ``vertices`` / ``links`` dataframes, and connected ``components``.
    """
    vertices = pd.read_csv(os.path.join(root, vertices_file))
    links = pd.read_csv(os.path.join(root, links_file))

    required_v = {"v_id", "lon", "lat"}
    required_l = {"v_id_1", "v_id_2"}
    if required_v - set(vertices.columns):
        raise ValueError(f"Missing vertex columns: {required_v - set(vertices.columns)}")
    if required_l - set(links.columns):
        raise ValueError(f"Missing link columns: {required_l - set(links.columns)}")

    vertices = vertices.dropna(subset=["v_id", "lon", "lat"]).copy()
    vertices["v_id"] = vertices["v_id"].astype(int)
    valid_ids = set(vertices["v_id"].values)

    links = links.dropna(subset=["v_id_1", "v_id_2"]).copy()
    links["v_id_1"] = links["v_id_1"].astype(int)
    links["v_id_2"] = links["v_id_2"].astype(int)
    links = links[
        links["v_id_1"].isin(valid_ids)
        & links["v_id_2"].isin(valid_ids)
        & (links["v_id_1"] != links["v_id_2"])
    ].copy()

    vertices = vertices.sort_values("v_id").reset_index(drop=True)
    id_to_idx = {v_id: i for i, v_id in enumerate(vertices["v_id"].values)}

    edges = np.column_stack([
        links["v_id_1"].map(id_to_idx).values,
        links["v_id_2"].map(id_to_idx).values,
    ]).astype(np.int64)

    if collapse_multiedges:
        edges = np.sort(edges, axis=1)
        edges = np.unique(edges, axis=0)

    points_xy_km = lonlat_to_xy_km(vertices["lon"].values, vertices["lat"].values)

    G = nx.Graph()
    G.add_nodes_from(range(len(vertices)))
    G.add_edges_from(map(tuple, edges))
    components = sorted(nx.connected_components(G), key=len, reverse=True)

    if largest_component:
        keep = list(components[0])
        old_to_new = -np.ones(len(vertices), dtype=np.int64)
        old_to_new[keep] = np.arange(len(keep), dtype=np.int64)

        mask = (old_to_new[edges[:, 0]] >= 0) & (old_to_new[edges[:, 1]] >= 0)
        edges = old_to_new[edges[mask]]

        points_xy_km = points_xy_km[keep]
        vertices = vertices.iloc[keep].reset_index(drop=True)

    return {
        "points_xy_km": points_xy_km,
        "edges": edges,
        "vertices": vertices,
        "links": links,
        "components": components,
    }


def graph_diagnostics(points, edges):
    """Basic graph-level diagnostics for an embedded network."""
    N = points.shape[0]
    degrees = np.zeros(N, dtype=np.int64)
    for i, j in edges:
        degrees[i] += 1
        degrees[j] += 1
    E = int(edges.shape[0])
    return {
        "N": int(N),
        "E": E,
        "mean_degree": float(degrees.mean()),
        "degree_std": float(degrees.std()),
        "degree_min": int(degrees.min()),
        "degree_max": int(degrees.max()),
        "edge_density": float(2 * E / (N * (N - 1))) if N > 1 else float("nan"),
        "degrees": degrees,
    }


def prepare_network(network_key, config, out_dir, raw_dir):
    """Load, clean, and persist one network; return its data dict + diagnostics."""
    net_cfg = config["networks"][network_key]
    data = load_gridkit(raw_dir, net_cfg["vertices"], net_cfg["links"])

    points = data["points_xy_km"]
    edges = data["edges"]
    diagnostics = graph_diagnostics(points, edges)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_df = pd.DataFrame({
        "idx": np.arange(points.shape[0]),
        "x_km": points[:, 0],
        "y_km": points[:, 1],
        "lon": data["vertices"]["lon"].values,
        "lat": data["vertices"]["lat"].values,
    })
    edges_df = pd.DataFrame({
        "i": edges[:, 0],
        "j": edges[:, 1],
    })

    nodes_df.to_csv(out_dir / f"{network_key}_nodes.csv", index=False)
    edges_df.to_csv(out_dir / f"{network_key}_edges.csv", index=False)

    return data, diagnostics
