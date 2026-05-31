"""State and map construction."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from .config import Config, MapData


@dataclass
class State:
    t: int                  # 1-indexed year (1..T)
    x: np.ndarray           # (n,) {0,1} protected
    d: np.ndarray           # (n,) {0,1} developed
    budget: float

    def copy(self) -> "State":
        return State(self.t, self.x.copy(), self.d.copy(), float(self.budget))


def _assign_clusters(coords: np.ndarray, h: int, w: int, cfg: Config):
    """Return (cluster_ids, n_clusters).

    The 10x10 toy keeps its hand-defined 3-cluster layout (top-left 5x5,
    top-right 5x5, bottom 5x10) so existing runs and sanity checks are unchanged.
    Any other grid is tiled into n_cluster_rows x n_cluster_cols rectangular
    blocks (clusters are contiguous, so 4-neighbor adjacency stays mostly
    in-cluster, preserving the same-cluster snowball dynamics).
    """
    n = h * w
    cluster = np.full(n, -1, dtype=np.int32)
    if h == 10 and w == 10 and cfg.n_cluster_rows == 0 and cfg.n_cluster_cols == 0:
        for idx, (i, j) in enumerate(coords):
            if i < 5 and j < 5:
                cluster[idx] = 0
            elif i < 5 and j >= 5:
                cluster[idx] = 1
            else:
                cluster[idx] = 2
        return cluster, 3

    # General rectangular tiling. Auto: one row of blocks, ~sqrt-ish count.
    n_rows = cfg.n_cluster_rows if cfg.n_cluster_rows > 0 else 1
    n_cols = cfg.n_cluster_cols if cfg.n_cluster_cols > 0 else max(1, min(w, round(np.sqrt(max(1, n) / 25.0))))
    n_rows = max(1, min(n_rows, h))
    n_cols = max(1, min(n_cols, w))
    for idx, (i, j) in enumerate(coords):
        rb = min(n_rows - 1, int(i * n_rows // h))
        cb = min(n_cols - 1, int(j * n_cols // w))
        cluster[idx] = rb * n_cols + cb
    return cluster, int(n_rows * n_cols)


def build_default_map(cfg: Config, rng: np.random.Generator) -> MapData:
    """Build the toy map per CLAUDE.md. Defaults to the 10x10 layout; supports
    arbitrary (grid_h, grid_w) with rectangular cluster tiling (see
    _assign_clusters) for larger settings such as the 692-parcel run."""
    h, w = cfg.grid_h, cfg.grid_w
    n = h * w
    coords = np.array([(i, j) for i in range(h) for j in range(w)], dtype=np.int32)
    cluster, n_clusters = _assign_clusters(coords, h, w, cfg)

    # Value: Gaussian "hotspots" plus mild uniform noise and a floor. If the
    # configured hotspots all fall outside the grid (e.g. the 10x10 defaults on a
    # larger grid), auto-place one bump near each cluster centroid so any grid
    # keeps high-value zones the controller should learn to defend.
    value = np.full(n, cfg.value_floor, dtype=np.float32)
    coords_f = coords.astype(np.float32)
    hotspots = [hs for hs in (cfg.value_hotspots or []) if 0 <= hs[0] < h and 0 <= hs[1] < w]
    if not hotspots and n != 100:
        std = max(1.5, min(h, w) / 3.0)
        for c in range(n_clusters):
            members = coords_f[cluster == c]
            if len(members) == 0:
                continue
            cr, cc = members.mean(axis=0)
            hotspots.append((float(cr), float(cc), 1.0, float(std)))
    if hotspots:
        for (cr, cc, peak, std) in hotspots:
            dist2 = (coords_f[:, 0] - cr) ** 2 + (coords_f[:, 1] - cc) ** 2
            bump = float(peak) * np.exp(-dist2 / (2.0 * float(std) ** 2))
            value = np.maximum(value, bump)
        value = value + cfg.value_noise * rng.uniform(size=n).astype(np.float32)
    else:
        value = rng.uniform(0.2, 1.0, size=n).astype(np.float32)
    value = value.astype(np.float32)
    cost = rng.uniform(0.2, 1.0, size=n).astype(np.float32)
    threat = rng.uniform(1.0, 9.0, size=n).astype(np.float32)

    # 4-neighbor adjacency, same-cluster only
    neighbors: list[np.ndarray] = []
    for idx, (i, j) in enumerate(coords):
        ns = []
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ii, jj = i + di, j + dj
            if 0 <= ii < h and 0 <= jj < w:
                jdx = ii * w + jj
                if cluster[jdx] == cluster[idx]:
                    ns.append(jdx)
        neighbors.append(np.array(ns, dtype=np.int32))

    adj = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        if len(neighbors[i]) > 0:
            adj[i, neighbors[i]] = 1.0
    deg = adj.sum(axis=1)  # (n,)
    cluster_onehot = np.zeros((n, n_clusters), dtype=np.float32)
    cluster_onehot[np.arange(n), cluster] = 1.0

    return MapData(
        n=n, h=h, w=w,
        coords=coords, cluster=cluster, n_clusters=n_clusters,
        value=value, cost=cost, threat=threat,
        neighbors=neighbors,
        adj=adj, deg=deg, cluster_onehot=cluster_onehot,
        total_value=float(value.sum()),
        total_cost=float(cost.sum()),
    )


def initial_state(cfg: Config, map_data: MapData) -> State:
    n = map_data.n
    return State(t=1, x=np.zeros(n, np.int8), d=np.zeros(n, np.int8), budget=cfg.budget_per_year)
