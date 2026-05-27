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


def build_default_map(cfg: Config, rng: np.random.Generator) -> MapData:
    """Build the 10x10 toy map per CLAUDE.md."""
    h, w = cfg.grid_h, cfg.grid_w
    n = h * w
    coords = np.array([(i, j) for i in range(h) for j in range(w)], dtype=np.int32)
    cluster = np.full(n, -1, dtype=np.int32)
    for idx, (i, j) in enumerate(coords):
        if i < 5 and j < 5:
            cluster[idx] = 0
        elif i < 5 and j >= 5:
            cluster[idx] = 1
        else:
            cluster[idx] = 2
    n_clusters = 3

    # Value: sum of Gaussian "hotspots" plus mild uniform noise and a floor.
    # This produces a few high-value zones that the controller should learn to
    # cluster around. If ``cfg.value_hotspots`` is empty, fall back to uniform.
    value = np.full(n, cfg.value_floor, dtype=np.float32)
    if cfg.value_hotspots:
        coords_f = coords.astype(np.float32)
        for (cr, cc, peak, std) in cfg.value_hotspots:
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
