"""Plot helpers for the toy grid with cluster borders."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from .config import MapData
from .state import State


def _cluster_borders(ax, map_data: MapData, color: str = "black", lw: float = 2.5):
    h, w = map_data.h, map_data.w
    cl = map_data.cluster.reshape(h, w)
    # vertical edges
    for i in range(h):
        for j in range(w - 1):
            if cl[i, j] != cl[i, j + 1]:
                ax.plot([j + 0.5, j + 0.5], [i - 0.5, i + 0.5], color=color, lw=lw)
    # horizontal edges
    for i in range(h - 1):
        for j in range(w):
            if cl[i, j] != cl[i + 1, j]:
                ax.plot([j - 0.5, j + 0.5], [i + 0.5, i + 0.5], color=color, lw=lw)
    # outer border
    ax.plot([-0.5, w - 0.5, w - 0.5, -0.5, -0.5],
            [-0.5, -0.5, h - 0.5, h - 0.5, -0.5], color=color, lw=lw)


def plot_heatmap(map_data: MapData, values: np.ndarray, title: str, ax=None, cmap="viridis"):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    grid = values.reshape(map_data.h, map_data.w)
    im = ax.imshow(grid, cmap=cmap, origin="upper")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _cluster_borders(ax, map_data)
    return ax


def plot_state(state: State, map_data: MapData, a: np.ndarray | None = None,
               k: np.ndarray | None = None, ax=None, title: str = ""):
    """Color cells by status: 0 free, 1 protected (old), 2 developed (old),
    3 newly protected, 4 newly developed.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    h, w = map_data.h, map_data.w
    grid = np.zeros((h, w), dtype=np.int32)
    grid[state.x.reshape(h, w) == 1] = 1
    grid[state.d.reshape(h, w) == 1] = 2
    if a is not None:
        new_prot = ((a == 1) & (state.x == 0)).reshape(h, w)
        grid[new_prot] = 3
    if k is not None:
        new_dev = ((k == 1) & (state.d == 0)).reshape(h, w)
        grid[new_dev] = 4
    colors = ["#dddddd", "#1f77b4", "#7f7f7f", "#2ca02c", "#d62728"]
    cmap = ListedColormap(colors)
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, origin="upper")
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    _cluster_borders(ax, map_data)
    return ax
