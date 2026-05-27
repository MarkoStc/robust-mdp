"""Sanity checks per CLAUDE.md."""
from __future__ import annotations
import numpy as np

from .config import Config, MapData
from .state import State, initial_state, build_default_map
from .dynamics import (
    compute_development_probabilities, log_likelihood,
    nature_feasible, controller_feasible, likelihood_slack,
)
from .candidates import generate_controller_candidates, generate_nature_candidates


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


def check_map(map_data: MapData):
    _assert(map_data.n == map_data.h * map_data.w, "map: n != h*w")
    # cluster layout for 10x10 default
    if map_data.h == 10 and map_data.w == 10 and map_data.n_clusters == 3:
        for idx, (i, j) in enumerate(map_data.coords):
            c = map_data.cluster[idx]
            if i < 5 and j < 5:
                _assert(c == 0, f"parcel {idx} cluster wrong")
            elif i < 5 and j >= 5:
                _assert(c == 1, f"parcel {idx} cluster wrong")
            else:
                _assert(c == 2, f"parcel {idx} cluster wrong")
    # Neighbors are same-cluster only
    for i in range(map_data.n):
        for j in map_data.neighbors[i]:
            _assert(map_data.cluster[j] == map_data.cluster[i], "neighbor cross-cluster")


def check_probabilities(map_data: MapData, cfg: Config):
    d = np.zeros(map_data.n, dtype=np.int8)
    p = compute_development_probabilities(d, map_data, cfg.eps)
    _assert(np.isfinite(p).all(), "probabilities not finite")
    _assert((p >= cfg.eps).all() and (p <= 1 - cfg.eps).all(), "probabilities not clipped")


def check_log_likelihood(map_data: MapData, cfg: Config, rng):
    d = (rng.uniform(size=map_data.n) < 0.2).astype(np.int8)
    p = compute_development_probabilities(d, map_data, cfg.eps)
    ll = log_likelihood(d, p)
    _assert(np.isfinite(ll), "log-likelihood not finite")


def check_controller_candidates(map_data: MapData, cfg: Config, rng):
    state = initial_state(cfg, map_data)
    A = generate_controller_candidates(state, map_data, cfg, rng)
    _assert(A.shape[1] == map_data.n, "controller cand shape wrong")
    _assert((A.sum(axis=1) == 0).any(), "controller zero action missing")
    # Dedup
    keys = set(a.tobytes() for a in A)
    _assert(len(keys) == A.shape[0], "controller candidates not deduplicated")
    # All feasible: cost <= budget; no protected/developed flips
    for a in A:
        _assert(float(map_data.cost[a == 1].sum()) <= state.budget + 1e-6,
                "controller candidate violates budget")
        _assert(((a == 1) & (state.x == 1)).sum() == 0, "controller protects already protected")
        _assert(((a == 1) & (state.d == 1)).sum() == 0, "controller protects developed")


def check_nature_candidates(map_data: MapData, cfg: Config, rng):
    state = initial_state(cfg, map_data)
    a_zero = np.zeros(map_data.n, dtype=np.int8)
    K = generate_nature_candidates(state, a_zero, map_data, cfg, rng)
    _assert(K.shape[1] == map_data.n, "nature cand shape wrong")
    _assert((K.sum(axis=1) == 0).any(), "nature zero action missing")
    keys = set(k.tobytes() for k in K)
    _assert(len(keys) == K.shape[0], "nature candidates not deduplicated")
    # Likelihood feasibility
    bad = 0
    for k in K:
        d_plus = state.d | k
        if likelihood_slack(state, d_plus, map_data, cfg) < -1e-6:
            bad += 1
    _assert(bad == 0, f"{bad}/{len(K)} nature candidates infeasible")


def check_matrix_game_shape(map_data: MapData, cfg: Config, rng):
    from .features import feature_dim, features_phi
    from .matrix_game import solve_matrix_game
    state = initial_state(cfg, map_data)
    A = generate_controller_candidates(state, map_data, cfg, rng)
    K = generate_nature_candidates(state, np.zeros(map_data.n, dtype=np.int8), map_data, cfg, rng)
    D = feature_dim(map_data, cfg)
    theta = np.zeros(D)
    J, L = len(A), len(K)
    Q = np.zeros((J, L))
    for j in range(J):
        for l in range(L):
            phi = features_phi(state, A[j], K[l], map_data, cfg)
            Q[j, l] = float(theta @ phi)
    _assert(Q.shape == (J, L), "Q matrix shape wrong")
    pi, omega, v = solve_matrix_game(Q)
    _assert(pi.shape == (J,) and omega.shape == (L,), "pi/omega shape wrong")
    _assert(np.isclose(pi.sum(), 1.0, atol=1e-6), "pi not simplex")
    _assert(np.isclose(omega.sum(), 1.0, atol=1e-6), "omega not simplex")
    _assert((pi >= -1e-9).all() and (omega >= -1e-9).all(), "negative simplex weights")


def sanity_check_all(cfg: Config, map_data: MapData, rng):
    check_map(map_data)
    check_probabilities(map_data, cfg)
    check_log_likelihood(map_data, cfg, rng)
    check_controller_candidates(map_data, cfg, rng)
    check_nature_candidates(map_data, cfg, rng)
    check_matrix_game_shape(map_data, cfg, rng)
    return True
