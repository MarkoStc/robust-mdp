"""Development probabilities, likelihood, and feasibility helpers (vectorized)."""
from __future__ import annotations
import numpy as np

from .config import Config, MapData
from .state import State


def compute_development_probabilities(d: np.ndarray, map_data: MapData, eps: float) -> np.ndarray:
    """Vectorized p_i = (TI_i / 10) * (1 + dev neighbors) / (1 + neighbors)."""
    d = d.astype(np.float32, copy=False)
    dev_neigh = map_data.adj @ d                  # (n,)
    denom = 1.0 + map_data.deg                    # (n,)
    numer = 1.0 + dev_neigh
    p = (map_data.threat / 10.0) * numer / denom
    return np.clip(p, eps, 1.0 - eps)


def frontier_pressure(d: np.ndarray, map_data: MapData) -> np.ndarray:
    d = d.astype(np.float32, copy=False)
    dev_neigh = map_data.adj @ d
    return (1.0 + dev_neigh) / (1.0 + map_data.deg)


def log_likelihood(d: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(d * np.log(p) + (1 - d) * np.log(1 - p)))


def likelihood_slack(state: State, d_plus: np.ndarray, map_data: MapData, cfg: Config) -> float:
    """Slack of nature feasibility, with baseline subtraction so zero is feasible.

    slack = (loglik(d_plus) - loglik(0)) - t * log(lambda)
           = sum_i d_plus_i * log(p_i / (1 - p_i)) - t * log(lambda).
    """
    if cfg.fixed_p_feasibility:
        p_check = compute_development_probabilities(state.d, map_data, cfg.eps)
    else:
        p_check = compute_development_probabilities(d_plus, map_data, cfg.eps)
    p_check = np.clip(p_check, 1e-12, 1 - 1e-12)
    rel_loglik = float(np.sum(d_plus * (np.log(p_check) - np.log(1.0 - p_check))))
    return rel_loglik - state.t * np.log(cfg.lambda_uncertainty)


def nature_feasible(state: State, k: np.ndarray, map_data: MapData, cfg: Config) -> bool:
    if np.any((k == 1) & (state.x == 1)):
        return False
    if np.any((k == 1) & (state.d == 1)):
        return False
    d_plus = state.d | k
    return likelihood_slack(state, d_plus, map_data, cfg) >= 0


def controller_feasible(state: State, a: np.ndarray, map_data: MapData) -> bool:
    if np.any((a == 1) & (state.x == 1)):
        return False
    if np.any((a == 1) & (state.d == 1)):
        return False
    if float(map_data.cost[a == 1].sum()) > state.budget + 1e-9:
        return False
    return True


def step(state: State, a: np.ndarray, k: np.ndarray, cfg: Config) -> State:
    """Apply controller and nature actions. Nature cannot develop just-protected parcels."""
    a = a.astype(np.int8, copy=False)
    k = k.astype(np.int8, copy=False)
    x_plus = (state.x | a).astype(np.int8)
    k_eff = (k & (1 - x_plus)).astype(np.int8)
    d_plus = (state.d | k_eff).astype(np.int8)
    return State(t=state.t + 1, x=x_plus, d=d_plus, budget=cfg.budget_per_year)
