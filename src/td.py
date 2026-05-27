"""Algorithm 2: temporal-difference learning of theta for fixed (pi, omega).

Inputs: pi and omega are oracles that, given a state, return candidate sets
and a probability vector over them.
"""
from __future__ import annotations
from typing import Callable
import numpy as np

from .config import Config, MapData
from .state import State, initial_state
from .candidates import generate_controller_candidates, generate_nature_candidates
from .dynamics import step
from .features import features_phi, feature_dim
from .rollout import step_reward


def _sample_index(rng: np.random.Generator, probs: np.ndarray) -> int:
    p = np.clip(probs, 0.0, None)
    s = p.sum()
    if s <= 0:
        return int(rng.integers(0, len(p)))
    return int(rng.choice(len(p), p=p / s))


def fit_linear_q(cfg: Config, map_data: MapData,
                 pi_oracle: Callable, omega_oracle: Callable,
                 theta_init: np.ndarray | None = None,
                 n_samples: int | None = None,
                 rng: np.random.Generator | None = None) -> np.ndarray:
    """TD with tau-skip. Returns theta of dimension feature_dim(map_data).

    pi_oracle(state, rng) -> (A_cand, pi_probs)
    omega_oracle(state, a_protect, rng) -> (K_cand, omega_probs)
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed)
    D = feature_dim(map_data, cfg)
    theta = np.zeros(D) if theta_init is None else theta_init.copy()
    eta = cfg.td_step_eta
    tau = max(0, cfg.td_skip_tau)
    N = n_samples if n_samples is not None else cfg.samples_per_iter
    gamma = cfg.gamma

    state = initial_state(cfg, map_data)

    def reset_state():
        return initial_state(cfg, map_data)

    def _sample_pair(s, rng):
        A_cand, pi_probs = pi_oracle(s, rng)
        j = _sample_index(rng, pi_probs)
        a = A_cand[j]
        K_cand, om_probs = omega_oracle(s, a, rng, pi_cache=(A_cand, pi_probs))
        l = _sample_index(rng, om_probs)
        k = K_cand[l]
        return a, k

    for t_iter in range(N):
        a, k = _sample_pair(state, rng)
        next_state = step(state, a, k, cfg)
        r = step_reward(state, a, k, next_state, map_data, cfg) if cfg.use_step_reward else 0.0
        phi = features_phi(state, a, k, map_data, cfg)

        if next_state.t > cfg.horizon_T:
            target = r
            td_error = (theta @ phi) - target
            theta = theta - eta * td_error * phi
            state = reset_state()
            continue

        a2, k2 = _sample_pair(next_state, rng)
        phi_next = features_phi(next_state, a2, k2, map_data, cfg)
        target = r + gamma * (theta @ phi_next)
        td_error = (theta @ phi) - target
        theta = theta - eta * td_error * phi

        cur = next_state
        for _ in range(tau):
            ax, kx = _sample_pair(cur, rng)
            cur = step(cur, ax, kx, cfg)
            if cur.t > cfg.horizon_T:
                cur = reset_state()
                break
        state = cur

    return theta
