"""Episode rollouts under (controller policy, nature policy) defined via candidate sets."""
from __future__ import annotations
from typing import Callable, Dict, Tuple
import numpy as np

from .config import Config, MapData
from .state import State, initial_state
from .candidates import generate_controller_candidates, generate_nature_candidates
from .dynamics import compute_development_probabilities, step
from .features import features_phi, q_matrix_batch
from .matrix_game import solve_matrix_game


def q_matrix(state: State, A_cand: np.ndarray, K_cand: np.ndarray,
             theta: np.ndarray, map_data: MapData, cfg: Config) -> np.ndarray:
    return q_matrix_batch(state, A_cand, K_cand, theta, map_data, cfg)


def policy_at_state(state: State, theta: np.ndarray, map_data: MapData, cfg: Config,
                    rng: np.random.Generator):
    """Returns (pi, omega, A_cand, K_cand, value, Q)."""
    A_cand = generate_controller_candidates(state, map_data, cfg, rng)
    K_cand = generate_nature_candidates(state, np.zeros_like(state.x), map_data, cfg, rng)
    Q = q_matrix(state, A_cand, K_cand, theta, map_data, cfg)
    span = float(Q.max() - Q.min())
    if span < 1e-9 or not np.isfinite(span):
        J, L = Q.shape
        pi = np.ones(J) / J
        omega = np.ones(L) / L
        v = float(Q.mean())
    else:
        pi, omega, v = solve_matrix_game(Q)
    return pi, omega, A_cand, K_cand, v, Q


def sample_action(rng: np.random.Generator, probs: np.ndarray) -> int:
    p = probs.copy()
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if s <= 0:
        return int(rng.integers(0, len(p)))
    return int(rng.choice(len(p), p=p / s))


def rollout_episode(cfg: Config, map_data: MapData, theta: np.ndarray,
                    rng: np.random.Generator,
                    nature_uses_omega: bool = True) -> Dict:
    """Roll out one episode; returns summary stats and per-step record."""
    state = initial_state(cfg, map_data)
    T = cfg.horizon_T

    history = []
    protected_value = []
    developed_value = []
    free_value = []

    for t in range(T):
        pi, omega, A_cand, K_cand, v, Q = policy_at_state(state, theta, map_data, cfg, rng)
        j = sample_action(rng, pi)
        a = A_cand[j]
        if nature_uses_omega:
            l = sample_action(rng, omega)
        else:
            # pick worst-case (argmin Q for chosen a, then sample)
            l = int(np.argmin(Q[j]))
        k = K_cand[l]
        # Apply
        # Track stats before step
        protected_value.append(float((map_data.value * (state.x | a)).sum()))
        developed_value.append(float((map_data.value * (state.d | k)).sum()))
        free_value.append(float((map_data.value * (((state.x | a) == 0) & ((state.d | k) == 0))).sum()))
        history.append({
            "t": state.t,
            "x": state.x.copy(), "d": state.d.copy(),
            "a": a.copy(), "k": k.copy(),
            "pi_mass_on_chosen": float(pi[j]),
            "omega_mass_on_chosen": float(omega[l]),
            "value_estimate": v,
        })
        state = step(state, a, k, cfg)

    return {
        "history": history,
        "final_protected_value": float((map_data.value * state.x).sum()),
        "final_developed_value": float((map_data.value * state.d).sum()),
        "final_free_value": float((map_data.value * ((state.x == 0) & (state.d == 0))).sum()),
        "protected_value_trace": protected_value,
        "developed_value_trace": developed_value,
        "free_value_trace": free_value,
    }


def step_reward(state: State, a: np.ndarray, k: np.ndarray, next_state: State,
                map_data: MapData, cfg: Config) -> float:
    """Reward = newly protected value / total_value (good-value convention)."""
    new_protected = ((state.x == 0) & (next_state.x == 1)).astype(np.float64)
    return float((map_data.value * new_protected).sum()) / max(1e-9, map_data.total_value)
