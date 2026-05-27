"""Action-span and optimality-gap diagnostics for the reduced action sets.

Compares the heuristic candidate pool to a much larger reference pool, in
Q-relevant action-effect embedding space (not raw binary span).
"""
from __future__ import annotations
import numpy as np

from .config import Config, MapData
from .state import State
from .candidates import (
    generate_controller_candidates, generate_nature_candidates,
)
from .dynamics import frontier_pressure as _frontier_pressure
from .features import features_phi


def _controller_action_embedding(state: State, a: np.ndarray, map_data: MapData, cfg: Config) -> np.ndarray:
    """Action-effect embedding for controller a in state s."""
    from .dynamics import compute_development_probabilities
    p = compute_development_probabilities(state.d, map_data, cfg.eps)
    q = _frontier_pressure(state.d, map_data)
    eps = cfg.eps
    value = map_data.value
    cost = map_data.cost
    feats = [
        float((value * a).sum()) / map_data.total_value,
        float((cost * a).sum()) / map_data.total_cost,
        float((value * p * a).sum()) / map_data.total_value,
        float((value * q * a).sum()) / map_data.total_value,
        float((value * a / (cost + eps)).sum()) / 100.0,
    ]
    # cluster split
    for c in range(map_data.n_clusters):
        mask = (map_data.cluster == c)
        feats.append(float((value * a * mask).sum()) / map_data.total_value)
        feats.append(float((value * p * a * mask).sum()) / map_data.total_value)
    return np.array(feats, dtype=np.float64)


def _nature_action_embedding(state: State, k: np.ndarray, map_data: MapData, cfg: Config) -> np.ndarray:
    from .dynamics import compute_development_probabilities, log_likelihood
    p = compute_development_probabilities(state.d, map_data, cfg.eps)
    q = _frontier_pressure(state.d, map_data)
    value = map_data.value
    d_plus = state.d | k
    p_plus = compute_development_probabilities(d_plus, map_data, cfg.eps)
    slack = (log_likelihood(d_plus, p_plus) - state.t * np.log(cfg.lambda_uncertainty)) / max(1, map_data.n)
    feats = [
        float((value * k).sum()) / map_data.total_value,
        float((value * p * k).sum()) / map_data.total_value,
        float((value * q * k).sum()) / map_data.total_value,
        float(slack),
    ]
    for c in range(map_data.n_clusters):
        mask = (map_data.cluster == c)
        feats.append(float((value * k * mask).sum()) / map_data.total_value)
        feats.append(float((value * p * k * mask).sum()) / map_data.total_value)
    return np.array(feats, dtype=np.float64)


def _projection_errors(reference: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """For each row z of reference, returns ||z - U U^T z|| / ||z|| where U is an
    orthonormal basis of the candidate row-span."""
    if candidates.shape[0] == 0 or reference.shape[0] == 0:
        return np.array([])
    # rank-truncated SVD of candidates -> orthonormal basis U (cols are basis)
    U, S, _ = np.linalg.svd(candidates.T, full_matrices=False)
    keep = S > (S.max() * 1e-9) if S.size else np.array([], dtype=bool)
    U = U[:, keep]
    if U.size == 0:
        norms = np.linalg.norm(reference, axis=1) + 1e-12
        return np.ones(reference.shape[0])
    P = U @ U.T
    proj = reference @ P.T
    diff = reference - proj
    num = np.linalg.norm(diff, axis=1)
    den = np.linalg.norm(reference, axis=1) + 1e-12
    return num / den


def action_span_diagnostics(state: State, map_data: MapData, cfg: Config,
                            rng: np.random.Generator, n_reference: int = 5000):
    """Return dict with median/p90/p95 projection errors for controller and nature."""
    # Build a much larger reference pool with the same generator but high target sizes
    cfg_big = type(cfg)(**{**cfg.__dict__,
                           "n_controller_candidates": n_reference,
                           "n_nature_candidates": n_reference})
    ref_A = generate_controller_candidates(state, map_data, cfg_big, rng)
    cand_A = generate_controller_candidates(state, map_data, cfg, rng)
    ref_K = generate_nature_candidates(state, np.zeros(map_data.n, dtype=np.int8), map_data, cfg_big, rng)
    cand_K = generate_nature_candidates(state, np.zeros(map_data.n, dtype=np.int8), map_data, cfg, rng)

    Z_ref_A = np.stack([_controller_action_embedding(state, a, map_data, cfg) for a in ref_A])
    Z_cand_A = np.stack([_controller_action_embedding(state, a, map_data, cfg) for a in cand_A])
    Z_ref_K = np.stack([_nature_action_embedding(state, k, map_data, cfg) for k in ref_K])
    Z_cand_K = np.stack([_nature_action_embedding(state, k, map_data, cfg) for k in cand_K])

    errA = _projection_errors(Z_ref_A, Z_cand_A)
    errK = _projection_errors(Z_ref_K, Z_cand_K)
    return {
        "controller": {
            "n_ref": int(Z_ref_A.shape[0]),
            "n_cand": int(Z_cand_A.shape[0]),
            "median_proj_err": float(np.median(errA)) if errA.size else None,
            "p90": float(np.percentile(errA, 90)) if errA.size else None,
            "p95": float(np.percentile(errA, 95)) if errA.size else None,
        },
        "nature": {
            "n_ref": int(Z_ref_K.shape[0]),
            "n_cand": int(Z_cand_K.shape[0]),
            "median_proj_err": float(np.median(errK)) if errK.size else None,
            "p90": float(np.percentile(errK, 90)) if errK.size else None,
            "p95": float(np.percentile(errK, 95)) if errK.size else None,
        },
    }
