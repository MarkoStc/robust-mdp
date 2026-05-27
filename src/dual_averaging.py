"""Algorithm 1: stochastic dual averaging for robust policy evaluation.

For a frozen controller policy pi (defined by an oracle over old theta), we update
a state-conditional nature policy omega via dual averaging with KL regularization.

To stay implementable for the continuous state space, omega(.|s) is not stored
explicitly; instead we represent it by the cumulative "score" vector g_cumulative
over the locally-generated candidate kernels at s, i.e. for query state s we
maintain g_t(k|s) = sum_{m<=t} alpha_m * pi(a|s)^T Q_theta_m(s, a, k) over the
**candidate K-set at s**. To avoid having to track per-state omega vectors, we
follow CLAUDE.md and use a simplified scheme:

- Average the theta estimates across inner iterations to obtain theta_bar.
- At each inner iter m, omega_m(.|s) is computed on-the-fly from the cumulative
  weighted estimate using the per-state g(k|s) computed at iteration m using
  theta_{m-1}.

We use a simpler per-call dual-averaging step: when omega is queried at a state,
we compute g(k|s) over the local candidate set using all previously stored
(theta_t, alpha_t), and return softmin( (1/lambda_m) g ).

For inner iterations, we maintain a list of (theta_t, alpha_t).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from .config import Config, MapData
from .state import State
from .candidates import generate_controller_candidates, generate_nature_candidates
from .features import features_phi, feature_dim, q_matrix_batch
from .matrix_game import solve_matrix_game
from .td import fit_linear_q


@dataclass
class NaturePolicyState:
    thetas: list           # list of theta vectors theta_0..theta_M-1
    alphas: list           # weights alpha_t = sqrt(t+1)
    lam: float             # current regularization strength lambda_M

    def is_empty(self) -> bool:
        return len(self.thetas) == 0


def _mix_uniform(pi: np.ndarray, eps: float) -> np.ndarray:
    if eps <= 0:
        return pi
    J = len(pi)
    return (1.0 - eps) * pi + eps * (np.ones(J) / J)


def _pi_probs_from_theta(state: State, A_cand: np.ndarray, K_cand: np.ndarray,
                         theta: np.ndarray, map_data: MapData, cfg: Config):
    Q = q_matrix_batch(state, A_cand, K_cand, theta, map_data, cfg)
    # If Q is essentially constant (e.g. theta=0 at the start), LP returns a degenerate
    # corner. Detect and fall back to uniform so the controller still explores.
    span = float(Q.max() - Q.min())
    if span < 1e-9 or not np.isfinite(span):
        pi = np.ones(Q.shape[0]) / Q.shape[0]
    else:
        pi, _, _ = solve_matrix_game(Q)
    return pi, Q


def make_pi_oracle(theta_old: np.ndarray, map_data: MapData, cfg: Config):
    """Oracle (state, rng) -> (A_cand, pi_probs). Controller frozen via theta_old; mixes with uniform."""
    def oracle(state: State, rng):
        A_cand = generate_controller_candidates(state, map_data, cfg, rng)
        K_cand = generate_nature_candidates(state, np.zeros_like(state.x), map_data, cfg, rng)
        pi, _ = _pi_probs_from_theta(state, A_cand, K_cand, theta_old, map_data, cfg)
        pi = _mix_uniform(pi, cfg.exploration_eps)
        return A_cand, pi
    return oracle


def make_omega_oracle(nature_state: NaturePolicyState, pi_oracle, map_data: MapData, cfg: Config):
    """omega(k|s) ∝ exp(- (1/lam) * sum_t alpha_t * E_a~pi [theta_t . phi(s, a, k)]).

    Accepts optional precomputed (A_cand, pi_probs) via `pi_cache` keyword to avoid
    redundant work when the caller already obtained these from pi_oracle.
    """
    def oracle(state: State, a_protect, rng, pi_cache=None):
        K_cand = generate_nature_candidates(state, a_protect, map_data, cfg, rng)
        L = len(K_cand)
        if nature_state.is_empty():
            return K_cand, np.ones(L) / L
        if pi_cache is None:
            A_cand, pi_probs = pi_oracle(state, rng)
        else:
            A_cand, pi_probs = pi_cache
        alphas = np.array(nature_state.alphas)
        theta_stack = np.stack(nature_state.thetas, axis=0)            # (M, D)
        theta_weighted = (alphas[:, None] * theta_stack).sum(axis=0)   # (D,)
        Q = q_matrix_batch(state, A_cand, K_cand, theta_weighted, map_data, cfg)  # (J, L)
        cum_g = pi_probs @ Q  # (L,)
        lam = max(1e-6, nature_state.lam)
        logits = -cum_g / lam
        logits = logits - logits.max()
        probs = np.exp(logits)
        s = probs.sum()
        if s <= 0:
            probs = np.ones(L) / L
        else:
            probs = probs / s
        probs = _mix_uniform(probs, cfg.exploration_eps)
        return K_cand, probs
    return oracle


def algorithm1_policy_eval(cfg: Config, map_data: MapData,
                           theta_old: np.ndarray,
                           rng: np.random.Generator) -> np.ndarray:
    """Inner loop: estimate robust Q for the controller policy defined by theta_old.

    Returns theta_bar = average of theta_m across inner iterations.
    """
    D = feature_dim(map_data, cfg)
    nature_state = NaturePolicyState(thetas=[], alphas=[], lam=1.0)
    pi_oracle = make_pi_oracle(theta_old, map_data, cfg)
    omega_oracle = make_omega_oracle(nature_state, pi_oracle, map_data, cfg)

    theta_accum = np.zeros(D)
    weight_accum = 0.0

    B_const = 1.0  # nominal scale used in paper; we expose lam = (m+1)*B / (2*sqrt(log|K|))

    for m in range(cfg.inner_M):
        alpha_m = np.sqrt(m + 1)
        # update lam following the paper (depends on K-size; we use n_nature_candidates)
        Kc = max(2, cfg.n_nature_candidates)
        nature_state.lam = (m + 1) * B_const / (2.0 * np.sqrt(np.log(Kc)))
        # Fit theta_m given current frozen omega oracle
        theta_m = fit_linear_q(cfg, map_data, pi_oracle, omega_oracle,
                               theta_init=None,
                               n_samples=cfg.samples_per_iter,
                               rng=rng)
        nature_state.thetas.append(theta_m)
        nature_state.alphas.append(alpha_m)
        theta_accum += alpha_m * theta_m
        weight_accum += alpha_m

    return theta_accum / max(weight_accum, 1e-9)
