"""Reduced-action candidate generation for controller and nature."""
from __future__ import annotations
import math
import numpy as np

from .config import Config, MapData
from .state import State
from .dynamics import (
    compute_development_probabilities,
    frontier_pressure as _frontier_pressure,
    likelihood_slack,
)


def _connectivity_bonus(x: np.ndarray, map_data: MapData) -> np.ndarray:
    prot_neigh = map_data.adj @ x.astype(np.float32)
    return 1.0 + 0.5 * prot_neigh / np.maximum(1.0, map_data.deg)


def _greedy_knapsack(scores: np.ndarray, costs: np.ndarray, budget: float,
                     mask_available: np.ndarray, rng: np.random.Generator,
                     temperature: float = 1.0) -> np.ndarray:
    """Score-perturbed greedy knapsack: sample one ordering using Gumbel noise on
    `scores`, then walk it once adding parcels whose cost fits the remaining budget.
    O(n log n) and amortizes much better than a per-pick softmax loop.
    """
    n = scores.shape[0]
    a = np.zeros(n, dtype=np.int8)
    feas0 = mask_available & (costs <= budget + 1e-9)
    if not feas0.any():
        return a
    if temperature > 0:
        noise = rng.gumbel(0.0, 1.0, size=n) * max(1e-6, temperature)
        perturbed = scores + noise
    else:
        perturbed = scores.copy()
    perturbed = np.where(mask_available, perturbed, -np.inf)
    order = np.argsort(-perturbed)
    remaining = float(budget)
    for j in order:
        if not mask_available[j]:
            continue
        c = float(costs[j])
        if c > remaining + 1e-9:
            continue
        a[j] = 1
        remaining -= c
        if remaining < float(costs[mask_available].min(initial=remaining + 1)) - 1e-9:
            break
    return a


def _stochastic_greedy_nature(scores: np.ndarray, mask_available: np.ndarray,
                              state: State, map_data: MapData, cfg: Config,
                              rng: np.random.Generator, temperature: float = 1.0,
                              log_ratio: np.ndarray | None = None,
                              slack_baseline: float | None = None) -> np.ndarray:
    """Greedy nature (FROZEN-p fast path): walk parcels in Gumbel-perturbed score
    order, adding each that keeps the cumulative log-likelihood-ratio above the
    t*log(lambda) cutoff, using state-only p (precomputed log_ratio).

    This is the O(1)-per-parcel approximation used only when
    ``cfg.fixed_p_feasibility`` is True. The default exact path is
    ``_stochastic_greedy_nature_exact`` below.
    """
    n = scores.shape[0]
    k = np.zeros(n, dtype=np.int8)
    if not mask_available.any():
        return k
    if log_ratio is None:
        from .dynamics import compute_development_probabilities
        p_check = compute_development_probabilities(state.d, map_data, cfg.eps)
        p_check = np.clip(p_check, 1e-9, 1 - 1e-9)
        log_ratio = np.log(p_check / (1.0 - p_check))
    if slack_baseline is None:
        slack_baseline = float((state.d.astype(np.float32) * log_ratio).sum()
                               - state.t * np.log(cfg.lambda_uncertainty))
    if temperature > 0:
        noise = rng.gumbel(0.0, 1.0, size=n) * max(1e-6, temperature)
        perturbed = scores + noise
    else:
        perturbed = scores.copy()
    perturbed = np.where(mask_available, perturbed, -np.inf)
    order = np.argsort(-perturbed)
    slack = float(slack_baseline)
    for j in order:
        if not mask_available[j]:
            continue
        lr = float(log_ratio[j])
        if slack + lr >= 0:
            k[j] = 1
            slack += lr
    return k


def _stochastic_greedy_nature_exact(scores: np.ndarray, mask_available: np.ndarray,
                                    state: State, map_data: MapData, cfg: Config,
                                    rng: np.random.Generator,
                                    temperature: float = 1.0) -> np.ndarray:
    """Greedy nature with EXACT immediate-p feasibility (CLAUDE.md default).

    Feasibility uses the relative log-likelihood slack evaluated on the *current*
    development pattern with p recomputed from it (immediate recomputation):

        slack(d) = sum_{i: d_i=1} log( p_i(d) / (1 - p_i(d)) ) - t * log(lambda)

    where p_i(d) = (TI_i/10) * (1 + dev_neighbors_i(d)) / (1 + neighbors_i).

    Developing parcel j only changes p for j's same-cluster neighbors, so we
    maintain the developed-neighbor counts and the running slack incrementally in
    O(deg(j)) per accepted/tested parcel rather than recomputing over all n. This
    is exact (matches dynamics.likelihood_slack with fixed_p_feasibility=False),
    not an approximation.
    """
    n = scores.shape[0]
    k = np.zeros(n, dtype=np.int8)
    if not mask_available.any():
        return k

    eps = cfg.eps
    threat = (map_data.threat / 10.0).astype(np.float64)
    deg = map_data.deg.astype(np.float64)
    neighbors = map_data.neighbors

    dev = state.d.astype(np.int8).copy()
    # developed-neighbor count per parcel (same-cluster), float for arithmetic
    dev_neigh = (map_data.adj @ dev.astype(np.float64))

    def log_ratio_of(i: int, extra: float = 0.0) -> float:
        # log(p/(1-p)) for parcel i with (dev_neigh[i] + extra) developed neighbors
        p = threat[i] * (1.0 + dev_neigh[i] + extra) / (1.0 + deg[i])
        if p < eps:
            p = eps
        elif p > 1.0 - eps:
            p = 1.0 - eps
        return math.log(p / (1.0 - p))

    # current slack over already-developed parcels
    slack = -state.t * math.log(cfg.lambda_uncertainty)
    for i in np.where(dev == 1)[0]:
        slack += log_ratio_of(int(i))

    if temperature > 0:
        perturbed = scores + rng.gumbel(0.0, 1.0, size=n) * max(1e-6, temperature)
    else:
        perturbed = scores.copy()
    perturbed = np.where(mask_available, perturbed, -np.inf)
    order = np.argsort(-perturbed)

    for j in order:
        j = int(j)
        if not mask_available[j] or dev[j] == 1:
            continue
        # delta to slack if we develop j: j's own term (its dev_neigh is unchanged
        # by developing itself) plus the increase for each already-developed neighbor
        # whose developed-neighbor count rises by 1.
        delta = log_ratio_of(j)
        for m in neighbors[j]:
            m = int(m)
            if dev[m] == 1:
                delta += log_ratio_of(m, extra=1.0) - log_ratio_of(m)
        if slack + delta >= 0:
            k[j] = 1
            dev[j] = 1
            slack += delta
            for m in neighbors[j]:
                dev_neigh[int(m)] += 1.0
    return k


def _dedup(actions: list[np.ndarray]) -> list[np.ndarray]:
    seen = set()
    out = []
    for a in actions:
        key = a.tobytes()
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def generate_controller_candidates(state: State, map_data: MapData, cfg: Config,
                                   rng: np.random.Generator) -> np.ndarray:
    """Return (M, n) int8 matrix of controller candidate actions."""
    n = map_data.n
    p = compute_development_probabilities(state.d, map_data, cfg.eps)
    q = _frontier_pressure(state.d, map_data)
    available = (state.x == 0) & (state.d == 0)
    conn = _connectivity_bonus(state.x, map_data)
    value, cost = map_data.value, map_data.cost
    eps = cfg.eps

    target = cfg.n_controller_candidates
    actions: list[np.ndarray] = []
    # always include zero action
    actions.append(np.zeros(n, dtype=np.int8))

    if not available.any() or state.budget <= 0:
        return np.stack(_dedup(actions), axis=0)

    # heuristic-score sweep
    n_sweep = max(50, target // 4)
    for _ in range(n_sweep):
        alpha_C = float(rng.uniform(0.0, 3.0))
        beta_C = float(rng.uniform(0.0, 3.0))
        noise = rng.gumbel(0.0, 1.0, size=n)
        score = (value / (cost + eps)) * (1.0 + alpha_C * p) * (1.0 + beta_C * q) * conn + 0.3 * noise
        a = _greedy_knapsack(score, cost, state.budget, available, rng,
                             temperature=float(rng.uniform(0.05, 1.0)))
        actions.append(a)

    # additive variant
    n_add = max(20, target // 10)
    for _ in range(n_add):
        a_v = float(rng.uniform(0.0, 1.0))
        a_p = float(rng.uniform(0.0, 1.0))
        a_q = float(rng.uniform(0.0, 1.0))
        noise = rng.gumbel(0.0, 1.0, size=n)
        score = a_v * value / (cost + eps) + a_p * p + a_q * q + 0.3 * noise
        a = _greedy_knapsack(score, cost, state.budget, available, rng,
                             temperature=float(rng.uniform(0.05, 1.0)))
        actions.append(a)

    # swap perturbations of top action
    base_score = (value / (cost + eps)) * (1.0 + p) * (1.0 + q) * conn
    a0 = _greedy_knapsack(base_score, cost, state.budget, available, rng, temperature=0.0)
    for _ in range(max(20, target // 10)):
        a = a0.copy()
        on = np.where(a == 1)[0]
        off = np.where((a == 0) & available)[0]
        if len(on) == 0 or len(off) == 0:
            actions.append(a)
            continue
        n_swap = int(rng.integers(1, max(2, min(5, len(on) + 1))))
        for _ in range(n_swap):
            if len(on) == 0:
                break
            i_off = on[int(rng.integers(0, len(on)))]
            a[i_off] = 0
            on = np.where(a == 1)[0]
        # then add as many as budget allows from remaining
        remaining = state.budget - float(cost[a == 1].sum())
        avail2 = available & (a == 0)
        rest = _greedy_knapsack(
            base_score + 0.3 * rng.gumbel(0.0, 1.0, size=n),
            cost, remaining, avail2, rng, temperature=0.3,
        )
        a = (a | rest).astype(np.int8)
        actions.append(a)

    # pure random feasible
    for _ in range(max(20, target // 10)):
        order = rng.permutation(n)
        a = np.zeros(n, dtype=np.int8)
        remaining = state.budget
        for j in order:
            if not available[j]:
                continue
            if cost[j] <= remaining + 1e-9:
                a[j] = 1
                remaining -= float(cost[j])
        actions.append(a)

    # truncate / pad
    actions = _dedup(actions)
    if len(actions) > target:
        rng.shuffle(actions)
        # always keep zero
        zero_idx = next((i for i, a in enumerate(actions) if a.sum() == 0), None)
        if zero_idx is not None and zero_idx >= target:
            actions[0], actions[zero_idx] = actions[zero_idx], actions[0]
        actions = actions[:target]
    return np.stack(actions, axis=0)


def generate_nature_candidates(state: State, a_protect: np.ndarray,
                               map_data: MapData, cfg: Config,
                               rng: np.random.Generator) -> np.ndarray:
    """Return (M, n) int8 matrix of nature candidates, given controller already protected `a_protect`."""
    n = map_data.n
    x_after = (state.x | a_protect).astype(np.int8)
    tent = State(t=state.t, x=x_after, d=state.d.copy(), budget=0.0)

    p = compute_development_probabilities(tent.d, map_data, cfg.eps)
    q = _frontier_pressure(tent.d, map_data)
    available = (tent.x == 0) & (tent.d == 0)

    value = map_data.value
    eps = cfg.eps
    Lpen = np.log(np.clip((1.0 - p) / p, 1e-12, None))

    # Precompute log_ratio and slack_baseline once for this state
    p_clip = np.clip(p, 1e-9, 1 - 1e-9)
    log_ratio = np.log(p_clip / (1.0 - p_clip))
    slack_baseline = float((tent.d.astype(np.float32) * log_ratio).sum()
                           - tent.t * np.log(cfg.lambda_uncertainty))

    target = cfg.n_nature_candidates
    actions: list[np.ndarray] = [np.zeros(n, dtype=np.int8)]

    if not available.any():
        return np.stack(_dedup(actions), axis=0)

    def gen_one(score, temperature):
        if cfg.fixed_p_feasibility:
            # frozen-p fast path (approximate); kept for ablations
            return _stochastic_greedy_nature(
                score, available, tent, map_data, cfg, rng,
                temperature=temperature,
                log_ratio=log_ratio, slack_baseline=slack_baseline,
            )
        # exact immediate-p feasibility (default, coherent with the paper)
        return _stochastic_greedy_nature_exact(
            score, available, tent, map_data, cfg, rng,
            temperature=temperature,
        )

    n_sweep = max(50, target // 4)
    for _ in range(n_sweep):
        alpha_N = float(rng.uniform(0.0, 3.0))
        beta_N = float(rng.uniform(0.0, 3.0))
        noise = rng.gumbel(0.0, 1.0, size=n)
        score = value * (1.0 + alpha_N * p) * (1.0 + beta_N * q) / (Lpen + eps) + 0.3 * noise
        actions.append(gen_one(score, float(rng.uniform(0.05, 1.0))))
    for _ in range(max(20, target // 10)):
        noise = rng.gumbel(0.0, 1.0, size=n)
        actions.append(gen_one(value + 0.5 * noise, float(rng.uniform(0.05, 0.5))))
    for _ in range(max(20, target // 10)):
        noise = rng.gumbel(0.0, 1.0, size=n)
        actions.append(gen_one(1.0 / (Lpen + eps) + 0.3 * noise, float(rng.uniform(0.05, 0.5))))
    for _ in range(max(20, target // 10)):
        noise = rng.gumbel(0.0, 1.0, size=n)
        actions.append(gen_one(q * value + 0.3 * noise, float(rng.uniform(0.05, 0.5))))
    for _ in range(max(20, target // 10)):
        actions.append(gen_one(rng.uniform(0.0, 1.0, size=n), float(rng.uniform(0.5, 2.0))))

    actions = _dedup(actions)
    if len(actions) > target:
        rng.shuffle(actions)
        zero_idx = next((i for i, a in enumerate(actions) if a.sum() == 0), None)
        if zero_idx is not None and zero_idx >= target:
            actions[0], actions[zero_idx] = actions[zero_idx], actions[0]
        actions = actions[:target]
    return np.stack(actions, axis=0)
