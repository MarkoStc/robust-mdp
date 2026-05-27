"""Paper baselines and a common adversarial nature, for the comparison study.

This module re-implements, inside *our* toy environment, the two non-RL methods
from Ye et al. "Conserving Biodiversity via Adjustable Robust Optimization"
(AAMAS 2022):

  * ``knapsack_protect``    -- their benchmark (Problem 8/9): each year protect the
                               value-maximizing set of parcels within budget,
                               ignoring development uncertainty entirely.
  * ``static_approx_schedule`` -- their *final proposed* method, StaticApprox
                               (Problem 7): a single, here-and-now (non-adaptive)
                               protection schedule for the whole horizon, chosen to
                               minimize the worst-case loss over the likelihood
                               uncertainty set. Solved by constraint generation
                               (master MILP via scipy.milp + greedy separation),
                               which is the algorithm described in their Section 6.

It also provides the two "natures" used to evaluate every controller on equal
footing:

  * ``greedy_adversary_develop``  -- worst-case nature: greedily develops the
                               parcels with the highest value-per-likelihood-cost
                               while staying inside the uncertainty set U
                               (immediate p recomputation, per CLAUDE.md).
  * ``bernoulli_develop``         -- average-case nature: the cellular-automata
                               Bernoulli(p_i) model the paper uses to *simulate*
                               realized developments (no robustness cap).

All of these operate on the same ``State``/``MapData``/``Config`` objects and the
same dynamics as the trained policy, so the comparison is apples-to-apples.
"""
from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .config import Config, MapData
from .state import State, initial_state
from .dynamics import (
    compute_development_probabilities, likelihood_slack, step,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def available_mask(state: State) -> np.ndarray:
    """Free parcels: neither protected nor developed."""
    return (state.x == 0) & (state.d == 0)


# ---------------------------------------------------------------------------
# Knapsack benchmark (paper Problem 8/9)
# ---------------------------------------------------------------------------

def knapsack_protect(state: State, map_data: MapData, cfg: Config,
                     budget: float | None = None) -> np.ndarray:
    """Protect the value-maximizing set of *available* parcels under budget.

    This is the paper's myopic knapsack: ignore uncertainty, maximize protected
    value subject to the per-year budget. Exact 0/1 knapsack via scipy.milp, with
    a value/cost-density greedy fallback.
    """
    n = map_data.n
    if budget is None:
        budget = state.budget
    free = available_mask(state)
    idx = np.where(free)[0]
    a = np.zeros(n, dtype=np.int8)
    if idx.size == 0 or budget <= 0:
        return a
    v = map_data.value[idx].astype(np.float64)
    c = map_data.cost[idx].astype(np.float64)

    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        res = milp(
            c=-v,                                   # maximize v.z  ->  minimize -v.z
            constraints=[LinearConstraint(c.reshape(1, -1), -np.inf, budget)],
            integrality=np.ones(idx.size),
            bounds=Bounds(0, 1),
        )
        if res.x is not None:
            z = np.asarray(res.x) > 0.5
            a[idx[z]] = 1
            return a
    except Exception:
        pass

    # greedy fallback: value/cost density
    order = idx[np.argsort(-(v / (c + 1e-9)))]
    spent = 0.0
    for i, cc in zip(order, map_data.cost[order]):
        if spent + cc <= budget + 1e-9:
            a[i] = 1
            spent += float(cc)
    return a


# ---------------------------------------------------------------------------
# Worst-case (adversarial) nature
# ---------------------------------------------------------------------------

def greedy_adversary_develop(state: State, a: np.ndarray, map_data: MapData,
                             cfg: Config) -> np.ndarray:
    """Greedy worst-case development given the controller already protected ``a``.

    Nature wants to destroy as much conservation value as possible while keeping
    the realization inside the likelihood uncertainty set U. Each parcel i costs
    ``L_i = log((1 - p_i) / p_i)`` of likelihood budget; nature greedily develops
    the parcel maximizing value-per-likelihood-cost, recomputing p after every
    addition (immediate recomputation: developing a parcel raises its neighbours'
    p, which loosens the constraint and lets the development snowball -- exactly
    the cellular-automata dynamic).

    Returns the newly developed mask k (excludes anything just protected).
    """
    n = map_data.n
    x_plus = (state.x | a).astype(np.int8)
    developable = (x_plus == 0) & (state.d == 0)
    k = np.zeros(n, dtype=np.int8)
    value = map_data.value.astype(np.float64)
    eps = cfg.eps

    while True:
        d_cur = (state.d | k).astype(np.int8)
        p = compute_development_probabilities(d_cur, map_data, eps).astype(np.float64)
        L = np.log((1.0 - p) / p)            # likelihood cost per parcel
        cand = np.where(developable & (k == 0))[0]
        if cand.size == 0:
            break
        best_i = -1
        best_ratio = -np.inf
        for i in cand:
            k_try = k.copy()
            k_try[i] = 1
            d_try = (state.d | k_try).astype(np.int8)
            if likelihood_slack(state, d_try, map_data, cfg) < 0:
                continue
            li = L[i]
            ratio = np.inf if li <= eps else value[i] / li   # cheap (or free) parcels first
            if ratio > best_ratio:
                best_ratio = ratio
                best_i = int(i)
        if best_i < 0:
            break
        k[best_i] = 1
    return k


# ---------------------------------------------------------------------------
# Average-case (stochastic) nature: cellular-automata Bernoulli
# ---------------------------------------------------------------------------

def bernoulli_develop(state: State, a: np.ndarray, map_data: MapData,
                      cfg: Config, rng: np.random.Generator) -> np.ndarray:
    """Sample each free parcel's development ~ Bernoulli(p_i), the cellular-automata
    model the paper uses to simulate realized human land use. No robustness cap."""
    n = map_data.n
    x_plus = (state.x | a).astype(np.int8)
    developable = (x_plus == 0) & (state.d == 0)
    p = compute_development_probabilities(state.d, map_data, cfg.eps).astype(np.float64)
    draws = rng.uniform(size=n) < p
    k = (draws & developable).astype(np.int8)
    return k


# ---------------------------------------------------------------------------
# StaticApprox (paper Problem 7) via constraint generation
# ---------------------------------------------------------------------------

def _adversary_against_schedule(schedule: np.ndarray, map_data: MapData,
                                cfg: Config) -> Tuple[np.ndarray, float]:
    """Separation oracle: roll the greedy worst-case nature forward against a
    *fixed* protection schedule and return the realized development trajectory
    (cumulative, shape (T, n)) and the realized lost value."""
    T, n = schedule.shape
    state = initial_state(cfg, map_data)
    xi = np.zeros((T, n), dtype=np.int8)
    prev = np.zeros(n, dtype=np.int8)
    for t in range(T):
        planned = schedule[t]
        # newly protected this year, masked to parcels that are still available
        a = ((planned == 1) & (prev == 0)).astype(np.int8)
        a = (a & available_mask(state).astype(np.int8)).astype(np.int8)
        k = greedy_adversary_develop(state, a, map_data, cfg)
        state = step(state, a, k, cfg)
        xi[t] = state.d
        prev = (prev | planned).astype(np.int8)
    loss = float((map_data.value * state.d).sum())
    return xi, loss


def static_approx_schedule(map_data: MapData, cfg: Config,
                           max_iters: int = 25, tol: float = 1e-6,
                           log_fn=None) -> np.ndarray:
    """Solve StaticApprox (Problem 7) by constraint generation.

    Returns a cumulative protection schedule X of shape (T, n): X[t, i] == 1 iff
    parcel i is planned to be protected on or before year t+1. The plan is
    here-and-now (non-adaptive): it does not depend on realized developments.
    """
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import csr_matrix, vstack

    n = map_data.n
    T = cfg.horizon_T
    b = cfg.budget_per_year
    v = map_data.value.astype(np.float64)
    c = map_data.cost.astype(np.float64)
    total_value = float(v.sum())
    nvar = T * n + 1                       # x[t,i] flattened + tau (last)
    tau_idx = T * n

    def var(t, i):
        return t * n + i

    # ---- fixed constraints: monotonicity and per-year budget -------------
    rows, cols, data, lb, ub = [], [], [], [], []
    r = 0
    # monotone: x[t,i] - x[t-1,i] >= 0  for t >= 1
    for t in range(1, T):
        for i in range(n):
            rows += [r, r]; cols += [var(t, i), var(t - 1, i)]; data += [1.0, -1.0]
            lb.append(0.0); ub.append(np.inf); r += 1
    # budget: sum_i c_i (x[t,i] - x[t-1,i]) <= b
    for t in range(T):
        for i in range(n):
            rows.append(r); cols.append(var(t, i)); data.append(float(c[i]))
            if t >= 1:
                rows.append(r); cols.append(var(t - 1, i)); data.append(-float(c[i]))
        lb.append(-np.inf); ub.append(b); r += 1
    A_fixed = csr_matrix((data, (rows, cols)), shape=(r, nvar))
    lb_fixed = np.array(lb); ub_fixed = np.array(ub)

    integrality = np.ones(nvar)
    integrality[tau_idx] = 0                      # tau is continuous
    bounds = Bounds(
        lb=np.concatenate([np.zeros(T * n), [0.0]]),
        ub=np.concatenate([np.ones(T * n), [total_value]]),
    )
    obj = np.zeros(nvar); obj[tau_idx] = 1.0      # minimize tau

    cut_A_rows: List[np.ndarray] = []
    cut_lb: List[float] = []

    schedule = np.zeros((T, n), dtype=np.int8)
    prev_schedule = None
    for it in range(max_iters):
        if cut_A_rows:
            A_cuts = csr_matrix(np.vstack(cut_A_rows))
            cons = [
                LinearConstraint(A_fixed, lb_fixed, ub_fixed),
                LinearConstraint(A_cuts, np.array(cut_lb), np.full(len(cut_lb), np.inf)),
            ]
        else:
            cons = [LinearConstraint(A_fixed, lb_fixed, ub_fixed)]
        res = milp(c=obj, constraints=cons, integrality=integrality, bounds=bounds)
        if res.x is None:
            if log_fn:
                log_fn(f"[staticapprox] iter {it}: MILP infeasible/failed, stopping")
            break
        x = np.asarray(res.x)
        schedule = (x[:T * n].reshape(T, n) > 0.5).astype(np.int8)
        tau = float(x[tau_idx])

        # separation: worst-case nature against this schedule
        xi, loss = _adversary_against_schedule(schedule, map_data, cfg)
        if log_fn:
            log_fn(f"[staticapprox] iter {it}: tau={tau:.4f}  worst_loss={loss:.4f}")
        if loss <= tau + tol:
            break
        # The separation oracle is deterministic given the schedule, so if the
        # master returns the same plan twice the cut would be a duplicate and we
        # have converged to the best plan this oracle can certify.
        if prev_schedule is not None and np.array_equal(schedule, prev_schedule):
            break
        prev_schedule = schedule.copy()

        # Add cut:  tau >= sum_{t,i} v_i * newly[t,i] * (1 - x[t,i])  ==>
        #           tau + sum_{t,i} v_i * newly[t,i] * x[t,i] >= S.
        # We credit protection at the SAME year it would be developed, because in
        # our simulator the controller protects before nature develops within a
        # year (so x[t] guards against developments at year t). This differs from
        # the paper's x[t-1] term, which matches their protect-after-observe timing.
        newly = xi.copy()
        newly[1:] = (xi[1:] - xi[:-1])
        S = float((v[None, :] * newly).sum())
        row = np.zeros(nvar)
        row[tau_idx] = 1.0
        for t in range(T):
            row[var(t, 0):var(t, 0) + n] += v * newly[t]
        cut_A_rows.append(row)
        cut_lb.append(S)

    return schedule


def schedule_action(schedule: np.ndarray, state: State) -> np.ndarray:
    """Per-year action from a cumulative schedule, masked to still-available
    parcels (you cannot buy land that was already lost)."""
    t = state.t - 1                               # state.t is 1-indexed
    prev = schedule[t - 1] if t >= 1 else np.zeros_like(schedule[0])
    a = ((schedule[t] == 1) & (prev == 0)).astype(np.int8)
    a = (a & available_mask(state).astype(np.int8)).astype(np.int8)
    return a
