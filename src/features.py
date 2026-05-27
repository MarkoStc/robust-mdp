"""Engineered features phi(s, a, k) per CLAUDE.md.

We provide:
- ``features_phi(s, a, k)`` for a single (a, k) pair.
- ``q_matrix_batch(s, A, K, theta, ...)`` that computes ``Q[j, l] = theta . phi(s, A[j], K[l])``
  for all pairs without materializing the full (J, L, D) tensor.

Feature blocks (in order, sizes use n parcels, C clusters):
  global      : 10
  controller  : 9
  nature      : 5
  post-action : 5
  cluster     : 6 * C
  parcel-id   : 8 * n
"""
from __future__ import annotations
from typing import Tuple
import numpy as np

from .config import Config, MapData
from .state import State
from .dynamics import compute_development_probabilities, frontier_pressure, log_likelihood


def _adjacent_to_protected(a: np.ndarray, x: np.ndarray, map_data: MapData) -> np.ndarray:
    """Return 0/1 vector: parcel i in a and adjacent to a protected parcel (vectorized)."""
    has_prot_neigh = (map_data.adj @ x.astype(np.float32)) > 0
    return (a.astype(np.int8) & has_prot_neigh.astype(np.int8))


def _isolated_protected(a: np.ndarray, x: np.ndarray, map_data: MapData) -> np.ndarray:
    """Protected parcels (in a) with no protected neighbor."""
    has_prot_neigh = (map_data.adj @ x.astype(np.float32)) > 0
    return (a.astype(np.int8) & (~has_prot_neigh).astype(np.int8))


def _state_pack(state: State, map_data: MapData, cfg: Config):
    """Precompute state-only quantities."""
    x = state.x.astype(np.float32)
    d = state.d.astype(np.float32)
    free = (1.0 - x) * (1.0 - d)
    p = compute_development_probabilities(d, map_data, cfg.eps).astype(np.float32)
    q = frontier_pressure(d, map_data).astype(np.float32)
    has_prot_neigh = (map_data.adj @ x) > 0  # bool (n,)
    return x, d, free, p, q, has_prot_neigh


def features_phi(state: State, a: np.ndarray, k: np.ndarray,
                 map_data: MapData, cfg: Config) -> np.ndarray:
    """Dispatcher: engineered phi or frozen-encoder phi (cfg.use_encoder)."""
    if getattr(cfg, "use_encoder", False):
        from .encoder import encoder_phi
        return encoder_phi(state, a, k, map_data, cfg)
    return _engineered_features_phi(state, a, k, map_data, cfg)


def _engineered_features_phi(state: State, a: np.ndarray, k: np.ndarray,
                             map_data: MapData, cfg: Config) -> np.ndarray:
    """Single-pair engineered feature vector. Mirrors q_matrix_batch's layout."""
    n = map_data.n
    T = cfg.horizon_T
    value = map_data.value.astype(np.float32)
    cost = map_data.cost.astype(np.float32)
    total_value = max(1e-9, map_data.total_value)
    total_cost = max(1e-9, map_data.total_cost)
    cluster_onehot = map_data.cluster_onehot  # (n, C)
    eps = cfg.eps

    x, d, free, p, q, has_prot_neigh = _state_pack(state, map_data, cfg)
    a_f = a.astype(np.float32)
    k_f = k.astype(np.float32)
    # honor "protected parcels cannot be developed" when computing post-action features
    x_plus = np.clip(x + a_f, 0, 1)
    k_eff = k_f * (1.0 - x_plus)
    d_plus = np.clip(d + k_eff, 0, 1)
    free_plus = (1.0 - x_plus) * (1.0 - d_plus)
    # Use state-only p, q for post-action features so the linear Q decomposition
    # in ``q_matrix_batch`` matches this routine exactly. (The exact p_plus, q_plus
    # are nonlinear in k via neighbor sums; we accept that approximation.)
    p_plus = p
    q_plus = q

    feats = []
    # --- Global (10) ---
    feats.append(1.0)
    feats.append(state.t / T)
    feats.append((T - state.t) / T)
    feats.append(state.budget / total_cost)
    feats.append(state.budget / total_cost)
    feats.append(float((value * x).sum()) / total_value)
    feats.append(float((value * d).sum()) / total_value)
    feats.append(float((value * free).sum()) / total_value)
    feats.append(float((value * p * free).sum()) / total_value)
    feats.append(float((value * q * free).sum()) / total_value)
    # --- Controller (9) ---
    feats.append(float((value * a_f).sum()) / total_value)
    feats.append(float((cost * a_f).sum()) / total_cost)
    feats.append(float((value * a_f / (cost + eps)).sum()) / n)
    feats.append(float((value * p * a_f).sum()) / total_value)
    feats.append(float((value * p * a_f / (cost + eps)).sum()) / n)
    feats.append(float((value * q * a_f).sum()) / total_value)
    feats.append(float(a_f.sum()) / n)
    iso = _isolated_protected(a, x.astype(np.int8), map_data).astype(np.float32)
    feats.append(float((value * iso).sum()) / total_value)
    adj_p = _adjacent_to_protected(a, x.astype(np.int8), map_data).astype(np.float32)
    feats.append(float((value * adj_p).sum()) / total_value)
    # --- Nature (5) ---
    feats.append(float((value * k_eff).sum()) / total_value)
    feats.append(float((value * p * k_eff).sum()) / total_value)
    feats.append(float((value * q * k_eff).sum()) / total_value)
    # slack feature: use state-only p for log-ratio (matches batched form).
    log_ratio = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1.0 - p, 1e-9, 1 - 1e-9))
    slack = ((d_plus * log_ratio).sum() - state.t * np.log(cfg.lambda_uncertainty)) / max(1, n)
    feats.append(float(slack))
    feats.append(float(k_eff.sum()) / n)
    # --- Post-action (5) ---
    feats.append(float((value * x_plus).sum()) / total_value)
    feats.append(float((value * d_plus).sum()) / total_value)
    feats.append(float((value * free_plus).sum()) / total_value)
    feats.append(float((value * p_plus * free_plus).sum()) / total_value)
    feats.append(float((value * q_plus * free_plus).sum()) / total_value)
    # --- Cluster (6 * C) ---
    for c in range(map_data.n_clusters):
        mask = cluster_onehot[:, c]
        feats.append(float((value * x * mask).sum()) / total_value)
        feats.append(float((value * d * mask).sum()) / total_value)
        feats.append(float((value * free * mask).sum()) / total_value)
        feats.append(float((value * p * free * mask).sum()) / total_value)
        feats.append(float((value * a_f * mask).sum()) / total_value)
        feats.append(float((value * k_eff * mask).sum()) / total_value)
    # --- Parcel-level (8 * n) ---
    f_xi = value * x / total_value
    f_di = value * d / total_value
    f_ai = value * a_f / total_value
    f_ki = value * k_eff / total_value
    f_pf = value * p * free / total_value
    f_pa = value * p * a_f / total_value
    f_pk = value * p * k_eff / total_value
    f_qf = value * q * free / total_value
    parcel_block = np.concatenate([f_xi, f_di, f_ai, f_ki, f_pf, f_pa, f_pk, f_qf])
    arr = np.concatenate([np.array(feats, dtype=np.float32), parcel_block.astype(np.float32)])
    return arr.astype(np.float64)


def feature_dim(map_data: MapData, cfg: Config | None = None) -> int:
    if cfg is not None and getattr(cfg, "use_encoder", False):
        return int(cfg.encoder_embed_dim)
    return 10 + 9 + 5 + 5 + 6 * map_data.n_clusters + 8 * map_data.n


def _slice_indices(map_data: MapData):
    """Return slice indices for the feature blocks."""
    n = map_data.n
    C = map_data.n_clusters
    i0 = 0
    i_global = (i0, i0 + 10); i0 += 10
    i_ctrl   = (i0, i0 + 9);  i0 += 9
    i_nat    = (i0, i0 + 5);  i0 += 5
    i_post   = (i0, i0 + 5);  i0 += 5
    i_clust  = (i0, i0 + 6 * C); i0 += 6 * C
    # parcel block: 8 sub-blocks of length n: f_xi, f_di, f_ai, f_ki, f_pf, f_pa, f_pk, f_qf
    i_parcel_start = i0
    return {
        "global": i_global, "ctrl": i_ctrl, "nat": i_nat, "post": i_post,
        "clust": i_clust, "parcel_start": i_parcel_start, "n": n, "C": C,
    }


def q_matrix_batch(state: State, A: np.ndarray, K: np.ndarray, theta: np.ndarray,
                   map_data: MapData, cfg: Config) -> np.ndarray:
    """Dispatcher: engineered low-rank Q matrix or frozen-encoder Q matrix."""
    if getattr(cfg, "use_encoder", False):
        from .encoder import encoder_q_matrix_batch
        return encoder_q_matrix_batch(state, A, K, theta, map_data, cfg)
    return _engineered_q_matrix_batch(state, A, K, theta, map_data, cfg)


def _engineered_q_matrix_batch(state: State, A: np.ndarray, K: np.ndarray, theta: np.ndarray,
                               map_data: MapData, cfg: Config) -> np.ndarray:
    """Compute Q[j, l] = theta . phi(s, A[j], K[l]) for all pairs without materializing phi.

    A: (J, n) int / float
    K: (L, n) int / float
    theta: (D,)
    """
    n = map_data.n
    T = cfg.horizon_T
    J = A.shape[0]
    L = K.shape[0]
    A_f = A.astype(np.float32)
    K_f = K.astype(np.float32)
    value = map_data.value.astype(np.float32)
    cost = map_data.cost.astype(np.float32)
    total_value = max(1e-9, map_data.total_value)
    total_cost = max(1e-9, map_data.total_cost)
    eps = cfg.eps
    cluster_onehot = map_data.cluster_onehot  # (n, C)

    x, d, free, p, q, has_prot_neigh = _state_pack(state, map_data, cfg)

    # post-action effective nature action: k_eff[l,j,i] = K[l,i] * (1 - x[i] - A[j,i]) clipped
    # Since x ⊥ a (controller can't protect already-protected) we have 1 - x_plus = (1-x)*(1-A) only
    # when A bits are restricted to (1-x). Candidate generation enforces this. So k_eff[l,j,i] = K[l,i] * (1-x)[i] * (1-A[j,i]).
    one_minus_x = (1.0 - x)
    # We'll need many quantities of the form theta_slice @ phi_block contributions; build the
    # contributions per (j, l) block by block.

    sl = _slice_indices(map_data)
    th = theta.astype(np.float32)

    # ---------- Global block (state-only): one number, broadcast ----------
    g0, g1 = sl["global"]
    gfeat = np.array([
        1.0,
        state.t / T,
        (T - state.t) / T,
        state.budget / total_cost,
        state.budget / total_cost,
        float((value * x).sum()) / total_value,
        float((value * d).sum()) / total_value,
        float((value * free).sum()) / total_value,
        float((value * p * free).sum()) / total_value,
        float((value * q * free).sum()) / total_value,
    ], dtype=np.float32)
    Q_global = float(th[g0:g1] @ gfeat)  # scalar
    # ---------- Controller block (depends on s, A): (J,) ----------
    c0, c1 = sl["ctrl"]
    tc = th[c0:c1]
    # ctrl[0]: value*a / tv
    v0 = (A_f @ value) / total_value
    v1 = (A_f @ cost) / total_cost
    v2 = (A_f @ (value / (cost + eps))) / n
    v3 = (A_f @ (value * p)) / total_value
    v4 = (A_f @ (value * p / (cost + eps))) / n
    v5 = (A_f @ (value * q)) / total_value
    v6 = A_f.sum(axis=1) / n
    # iso & adj
    has_prot = has_prot_neigh.astype(np.float32)  # (n,)
    v_iso = (A_f @ (value * (1.0 - has_prot))) / total_value
    v_adj = (A_f @ (value * has_prot)) / total_value
    ctrl_block = np.stack([v0, v1, v2, v3, v4, v5, v6, v_iso, v_adj], axis=1)  # (J, 9)
    Q_ctrl = ctrl_block @ tc  # (J,)
    # ---------- Nature block (depends on s, k_eff): k_eff = K * (1-x) * (1-A)
    # Define K_eff_jl = K[l] * (1-x) - K[l]*(1-x)*A[j]  (since 1-A is binary).
    # For features sum_i value*k_eff: we get value@K_eff = value@K*(1-x) - value*A@K*(1-x)
    # i.e. (L,) vector minus (J, L) matrix → (J, L).
    n0, n1 = sl["nat"]
    tn = th[n0:n1]
    # Precompute per-l projections of K * (1-x) against various weights w:
    Kx = K_f * one_minus_x[None, :]  # (L, n) — k_eff before subtracting A overlap
    # For each weight w (n,), nat_w[j, l] = K_eff[l, j] @ w = (Kx @ w)[l] - A[j] @ (Kx_l * ... )
    # But A varies. Use: nat_w[j,l] = sum_i Kx[l,i] * (1 - A[j,i]) * w[i]
    #                              = (Kx * w[None,:]) @ (1 - A.T)  -> (L, J)
    # So: nat_w = (Kx * w).sum(axis=1) - (Kx * w) @ A.T  → easier as broadcast
    def nat_w(w):
        Kxw = Kx * w[None, :]  # (L, n)
        s_l = Kxw.sum(axis=1)  # (L,)
        cross = Kxw @ A_f.T    # (L, J)
        return (s_l[:, None] - cross).T  # (J, L)
    Vn0 = nat_w(value) / total_value
    Vn1 = nat_w(value * p) / total_value
    Vn2 = nat_w(value * q) / total_value
    # slack: a state-summary minus k_eff-dependent log term. The literal feature uses p_plus(d_plus),
    # which depends on k_eff (nonlinear). Use p (state-only) approximation for slack here.
    # slack_per_n = (sum_i d_plus * log(p/(1-p)) - t * log(lambda)) / n   (baseline-corrected)
    log_ratio = np.log(np.clip(p, 1e-9, 1 - 1e-9) / np.clip(1.0 - p, 1e-9, 1 - 1e-9))  # (n,)
    # d_plus = d + k_eff (no overlap with d guaranteed by feasibility)
    base = float((d * log_ratio).sum())
    slack_const = (base - state.t * np.log(cfg.lambda_uncertainty)) / n
    Vn3_extra = nat_w(log_ratio) / n  # (J, L)
    Vn3 = slack_const + Vn3_extra
    Vn4 = nat_w(np.ones(n, dtype=np.float32)) / n
    # nat features matrix (J, L, 5); apply theta slice tn:
    Q_nat = (Vn0 * tn[0] + Vn1 * tn[1] + Vn2 * tn[2] + Vn3 * tn[3] + Vn4 * tn[4])  # (J, L)

    # ---------- Post-action block (depends on s, A, K_eff): (J, L) ----------
    p0, p1 = sl["post"]
    tp = th[p0:p1]
    # x_plus = x + A (no overlap), d_plus = d + K_eff
    # value * x_plus per j: (A_f @ value) + (value*x).sum()  → (J,)
    Va_val = A_f @ value
    sum_vx = float((value * x).sum())
    Vp_x = (sum_vx + Va_val[:, None]) / total_value  # (J,1) broadcast to (J,L)
    # value * d_plus per (j,l): sum value*d + nat_w(value)[j,l]
    sum_vd = float((value * d).sum())
    Vp_d = (sum_vd + Vn0 * total_value) / total_value  # (J, L)  (note Vn0 already / total_value)
    # value * free_plus = total_value - value*x_plus - value*d_plus
    Vp_free = 1.0 - Vp_x - Vp_d
    # For Vp_free * p_plus, etc., we need p_plus & q_plus which depend on d_plus through neighbors.
    # Approximation: use p, q (state-only) instead of p_plus, q_plus. This loses some nonlinearity
    # but is consistent with the slack approximation above and keeps Q low-rank.
    vpf = (value * p * free).sum() / total_value  # state baseline
    vqf = (value * q * free).sum() / total_value
    # Approximate: post-free risk-weighted = vpf - (A_f @ (value * p)) - nat_w(value*p)
    Va_vp = A_f @ (value * p) / total_value
    Vp_pfree = vpf - Va_vp[:, None] - Vn1                  # (J, L)
    Va_vq = A_f @ (value * q) / total_value
    Vp_qfree = vqf - Va_vq[:, None] - Vn2
    Q_post = (Vp_x * tp[0] + Vp_d * tp[1] + Vp_free * tp[2]
              + Vp_pfree * tp[3] + Vp_qfree * tp[4])  # (J, L)
    # ---------- Cluster block (6*C, depends on s, A, K_eff) ----------
    cl0, cl1 = sl["clust"]
    tcl = th[cl0:cl1]  # (6C,)
    Q_clust = np.zeros((J, L), dtype=np.float32)
    for c in range(map_data.n_clusters):
        mask = cluster_onehot[:, c]
        base_idx = 6 * c
        # f0: value*x*mask (state) — broadcast scalar
        f0 = float((value * x * mask).sum()) / total_value
        f1 = float((value * d * mask).sum()) / total_value
        f2 = float((value * free * mask).sum()) / total_value
        f3 = float((value * p * free * mask).sum()) / total_value
        # f4: value*a*mask depends on A (J,)
        Va_m = A_f @ (value * mask) / total_value
        # f5: value*k_eff*mask depends on (J,L)
        Vk_m = nat_w(value * mask) / total_value
        Q_clust += (tcl[base_idx] * f0
                    + tcl[base_idx + 1] * f1
                    + tcl[base_idx + 2] * f2
                    + tcl[base_idx + 3] * f3
                    + tcl[base_idx + 4] * Va_m[:, None]
                    + tcl[base_idx + 5] * Vk_m)
    # ---------- Parcel-level block (8*n) ----------
    # Build per-block weighted theta vectors (n,):
    ps = sl["parcel_start"]
    tp_xi = th[ps:ps + n]; ps2 = ps + n
    tp_di = th[ps2:ps2 + n]; ps2 += n
    tp_ai = th[ps2:ps2 + n]; ps2 += n
    tp_ki = th[ps2:ps2 + n]; ps2 += n
    tp_pf = th[ps2:ps2 + n]; ps2 += n
    tp_pa = th[ps2:ps2 + n]; ps2 += n
    tp_pk = th[ps2:ps2 + n]; ps2 += n
    tp_qf = th[ps2:ps2 + n]
    # State-only: const
    parcel_state = float((tp_xi * value * x).sum() + (tp_di * value * d).sum()
                         + (tp_pf * value * p * free).sum() + (tp_qf * value * q * free).sum()) / total_value
    # A-dependent (J,): tp_ai . (value*A_f)  +  tp_pa . (value*p*A_f)
    parcel_ctrl = (A_f @ (tp_ai * value) + A_f @ (tp_pa * value * p)) / total_value  # (J,)
    # K_eff-dependent (J, L): tp_ki . (value*K_eff) + tp_pk . (value*p*K_eff)
    Vp_ki = nat_w(tp_ki * value) / total_value         # uses our nat_w with weight=tp_ki*value
    Vp_pk = nat_w(tp_pk * value * p) / total_value
    # NOTE: nat_w divides by total_value here but it's applied to a weight that is not value;
    # we want sum_i tp_ki[i] * value[i] * k_eff[i] / total_value. nat_w(w)[j,l] = sum_i w[i]*K_eff[l,j,i].
    # So we need nat_w(tp_ki*value) / total_value. The function nat_w returns sum_i w[i]*K_eff. ✓
    # Fix: my nat_w above is correct (returns dot of K_eff with w). The "/ total_value" was applied
    # for the nat block features that are scaled by total_value. Here we want it raw, so undo division:
    # Actually I already wrote "nat_w(...) / total_value" which is consistent — we want / total_value
    # for the normalization. ✓
    Q_parcel = parcel_state + parcel_ctrl[:, None] + Vp_ki + Vp_pk  # (J, L)

    Q = Q_global + Q_ctrl[:, None] + Q_nat + Q_post + Q_clust + Q_parcel
    return Q.astype(np.float64)
