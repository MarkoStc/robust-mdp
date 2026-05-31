"""Interpret the frozen encoder's features for an existing trained run.

Two analyses (see session discussion):

  (1) Q-fit comparison, encoder phi vs engineered phi.
      Roll out episodes under the trained policy, record each visited
      (state, a, k) plus its Monte-Carlo return y = discounted sum of
      step_reward (surviving conservation value). Fit Ridge twice -- once on
      encoder phi (64-dim), once on the hand-engineered phi -- with a held-out
      split, and compare held-out R-squared. If they tie, the encoder is not
      buying predictive power over the cheap features.

  (2) Effective rank of phi.
      Collect encoder phi over the same buffer (rows = situations, cols = 64),
      run SVD, report the singular-value spectrum, the effective rank
      (participation ratio + 90/99% energy thresholds) -- diagnoses feature
      collapse from the global-average-pool bottleneck.

Usage:
  python tools/analyze_features.py --run-dir outputs/slurm/biodiv-2312346
"""
from __future__ import annotations
import argparse
import json
import os
import numpy as np

from src.config import Config
from src.state import build_default_map, initial_state
from src.dynamics import step
from src.rollout import policy_at_state, step_reward


def load_cfg(run_dir: str) -> Config:
    cfg = Config()
    with open(os.path.join(run_dir, "config.json")) as f:
        saved = json.load(f)
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def collect_buffer(cfg, md, theta, n_episodes, rng):
    """Roll out under the trained policy; return list of (state,a,k) and MC returns y."""
    samples = []          # (state_copy, a, k)
    rewards_per_ep = []
    states_per_ep = []
    for ep in range(n_episodes):
        state = initial_state(cfg, md)
        ep_sak, ep_r = [], []
        for _ in range(cfg.horizon_T):
            pi, omega, A_cand, K_cand, _, _ = policy_at_state(state, theta, md, cfg, rng)
            a = A_cand[int(np.argmax(pi))]
            # sample nature from omega (its mixed strategy) for buffer diversity
            l = int(rng.choice(len(omega), p=np.clip(omega, 0, None) / max(1e-12, omega.sum())))
            k = K_cand[l]
            nxt = step(state, a, k, cfg)
            r = step_reward(state, a, k, nxt, md, cfg)
            ep_sak.append((state.copy(), a.copy(), k.copy()))
            ep_r.append(r)
            state = nxt
        # discounted MC return from each step to end
        G = 0.0
        ep_y = [0.0] * len(ep_r)
        for t in reversed(range(len(ep_r))):
            G = ep_r[t] + cfg.gamma * G
            ep_y[t] = G
        samples.extend(ep_sak)
        rewards_per_ep.append(ep_y)
        states_per_ep.append(ep_sak)
    y = np.array([g for ep in rewards_per_ep for g in ep], dtype=np.float64)
    return samples, y


def embed(samples, md, cfg, use_encoder):
    """Return (N, D) feature matrix using encoder phi or engineered phi."""
    from src.features import features_phi
    c = Config(**{k: getattr(cfg, k) for k in cfg.__dict__})
    c.use_encoder = use_encoder
    if use_encoder:
        c.encoder_path = cfg.encoder_path
        c.encoder_device = cfg.encoder_device
    rows = [features_phi(s, a, k, md, c) for (s, a, k) in samples]
    return np.asarray(rows, dtype=np.float64)


def ridge_r2(X, y, alpha, seed=0, test_frac=0.25):
    """Fit Ridge on a train split, return (r2_train, r2_test, n_train, n_test)."""
    from sklearn.linear_model import Ridge
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_test = int(n * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    # standardize features on train stats
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-9
    Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
    best = None
    for a in (alpha if isinstance(alpha, (list, tuple)) else [alpha]):
        m = Ridge(alpha=a).fit(Xtr, y[tr])
        from sklearn.metrics import r2_score
        r2te = r2_score(y[te], m.predict(Xte))
        r2tr = r2_score(y[tr], m.predict(Xtr))
        if best is None or r2te > best[1]:
            best = (r2tr, r2te, a)
    return best[0], best[1], best[2], len(tr), len(te)


def effective_rank(Phi):
    """SVD-based diagnostics of the encoder feature matrix (N, D)."""
    Xc = Phi - Phi.mean(0, keepdims=True)
    s = np.linalg.svd(Xc, compute_uv=False)
    ev = s ** 2
    ev_sum = ev.sum() + 1e-12
    p = ev / ev_sum
    # participation ratio (a.k.a. effective dimension)
    pr = (ev.sum() ** 2) / (np.sum(ev ** 2) + 1e-12)
    # Shannon effective rank
    pe = p[p > 0]
    shannon = float(np.exp(-(pe * np.log(pe)).sum()))
    cum = np.cumsum(p)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k99 = int(np.searchsorted(cum, 0.99) + 1)
    return {
        "n_dims": int(Phi.shape[1]),
        "singular_values_top12": [float(x) for x in s[:12]],
        "energy_frac_top12": [float(x) for x in p[:12]],
        "participation_ratio": float(pr),
        "shannon_effective_rank": shannon,
        "dims_for_90pct_energy": k90,
        "dims_for_99pct_energy": k99,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/slurm/biodiv-2312346")
    ap.add_argument("--theta", default="theta_final.npy")
    ap.add_argument("--n-episodes", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg = load_cfg(args.run_dir)
    cfg.encoder_path = os.path.join(args.run_dir, "encoder.pt")
    if cfg.use_encoder:
        try:
            import torch
            cfg.encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.encoder_device = "cpu"
    md = build_default_map(cfg, np.random.default_rng(cfg.seed))
    theta = np.load(os.path.join(args.run_dir, args.theta))
    rng = np.random.default_rng(args.seed)

    print(f"== run={args.run_dir} use_encoder={cfg.use_encoder} n={md.n} "
          f"embed_dim={cfg.encoder_embed_dim} device={cfg.encoder_device}", flush=True)
    print(f"== collecting buffer: {args.n_episodes} episodes x T={cfg.horizon_T}", flush=True)
    samples, y = collect_buffer(cfg, md, theta, args.n_episodes, rng)
    print(f"== buffer: N={len(samples)} situations, "
          f"y mean={y.mean():.3f} std={y.std():.3f} min={y.min():.3f} max={y.max():.3f}", flush=True)

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    out = {"run_dir": args.run_dir, "N": len(samples),
           "y_stats": {"mean": float(y.mean()), "std": float(y.std())}}

    # ---- (1) Q-fit comparison -------------------------------------------
    print("\n== (1) Q-fit: encoder phi vs engineered phi (held-out R^2) ==", flush=True)
    Xenc = embed(samples, md, cfg, use_encoder=True)
    r2tr_e, r2te_e, a_e, ntr, nte = ridge_r2(Xenc, y, alphas, seed=args.seed)
    print(f"  encoder    phi: dim={Xenc.shape[1]:4d}  train R2={r2tr_e:.4f}  "
          f"HELD-OUT R2={r2te_e:.4f}  (alpha={a_e}, n_tr={ntr} n_te={nte})", flush=True)

    Xeng = embed(samples, md, cfg, use_encoder=False)
    r2tr_g, r2te_g, a_g, _, _ = ridge_r2(Xeng, y, alphas, seed=args.seed)
    print(f"  engineered phi: dim={Xeng.shape[1]:4d}  train R2={r2tr_g:.4f}  "
          f"HELD-OUT R2={r2te_g:.4f}  (alpha={a_g})", flush=True)

    gap = r2te_e - r2te_g
    verdict = ("encoder clearly better" if gap > 0.05 else
               "engineered clearly better" if gap < -0.05 else
               "tie -> encoder adds little over engineered features")
    print(f"  --> held-out R2 gap (enc - eng) = {gap:+.4f}  [{verdict}]", flush=True)
    out["qfit"] = {
        "encoder": {"dim": int(Xenc.shape[1]), "r2_train": r2tr_e, "r2_heldout": r2te_e, "alpha": a_e},
        "engineered": {"dim": int(Xeng.shape[1]), "r2_train": r2tr_g, "r2_heldout": r2te_g, "alpha": a_g},
        "heldout_gap_enc_minus_eng": gap, "verdict": verdict,
    }

    # ---- (2) Effective rank of encoder phi ------------------------------
    print("\n== (2) Effective rank of encoder phi ==", flush=True)
    er = effective_rank(Xenc)
    print(f"  nominal dims         : {er['n_dims']}", flush=True)
    print(f"  participation ratio  : {er['participation_ratio']:.2f}", flush=True)
    print(f"  Shannon eff. rank    : {er['shannon_effective_rank']:.2f}", flush=True)
    print(f"  dims for 90% energy  : {er['dims_for_90pct_energy']}", flush=True)
    print(f"  dims for 99% energy  : {er['dims_for_99pct_energy']}", flush=True)
    print(f"  top-12 energy frac   : {[round(x,3) for x in er['energy_frac_top12']]}", flush=True)
    collapse = ("SEVERE collapse" if er['participation_ratio'] < er['n_dims'] * 0.15 else
                "moderate collapse" if er['participation_ratio'] < er['n_dims'] * 0.4 else
                "healthy spread")
    print(f"  --> {collapse} (PR={er['participation_ratio']:.1f} of {er['n_dims']})", flush=True)
    out["effective_rank"] = er
    out["effective_rank"]["assessment"] = collapse

    out_path = args.out or os.path.join(args.run_dir, "feature_analysis.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n== saved {out_path} ==", flush=True)


if __name__ == "__main__":
    main()
