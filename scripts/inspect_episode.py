"""Run one episode under a trained theta and print qualitative picks per year.

For each year prints:
  - hotspot map header on year 1
  - controller picks: (row,col), value v_i, cost c_i, p_i, frontier q_i, dev-nbrs
  - nature picks: same fields
  - summary: protected/developed/free value and likelihood slack
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np

from src.config import Config
from src.state import build_default_map, initial_state
from src.dynamics import (
    compute_development_probabilities,
    frontier_pressure,
    likelihood_slack,
    step,
)
from src.rollout import policy_at_state, sample_action


def coerce_cfg_dict(d: dict) -> dict:
    """Match notebook coercion: restore tuple hotspots, drop unknown keys."""
    out = {}
    valid = set(Config().__dict__.keys())
    for k, v in d.items():
        if k not in valid:
            continue
        if k == "value_hotspots" and isinstance(v, list):
            out[k] = [tuple(t) for t in v]
        else:
            out[k] = v
    return out


def fmt_idx(idx: int, w: int) -> str:
    return f"({idx // w:>2},{idx % w:>2})"


def grid_ascii(x: np.ndarray, d: np.ndarray, h: int, w: int,
               a: np.ndarray | None = None, k: np.ndarray | None = None,
               hotspots=()) -> str:
    """Return an ASCII grid:
      .  free
      P  protected (already)
      D  developed (already)
      p  newly protected this step (a==1)
      d  newly developed this step (k==1)
      *  hotspot center
    """
    grid = [["." for _ in range(w)] for _ in range(h)]
    for i in range(h * w):
        r, c = i // w, i % w
        if x[i] == 1:
            grid[r][c] = "P"
        elif d[i] == 1:
            grid[r][c] = "D"
    if a is not None:
        for i in range(h * w):
            if a[i] == 1:
                r, c = i // w, i % w
                grid[r][c] = "p"
    if k is not None:
        for i in range(h * w):
            if k[i] == 1:
                r, c = i // w, i % w
                if grid[r][c] != "p":  # protection wins display
                    grid[r][c] = "d"
    # mark hotspot centers
    for (cr, cc, _, _) in hotspots:
        ir, ic = int(round(cr)), int(round(cc))
        if 0 <= ir < h and 0 <= ic < w:
            if grid[ir][ic] == ".":
                grid[ir][ic] = "*"
    rows = ["    " + " ".join(f"{c:>2}" for c in range(w))]
    for r in range(h):
        rows.append(f"{r:>2}  " + "  ".join(grid[r]))
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str,
                    default="outputs/slurm/biodiv-2312346")
    ap.add_argument("--theta", type=str, default="theta_final.npy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nature-mode", type=str, default="omega",
                    choices=["omega", "worst"])
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.json")
    theta_path = os.path.join(args.run_dir, args.theta)
    with open(cfg_path) as f:
        raw = json.load(f)
    cfg = Config(**coerce_cfg_dict(raw))
    # The encoder is loaded via cfg.encoder_path; ensure it points at this run.
    if cfg.use_encoder:
        cfg.encoder_path = os.path.join(args.run_dir, "encoder.pt")
        try:
            import torch
            cfg.encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.encoder_device = "cpu"

    theta = np.load(theta_path)
    print(f"[inspect] cfg: lambda={cfg.lambda_uncertainty}, budget={cfg.budget_per_year}, "
          f"T={cfg.horizon_T}, use_encoder={cfg.use_encoder}, embed_dim={cfg.encoder_embed_dim}")
    print(f"[inspect] theta shape={theta.shape}, ||theta||={np.linalg.norm(theta):.4f}")
    print(f"[inspect] nature mode = {args.nature_mode}")

    rng = np.random.default_rng(args.seed)
    md = build_default_map(cfg, rng)
    state = initial_state(cfg, md)
    print(f"[inspect] total_value={md.total_value:.3f}, total_cost={md.total_cost:.3f}")
    print(f"[inspect] hotspots (row,col,peak,std): {cfg.value_hotspots}")

    # Show top-10 value parcels so we know what "should" be defended.
    top_val = np.argsort(md.value)[::-1][:10]
    print("\nTop-10 value parcels:")
    for i in top_val:
        print(f"  {fmt_idx(int(i), md.w)}  v={md.value[i]:.3f}  c={md.cost[i]:.3f}  "
              f"TI={md.threat[i]:.2f}  cluster={md.cluster[i]}")

    for t_idx in range(cfg.horizon_T):
        print("\n" + "=" * 78)
        print(f"YEAR t={state.t}    budget={state.budget:.2f}")
        print("=" * 78)

        p = compute_development_probabilities(state.d, md, cfg.eps)
        q = frontier_pressure(state.d, md)

        pi, omega, A_cand, K_cand, v_est, Q = policy_at_state(state, theta, md, cfg, rng)
        # Pick the *mode* (most likely action) for both, plus also note worst-case for nature.
        j = int(np.argmax(pi))
        a = A_cand[j]
        if args.nature_mode == "omega":
            l = int(np.argmax(omega))
        else:
            l = int(np.argmin(Q[j]))
        k = K_cand[l]

        # Show grid BEFORE step (with picks marked)
        print(f"V-est={v_est:.4f}   pi_mass(chosen a)={pi[j]:.3f}   "
              f"omega_mass(chosen k)={omega[l]:.3f}   "
              f"|A|={len(A_cand)} |K|={len(K_cand)}")
        print()
        print(grid_ascii(state.x, state.d, md.h, md.w, a=a, k=k,
                        hotspots=cfg.value_hotspots))

        # Controller picks detail
        a_idx = np.where(a == 1)[0]
        print(f"\nController picks ({len(a_idx)} parcels, cost={md.cost[a_idx].sum():.2f}/{state.budget:.2f}):")
        for i in a_idx:
            i = int(i)
            dev_nbrs = int(md.adj[i] @ state.d.astype(np.float32))
            nbrs = int(md.deg[i])
            print(f"  {fmt_idx(i, md.w)}  v={md.value[i]:.3f}  c={md.cost[i]:.3f}  "
                  f"TI={md.threat[i]:.2f}  p={p[i]:.3f}  q={q[i]:.3f}  "
                  f"dev_nbrs={dev_nbrs}/{nbrs}  cluster={md.cluster[i]}")

        # Nature picks detail
        k_idx = np.where(k == 1)[0]
        d_plus = state.d | k
        slack = likelihood_slack(state, d_plus, md, cfg)
        print(f"\nNature picks ({len(k_idx)} parcels, lik-slack={slack:+.3f}):")
        for i in k_idx:
            i = int(i)
            dev_nbrs = int(md.adj[i] @ state.d.astype(np.float32))
            nbrs = int(md.deg[i])
            print(f"  {fmt_idx(i, md.w)}  v={md.value[i]:.3f}  TI={md.threat[i]:.2f}  "
                  f"p={p[i]:.3f}  q={q[i]:.3f}  "
                  f"dev_nbrs={dev_nbrs}/{nbrs}  cluster={md.cluster[i]}")

        # Step
        state = step(state, a, k, cfg)
        prot_v = float((md.value * state.x).sum())
        dev_v = float((md.value * state.d).sum())
        free_v = float((md.value * ((state.x == 0) & (state.d == 0))).sum())
        print(f"\n-> after step: protected_v={prot_v:.3f}  developed_v={dev_v:.3f}  "
              f"free_v={free_v:.3f}  (total={md.total_value:.3f})")

    print("\n=== Final ===")
    print(grid_ascii(state.x, state.d, md.h, md.w, hotspots=cfg.value_hotspots))


if __name__ == "__main__":
    main()
