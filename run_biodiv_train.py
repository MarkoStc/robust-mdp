#!/usr/bin/env python
"""Non-interactive training entry point for the biodiversity robust MDP."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import numpy as np

from src.config import Config
from src.state import build_default_map, initial_state
from src.policy_iteration import train
from src.sanity import sanity_check_all
from src.diagnostics import action_span_diagnostics


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-uncertainty", type=float, default=0.20)
    ap.add_argument("--samples-per-iter", type=int, default=2000)
    ap.add_argument("--outer-iters", type=int, default=10)
    ap.add_argument("--n-controller-candidates", type=int, default=500)
    ap.add_argument("--n-nature-candidates", type=int, default=500)
    ap.add_argument("--inner-M", type=int, default=8)
    ap.add_argument("--horizon-T", type=int, default=10)
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--budget-per-year", type=float, default=2.0)
    ap.add_argument("--rollout-episodes", type=int, default=20)
    ap.add_argument("--td-step-eta", type=float, default=0.02)
    ap.add_argument("--td-skip-tau", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, default="outputs/run")
    ap.add_argument("--run-diagnostics", action="store_true",
                    help="Run action-span diagnostics at final theta")
    # Encoder mode
    ap.add_argument("--use-encoder", action="store_true",
                    help="Use a pretrained-then-frozen CNN encoder for phi")
    ap.add_argument("--encoder-path", type=str, default="",
                    help="Existing encoder checkpoint; pretrained on the fly if empty")
    ap.add_argument("--encoder-embed-dim", type=int, default=64)
    ap.add_argument("--encoder-pretrain-episodes", type=int, default=400)
    ap.add_argument("--encoder-pretrain-epochs", type=int, default=30)
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        lambda_uncertainty=args.lambda_uncertainty,
        samples_per_iter=args.samples_per_iter,
        outer_iters=args.outer_iters,
        n_controller_candidates=args.n_controller_candidates,
        n_nature_candidates=args.n_nature_candidates,
        inner_M=args.inner_M,
        horizon_T=args.horizon_T,
        gamma=args.gamma,
        budget_per_year=args.budget_per_year,
        rollout_episodes=args.rollout_episodes,
        td_step_eta=args.td_step_eta,
        td_skip_tau=args.td_skip_tau,
        seed=args.seed,
        out_dir=args.out_dir,
        use_encoder=args.use_encoder,
        encoder_path=args.encoder_path,
        encoder_embed_dim=args.encoder_embed_dim,
        encoder_pretrain_episodes=args.encoder_pretrain_episodes,
        encoder_pretrain_epochs=args.encoder_pretrain_epochs,
    )
    if cfg.use_encoder:
        try:
            import torch
            cfg.encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.encoder_device = "cpu"

    os.makedirs(cfg.out_dir, exist_ok=True)
    # serialize the dataclass safely (drop non-JSON values)
    cfg_dict = {k: v for k, v in cfg.__dict__.items() if isinstance(v, (int, float, str, bool, list, tuple, type(None)))}
    with open(os.path.join(cfg.out_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2, default=str)

    rng = np.random.default_rng(cfg.seed)
    md = build_default_map(cfg, rng)

    if cfg.use_encoder:
        print(f"=== Encoder mode (device={cfg.encoder_device}) ===", flush=True)
        from src.encoder import pretrain_encoder
        if not cfg.encoder_path:
            cfg.encoder_path = os.path.join(cfg.out_dir, "encoder.pt")
        if not os.path.exists(cfg.encoder_path):
            t_pre = time.time()
            pretrain_encoder(cfg, md, rng, cfg.encoder_path,
                             log_fn=lambda *a, **kw: print(*a, **kw, flush=True))
            print(f"=== Pretrain done in {time.time()-t_pre:.1f}s ===", flush=True)
        else:
            print(f"[encoder] using existing checkpoint {cfg.encoder_path}", flush=True)

    print("=== Sanity checks ===", flush=True)
    sanity_check_all(cfg, md, rng)
    print("sanity ok", flush=True)

    print("=== Training ===", flush=True)
    t0 = time.time()
    out = train(cfg, md, log_fn=lambda *a, **kw: print(*a, **kw, flush=True))
    elapsed = time.time() - t0
    print(f"=== Training done in {elapsed:.1f}s ===", flush=True)

    if args.run_diagnostics:
        print("=== Action-span diagnostics (full horizon, on-policy) ===", flush=True)
        from src.dynamics import step
        from src.rollout import policy_at_state
        theta = out["theta"]
        state = initial_state(cfg, md)
        diags = []
        # Walk the full horizon along the learned robust policy so the diagnostic
        # measures candidate coverage at states a real episode actually visits
        # (not idle zero-action grids).
        for _ in range(cfg.horizon_T):
            try:
                d = action_span_diagnostics(state, md, cfg, rng, n_reference=3000)
            except Exception as e:
                d = {"error": str(e)}
            d["t"] = int(state.t)
            diags.append(d)
            # advance one tick using the mode (most likely) action for each player
            try:
                pi, omega, A_cand, K_cand, _, _ = policy_at_state(state, theta, md, cfg, rng)
                a = A_cand[int(np.argmax(pi))]
                k = K_cand[int(np.argmax(omega))]
            except Exception:
                a = np.zeros(md.n, dtype=np.int8)
                k = np.zeros(md.n, dtype=np.int8)
            state = step(state, a, k, cfg)
        with open(os.path.join(cfg.out_dir, "diagnostics.json"), "w") as f:
            json.dump(diags, f, indent=2)
        for d in diags:
            print(f"diag t={d.get('t')}: {d}", flush=True)
    print("=== All done ===", flush=True)


if __name__ == "__main__":
    main()
