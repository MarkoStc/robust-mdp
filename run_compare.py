#!/usr/bin/env python
"""Compare our learned robust-MDP policy against the paper's methods.

Reference: Ye et al., "Conserving Biodiversity via Adjustable Robust
Optimization" (AAMAS 2022). Their *final proposed* method is StaticApprox
(Problem 7); their benchmark is the Knapsack method (Problem 8/9).

We evaluate three controllers inside our toy environment, all on equal footing:

    Ours          -- trained robust-MDP policy (adaptive matrix game over
                     candidate actions, frozen encoder + learned theta)
    StaticApprox  -- paper's proposal: one non-adaptive protection schedule for
                     the whole horizon, robust to the worst-case in U
    Knapsack      -- paper's benchmark: greedy value maximization per year

against two natures:

    worst-case  -- greedy adversary inside the likelihood uncertainty set U
                   (this is the "adversarial nature" robustness comparison)
    stochastic  -- cellular-automata Bernoulli(p_i) realized developments
                   (the average-case the paper simulates in Figure 7)

Outputs: <out-dir>/comparison_results.json and <out-dir>/comparison_vs_paper.png
"""
from __future__ import annotations
import argparse
import json
import os
import time
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import Config
from src.state import build_default_map, initial_state
from src.dynamics import step
from src.rollout import policy_at_state
from src.baselines import (
    knapsack_protect, static_approx_schedule, schedule_action,
    greedy_adversary_develop, bernoulli_develop,
)


def load_config(run_dir: str) -> Config:
    """Rebuild the training Config from the saved config.json."""
    cfg = Config()
    with open(os.path.join(run_dir, "config.json")) as f:
        saved = json.load(f)
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def final_stats(state, map_data):
    free = (state.x == 0) & (state.d == 0)
    return {
        "protected_value": float((map_data.value * state.x).sum()),
        "lost_value": float((map_data.value * state.d).sum()),
        "free_value": float((map_data.value * free).sum()),
    }


def run_episode(controller_fn, nature_fn, cfg, map_data, rng_ctrl, rng_nat):
    state = initial_state(cfg, map_data)
    for _ in range(cfg.horizon_T):
        a = controller_fn(state, rng_ctrl)
        k = nature_fn(state, a, map_data, cfg, rng_nat)
        state = step(state, a, k, cfg)
    return final_stats(state, map_data)


def worst_case_nature(state, a, map_data, cfg, rng):
    return greedy_adversary_develop(state, a, map_data, cfg)


def aggregate(samples):
    arr = {key: np.array([s[key] for s in samples]) for key in samples[0]}
    out = {}
    for key, vals in arr.items():
        out[key] = {
            "mean": float(vals.mean()), "std": float(vals.std()),
            "min": float(vals.min()), "max": float(vals.max()),
            "median": float(np.median(vals)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/slurm/biodiv-2312346",
                    help="Completed training run with config.json + encoder.pt + theta")
    ap.add_argument("--theta", default="theta_final.npy",
                    help="theta filename inside run-dir")
    ap.add_argument("--n-stochastic", type=int, default=40,
                    help="episodes for the average-case stochastic nature")
    ap.add_argument("--static-max-iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    run_dir = args.run_dir
    out_dir = args.out_dir or os.path.join(run_dir, "compare")
    os.makedirs(out_dir, exist_ok=True)

    cfg = load_config(run_dir)
    cfg.encoder_path = os.path.join(run_dir, "encoder.pt")
    if cfg.use_encoder:
        try:
            import torch
            cfg.encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.encoder_device = "cpu"
    print(f"== config: lambda={cfg.lambda_uncertainty} budget={cfg.budget_per_year} "
          f"T={cfg.horizon_T} use_encoder={cfg.use_encoder} device={cfg.encoder_device}", flush=True)

    # Rebuild the SAME map the policy was trained on (seed-matched).
    md = build_default_map(cfg, np.random.default_rng(cfg.seed))
    theta = np.load(os.path.join(run_dir, args.theta))
    print(f"== loaded theta {args.theta}: shape={theta.shape} norm={np.linalg.norm(theta):.4f}", flush=True)

    # ---- controllers ----------------------------------------------------
    def ours_fn(state, rng):
        pi, omega, A_cand, K_cand, v, Q = policy_at_state(state, theta, md, cfg, rng)
        return A_cand[int(np.argmax(pi))]

    def knapsack_fn(state, rng):
        return knapsack_protect(state, md, cfg)

    print("== solving StaticApprox (constraint generation) ...", flush=True)
    t0 = time.time()
    schedule = static_approx_schedule(md, cfg, max_iters=args.static_max_iters,
                                      log_fn=lambda *a: print(*a, flush=True))
    print(f"== StaticApprox solved in {time.time()-t0:.1f}s "
          f"(plans to protect {int(schedule[-1].sum())} parcels)", flush=True)

    def static_fn(state, rng):
        return schedule_action(schedule, state)

    controllers = {"Ours": ours_fn, "StaticApprox": static_fn, "Knapsack": knapsack_fn}

    # ---- worst-case (adversarial) evaluation ----------------------------
    results = {}
    print("\n== Worst-case adversarial nature ==", flush=True)
    for name, fn in controllers.items():
        rng_ctrl = np.random.default_rng(args.seed)
        stats = run_episode(fn, worst_case_nature, cfg, md, rng_ctrl, None)
        results.setdefault(name, {})["worst_case"] = stats
        print(f"  {name:12s}  lost={stats['lost_value']:.3f}  "
              f"protected={stats['protected_value']:.3f}  free={stats['free_value']:.3f}", flush=True)

    # ---- average-case (stochastic) evaluation ---------------------------
    print(f"\n== Average-case stochastic nature ({args.n_stochastic} episodes) ==", flush=True)
    for name, fn in controllers.items():
        samples = []
        for s in range(args.n_stochastic):
            rng_ctrl = np.random.default_rng(args.seed + 1000 + s)
            rng_nat = np.random.default_rng(args.seed + 5000 + s)
            samples.append(run_episode(fn, bernoulli_develop, cfg, md, rng_ctrl, rng_nat))
        agg = aggregate(samples)
        results[name]["stochastic"] = agg
        print(f"  {name:12s}  lost mean={agg['lost_value']['mean']:.3f}"
              f" [{agg['lost_value']['min']:.2f},{agg['lost_value']['max']:.2f}]"
              f"  protected mean={agg['protected_value']['mean']:.3f}", flush=True)

    # ---- save + plot ----------------------------------------------------
    meta = {
        "run_dir": run_dir, "theta": args.theta,
        "lambda_uncertainty": cfg.lambda_uncertainty,
        "budget_per_year": cfg.budget_per_year, "horizon_T": cfg.horizon_T,
        "total_value": float(md.total_value), "n_stochastic": args.n_stochastic,
        "static_plan_parcels": int(schedule[-1].sum()),
    }
    with open(os.path.join(out_dir, "comparison_results.json"), "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)

    make_plot(results, md, out_dir)
    print(f"\n== saved results + plot to {out_dir} ==", flush=True)
    print_verdict(results)


def make_plot(results, md, out_dir):
    names = list(results.keys())
    colors = {"Ours": "#2c7fb8", "StaticApprox": "#d95f0e", "Knapsack": "#7a7a7a"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(names))
    width = 0.38

    # panel 0: lost value (lower is better)  -- the robustness metric
    ax = axes[0]
    wc = [results[n]["worst_case"]["lost_value"] for n in names]
    st = [results[n]["stochastic"]["lost_value"]["mean"] for n in names]
    st_lo = [results[n]["stochastic"]["lost_value"]["mean"]
             - results[n]["stochastic"]["lost_value"]["min"] for n in names]
    st_hi = [results[n]["stochastic"]["lost_value"]["max"]
             - results[n]["stochastic"]["lost_value"]["mean"] for n in names]
    ax.bar(x - width / 2, wc, width, label="worst-case (adversarial)",
           color=[colors[n] for n in names])
    ax.bar(x + width / 2, st, width, yerr=[st_lo, st_hi], capsize=4,
           label="stochastic (avg ± range)", color=[colors[n] for n in names], alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Lost conservation value")
    ax.set_title("Loss to development (lower = better)")
    ax.legend(fontsize=8)

    # panel 1: protected value (higher is better)
    ax = axes[1]
    wc = [results[n]["worst_case"]["protected_value"] for n in names]
    st = [results[n]["stochastic"]["protected_value"]["mean"] for n in names]
    ax.bar(x - width / 2, wc, width, label="worst-case (adversarial)",
           color=[colors[n] for n in names])
    ax.bar(x + width / 2, st, width, label="stochastic (avg)",
           color=[colors[n] for n in names], alpha=0.5)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Protected conservation value")
    ax.set_title("Value preserved (higher = better)")
    ax.legend(fontsize=8)

    fig.suptitle(f"Our robust-MDP policy vs paper methods  "
                 f"(total value = {md.total_value:.1f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "comparison_vs_paper.png"), dpi=130)
    plt.close(fig)


def print_verdict(results):
    print("\n== Verdict (worst-case adversarial lost value, lower=better) ==", flush=True)
    ranked = sorted(results.items(), key=lambda kv: kv[1]["worst_case"]["lost_value"])
    for rank, (name, r) in enumerate(ranked, 1):
        print(f"  {rank}. {name:12s}  lost={r['worst_case']['lost_value']:.3f}", flush=True)
    best = ranked[0][0]
    ours = results["Ours"]["worst_case"]["lost_value"]
    for name in results:
        if name == "Ours":
            continue
        other = results[name]["worst_case"]["lost_value"]
        if other > 0:
            print(f"  Ours vs {name}: {(other - ours) / other * 100:+.1f}% less worst-case loss", flush=True)


if __name__ == "__main__":
    main()
