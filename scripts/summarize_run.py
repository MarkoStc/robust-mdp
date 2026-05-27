#!/usr/bin/env python
"""Render a quick summary of a training run directory."""
from __future__ import annotations
import argparse
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.json")
    log_path = os.path.join(args.run_dir, "train_log.json")
    diag_path = os.path.join(args.run_dir, "diagnostics.json")

    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
        print("=== Config ===")
        for k in ("lambda_uncertainty", "outer_iters", "inner_M", "samples_per_iter",
                  "n_controller_candidates", "n_nature_candidates", "horizon_T",
                  "gamma", "budget_per_year", "rollout_episodes"):
            print(f"  {k} = {cfg.get(k)}")
    if os.path.exists(log_path):
        log = json.load(open(log_path))
        print("\n=== Training curves ===")
        n = len(log["outer_iter"])
        print(f"  outer | protected | developed | free | ||theta||")
        for i in range(n):
            print(f"  {log['outer_iter'][i]:>5} | "
                  f"{log['avg_protected_value'][i]:>9.3f} | "
                  f"{log['avg_developed_value'][i]:>9.3f} | "
                  f"{log['avg_free_value'][i]:>5.3f} | "
                  f"{log['theta_norm'][i]:>8.3f}")
        # quick diagnostic: did learning improve?
        if n >= 2:
            best = max(log['avg_protected_value'])
            first = log['avg_protected_value'][0]
            print(f"\n  protected_value (best vs first): {best:.3f} vs {first:.3f}")
    if os.path.exists(diag_path):
        diag = json.load(open(diag_path))
        print("\n=== Action-span diagnostics (per t) ===")
        for i, d in enumerate(diag):
            print(f"  t={i+1}:")
            for who in ("controller", "nature"):
                if who in d:
                    w = d[who]
                    print(f"    {who}: n_cand={w.get('n_cand')}, n_ref={w.get('n_ref')}, "
                          f"median_err={w.get('median_proj_err'):.2e}, p95={w.get('p95'):.2e}")
                elif "error" in d:
                    print(f"    error: {d['error']}")


if __name__ == "__main__":
    main()
