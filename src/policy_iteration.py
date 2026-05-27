"""Algorithm 4: approximate policy iteration loop (outer).

For each outer iter n:
  1. Freeze controller policy as the matrix-game policy from theta_n.
  2. Run Algorithm 1 to obtain theta_{n+1} approximating Q_{pi_{n+1}}.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import json
import os
import numpy as np

from .config import Config, MapData
from .features import feature_dim
from .dual_averaging import algorithm1_policy_eval
from .rollout import rollout_episode
from .sanity import sanity_check_all


@dataclass
class TrainLog:
    outer_iter: List[int] = field(default_factory=list)
    avg_protected_value: List[float] = field(default_factory=list)
    avg_developed_value: List[float] = field(default_factory=list)
    avg_free_value: List[float] = field(default_factory=list)
    theta_norm: List[float] = field(default_factory=list)


def train(cfg: Config, map_data: MapData, log_fn=print) -> dict:
    rng = np.random.default_rng(cfg.seed)
    D = feature_dim(map_data, cfg)
    theta = np.zeros(D)
    log = TrainLog()
    out_dir = cfg.out_dir
    os.makedirs(out_dir, exist_ok=True)

    log_fn(f"[init] feature_dim={D}, n={map_data.n}, "
           f"controller_cand_target={cfg.n_controller_candidates}, "
           f"nature_cand_target={cfg.n_nature_candidates}")

    for n in range(cfg.outer_iters):
        log_fn(f"\n[outer {n+1}/{cfg.outer_iters}] running Algorithm 1 ...")
        theta = algorithm1_policy_eval(cfg, map_data, theta_old=theta, rng=rng)
        log_fn(f"[outer {n+1}] theta norm = {np.linalg.norm(theta):.4f}")

        # Evaluate via rollouts
        prot, dev, free = [], [], []
        for _ in range(cfg.rollout_episodes):
            res = rollout_episode(cfg, map_data, theta, rng, nature_uses_omega=True)
            prot.append(res["final_protected_value"])
            dev.append(res["final_developed_value"])
            free.append(res["final_free_value"])
        log.outer_iter.append(n + 1)
        log.avg_protected_value.append(float(np.mean(prot)))
        log.avg_developed_value.append(float(np.mean(dev)))
        log.avg_free_value.append(float(np.mean(free)))
        log.theta_norm.append(float(np.linalg.norm(theta)))
        log_fn(f"[outer {n+1}] avg_protected={np.mean(prot):.3f}  "
               f"avg_developed={np.mean(dev):.3f}  avg_free={np.mean(free):.3f} "
               f"(total value = {map_data.total_value:.2f})")

        # Save checkpoint
        np.save(os.path.join(out_dir, f"theta_iter{n+1}.npy"), theta)
        with open(os.path.join(out_dir, "train_log.json"), "w") as f:
            json.dump(log.__dict__, f, indent=2)

    np.save(os.path.join(out_dir, "theta_final.npy"), theta)
    with open(os.path.join(out_dir, "train_log.json"), "w") as f:
        json.dump(log.__dict__, f, indent=2)
    return {"theta": theta, "log": log.__dict__}
