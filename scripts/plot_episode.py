"""Run one episode and save matplotlib plots showing the per-year picks.

Generates:
  - episode_grid.png : 2x5 multi-panel, each year showing state+picks
  - episode_value_bg.png : 2x5 multi-panel with value heatmap underneath
  - per_year/year_{t}.png : individual per-year frames
  - map_overview.png : value/threat/cost heatmaps with cluster borders
"""
from __future__ import annotations
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.config import Config
from src.state import build_default_map, initial_state
from src.dynamics import step
from src.rollout import policy_at_state
from src.viz import plot_heatmap, _cluster_borders


def coerce_cfg_dict(d: dict) -> dict:
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


STATUS_COLORS = ["#eeeeee", "#1f77b4", "#7f7f7f", "#2ca02c", "#d62728"]
STATUS_LABELS = ["free", "protected (old)", "developed (old)",
                 "newly protected", "newly developed"]


def status_grid(state_x, state_d, a, k, h, w):
    grid = np.zeros((h, w), dtype=np.int32)
    grid[state_x.reshape(h, w) == 1] = 1
    grid[state_d.reshape(h, w) == 1] = 2
    if a is not None:
        new_prot = ((a == 1) & (state_x == 0)).reshape(h, w)
        grid[new_prot] = 3
    if k is not None:
        new_dev = ((k == 1) & (state_d == 0)).reshape(h, w)
        grid[new_dev] = 4
    return grid


def draw_year(ax, state_x, state_d, a, k, md, cfg, title,
              value_background=False):
    h, w = md.h, md.w
    if value_background:
        vgrid = md.value.reshape(h, w)
        ax.imshow(vgrid, cmap="Greys", origin="upper", alpha=0.55,
                  vmin=0, vmax=float(vgrid.max()))
    grid = status_grid(state_x, state_d, a, k, h, w)
    cmap = ListedColormap(STATUS_COLORS)
    alpha = 0.55 if value_background else 1.0
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=4, origin="upper", alpha=alpha)
    # hotspot markers
    for (cr, cc, _, _) in cfg.value_hotspots:
        ax.plot(cc, cr, marker="*", color="gold", markersize=14,
                markeredgecolor="black", markeredgewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    _cluster_borders(ax, md, lw=2.0)


def make_legend(fig):
    handles = [Patch(facecolor=STATUS_COLORS[i], edgecolor="black", label=lbl)
               for i, lbl in enumerate(STATUS_LABELS)]
    handles.append(plt.Line2D([0], [0], marker="*", color="w",
                              markerfacecolor="gold", markeredgecolor="black",
                              markersize=12, label="hotspot center"))
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=str,
                    default="outputs/slurm/biodiv-2312346")
    ap.add_argument("--theta", type=str, default="theta_final.npy")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nature-mode", type=str, default="omega",
                    choices=["omega", "worst"])
    ap.add_argument("--out-sub", type=str, default="episode_seed42")
    args = ap.parse_args()

    cfg_path = os.path.join(args.run_dir, "config.json")
    theta_path = os.path.join(args.run_dir, args.theta)
    with open(cfg_path) as f:
        raw = json.load(f)
    cfg = Config(**coerce_cfg_dict(raw))
    if cfg.use_encoder:
        cfg.encoder_path = os.path.join(args.run_dir, "encoder.pt")
        try:
            import torch
            cfg.encoder_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            cfg.encoder_device = "cpu"

    theta = np.load(theta_path)
    rng = np.random.default_rng(args.seed)
    md = build_default_map(cfg, rng)
    state = initial_state(cfg, md)

    out_dir = os.path.join(args.run_dir, args.out_sub)
    per_year_dir = os.path.join(out_dir, "per_year")
    os.makedirs(per_year_dir, exist_ok=True)

    # --- map overview (value / threat / cost)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    plot_heatmap(md, md.value, "Value", ax=axes[0], cmap="viridis")
    plot_heatmap(md, md.threat, "Threat index", ax=axes[1], cmap="magma")
    plot_heatmap(md, md.cost, "Cost", ax=axes[2], cmap="cividis")
    for (cr, cc, _, _) in cfg.value_hotspots:
        for ax in axes:
            ax.plot(cc, cr, marker="*", color="gold", markersize=12,
                    markeredgecolor="black", markeredgewidth=0.8)
    fig.suptitle("Map overview (gold stars = value hotspot centers)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "map_overview.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir}/map_overview.png")

    # --- per-year run
    records = []
    for t_idx in range(cfg.horizon_T):
        pi, omega, A_cand, K_cand, v_est, Q = policy_at_state(state, theta, md, cfg, rng)
        j = int(np.argmax(pi))
        a = A_cand[j]
        if args.nature_mode == "omega":
            l = int(np.argmax(omega))
        else:
            l = int(np.argmin(Q[j]))
        k = K_cand[l]
        records.append({
            "t": state.t,
            "x_before": state.x.copy(),
            "d_before": state.d.copy(),
            "a": a.copy(),
            "k": k.copy(),
            "v_est": float(v_est),
        })
        # individual year plot
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
        draw_year(ax[0], state.x, state.d, a, k, md, cfg,
                  f"Year {state.t}: status overlay", value_background=False)
        draw_year(ax[1], state.x, state.d, a, k, md, cfg,
                  f"Year {state.t}: value background", value_background=True)
        make_legend(fig)
        fig.tight_layout(rect=[0, 0.04, 1, 1])
        fig.savefig(os.path.join(per_year_dir, f"year_{state.t:02d}.png"),
                    dpi=120, bbox_inches="tight")
        plt.close(fig)
        state = step(state, a, k, cfg)

    # --- multi-panel summary
    for bg, suffix in [(False, "episode_grid.png"),
                       (True, "episode_value_bg.png")]:
        fig, axes = plt.subplots(2, 5, figsize=(20, 9))
        for idx, rec in enumerate(records):
            ax = axes[idx // 5, idx % 5]
            draw_year(ax, rec["x_before"], rec["d_before"], rec["a"], rec["k"],
                      md, cfg, f"t={rec['t']}  V={rec['v_est']:.3f}",
                      value_background=bg)
        make_legend(fig)
        fig.suptitle(
            f"Seed {args.seed}, nature={args.nature_mode}, "
            f"||theta||={np.linalg.norm(theta):.3f}, lambda={cfg.lambda_uncertainty}",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.97])
        fig.savefig(os.path.join(out_dir, suffix), dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"saved {out_dir}/{suffix}")

    # --- final state plot
    fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
    draw_year(ax[0], state.x, state.d, None, None, md, cfg,
              "Final: status overlay", value_background=False)
    draw_year(ax[1], state.x, state.d, None, None, md, cfg,
              "Final: value background", value_background=True)
    prot_v = float((md.value * state.x).sum())
    dev_v = float((md.value * state.d).sum())
    free_v = float((md.value * ((state.x == 0) & (state.d == 0))).sum())
    fig.suptitle(
        f"Final state — protected={prot_v:.2f}, developed={dev_v:.2f}, "
        f"free={free_v:.2f} (total={md.total_value:.2f})",
        fontsize=12,
    )
    make_legend(fig)
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(os.path.join(out_dir, "final_state.png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_dir}/final_state.png")
    print("done.")


if __name__ == "__main__":
    main()
