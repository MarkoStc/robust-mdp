"""Pretrained-then-frozen grid encoder for phi(s, a, k).

CLAUDE.md path "Frozen neural encoder option": the encoder takes the full
(s, a, k) triple as a (H, W, C) channel stack plus global scalars and produces
a fixed-dim embedding ``phi``. The robust-MDP linear head ``theta`` is the only
weight updated inside Algorithm 1 / Algorithm 2 — encoder weights are loaded
from a checkpoint and frozen.

Channels (per parcel, on the grid):
    0  x              protected
    1  d              developed
    2  a              controller action being scored
    3  k              nature action being scored (already AND'd with (1-x-a))
    4  value          v_i (normalized to [0,1])
    5  cost           c_i / max_cost
    6  threat         TI_i / 10
    7  p              current development probability
    8  q              frontier pressure
    9  has_prot_neigh whether i has a protected neighbor
    10..10+C-1        cluster one-hot
    -2 row_norm       i / H
    -1 col_norm       j / W

Global scalars (concatenated to the CNN flat output):
    t/T, (T-t)/T, budget/total_cost, lambda_uncertainty, gamma
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import os
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

from .config import Config, MapData
from .state import State
from .dynamics import compute_development_probabilities, frontier_pressure


N_GLOBAL_SCALARS = 5


def n_channels(map_data: MapData) -> int:
    # 10 base channels + cluster one-hot + row/col coords
    return 10 + map_data.n_clusters + 2


def _state_grids(state: State, map_data: MapData, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Returns (state_channels, global_scalars).

    state_channels: (Cs, H, W) numpy float32 — channels that depend only on s
                    (everything except a, k). Channels 2 and 3 are filled with
                    zeros for now; the per-(a,k) builder fills them in.
    global_scalars: (G,) numpy float32
    """
    h, w = map_data.h, map_data.w
    n = map_data.n
    Cs = n_channels(map_data)

    x = state.x.astype(np.float32)
    d = state.d.astype(np.float32)
    p = compute_development_probabilities(d, map_data, cfg.eps).astype(np.float32)
    q = frontier_pressure(d, map_data).astype(np.float32)
    has_prot_neigh = ((map_data.adj @ x) > 0).astype(np.float32)
    v_norm = (map_data.value / max(1e-9, float(map_data.value.max()))).astype(np.float32)
    c_norm = (map_data.cost / max(1e-9, float(map_data.cost.max()))).astype(np.float32)
    tr_norm = (map_data.threat / 10.0).astype(np.float32)

    flat = np.zeros((Cs, n), dtype=np.float32)
    flat[0] = x
    flat[1] = d
    # 2,3 are a,k — filled later
    flat[4] = v_norm
    flat[5] = c_norm
    flat[6] = tr_norm
    flat[7] = p
    flat[8] = q
    flat[9] = has_prot_neigh
    # cluster one-hot
    for c in range(map_data.n_clusters):
        flat[10 + c] = (map_data.cluster == c).astype(np.float32)
    # coords
    flat[-2] = map_data.coords[:, 0].astype(np.float32) / max(1.0, h - 1)
    flat[-1] = map_data.coords[:, 1].astype(np.float32) / max(1.0, w - 1)

    state_channels = flat.reshape(Cs, h, w)
    globals_vec = np.array([
        state.t / cfg.horizon_T,
        (cfg.horizon_T - state.t) / cfg.horizon_T,
        state.budget / max(1e-9, map_data.total_cost),
        float(cfg.lambda_uncertainty),
        float(cfg.gamma),
    ], dtype=np.float32)
    return state_channels, globals_vec


def _ak_effective(state: State, a: np.ndarray, k: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mask a/k to feasibility-consistent versions (a only on free cells, k on
    not-already-protected after controller). Returns float32 (n,) arrays."""
    free = (state.x == 0) & (state.d == 0)
    a_f = (a.astype(np.int8) & free.astype(np.int8)).astype(np.float32)
    x_plus = np.clip(state.x.astype(np.int8) + a_f.astype(np.int8), 0, 1)
    k_f = (k.astype(np.int8) & (1 - x_plus)).astype(np.float32)
    return a_f, k_f


def build_input_batch(state: State, A: np.ndarray, K: np.ndarray,
                      map_data: MapData, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    """For all (j, l) pairs build the (J*L, Cs, H, W) channel stack and the
    matching (J*L, G) globals. Returns numpy arrays."""
    h, w = map_data.h, map_data.w
    Cs = n_channels(map_data)
    state_ch, globals_vec = _state_grids(state, map_data, cfg)  # (Cs,H,W), (G,)
    J = A.shape[0]
    L = K.shape[0]
    # Effective k_eff per (j, l): k & (1 - x_plus) where x_plus = x | A[j]
    A_i8 = A.astype(np.int8)
    K_i8 = K.astype(np.int8)
    x = state.x.astype(np.int8)
    free = ((x == 0) & (state.d.astype(np.int8) == 0)).astype(np.int8)
    A_eff = (A_i8 & free[None, :]).astype(np.float32)            # (J, n)
    # k_eff[j, l] = K[l] & (1 - x - A_eff[j])
    one_minus_x = (1 - x).astype(np.int8)
    # (J, L, n)
    K_eff = (K_i8[None, :, :] & (one_minus_x[None, None, :] - A_i8[:, None, :].clip(0, 1)).clip(0, 1)).astype(np.float32)
    # Broadcast state channels to (J, L, Cs, n) then fill a, k
    base = np.broadcast_to(state_ch.reshape(Cs, -1), (J, L, Cs, h * w)).copy()
    base[:, :, 2, :] = A_eff[:, None, :]
    base[:, :, 3, :] = K_eff
    out = base.reshape(J * L, Cs, h, w)
    g = np.broadcast_to(globals_vec, (J * L, N_GLOBAL_SCALARS)).copy()
    return out, g


def build_input_single(state: State, a: np.ndarray, k: np.ndarray,
                       map_data: MapData, cfg: Config) -> Tuple[np.ndarray, np.ndarray]:
    h, w = map_data.h, map_data.w
    Cs = n_channels(map_data)
    state_ch, globals_vec = _state_grids(state, map_data, cfg)
    a_f, k_f = _ak_effective(state, a, k)
    grid = state_ch.copy()
    grid[2] = a_f.reshape(h, w)
    grid[3] = k_f.reshape(h, w)
    return grid, globals_vec


# ---------------------------------------------------------------------------
# Torch encoder
# ---------------------------------------------------------------------------

if _HAS_TORCH:

    class GridEncoder(nn.Module):
        """Small CNN over (Cs, H, W) + globals -> ``embed_dim`` feature vector."""

        def __init__(self, in_channels: int, n_globals: int = N_GLOBAL_SCALARS,
                     embed_dim: int = 64):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
            # After three conv + global avg pool we have a 32-vector
            self.fc1 = nn.Linear(32 + n_globals, embed_dim)
            self.fc2 = nn.Linear(embed_dim, embed_dim)
            self.embed_dim = embed_dim
            self.in_channels = in_channels
            self.n_globals = n_globals

        def forward(self, grid: "torch.Tensor", globals_vec: "torch.Tensor") -> "torch.Tensor":
            h = F.relu(self.conv1(grid))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            h = h.mean(dim=(-1, -2))   # global avg pool -> (B, 32)
            h = torch.cat([h, globals_vec], dim=-1)
            h = F.relu(self.fc1(h))
            h = self.fc2(h)
            # L2 normalize to keep feature norms controlled (CLAUDE.md asks
            # for normalized/clipped features when using an encoder).
            h = h / (h.norm(dim=-1, keepdim=True) + 1e-6)
            return h

    def make_encoder(map_data: MapData, cfg: Config) -> "GridEncoder":
        return GridEncoder(
            in_channels=n_channels(map_data),
            n_globals=N_GLOBAL_SCALARS,
            embed_dim=cfg.encoder_embed_dim,
        )

    # ----------------------------------------------------------------- cache
    _ENCODER_CACHE = {}

    def get_encoder(cfg: Config, map_data: MapData) -> "GridEncoder":
        """Load (cached) encoder from cfg.encoder_path; build a fresh one if
        no path is set (used for pretraining)."""
        path = cfg.encoder_path
        key = (path, cfg.encoder_embed_dim, n_channels(map_data))
        if key in _ENCODER_CACHE:
            return _ENCODER_CACHE[key]
        net = make_encoder(map_data, cfg)
        if path and os.path.exists(path):
            sd = torch.load(path, map_location="cpu", weights_only=True)
            net.load_state_dict(sd)
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        net.to(cfg.encoder_device)
        _ENCODER_CACHE[key] = net
        return net

    # ----------------------------------------------------------------- phi/Q
    @torch.no_grad()
    def encoder_phi(state: State, a: np.ndarray, k: np.ndarray,
                    map_data: MapData, cfg: Config) -> np.ndarray:
        net = get_encoder(cfg, map_data)
        grid, glob = build_input_single(state, a, k, map_data, cfg)
        g = torch.from_numpy(grid).unsqueeze(0).to(cfg.encoder_device)
        v = torch.from_numpy(glob).unsqueeze(0).to(cfg.encoder_device)
        out = net(g, v).squeeze(0).cpu().numpy()
        return out.astype(np.float64)

    @torch.no_grad()
    def encoder_q_matrix_batch(state: State, A: np.ndarray, K: np.ndarray,
                               theta: np.ndarray, map_data: MapData, cfg: Config,
                               chunk: int = 4096) -> np.ndarray:
        net = get_encoder(cfg, map_data)
        J, L = A.shape[0], K.shape[0]
        grids, globs = build_input_batch(state, A, K, map_data, cfg)
        N = grids.shape[0]
        theta_t = torch.from_numpy(theta.astype(np.float32)).to(cfg.encoder_device)
        out = np.zeros(N, dtype=np.float32)
        for i in range(0, N, chunk):
            j = min(N, i + chunk)
            g = torch.from_numpy(grids[i:j]).to(cfg.encoder_device)
            v = torch.from_numpy(globs[i:j]).to(cfg.encoder_device)
            phi = net(g, v)
            q = phi @ theta_t
            out[i:j] = q.cpu().numpy()
        return out.reshape(J, L).astype(np.float64)

    # ----------------------------------------------------------------- pretrain
    def _random_policy_step(state: State, map_data: MapData, cfg: Config,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, float]:
        """Pick a random feasible (a, k) and return them with the step reward.

        We use a cheap heuristic: a -> sample a subset of available parcels
        greedily until budget exhausted; k -> sample developments while the
        likelihood slack remains feasible. The exact identity of the policy
        does not matter; we just need a realistic data distribution.
        """
        from .candidates import generate_controller_candidates, generate_nature_candidates
        A = generate_controller_candidates(state, map_data, cfg, rng)
        K = generate_nature_candidates(state, np.zeros(map_data.n, dtype=np.int8),
                                       map_data, cfg, rng)
        ia = int(rng.integers(0, A.shape[0]))
        a = A[ia]
        # Re-generate nature using the chosen a so feasibility uses the post-a state.
        K = generate_nature_candidates(state, a, map_data, cfg, rng)
        ik = int(rng.integers(0, K.shape[0]))
        k = K[ik]
        return a, k

    def collect_pretrain_dataset(cfg: Config, map_data: MapData,
                                 rng: np.random.Generator,
                                 n_episodes: int,
                                 log_fn=print):
        """Generate (grid, globals, return) samples by rolling out a random
        policy. ``return`` is the discounted cumulative step reward from the
        current step to terminal.
        """
        from .dynamics import step as dyn_step
        from .state import initial_state
        from .rollout import step_reward

        grids_list = []
        globs_list = []
        rets_list = []

        h, w = map_data.h, map_data.w
        Cs = n_channels(map_data)

        for ep in range(n_episodes):
            state = initial_state(cfg, map_data)
            states = []
            actions = []
            rewards = []
            for t in range(cfg.horizon_T):
                a, k = _random_policy_step(state, map_data, cfg, rng)
                next_state = dyn_step(state, a, k, cfg)
                r = step_reward(state, a, k, next_state, map_data, cfg)
                states.append(state)
                actions.append((a, k))
                rewards.append(r)
                state = next_state
            # Compute discounted returns from each step
            G = 0.0
            for t in reversed(range(len(rewards))):
                G = rewards[t] + cfg.gamma * G
                s = states[t]
                a, k = actions[t]
                grid, glob = build_input_single(s, a, k, map_data, cfg)
                grids_list.append(grid)
                globs_list.append(glob)
                rets_list.append(G)
            if log_fn is not None and (ep + 1) % max(1, n_episodes // 10) == 0:
                log_fn(f"  [pretrain-collect] episode {ep+1}/{n_episodes}")
        grids = np.stack(grids_list, axis=0)
        globs = np.stack(globs_list, axis=0)
        rets = np.array(rets_list, dtype=np.float32)
        return grids, globs, rets

    def pretrain_encoder(cfg: Config, map_data: MapData,
                         rng: np.random.Generator,
                         out_path: str,
                         log_fn=print) -> "GridEncoder":
        """Pretrain encoder + linear head on Monte-Carlo returns, then save
        encoder weights (head is discarded). Returns the trained encoder
        in eval mode with parameters frozen."""
        log_fn(f"[pretrain] collecting {cfg.encoder_pretrain_episodes} episodes ...")
        grids, globs, rets = collect_pretrain_dataset(
            cfg, map_data, rng,
            n_episodes=cfg.encoder_pretrain_episodes,
            log_fn=log_fn,
        )
        log_fn(f"[pretrain] dataset shape: grids={grids.shape}, returns std={rets.std():.3f}")

        device = cfg.encoder_device
        net = make_encoder(map_data, cfg).to(device)
        head = nn.Linear(cfg.encoder_embed_dim, 1).to(device)
        opt = torch.optim.Adam(
            list(net.parameters()) + list(head.parameters()),
            lr=cfg.encoder_pretrain_lr,
        )

        N = grids.shape[0]
        grids_t = torch.from_numpy(grids)
        globs_t = torch.from_numpy(globs)
        rets_t = torch.from_numpy(rets)
        bs = max(1, cfg.encoder_pretrain_batch)
        last_loss = float("inf")
        for epoch in range(cfg.encoder_pretrain_epochs):
            perm = torch.randperm(N)
            total = 0.0
            count = 0
            net.train()
            for i in range(0, N, bs):
                idx = perm[i:i + bs]
                g = grids_t[idx].to(device)
                v = globs_t[idx].to(device)
                y = rets_t[idx].to(device)
                phi = net(g, v)
                pred = head(phi).squeeze(-1)
                loss = F.mse_loss(pred, y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += float(loss.item()) * idx.numel()
                count += idx.numel()
            last_loss = total / max(1, count)
            if (epoch + 1) % max(1, cfg.encoder_pretrain_epochs // 10) == 0:
                log_fn(f"[pretrain] epoch {epoch+1}/{cfg.encoder_pretrain_epochs}  loss={last_loss:.5f}")

        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torch.save(net.state_dict(), out_path)
        log_fn(f"[pretrain] saved encoder to {out_path}  final_loss={last_loss:.5f}")
        # Reset cache so future ``get_encoder`` reads the new file.
        _ENCODER_CACHE.clear()
        return net


else:  # pragma: no cover — module still importable if torch is missing

    def get_encoder(*args, **kwargs):
        raise RuntimeError("torch is required for the encoder path")

    def encoder_phi(*args, **kwargs):
        raise RuntimeError("torch is required for the encoder path")

    def encoder_q_matrix_batch(*args, **kwargs):
        raise RuntimeError("torch is required for the encoder path")

    def pretrain_encoder(*args, **kwargs):
        raise RuntimeError("torch is required for the encoder path")
