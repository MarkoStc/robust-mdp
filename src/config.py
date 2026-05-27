"""Config dataclasses for the biodiversity robust MDP."""
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Config:
    # Grid / map
    grid_h: int = 10
    grid_w: int = 10
    horizon_T: int = 10
    gamma: float = 0.95

    # Budget (per year) — tightened so the controller can no longer trivially
    # cover the entire grid; this forces it to learn where to spend.
    budget_per_year: float = 2.0

    # Robustness — lambda lowered so nature's likelihood budget is meaningfully
    # large and we see several developments per episode instead of ~1.
    # Note: the multiplicative neighbor formula keeps p small until the first
    # cluster of developments seeds neighbor pressure, so lambda has to be quite
    # small to bootstrap the snowball.
    lambda_uncertainty: float = 0.20
    eps: float = 1e-3
    fixed_p_feasibility: bool = False

    # Candidate generation
    n_controller_candidates: int = 500
    n_nature_candidates: int = 500
    candidate_seed: int = 0

    # Outer / inner loop sizes
    outer_iters: int = 10
    inner_M: int = 8                  # nature dual-averaging iters
    samples_per_iter: int = 2000      # TD samples for Algorithm 2
    td_step_eta: float = 0.02
    td_skip_tau: int = 2

    # Sampled states per outer iter (we only learn theta over visited states)
    n_eval_states: int = 200
    rollout_episodes: int = 20

    # Reward shape: terminal protected-value reward (good-value convention)
    use_step_reward: bool = True

    # Exploration: mix solved policies with uniform
    exploration_eps: float = 0.1

    # ------------------------------------------------------------------
    # Value distribution: Gaussian "hotspots". A small number of high-value
    # bumps create the strategic structure the controller should learn to
    # cluster around. Each bump is a 2-D Gaussian centered at (row, col).
    # If `value_hotspots` is empty we fall back to uniform values.
    # ------------------------------------------------------------------
    value_hotspots: List[Tuple[float, float, float, float]] = field(
        # (center_row, center_col, peak, std)
        default_factory=lambda: [
            (2.0, 2.0, 1.0, 1.2),  # cluster 0 (top-left)
            (7.0, 7.0, 1.0, 1.5),  # cluster 2 (bottom)
        ]
    )
    value_floor: float = 0.05
    value_noise: float = 0.05      # uniform noise added to bumps

    # ------------------------------------------------------------------
    # Encoder option (CLAUDE.md "frozen neural encoder" path). When
    # ``use_encoder`` is True, phi(s,a,k) is produced by a pretrained-then-
    # frozen CNN rather than the engineered features.
    # ------------------------------------------------------------------
    use_encoder: bool = False
    encoder_path: str = ""             # path to a saved encoder checkpoint
    encoder_embed_dim: int = 64
    encoder_pretrain_episodes: int = 400
    encoder_pretrain_epochs: int = 30
    encoder_pretrain_lr: float = 1e-3
    encoder_pretrain_batch: int = 256
    encoder_device: str = "cpu"        # "cuda" if available — set by runner

    # Misc
    seed: int = 0
    out_dir: str = "outputs"


@dataclass
class MapData:
    n: int
    h: int
    w: int
    coords: 'np.ndarray'            # (n, 2)
    cluster: 'np.ndarray'           # (n,) int
    n_clusters: int
    value: 'np.ndarray'             # (n,)
    cost: 'np.ndarray'              # (n,)
    threat: 'np.ndarray'            # (n,) in [0,10]
    neighbors: 'list[np.ndarray]'   # neighbor indices per parcel (same-cluster)
    adj: 'np.ndarray'               # (n, n) 0/1 same-cluster neighbor matrix
    deg: 'np.ndarray'               # (n,) number of same-cluster neighbors
    cluster_onehot: 'np.ndarray'    # (n, C) 0/1
    total_value: float
    total_cost: float
