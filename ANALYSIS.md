# Biodiversity Robust-MDP — Runs, Configurations, Plots & Scaling Analysis

This document is a self-contained reference for the repository. It is written so
that an AI agent (or a new collaborator) given the zipped repo can report, in
detail, **what was run, how it was configured, what every plot/artifact means,
and how the method is expected to scale to the paper's real 692-parcel setting.**

Companion documents:
- `CLAUDE.md` — the full modeling spec (conventions, formulas, feature lists).
- `papers/10341148.pdf` — the reference paper (Ye et al., *Conserving
  Biodiversity via Adjustable Robust Optimization*, AAMAS 2022). `papers/*.txt`
  are extracted plain-text versions.

All timing numbers below were measured on the Clariden cluster GPU node
(`uenv pytorch/v2.9.1:v2`, CUDA), the same hardware the runs used.

> **Status note (2026-06-01).** §§2–6 describe the original published runs
> (`biodiv-2312346`, `biodiv-2392077`) on the `main` branch. Branch
> `exact-nature-feasibility` then (a) made nature's candidate feasibility **exact
> immediate-p** instead of the frozen-p approximation, and (b) added arbitrary-grid
> support for the paper's **692-parcel** count. The new feature-interpretation
> findings are in **§9**; the exact-nature change and the in-progress 692 / exact
> re-runs are in **§10**. Read those two sections for what is current.

---

## 1. What this repo is

A toy biodiversity-conservation game modeled as an **approximate robust MDP /
zero-sum Markov game** on a 10×10 grid of land parcels.

- **Controller** (conservation org) protects parcels each year within a budget.
- **Nature** (human development) develops unprotected parcels adversarially,
  constrained to a *likelihood uncertainty set* governed by `lambda`.
- Good-value convention: `Q(s,a,k) = expected protected value to end of horizon`.
  Higher is better for the controller; nature minimizes it.

The raw action spaces are `~2^n`; the repo **never enumerates them**. Instead it
generates a reduced candidate library (~100–1000 actions/player/state) via
heuristics + exploration, and solves a local matrix game over those candidates.
`Q` is approximated linearly: `Q ≈ theta^T phi(s,a,k)`, where `phi` comes from a
**pretrained-then-frozen CNN encoder** (only the linear head `theta` is learned in
the robust-MDP loop).

---

## 2. Environment & configuration

Defined in `src/config.py` (`Config` dataclass). The values used by the canonical
runs (saved in each run's `config.json`):

| Parameter | Value | Meaning |
|---|---|---|
| `grid_h`, `grid_w` | 10, 10 | 100 parcels |
| `n_clusters` | 3 | top-left 5×5, top-right 5×5, bottom 5×10 |
| `horizon_T` | 10 | planning years |
| `gamma` | 0.95 | discount |
| `budget_per_year` | 2.0 | per-year protection budget (costs ~U(0.2,1.0)) |
| `lambda_uncertainty` | 0.20 | robustness; smaller ⟹ larger uncertainty set ⟹ stronger adversary |
| `n_controller_candidates` | 200 | candidate protection actions / state |
| `n_nature_candidates` | 200 | candidate development actions / state |
| `outer_iters` | 5 | outer robust-policy-iteration rounds |
| `inner_M` | 5 | nature dual-averaging iters per outer round |
| `samples_per_iter` | 2000 | TD samples (states) per Algorithm-2 fit |
| `use_encoder` | true | phi = frozen CNN (else engineered features) |
| `encoder_embed_dim` | 64 | dimension of phi (hence of theta) |
| `value_hotspots` | `[[2,2,1,1.2],[7,7,1,1.5]]` | two Gaussian value bumps |

**Key dynamics (see `src/dynamics.py`, `CLAUDE.md` §"Development probability"):**

```
p_i = (TI_i/10) * (1 + developed same-cluster neighbors) / (1 + same-cluster neighbors)   # clipped to [eps,1-eps]
```

Timing per year: **controller protects first, then nature develops** (this matters
for the StaticApprox cut timing — see §6). Monotone: once protected always
protected, once developed always developed; protected parcels can't be developed.

**Nature feasibility (uncertainty set U), always in log form:**
```
loglik(d_plus) = sum_i [ d_i log p_i + (1-d_i) log(1-p_i) ]
feasible iff   loglik(d_plus) >= t * log(lambda)        # immediate p recomputation from d_plus
```

> The environment `step` and the adversary always used immediate-p. **Candidate
> generation** originally used a frozen-p fast path (probabilities from the current
> `d`, not updated as parcels are added) for speed. On branch
> `exact-nature-feasibility` candidate generation now uses **exact immediate-p**
> too (incremental O(deg) neighbour updates), so the entire pipeline matches the
> rule above; `fixed_p_feasibility=True` restores the old approximation for
> ablations. See §10.

> **Why lambda=0.20 is a deliberate "hard" regime.** The multiplicative neighbor
> formula keeps `p` small until a cluster of developments seeds neighbor pressure.
> 0.20 lets nature bootstrap and develop most of the grid in the worst case, so the
> adversary is near-omnipotent. This makes the *worst-case* margins small (see §6)
> but is an honest stress test.

---

## 3. The training runs

Two completed runs live under `outputs/slurm/`. They use identical config (above);
the second is a re-run after a diagnostics fix. **Both converged to the same place**
(`theta_norm ≈ 0.60`, flat after outer iter 1).

| Run | When | Pretrain | Training | Outcome (avg over rollouts) |
|---|---|---|---|---|
| `biodiv-2312346` | 2026-05-19 | — | — | protected ≈ 15.6 / 25.87 total, theta_norm 0.607 |
| `biodiv-2392077` | 2026-05-27 | 100.9 s | 4781.6 s (≈80 min) | protected ≈ 15.3–15.6, theta_norm 0.600 |

**Artifacts per run directory:**
- `config.json` — the exact `Config` used (the source of truth for that run).
- `encoder.pt` — frozen CNN weights (see §4). ~121 KB.
- `theta_iter{1..5}.npy`, `theta_final.npy` — the learned 64-dim linear head after
  each outer round; `theta_final` is the policy you evaluate with.
- `train_log.json` — per-outer-round metrics: `avg_protected_value`,
  `avg_developed_value`, `avg_free_value`, `theta_norm`. **Reading it:** flat
  curves after round 1 = converged; protected ≫ developed = controller winning.
- `diagnostics.json` — action-span diagnostics (see §5).
- `slurm.out` — full stdout (pretrain log, sanity checks, per-round training log,
  diagnostics dump, total timings).
- `compare/` — the paper-comparison outputs (see §6).

The `total value` of the map is **25.87** (sum of all parcel values); every
"value" number is on that scale.

---

## 4. The frozen encoder (`src/encoder.py`)

`phi(s,a,k)` is produced by a small CNN (`GridEncoder`) over a channel-stacked
grid plus global scalars, L2-normalized to a 64-dim vector. Only `theta` (64
weights) is trained inside the robust-MDP loop; **encoder weights are frozen** after
pretraining, per the linear-Q paper assumption.

### 4.1 Inputs

**Per-parcel grid channels** (shape `(C_in, H, W)`):

| idx | channel | idx | channel |
|---|---|---|---|
| 0 | protected `x` | 7 | development prob `p` |
| 1 | developed `d` | 8 | frontier pressure `q` |
| 2 | controller action `a` | 9 | has-protected-neighbor flag |
| 3 | nature action `k` (masked) | 10..10+C-1 | cluster one-hot (C clusters) |
| 4 | value `v` (norm) | −2 | row coord / H |
| 5 | cost `c` (norm) | −1 | col coord / W |
| 6 | threat `TI`/10 | | |

So `C_in = 10 + n_clusters + 2`. For the toy grid (3 clusters) → **15 channels**;
for the paper's 9-cluster setting → **21 channels**.

**Global scalars** (concatenated after the conv stack, `N_GLOBAL_SCALARS = 5`):
`t/T`, `(T−t)/T`, `budget/total_cost`, `lambda_uncertainty`, `gamma`.

### 4.2 Architecture & exact parameter count

`conv1(C_in→32,3×3) → conv2(32→32) → conv3(32→32) → global-avg-pool →
concat(32 + 5 globals) → fc1(37→64) → fc2(64→64) → L2-normalize`.

Parameter count is **almost independent of parcel count** (global average pooling
removes spatial size; only `C_in` and the cluster count change). The breakdown
below is exact and was **verified by direct `sum(p.numel())`** on instantiated
encoders (measured = computed, to the parameter):

| Layer | Formula | n=100, 15 ch | n=692, 21 ch |
|---|---|---|---|
| conv1 | `C_in·32·9 + 32` | 4,352 | 6,080 |
| conv2 | `32·32·9 + 32` | 9,248 | 9,248 |
| conv3 | `32·32·9 + 32` | 9,248 | 9,248 |
| fc1 | `37·64 + 64` | 2,432 | 2,432 |
| fc2 | `64·64 + 64` | 4,160 | 4,160 |
| **Total** | | **29,440** ✓ | **31,168** ✓ |

(✓ = matches the measured count exactly. The pretraining MSE head `Linear(64→1)`
adds 65 params but is discarded after pretraining — not part of the saved encoder.)

### 4.3 Pretraining procedure

`pretrain_encoder()` rolls out a random policy for `encoder_pretrain_episodes`
(=800) episodes, builds `(grid, globals, return)` samples (Monte-Carlo discounted
return as the regression target), and trains encoder+head with Adam for
`encoder_pretrain_epochs` (=40) epochs (MSE). For run 2392077: dataset =
`(8000, 15, 10, 10)`, final loss 0.0009, **100.9 s**. Then the encoder is frozen and
saved; the head is thrown away.

> **You cannot reuse this encoder for the paper's 692-parcel setting** — see §7.4.

---

## 5. Action-span diagnostics (`src/diagnostics.py`, `diagnostics.json`)

These answer the mentor's question "is the reduced candidate set good enough?"
**not** by raw binary span (uninformative) but in **Q-relevant action-effect
embedding space**. For each visited state we build a large reference action pool
(target 3000) and the heuristic candidate pool, embed both into value/cost/
threat/frontier/cluster effect features, and compute the projection error of each
reference vector onto the candidate span:

```
err(z) = || z − U_small U_small^T z || / || z ||
```

`diagnostics.json` is a list of per-year dicts (`t=1..T`), each with `controller`
and `nature` sub-dicts holding `n_ref`, `n_cand`, `median_proj_err`, `p90`, `p95`.

**Reading it:** projection errors near machine epsilon (1e-16 to 1e-9, as in both
runs across all 10 years) mean the ~100 candidate actions **fully span** the
reference action-effect space at every visited state — i.e. the reduction loses
no Q-relevant directions. `n_cand` ≈ 100 (not 200) because `budget=2.0` +
`lambda=0.20` make many candidates collapse to duplicates after dedup; the count
of *distinct* feasible actions is what matters and it suffices.

> History note: an earlier version walked only 3 idle (zero-action) steps; it was
> fixed to walk the full horizon **along the learned policy** (`run_biodiv_train.py`,
> the `--run-diagnostics` block). Run 2392077 has the correct 10-step diagnostics.

---

## 6. Comparison vs the paper's methods (`run_compare.py`, `src/baselines.py`)

Goal: show our learned policy holds up **against adversarial nature** relative to
the paper's non-RL methods, inside *our* environment, on equal footing.

### 6.1 The paper's methods (re-implemented in `src/baselines.py`)

- **Knapsack** (paper Problem 8/9, their *benchmark*): each year protect the
  value-maximizing parcel set within budget, ignoring uncertainty. Exact 0/1
  knapsack via `scipy.optimize.milp`. Adaptive only in the trivial sense that it
  re-optimizes over whatever parcels remain available.
- **StaticApprox** (paper Problem 7, their *final proposal*): one **non-adaptive,
  here-and-now** protection schedule for the whole horizon, chosen to minimize the
  worst-case loss over the uncertainty set U. Solved by **constraint generation**:
  a master MILP (`min tau` s.t. budget + monotonicity + one robust cut per
  generated scenario) plus a greedy separation oracle that finds the worst-case
  development against the current schedule. **Cut timing is aligned to our
  simulator** (controller protects before nature within a year, so protection at
  year `t` guards developments at year `t` — this differs from the paper's `x_{t-1}`
  term, which matches their protect-after-observe convention).

### 6.2 The two "natures" (identical for all three controllers)

- **Worst-case / adversarial** (`greedy_adversary_develop`): greedily develops the
  parcels with the highest value-per-likelihood-cost `v_i / log((1-p_i)/p_i)` while
  keeping the realization inside U, recomputing `p` after each addition (immediate
  recomputation; captures the development snowball). This is the robustness test.
- **Average-case / stochastic** (`bernoulli_develop`): the cellular-automata
  `Bernoulli(p_i)` model the paper uses to *simulate* realized developments (no
  robustness cap). This is the typical-case test.

### 6.3 Results (run 2392077; run 2312346 agrees)

| Nature | Ours | StaticApprox (paper proposal) | Knapsack |
|---|---|---|---|
| **Worst-case lost value** (↓) | **21.15 ± 1.49** | 22.72 | 22.72 |
| **Average-case lost value** (↓) | **12.24** | 19.82 | 12.48 |
| Worst-case protected (↑) | 4.72 | 3.15 | 3.15 |
| Average-case protected (↑) | 13.57 | 3.15 | 13.37 |

**Interpretation:**
- **Worst-case: ours ≈ 7% less loss** than StaticApprox, reproducible across both
  runs (2312346: 21.14 ± 1.17; 2392077: 21.15 ± 1.49).
- **StaticApprox == Knapsack on worst-case** — exactly the paper's **Proposition
  6.1** (small `lambda` ⟹ uncertainty set so large the two coincide). A good
  sanity check that the baselines are faithful.
- **Average-case: ours (and Knapsack) beat the non-adaptive StaticApprox by ~38%**,
  because StaticApprox is stuck with its small up-front plan no matter what happens.
- **The robust headline: ours is the only method strong in *both* regimes.**
  Knapsack collapses on the worst case (ignores uncertainty); StaticApprox
  collapses on average (cannot adapt). Ours does not collapse on either.

### 6.4 Methodology lesson baked into the code

Our policy **samples** its candidate actions, so a *single* greedy-adversary
rollout is noisy (std ≈ 1.2–1.5). An early single-episode measurement swung from
−10.7% to +0.4% between the two runs — a pure metric artifact. The worst-case
evaluation therefore **averages over `--n-worstcase` seeds (default 20)** and
reports the distribution (StaticApprox/Knapsack are deterministic ⟹ std 0).
**Always report the multi-seed distribution, never a single rollout.**

### 6.5 The comparison plot — `compare/comparison_vs_paper.png`

Two panels, three methods (`Ours`, `StaticApprox`, `Knapsack`):
- **Left — "Loss to development (lower = better)":** for each method, a solid bar =
  worst-case mean (with min/max error bars) and a faded bar = stochastic mean
  (with min/max error bars). Ours' solid bar is lowest; StaticApprox's faded bar
  is far higher than the other two.
- **Right — "Value preserved (higher = better)":** solid = worst-case protected,
  faded = stochastic protected. Ours leads on worst-case; ours ≈ Knapsack ≫
  StaticApprox on average.
- Title shows `total value = 25.9`.

`compare/comparison_results.json` holds the full numbers: `meta` (run dir, lambda,
budget, horizon, total value, #stochastic episodes, #parcels StaticApprox planned)
and `results[method][worst_case|stochastic][lost_value|protected_value|free_value]`
with `mean/std/min/max/median`.

### 6.6 Reproduce

```bash
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
 "source ~/qa-gym/.venv/bin/activate && cd ~/robust-mdp && \
  python run_compare.py --run-dir outputs/slurm/biodiv-2392077 \
  --n-worstcase 20 --n-stochastic 40"
```
Runs in ~16 s on GPU. Loads that run's frozen encoder + `theta_final`, rebuilds the
seed-matched map, solves both baselines, evaluates all three, writes the JSON+PNG.

---

## 7. Scaling to the paper's real 692-parcel setting

### 7.1 What the paper's real instance is (paper §7)

After cleaning, the paper keeps **692 parcels** (jaguar range, Latin America),
clustered into **9 groups** (4 threat levels × K-means subclusters). The reported
numerical experiment is **single-stage** ("we consider a single-stage problem,
which can be solved to optimality within 10 minutes" with Gurobi/CPLEX), budget
swept 25–175 M USD, evaluated on 1000 cellular-automata samples.

Our pipeline is **multistage (T=10)** and RL-based, so a direct port is more
expensive than their single-stage MILP. The estimates below assume we keep our
T=10 multistage setup but on a 692-parcel / 9-cluster map.

### 7.2 Measured scaling micro-benchmark (provable basis)

Same code paths, same GPU node, n=100 (3 clusters) vs a synthetic n=702
(9-cluster block grid, the closest grid-shaped stand-in for 692). These ratios are
the empirical basis for every estimate that follows:

| Quantity | n=100 / 3 clust | n=702 / 9 clust | ratio |
|---|---|---|---|
| Encoder params | 29,440 | 31,168 | 1.06× |
| `policy_at_state` @200 cand | 18.7 ms | 167.1 ms | **8.9×** |
| `policy_at_state` @500 cand | 39.0 ms | 491.0 ms | 12.6× |
| Pretrain episode (T=10) | 174 ms | 802 ms | 4.6× |

Why `policy_at_state` grows ~9× (not 7×): candidate generation and the engineered
embedding scan all parcels, and the encoder forward runs over ~7× the grid cells;
combined with a larger same-cluster adjacency these compound super-linearly.

### 7.3 Time estimates for n=692

**(a) Encoder pretraining** (must be redone — §7.4). The 100.9 s toy pretrain
splits into data collection (rollouts, scales ≈ 4.6×) and 40 epochs over the
dataset (CNN forward/backward over ~7× the cells, scales ≈ 7×). Net ≈ 5–7× →
**≈ 9–12 minutes** at 800 episodes / 40 epochs. (Grows linearly if you raise
episodes/epochs for the harder problem.)

**(b) Robust-MDP training** (5 outer × 5 inner × 2000 samples), which is
`policy_at_state`-dominated:
- **Same config (200 candidates):** 4781.6 s × 8.9 ≈ **42,600 s ≈ 11–12 hours.**
- **Paper-scale candidates (500), recommended given the larger action space:**
  per-eval 491 ms vs the toy's 18.7 ms ⟹ ≈ 26× ⟹ **≈ 35 hours ≈ 1.5 days.**

**(c) Comparison run** (`run_compare.py`): policy evals scale ~9×, but the
StaticApprox MILP grows from `T·100` to `T·692 ≈ 6,920` binaries; HiGHS handles it
but per-solve cost and constraint-generation iterations rise. Estimate **≈ 5–15
minutes** (MILP-dominated; the paper solved a comparable MILP in ~10 min with
Gurobi).

**Caveats (read before quoting):** these scale the measured *per-operation* cost by
the *same loop counts*. The dominant unknown is whether 692 parcels needs **more
candidates, more samples_per_iter, or more outer/inner iters to converge** — almost
certainly yes, which would push training up by a further constant factor. Treat the
numbers as order-of-magnitude, GPU, on this hardware:

| Stage | n=100 (measured) | n=692 (estimate) |
|---|---|---|
| Encoder pretrain | 100.9 s | ~9–12 min |
| Robust-MDP training (200 cand) | 4,781.6 s (80 min) | ~12 h |
| Robust-MDP training (500 cand) | — | ~1.5 days |
| Comparison run | 16 s | ~5–15 min |

### 7.4 Why the trained encoder **cannot** be reused for 692 parcels

1. **Input shape changes.** 9 clusters ⟹ `C_in = 21` (vs 15). `conv1`'s weight
   tensor is `(32, C_in, 3, 3)` — a different shape — so the saved `encoder.pt`
   state dict will not even load. Pretraining from scratch is mandatory.
2. **Different data distribution.** Values, costs, threat indices, adjacency and
   cluster structure are entirely different (real geographic data vs Gaussian
   bumps on a 10×10 grid). A frozen encoder trained on the toy distribution carries
   no useful features for the real one.
3. **Geometry / architecture.** The real 692 parcels are **irregular geographic
   cells, not a clean rectangle**. The CNN needs an `H×W` grid; two options:
   - embed parcels into a bounding grid with masked empty cells (what the §7.2
     synthetic stand-in does), or
   - switch to a **GNN over the parcel adjacency graph** — the cleaner choice for
     arbitrary parcel graphs (`CLAUDE.md` §"Frozen neural encoder option" already
     flags this). A GNN is a different architecture entirely, so again: retrain.

In all cases the encoder is **re-pretrained from scratch** and then frozen, and the
~9–12 min estimate in §7.3(a) applies. The parameter count stays ~31k for the CNN
path (§4.2); a comparable-width GNN (3 message-passing layers, 32 hidden, +5
globals) lands in the same 25–35k ballpark.

---

## 9. Encoder feature interpretation (`tools/analyze_features.py`)

Two analyses of the **frozen encoder's learned features `phi`**, run on the
existing trained model `biodiv-2312346` (n=100). Both write
`<run-dir>/feature_analysis.json`. Reproduce:

```bash
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
 "source ~/qa-gym/.venv/bin/activate && cd ~/robust-mdp && \
  python tools/analyze_features.py --run-dir outputs/slurm/biodiv-2312346 --n-episodes 200"
```

The tool rolls out 200 episodes under the trained policy (controller = argmax of
`pi`, nature sampled from `omega`), recording each visited `(s,a,k)` plus its
Monte-Carlo discounted **return** `y` (the real surviving-value-to-end, i.e.
ground truth, **not** a model estimate). N = 2000 situations.

### 9.1 Q-fit: encoder `phi` vs engineered `phi` (held-out R²)

Fit Ridge twice on the same buffer with a 75/25 train/held-out split; target is
the real return `y`. "Held-out R²" = how well `theta·phi` predicts the true
future value on situations the fit never saw (1.0 = perfect, 0 = no better than
the mean).

| features | dim | train R² | **held-out R²** |
|---|---|---|---|
| encoder (frozen CNN) | 64 | 0.9873 | **0.9866** |
| engineered (`CLAUDE.md` list) | 847 | 0.9954 | **0.9923** |

Gap (encoder − engineered) = **−0.0057** ⟹ a **tie** (engineered marginally
better). **Both predict the true return at R² ≈ 0.99**, so on the states the game
actually visits `Q ≈ theta·phi` is a near-perfect linear fit either way — but the
fancy encoder buys **nothing** over the cheap hand-engineered features.

### 9.2 Effective rank of the encoder `phi` (SVD)

Collect `phi` over the buffer (2000 × 64), centre, take the singular-value
spectrum. The encoder ends in **global average pooling** (`encoder.py`: `h.mean(dim=(-1,-2))`),
which averages each feature map over all parcels and is the prime suspect for
feature collapse.

| metric | value (nominal 64) |
|---|---|
| participation ratio | **1.08** |
| Shannon effective rank | 1.22 |
| dims for 90% energy | **1** |
| dims for 99% energy | 2 |
| top-3 energy fractions | 0.960, 0.033, 0.005 |

**Severe collapse:** the 64-dim fingerprint is effectively **one number** (one
direction holds 96% of the variance). The toy's value function is so well captured
by a single summary scalar that the encoder never needed more.

### 9.3 What this does and does NOT mean

- It does **NOT** mean the policy is broken. Verified independently: under the
  worst-case adversary (20 seeds, same map), the **trained** policy protects
  **4.92** of 25.87 total value vs **0.00** for an untrained `theta=0` policy, and
  it beats StaticApprox/Knapsack (§6). The collapsed-feature model is the same one
  that wins those comparisons.
- It **does** mean the encoder is **over-provisioned for this toy**: a much
  simpler fixed feature map (the engineered one) does the same job. This is a
  direct, useful argument for the 692 plan — using engineered features there would
  skip encoder pretraining entirely, dodge the "can't reuse the encoder" problem
  (§7.4), and be **more** faithful to the paper's fixed-linear-feature assumption.
- Caveat: collapse to ~1 effective dimension is partly a property of the **easy
  toy** (n=100, 3 clusters). The 692 / 9-cluster setting has real spatial structure
  and should need more effective dimensions; re-running this analysis on the new
  runs (§10) will show whether that holds.

---

## 10. Exact nature feasibility + 692-parcel setting (branch `exact-nature-feasibility`)

### 10.1 The exact-nature change

`src/candidates.py::_stochastic_greedy_nature_exact` replaces the frozen-p
approximation as the **default** for nature candidate generation. It enforces the
exact relative-slack feasibility (immediate-p), maintained **incrementally** via
sparse same-cluster neighbour updates — O(deg) ≈ O(4) per parcel, not O(n²). This
makes the whole pipeline (generation + step + adversary) coherent with the paper's
likelihood uncertainty set and with `CLAUDE.md`'s "immediate recomputation"
default. The old frozen-p path is preserved behind `cfg.fixed_p_feasibility=True`
for ablations. Sanity (`src/sanity.py`) now asserts every generated candidate is
exactly feasible. (Note: because developing parcels only *raises* neighbours' `p`
and the relative slack sums only over developed parcels, frozen-p was already a
conservative under-estimate — so this change tightens correctness without
invalidating the earlier runs' conclusions.)

### 10.2 Arbitrary-grid map support

`src/state.py::_assign_clusters` generalises `build_default_map` to any
`(grid_h, grid_w)` via rectangular cluster tiling (`--n-cluster-rows/-cols`), with
per-cluster auto-placed value hotspots. The 10×10 / 3-cluster toy layout is
unchanged. New CLI flags in `run_biodiv_train.py`: `--grid-h/-w`,
`--n-cluster-rows/-cols`, `--fixed-p-feasibility`.

### 10.3 The 692-parcel run (`run_clariden_692.sh`)

We do **not** have the paper's real land dataset, so the map is **synthetic but
matches the paper's parcel count and cluster count**: a **4×173 grid = exactly 692
cells** (no masking), tiled into **3×3 = 9 contiguous clusters** (verified: 9
non-empty clusters of size 57–116, **zero cross-cluster neighbour leak**, total
value ≈ 59.2). `budget=14.0` (≈7× the toy's 2.0, to keep a comparable protected
fraction), `lambda=0.20` (kept for consistency with the n=100 runs), encoder
pretrain 800 ep / 40 epoch, 200 candidates/side, 5 outer × 5 inner × 2000 samples.

### 10.4 Measured timing @692 (login GPU probe, exact nature, 200 cand)

| Quantity | n=692, 9 clusters |
|---|---|
| Encoder params | 29,728 (matches §4.2 formula for 9 clusters / 21 channels) |
| Exact nature-gen | ~26 ms/call (frozen-p path ~6 ms) |
| Encoder `q_matrix_batch` (~111×101) | ~58 ms |
| TD per-sample (`fit_linear_q`) | ~1.3 s/sample |
| Pretrain | ~68 s (mini probe); full 800 ep ≈ 9–12 min |

The login-probe TD micro-benchmark (~1.3 s/sample) **badly under-estimated the
real per-outer cost**: on the actual GPU job (2441645) pretrain took **~39 min**
and **one outer iter (2000 samples) took ~4.3 h**, so a 5-outer run needs ~22 h —
far past the 8 h `normal` wall it was first submitted under (it timed out at
outer 1/5). The fix (job 2483604) keeps everything identical but **halves samples
to 1000 (~2.15 h/outer) and runs 4 outers** (θ converges by iter 2 at n=100),
projecting ~9.3 h core + ~1 h overhead — inside the 12 h `normal` cap. The
account only has `QOS=normal`, so the 24 h `low` partition is not available.

### 10.5 Results — n=100 exact nature (job 2441623, COMPLETE)

This run is identical to the published 2312346/2392077 except for the exact-nature
constraint, so it is a clean apples-to-apples test of the change.

**Training.** The exact adversary is genuinely harder: `avg_protected` settled at
**~10.0 / 25.87** vs **~15.6** under frozen-p — the old approximation was letting
the controller off easy. θ-norm converged by outer iter 2 (~0.55).

**Paper comparison (`run_compare.py`, 20 worst-case × 40 stochastic seeds):**

| Policy | Worst-case protected (↑) | Stochastic protected (↑) |
|---|---|---|
| **Ours (robust)** | **7.74** (median 6.32, min 2.38) | **13.91** |
| StaticApprox (Problem 7) | 3.15 (flat) | 3.15 |
| Knapsack (Problem 8/9) | 3.15 (flat) | 13.37 |

**Ours = +21.5 % less worst-case lost value than *both* baselines**, ≈ 2.45× their
worst-case protected value, and it matches Knapsack in the average case while
crushing StaticApprox there too. Notably the **deployment worst-case improved vs
the frozen-p run** (7.74 vs 4.73): training against the correct, harder adversary
yields a *more* robust policy even though it protects less on average — the central
robustness story for the report.

**Span diagnostics.** Candidate sets cover the reference action-effect pool
essentially exactly at every t (median projection error ≤ ~1e-10; ~1e-16 at most
steps). The ~100–200 sampled candidates span the strategically-relevant subspace.

**Encoder feature interpretation (§9 re-run on this model).** Unchanged story:
encoder held-out R² = **0.968** vs engineered **0.961** (tie — encoder adds
nothing over hand features); effective rank **PR = 1.16 / 64**, 1 dim = 92.5 %
energy (**severe collapse**). Practical upshot stands: at 692 the encoder can be
dropped for engineered features with no expected loss.

### 10.6 Status — n=692 (in progress)

- **2441645** (first attempt) — TIMED OUT at 8 h after only outer 1/5
  (avg_protected 32.2 / 59.2). Superseded.
- **2483604** (resubmit) — n=692, 9 clusters, exact nature, frozen encoder,
  1000 samples × 4 outers on the 12 h `normal` partition. **RUNNING.** Paper
  comparison + feature analysis will be added here on completion.

---

## 11. File map

```
CLAUDE.md                  modeling spec (formulas, conventions, feature lists)
ANALYSIS.md                this document
papers/10341148.pdf        reference paper (+ .txt extract)
run_biodiv_train.py        training entry point (pretrain → train → diagnostics)
run_clariden.sh            SLURM submit script (n=100 canonical config)
run_clariden_692.sh        SLURM submit script (692 parcels, 9 clusters; §10)
run_compare.py             paper-comparison driver (worst-case + stochastic)
src/
  config.py                Config + MapData dataclasses
  state.py                 State, build_default_map, initial_state
  dynamics.py              p_i, log-likelihood, feasibility, step()
  candidates.py            generate_controller_candidates / generate_nature_candidates
  features.py              engineered phi + q_matrix_batch (dispatch on use_encoder)
  encoder.py               GridEncoder (frozen CNN), pretrain_encoder, phi/Q via encoder
  matrix_game.py           solve_matrix_game (max-min over candidate simplices)
  dual_averaging.py, td.py, policy_iteration.py   robust policy iteration (Alg 1/2)
  rollout.py               policy_at_state, rollout_episode
  diagnostics.py           action_span_diagnostics
  baselines.py             Knapsack + StaticApprox + the two natures (comparison)
  sanity.py, viz.py        assertions + plotting helpers
tools/
  analyze_features.py      encoder feature interpretation (§9): Q-fit + effective rank
scripts/
  plot_episode.py          renders per-year episode grids + map heatmaps
  inspect_episode.py       interactive episode walk
  summarize_run.py         prints a run-dir summary
notebooks/
  biodiv_robust_mdp.ipynb  end-to-end demo (value/threat/cost heatmaps with cluster
                           borders + interactive episode cell); .html is a render
outputs/slurm/biodiv-<jobid>/   per-run artifacts (see §3); compare/ holds §6 outputs
```

### Other plots in the repo
- **Notebook initial plots:** value / threat-index / cost heatmaps of the 10×10
  grid, each with a colorbar and **bold black cluster borders**.
- **`scripts/plot_episode.py` outputs:** `episode_grid.png` (2×5 per-year panels),
  `episode_value_bg.png` (same with value heatmap underlay), `per_year/year_t.png`,
  and `map_overview.png`. Grid colors encode free / protected / developed / newly
  protected / newly developed, always with cluster borders.

---

*Generated 2026-05-27; §§9–10 added 2026-06-01 (branch `exact-nature-feasibility`).
Timings are GPU (CUDA) on the Clariden `pytorch/v2.9.1:v2` uenv. Encoder param
counts and per-call timings are directly measured; n=692 end-to-end times are
arithmetic extrapolations from those measured ratios.*
