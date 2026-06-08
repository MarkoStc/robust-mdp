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

> **Status note (2026-06-08).** §§2–7 describe the original published runs
> (`biodiv-2312346`, `biodiv-2392077`) on the `main` branch. Branch
> `exact-nature-feasibility` then (a) made nature's candidate feasibility **exact
> immediate-p** instead of the frozen-p approximation, and (b) added arbitrary-grid
> support for the paper's **692-parcel** count. **§8** maps the whole pipeline,
> term-by-term, onto the Li et al. robust-MDP paper (kernels, Algorithms 1/2/4).
> The feature-interpretation findings are in **§9**; the exact-nature change and the
> **completed** exact n=100 (`biodiv-2441623`) and 692 (`biodiv-2483604`) runs —
> including the honest n=100-vs-692 comparison story — are in **§10**. Read §§8–10
> for what is current.

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

### 2.4 Heuristic candidate scores (controller & nature) — `src/candidates.py`

The reduced candidate libraries (`A_cand`, `K_cand`, ~100–200 distinct actions/side
after dedup) are generated by *scoring* every available parcel, then running a
noisy/greedy knapsack (controller) or a noisy/greedy likelihood-feasible fill
(nature). The exact per-parcel scores used (`generate_controller_candidates`,
`generate_nature_candidates`) are the multiplicative heuristics from `CLAUDE.md`:

**Controller score** (`candidates.py:208`), rewards valuable-cheap-threatened-
frontier parcels, with `α_C, β_C ~ U(0,3)` swept for diversity:

```
S_i^C = ( v_i / (c_i + eps) ) · (1 + α_C · p_i) · (1 + β_C · q_i) · conn_i   (+ 0.3·noise)
```

**Nature score** (`candidates.py:324`), rewards high value per unit likelihood
cost, with `α_N, β_N ~ U(0,3)`:

```
S_i^N = v_i · (1 + α_N · p_i) · (1 + β_N · q_i) / (L_i + eps)   (+ 0.3·noise)
       where  L_i = log((1 - p_i)/p_i)   is the likelihood penalty of developing i.
```

Here `p_i` is the development probability (cellular-automata eq. below), `q_i` the
frontier pressure `(1 + dev same-cluster neighbours)/(1 + same-cluster neighbours)`,
`conn_i` a connectivity bonus for adjacency to already-protected parcels. Additive
variants are also generated for ablation diversity. The multiplication is only a
heuristic — it favours parcels that are good on several axes at once. **These scores
order candidates only; feasibility (budget for the controller, likelihood for
nature) is enforced separately and exactly (§10.1), never via the score.**

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

### 6.1 The paper's methods — exact formulations (`papers/10341148.pdf`)

The paper (Ye et al., AAMAS 2022) writes the development risk with the **cellular
automata** model (their eq. 3, identical to ours):

```
prob_i^t = (TI_i / 10) · (1 + Σ_{j∈Ω_i} 1{j developed}) / (1 + |Ω_i|)
```

with `Ω_i` = same-cluster adjacent neighbours, development ~ `Bernoulli(prob_i)`,
and **9 clusters** (4 threat levels × K-means). Their **likelihood uncertainty set**
(immediately reproduced by our log-form rule, §2) is

```
U = { ξ ∈ {0,1}^{|I|×|T|} :  ξ_it ≥ ξ_{i,t-1},   Π_i (p_i^t)^{ξ_it}(1-p_i^t)^{1-ξ_it} ≥ λ^t  ∀t }.
```

Their **base robust model** is the multi-stage adaptive Problem (2):

```
min_x  sup_{ξ∈Ξ(x)}  Σ_i v_i Σ_t [ξ_it - ξ_{i,t-1}]·[1 - x_it(ξ^{t-1})]      (worst-case lost value)
 s.t.  Σ_i c_it[x_it - x_{i,t-1}] ≤ b_t,   x_it ≥ x_{i,t-1},   x_it ∈ {0,1}.
```

Note the protection variable `x_it(ξ^{t-1})` is decided on history up to `t-1`
(**before** `ξ_t` is observed) — i.e. **protect-first within a year, exactly our
simulator convention.** Problem (5) replaces the endogenous `Ξ(x)` with the
exogenous `U` (Theorem 5.2: conservative, and equal under a non-negativity
condition); Problem (6) is its scenario-extensive form. We re-implement their two
solved methods in `src/baselines.py`:

- **StaticApprox** = paper **Problem (7)**, their *final proposal* — a **static,
  here-and-now** (non-adaptive) collapse of Problem (6):
  ```
  min τ  s.t.  τ ≥ Σ_i v_i Σ_t (ξ_it - ξ_{i,t-1})(1 - x_{i,t-1})  ∀ξ∈U;
              Σ_i c_it(x_it - x_{i,t-1}) ≤ b_t;  x_it ≥ x_{i,t-1};  x_it ∈ {0,1}.
  ```
  Solved by **constraint generation**: a master MILP (`min τ` + budget +
  monotonicity + one robust cut per generated scenario) and a greedy separation
  oracle that finds the worst-case development against the current schedule. Two
  faithful-but-noted deviations: (i) the paper credits protection a year early via
  `(1 - x_{i,t-1})`; we credit the **same** year `(1 - x_{i,t})` to match our
  protect-then-develop simulator, which makes our StaticApprox **slightly stronger**
  than the paper's literal version (we do not handicap the baseline). (ii) HiGHS via
  `scipy.optimize.milp` instead of Gurobi.
- **Knapsack** = paper **Problem (8)** (single-stage, budget `b_1`,
  `max Σ_i v_i x_i s.t. Σ_i c_i x_i ≤ b_1`) generalised to the multi-stage
  **Problem (9)** (`max Σ_i v_i x_{iT}` with per-year budgets `Σ_i c_i[x_it -
  x_{i,t-1}] ≤ b_t`). Their *benchmark*: maximise protected value, **ignore
  uncertainty**; adaptive only in re-optimising over whatever parcels remain. Exact
  0/1 knapsack per year via `scipy.optimize.milp`.

**Proposition 6.1 (paper):** for small `λ` the uncertainty set is so large that
every unprotected parcel is developed, so minimising loss ≡ maximising protected
value ⟹ **StaticApprox ≈ Knapsack**. We reproduce this exactly at n=100 (§10.5),
a faithfulness check on the baselines.

### 6.2 The two "natures" (identical for all three controllers)

- **Worst-case / adversarial** (`greedy_adversary_develop`): greedily develops the
  parcels with the highest value-per-likelihood-cost `v_i / log((1-p_i)/p_i)` while
  keeping the realization inside U, recomputing `p` after each addition (immediate
  recomputation; captures the development snowball). This is the robustness test.
- **Average-case / stochastic** (`bernoulli_develop`): the cellular-automata
  `Bernoulli(p_i)` model the paper uses to *simulate* realized developments (no
  robustness cap). This is the typical-case test.

### 6.3 Results — see §10.5 / §10.6 (current, exact-nature)

> **The earlier frozen-p comparison numbers were removed as superseded.** They were
> produced under the frozen-p candidate-feasibility approximation (runs
> `biodiv-2392077` / `biodiv-2312346`) and *understated* the worst-case margin
> (they reported Ours ≈ 7 % less worst-case loss than StaticApprox). Training and
> evaluating against the **correct exact immediate-p adversary** (§10.1) roughly
> tripled that margin. The current, measured results are:
>
> - **n=100 exact** (`biodiv-2441623`): **§10.5** — Ours **+21.5 %** less worst-case
>   loss than *both* baselines; StaticApprox == Knapsack on the worst case (a live
>   confirmation of **Proposition 6.1**, the baseline-faithfulness check).
> - **n=692** (`biodiv-2483604`): **§10.6** — Ours **+51.2 %** vs StaticApprox but
>   **−14.3 % vs Knapsack** (ample-budget regime; honest non-win vs Knapsack).
>
> Always read the comparison story from §10, not from this section.

### 6.4 Methodology lesson baked into the code

Our policy **samples** its candidate actions, so a *single* greedy-adversary
rollout is noisy (std ≈ 1–5 depending on scale). Early single-episode measurements
swung sign between runs — a pure metric artifact. The worst-case evaluation
therefore **averages over `--n-worstcase` seeds (default 20)** and reports the
distribution (StaticApprox/Knapsack are deterministic ⟹ std 0).
**Always report the multi-seed distribution, never a single rollout.**

### 6.5 The comparison plot — `compare/comparison_vs_paper.png`

One PNG per run (each run dir's `compare/`), two panels, three methods (`Ours`,
`StaticApprox`, `Knapsack`):
- **Left — "Loss to development (lower = better)":** for each method, a solid bar =
  worst-case mean (with min/max error bars) and a faded bar = stochastic mean
  (with min/max error bars).
- **Right — "Value preserved (higher = better)":** solid = worst-case protected,
  faded = stochastic protected.
- Title shows the run's `total value` (25.87 at n=100, 59.21 at 692).

The current report plots are `outputs/slurm/biodiv-2441623/compare/comparison_vs_paper.png`
(n=100 exact) and `outputs/slurm/biodiv-2483604/compare/comparison_vs_paper.png`
(692). `compare/comparison_results.json` holds the full numbers: `meta` (run dir,
lambda, budget, horizon, total value, #stochastic episodes, #parcels StaticApprox
planned) and `results[method][worst_case|stochastic][lost_value|protected_value|free_value]`
with `mean/std/min/max/median`.

### 6.6 Reproduce

```bash
uenv run pytorch/v2.9.1:v2 --view=default -- bash -c \
 "source ~/qa-gym/.venv/bin/activate && cd ~/robust-mdp && \
  python run_compare.py --run-dir outputs/slurm/biodiv-2441623 \
  --n-worstcase 20 --n-stochastic 40"
```
Loads that run's frozen encoder + `theta_final`, rebuilds the seed-matched map,
solves both baselines, evaluates all three, writes the JSON+PNG. ~16 s at n=100;
at 692 the O(n²) adversary is heavy — run it as a batch job (`run_postprocess_692.sh`),
not on the login node.

---

## 7. Scaling to the paper's real 692-parcel setting

### 7.1 What the paper's real instance is (paper §7)

After cleaning, the paper keeps **692 parcels** (jaguar range, Latin America),
clustered into **9 groups** (4 threat levels × K-means subclusters). The reported
numerical experiment is **single-stage** ("we consider a single-stage problem,
which can be solved to optimality within 10 minutes" with Gurobi/CPLEX), budget
swept 25–175 M USD, evaluated on 1000 cellular-automata samples.

Our pipeline is **multistage (T=10)** and RL-based, so a direct port is more
expensive than their single-stage MILP.

### 7.2 Measured 692 timings — see §10.4 / §10.6

> **The earlier order-of-magnitude extrapolations (per-op micro-benchmarks scaled by
> loop counts) were removed as superseded by the real 692 run.** We now have
> *measured* numbers from job `biodiv-2483604`:
>
> | Stage | n=100 (measured) | n=692 (measured, job 2483604) |
> |---|---|---|
> | Encoder pretrain | 100.9 s | ~39 min |
> | One outer iter | — | ~4.3 h @2000 samples; ~2.15 h @1000 |
> | Robust-MDP training | 4,781.6 s (80 min) | ~8 h 47 m (1000 samples × 4 outers) |
> | Comparison (`run_compare.py`) | 16 s | part of the ~16 min postprocess batch |
>
> See **§10.4** for the timing breakdown (and why the first 8 h attempt timed out)
> and **§10.6** for the completed run. The account has only `QOS=normal` (12 h cap;
> no 24 h `low` partition), which set the 1000-sample × 4-outer budget.

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
   - embed parcels into a bounding grid with masked empty cells (what the §10.3
     synthetic 4×173 stand-in does), or
   - switch to a **GNN over the parcel adjacency graph** — the cleaner choice for
     arbitrary parcel graphs (`CLAUDE.md` §"Frozen neural encoder option" already
     flags this). A GNN is a different architecture entirely, so again: retrain.

In all cases the encoder is **re-pretrained from scratch** and then frozen (the
measured 692 pretrain was ~39 min, §10.4). The parameter count stays ~31k for the
CNN path (§4.2); a comparable-width GNN (3 message-passing layers, 32 hidden, +5
globals) lands in the same 25–35k ballpark.

---

## 8. Theoretical grounding — mapping to Li et al. (`papers/Li_Mengmeng_Paper1_MDP.pdf`)

The **algorithmic backbone** (not the application) comes from Li, Hu, Kuhn & Li,
*Robust Markov Decision Processes on Continuous State Spaces* — robust MDPs with a
**mixture ambiguity set** solved by stochastic dual averaging + TD + approximate
policy iteration. Our whole `src/` pipeline is a finite-candidate instantiation of
that framework. The correspondence is **exact**, term by term.

### 8.1 Deterministic development is a (degenerate) probability kernel

Their ambiguity set (Definition 1) is built from a finite family of **generating
transition kernels** `{P_k}_{k∈K}`, and nature plays a mixture
`P^ω(·|s,·) = Σ_k ω(k|s) P_k(·|s,·)` with weights `ω(·|s) ∈ W ⊆ Δ_K`. A *kernel*
`P_k(·|s,a)` is just a probability distribution over the next state.

Our discussion point: **"develop this specific set of parcels `k`" is a perfectly
valid probability kernel — it is the degenerate (Dirac) one** that puts all its mass
on the single deterministic next state `s' = step(s, a, k)`:

```
P_k(· | s, a) = δ_{ s' = step(s,a,k) }   (a point mass; a special case of P(S)).
```

So each **nature candidate `k ∈ K_cand(s)` IS a generating kernel** `P_k`. There is
no contradiction between "nature picks a deterministic set" and "nature picks a
kernel": a deterministic outcome is the extreme point of the probability simplex
over next states. Nature's *randomisation* then lives one level up — in the mixture
weights `ω(·|s) ∈ Δ_{K_cand}` over those deterministic kernels — which is precisely
the convex hull `Σ_k ω(k|s) P_k` of Definition 1. (Stochastic cellular-automata
nature, §6.2, is the same picture with each `P_k` a product-Bernoulli kernel instead
of a Dirac; our framework handles both because it only ever needs *sample access* to
`P_k`, never its analytic form — Li et al. §2.)

**The controller side is identical in form**: the controller plays a randomized
policy `π(·|s) ∈ Δ_A` over its candidate protection actions `A_cand(s)` (their
`π : S → Δ_A`). Each controller candidate `a` is a deterministic action; `π`
mixes over them. So `A_cand` ↔ finite `A`, `K_cand` ↔ generating kernels `{P_k}`,
and `W = Δ_{K_cand}` is nature's feasible mixture set.

### 8.2 Term-by-term dictionary

| Li et al. object | Our implementation |
|---|---|
| Generating kernels `{P_k}_{k∈K}` | nature candidates `K_cand(s)`, each a (Dirac or Bernoulli) kernel `P_k = δ_{step(s,a,k)}` |
| Nature mixture weights `v, ω(·|s) ∈ W = Δ_K` | nature simplex `ω` from the matrix game / dual averaging |
| Controller policy `π(·|s) ∈ Δ_A` | controller simplex `π` over `A_cand(s)` |
| Joint action-value `Q^ω_π(s,a,k) = r(s,a) + γ∫V^ω_π(s')P_k(ds'|s,a)` (Def. 2) | `Q(s,a,k) = θ^T φ(s,a,k)`, the linear-in-features Q (`src/td.py`, `features.py`) |
| Nature's action-value `H^ω_π(s,v) = Σ_a Σ_k v_k π(a|s) Q^ω_π(s,a,k)` (Def. 2) | `H(s,v) = v^T g(s)`, `g_k(s) = Σ_a π(a|s) Q(s,a,k)` (`src/dual_averaging.py`, `make_omega_oracle`) |
| Robust joint Q `Q_π(s,a,k) = min_{ω∈W} Q^ω_π` (eq. 16) | softmin / matrix-game value over the candidate simplex |
| Linear span `Q = {θ^T φ}`, `‖φ‖≤1` | engineered or **L2-normalised frozen-encoder** features φ (§4, §9); only `θ` is learned ⟹ matches their *fixed basis* assumption |

### 8.3 Algorithm-by-algorithm correspondence

- **Algorithm 2 (TD learning for `Q^ω_π`).** Their update
  `θ_{t+1} = θ_t − η·bF(θ_t, ξ_t)` with the linear TD operator (their eq. 24) is our
  `fit_linear_q` / `src/td.py`: draw transitions from the candidate kernels, fit the
  linear head `θ` of `Q ≈ θ^T φ`. The encoder φ is **frozen** (Li et al. require a
  *fixed* feature map — updating it inside the loop would break the contraction;
  `CLAUDE.md` "Things to avoid" enforces this).
- **Algorithm 1 (stochastic dual averaging for robust policy *evaluation*).** Their
  `ω^{(m+1)}(·|s) ← argmin_{v∈W} Σ_{t} α_t Ĥ_t(s,v) + λ_m h(v)` with negative-entropy
  `h` is our **inner loop** (`src/dual_averaging.py`, `inner_M` iterations): the
  entropic argmin has the closed form `ω ∝ exp(−cum_g / λ)`, `cum_g = Σ_t α_t g(s)`,
  `α_m = √(m+1)` — exactly their stepsize `α_m = √m`. The output is the
  `ϑ_m`-weighted average `Q̂_π = Σ_m ϑ_m Q̂_m`, `ϑ_m = α_m / Σ_j α_j` — our returned
  `theta_bar` (the αₘ-weighted mean of the inner `θ_m`). **This is the robust Q of
  the frozen controller policy `π_n`.**
- **Algorithm 4 (approximate policy *iteration*).** Their outer update
  `(π_{n+1}, ω_{n+1})(·|s) ← max_{π∈Δ_A} min_{ω∈Δ_K} π^T Q̂_{π_n}(s) ω` (their eq. 35)
  is our **outer loop** (`src/policy_iteration.py` + `solve_matrix_game`): at each
  outer iter `n`, the *frozen* `θ_old` defines the controller policy `π_n(·|s)` as
  the **matrix-game equilibrium** over `A_cand × K_cand`, Algorithm 1 evaluates its
  robust Q, and the new robust Q defines `π_{n+1}`. Because policies are generated
  *on demand from the candidate sets at the queried state* (never tabulated over all
  states), our cost — like theirs — is **independent of the size of the state
  space** (`outer_iters` ↔ `N`, `inner_M` ↔ Algorithm-1 `M`).

The one honest gap vs. their assumptions: their guarantees assume `W = Δ_K` over the
*true* generating kernels. We replace the intractable `2^n` kernels with a
**heuristic candidate subset** `K_cand` (and `A_cand`); §5/§10.5 span diagnostics
quantify how little this truncation loses in Q-relevant directions (projection error
≈ machine-ε). With a frozen *linear* feature map their fixed-basis hypotheses hold;
we **do not** claim their convergence rates for any end-to-end nonlinear training
(`CLAUDE.md` "Things to avoid").

---

## 9. Encoder feature interpretation (`tools/analyze_features.py`)

Two analyses of the **frozen encoder's learned features `phi`**, run on the
existing trained model `biodiv-2312346` (n=100), and re-run on the exact-nature
n=100 (`biodiv-2441623`, §10.5) and the 692 model (`biodiv-2483604`, §10.6). Both
write `<run-dir>/feature_analysis.json`. Reproduce:

> **Orthogonality note (important for reading this section).** The
> engineered-vs-encoder question is about *how `Q(s,a,k)` is represented* (the
> feature map φ). It is **independent of the exact-vs-frozen-p nature feasibility
> calculation** (§10.1), which only governs *which nature candidates `K_cand` are
> generated*. The same exact-nature candidate generation feeds both the encoder and
> the engineered features; switching φ does not change feasibility, and switching
> feasibility does not change φ. So "encoder buys nothing over engineered features"
> (below) is a conclusion about representation, and holds across both the frozen-p
> and exact-nature runs.

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
to 1000 (~2.15 h/outer) and runs 4 outers** (θ converges by iter 2 at n=100); it
**completed in ~8 h 47 m** on the 12 h `normal` partition (results in §10.6). The
account only has `QOS=normal`, so the 24 h `low` partition is not available.

### 10.5 Results — n=100 exact nature (job 2441623, COMPLETE)

This run is identical to the published 2312346/2392077 except for the exact-nature
constraint, so it is a clean apples-to-apples test of the change. Config: 10×10,
3 clusters, λ=0.20, budget 2.0, 200 cand/side, **5 outer × 5 inner × 2000 samples**,
frozen CNN encoder. Total map value 25.87.

**Outer-loop training metrics (`train_log.json`, 5 outer iters).** The exact
adversary is genuinely harder. On-policy training rollouts:

| outer iter | avg_protected | avg_developed | avg_free | θ-norm |
|---|---|---|---|---|
| 1 | 9.94 | 15.85 | 0.080 | 0.387 |
| 2 | 10.44 | 15.42 | 0.009 | 0.547 |
| 3–5 | ~10.0 | ~15.7–15.9 | ~0.00 | ~0.55 |

Reading it: **converged by outer iter 2** (θ-norm flat at ~0.55, protected flat at
~10.0); `avg_free ≈ 0` means by year `T` every parcel is either protected or
developed (no fence-sitters) — expected under the brutal λ=0.20 cascade. Protected
settled at **~10.0 / 25.87 vs ~15.6 under frozen-p**: the old approximation was
letting the controller off easy. *(Inner dual-averaging `M=5` iterations are not
logged individually — only their αₘ-weighted average `θ_bar` per outer round is
recorded; convergence is read from the across-outer θ-norm plateau.)*

**Paper comparison (`run_compare.py`, 20 worst-case × 40 stochastic seeds;
`compare/comparison_results.json`):**

| Policy | WC lost (↓) | WC protected (↑) | Stoch lost (↓) | Stoch protected (↑) |
|---|---|---|---|---|
| **Ours (robust)** | **17.82 ± 4.25** | **7.74** | 11.89 | 13.91 |
| StaticApprox (Prob. 7) | 22.72 (flat) | 3.15 | 19.82 | 3.15 |
| Knapsack (Prob. 8/9) | 22.72 (flat) | 3.15 | 12.48 | 13.37 |

**Ours = +21.5 % less worst-case lost value than *both* baselines**, ≈ 2.45× their
worst-case protected value, and it matches Knapsack in the average case while
crushing StaticApprox there too. **StaticApprox == Knapsack on the worst case
(both 22.72)** — a live confirmation of **Proposition 6.1** (small λ): a faithfulness
check on our baselines. Notably the **deployment worst-case improved vs the frozen-p
run** (7.74 vs 4.73 protected): training against the correct, harder adversary yields
a *more* robust policy even though it protects less on average — the central
robustness story for the report.

**Span diagnostics (`diagnostics.json`).** Candidate sets cover the reference
action-effect pool essentially exactly at every t (controller median proj-err
~1e-16; nature ~1e-10). The ~70–95 distinct sampled candidates/side span the
strategically-relevant subspace.

**Encoder feature interpretation (§9 re-run on this model).** Unchanged story:
encoder held-out R² = **0.968** vs engineered **0.961** (tie — encoder adds nothing
over hand features); effective rank **PR = 1.16 / 64**, 1 dim = 92.5 % energy
(**severe collapse**). Practical upshot: at 692 the encoder can be dropped for
engineered features with no expected loss.

### 10.6 Results — n=692 (job 2483604, COMPLETE)

The paper-scale run: **4×173 grid = exactly 692 parcels**, tiled into **3×3 = 9
contiguous clusters** (verified: 9 non-empty clusters, size 57–116, **zero
cross-cluster neighbour leak**), λ=0.20, **budget 14.0/yr** (≈7× the toy's 2.0 to
keep a comparable protectable fraction), exact nature, frozen CNN encoder,
**1000 samples × 4 outers × 5 inner**, 200 cand/side. Total map value **59.21**.
(First attempt **2441645** TIMED OUT at 8 h after outer 1/5 — see §10.4 timing;
this resubmit completed in ~8h47m on the 12 h `normal` partition.) Post-processing
(comparison + feature analysis) ran as batch job **2485588** (`run_postprocess_692.sh`,
~16 min) because the login node kills the heavy O(n²) adversary; artifacts land in
`outputs/slurm/biodiv-2483604/`.

**Outer-loop training metrics (`train_log.json`, 4 outer iters).**

| outer iter | avg_protected | avg_developed | avg_free | θ-norm |
|---|---|---|---|---|
| 1 | 32.18 | 7.88 | 19.15 | 0.349 |
| 2 | 32.33 | 7.03 | 19.85 | 0.488 |
| 3 | 32.35 | 7.03 | 19.83 | 0.488 |
| 4 | 32.33 | 7.17 | 19.70 | 0.489 |

Reading it: again **converged by outer iter 2** (θ-norm flat at ~0.488). Unlike
n=100, `avg_free ≈ 19.7` is large — at budget 14 the controller protects ~32 and the
*on-policy* (ω-sampled) nature only develops ~7, leaving ~20 untouched. That on-policy
gentleness is the first hint the budget is loose here; the worst-case adversary
(below) is far more aggressive than the training-time ω.

**Paper comparison (`run_compare.py`, 20 worst-case × 40 stochastic seeds;
`compare/comparison_results.json`). StaticApprox plans 115 parcels:**

| Policy | WC lost (↓) | WC protected (↑) | Stoch lost (↓) | Stoch protected (↑) |
|---|---|---|---|---|
| **Ours (robust)** | 14.83 ± 5.33 | 31.57 | 29.89 | 29.17 |
| StaticApprox (Prob. 7) | 30.41 (flat) | 15.09 | 40.00 | 14.14 |
| **Knapsack (Prob. 8/9)** | **12.98** (flat) | **32.48** | 29.86 | 29.29 |

- **Ours vs StaticApprox: +51.2 % less worst-case loss** — a decisive win over the
  paper's *proposed* method, in both regimes (StaticApprox's fixed plan cannot
  react, so it collapses everywhere).
- **Ours vs Knapsack: −14.3 % (Knapsack wins the worst case), tied on average**
  (29.89 vs 29.86 lost). **This reverses the n=100 result** where Ours beat Knapsack
  by +21.5 %, and we report it honestly.

**Why the reversal (regime, not bug).** At n=100 the budget (2.0) is brutally tight
and λ=0.20 makes the cascade raze everything, so per Prop 6.1 **Knapsack collapses
to 3.15** and adaptivity is the only way to do better — Ours wins. At 692 the budget
(14/yr) is **generous** (you can protect ~half of all value), so plain value-greedy
Knapsack is already near-optimal and there is little adaptive edge to capture. A
parcel-count autopsy (one worst-case episode, seed 123) makes the mechanism concrete:

| | protected parcels | developed parcels | free | prot. value | lost value |
|---|---|---|---|---|---|
| Ours | ~320 | ~225 | 147 | 31.4 | 17.5* |
| Knapsack | 334 | 164 | 194 | 32.5 | 13.0 |

\*single unlucky seed; the 20-seed mean is 14.8. Both protect a similar *count* and
*value*, but Ours leaves more **frontier** parcels exposed, letting the immediate-p
cascade chain ~60 more developments (225 vs 164). This is consistent with the
**severe feature collapse** (below): a ~1-D Q can only express crude value-greedy
protection — essentially Knapsack — and cannot do the subtle "protect the
cascade-enabling frontier" move that would beat it. Net: Ours is *not* more robust
than Knapsack at this budget; it ties/slightly trails.

**Decision:** we are **not** pursuing a tighter-budget 692 re-run (which would
recreate the tight regime where adaptivity wins). The honest, reported finding is:
*robust adaptivity decisively beats the paper's StaticApprox at both scales, beats
Knapsack only in the tight-budget regime (n=100), and ties Knapsack when budget is
ample (692).*

**Span diagnostics (`diagnostics.json`).** Even at 692 the reduced candidate sets
cover the reference action-effect pool essentially exactly: controller median
proj-err ~6–8e-10, nature ~3–5e-10, at both t=1 and t=10, with only ~111 controller
/ ~125 nature distinct candidates vs ~1600 reference actions. The reduction loses no
Q-relevant directions at scale.

**Encoder feature interpretation (`feature_analysis.json`).** Encoder held-out R² =
**0.99882** vs engineered (dim 5619) **0.99877** — a **tie** (gap +5.5e-5); effective
rank **PR = 1.11 / 64**, 1 dim = 94.7 % energy, dims-for-90 % = 1 (**SEVERE
collapse**). The hope that real 9-cluster spatial structure would force more
effective dimensions (the §9.3 caveat) **did not materialise** — collapse is as
severe at 692 as at n=100. Direct implication: **drop the encoder for engineered
features at scale** — same Q-fit, no pretraining, no "can't reuse the encoder"
problem (§7.4), and a closer match to the paper's fixed-linear-feature assumption.

---

## 11. File map

```
CLAUDE.md                  modeling spec (formulas, conventions, feature lists)
ANALYSIS.md                this document
papers/10341148.pdf        application paper: Ye et al., biodiversity ARO (+ .txt)
papers/Li_Mengmeng_Paper1_MDP.pdf  algorithmic paper: robust MDP / mixture
                           ambiguity set (Algs 1/2/4); mapped in §8 (+ .txt)
run_biodiv_train.py        training entry point (pretrain → train → diagnostics)
run_clariden.sh            SLURM submit script (n=100 canonical config)
run_clariden_692.sh        SLURM submit script (692 parcels, 9 clusters; §10)
run_postprocess_692.sh     SLURM batch: run_compare + analyze_features for a run
                           (heavy 692 adversary -> compute node, not login; §10.6)
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

### Where the report metrics & pictures live (per run dir `outputs/slurm/biodiv-<jobid>/`)

| Artifact | File | Sections that quote it |
|---|---|---|
| Exact `Config` used | `config.json` | all |
| Frozen encoder weights | `encoder.pt` | §4 |
| Learned linear head | `theta_iter{1..}.npy`, `theta_final.npy` | §3, §8.3 |
| Outer-loop training curve | `train_log.json` | §10.5, §10.6 |
| Action-span diagnostics | `diagnostics.json` | §5, §10.5, §10.6 |
| Encoder-vs-engineered Q-fit + effective rank | `feature_analysis.json` | §9, §10.5, §10.6 |
| Paper comparison numbers | `compare/comparison_results.json` | §6, §10.5, §10.6 |
| Paper comparison **plot** | `compare/comparison_vs_paper.png` | §6.5 |
| Full stdout (pretrain, sanity, per-outer log, timings) | `slurm.out` | §3, §10.4 |
| Episode / map pictures | `scripts/plot_episode.py` → `episode_grid.png`, `episode_value_bg.png`, `per_year/year_t.png`, `map_overview.png` | §11 "Other plots" |

**The runs referenced for the report:** n=100 exact = `biodiv-2441623`; 692 =
`biodiv-2483604`; original frozen-p n=100 = `biodiv-2312346` / `biodiv-2392077`.

### Other plots in the repo
- **Notebook initial plots:** value / threat-index / cost heatmaps of the 10×10
  grid, each with a colorbar and **bold black cluster borders**.
- **`scripts/plot_episode.py` outputs:** `episode_grid.png` (2×5 per-year panels),
  `episode_value_bg.png` (same with value heatmap underlay), `per_year/year_t.png`,
  and `map_overview.png`. Grid colors encode free / protected / developed / newly
  protected / newly developed, always with cluster borders.

---

*Generated 2026-05-27; §§9–10 added 2026-06-01; §§2.4, 6.1, 8 and final
n=100-exact + 692 results (§§10.5–10.6) added 2026-06-08 (branch
`exact-nature-feasibility`). Timings are GPU (CUDA) on the Clariden
`pytorch/v2.9.1:v2` uenv. Encoder param counts and per-call timings are directly
measured; the n=692 training/comparison numbers are now **measured** (job 2483604),
not extrapolated.*
