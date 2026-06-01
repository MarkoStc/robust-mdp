# CLAUDE.md

## Project overview

This repo implements a toy biodiversity conservation game as an approximate robust MDP / zero-sum Markov game.

The original parcel action spaces are enormous:

- controller action: choose a subset of parcels to protect;
- nature action: choose a subset of parcels to develop;
- raw size is roughly `2^n` for `n` parcels.

The repo must **not** enumerate the raw action spaces. Instead, it should generate a reduced candidate library of about **500--1000 actions per player per queried state**, using heuristic scoring plus exploration. The local game is then solved only over these candidate sets.

Use the “good-value” convention:

```text
Q(s, a, k) = expected protected / preserved value from now to the end
```

Higher Q is better for the controller. Nature tries to minimize Q.

---

## Core modeling conventions

### State

A state `s` contains at least:

- current time/year `t`;
- protected vector `x in {0,1}^n`;
- developed/lost vector `d in {0,1}^n`;
- current and remaining budgets;
- fixed parcel data: value `v_i`, cost `c_i`, threat index `TI_i`, cluster id, coordinates, adjacency.

For the default toy notebook:

- grid size: `10 x 10 = 100 parcels`;
- clusters:
  - cluster 0: top-left `5 x 5` block;
  - cluster 1: top-right `5 x 5` block;
  - cluster 2: bottom `5 x 10` block;
- adjacency: 4-neighbor grid, up/down/left/right only;
- neighbors count only if adjacent **and in the same cluster**.

### Timing

At each year:

1. controller protects parcels;
2. nature develops parcels;
3. protected parcels cannot be developed;
4. developed parcels cannot later be protected;
5. once protected, always protected;
6. once developed, always developed.

Use:

```python
x_plus = x | a
d_plus = d | k
```

with feasibility masks so `a` and `k` cannot violate rules.

### Development probability

For parcel `i`, compute:

```text
p_i = (TI_i / 10) * (1 + number of developed same-cluster neighbors) / (1 + number of same-cluster neighbors)
```

Clip probabilities into `[eps, 1-eps]`.

### Lambda and nature feasibility

`lambda_uncertainty` is fixed before the game. It controls robustness.

Nature feasibility uses the likelihood cutoff:

```text
prod_i (p_i^t)^(d_i) * (1-p_i^t)^(1-d_i) >= lambda^t
```

Always implement this in log form:

```text
loglik(d) = sum_i [d_i log p_i + (1-d_i) log(1-p_i)]
feasible iff loglik(d_plus) >= t * log(lambda_uncertainty)
```

Default interpretation for this repo: **immediate recomputation**.

That means when checking nature action `k`, compute probabilities from `d_plus`, not old `d`:

```python
p_plus = compute_p(d_plus)
loglik = log_likelihood(d_plus, p_plus)
```

Also keep a config option:

```python
FIXED_P_FEASIBILITY = False
```

If true, use `p = compute_p(d)` for the likelihood check.

---

## Reduced action spaces

Do not enumerate `2^n` actions.

At each queried state, generate:

```text
A_cand(s): 500--1000 controller candidate actions
K_cand(s): 500--1000 nature candidate actions / kernels
```

All other actions are excluded from the local game. Equivalently, nature’s simplex is over the selected candidate kernels only.

Important: do **not** treat this as `K=2^n` with most weights set to zero. Instead define the finite set actually used by the algorithm as the candidate set.

---

## Controller heuristic candidate generation

Controller wants to protect valuable, cheap, threatened, frontier-relevant parcels.

For available parcel `i`:

```text
available_i = (x_i == 0) and (d_i == 0)
```

Define frontier pressure:

```text
q_i = (1 + developed same-cluster neighbors of i) / (1 + same-cluster neighbors of i)
```

A useful controller score is:

```text
S_i^C = (v_i / (c_i + eps)) * (1 + alpha_C * p_i) * (1 + beta_C * q_i) * connectivity_bonus_i
```

The multiplication is only a heuristic. It rewards parcels that are good on several dimensions at once. Also implement additive variants for ablations.

Generate diverse actions by:

- sampling many `(alpha_C, beta_C)` values;
- adding Gumbel/Gaussian noise to parcel scores;
- randomized greedy knapsack under budget;
- random swaps / perturbations;
- some pure exploration actions;
- always include the zero action;
- deduplicate actions.

Controller feasibility:

```text
sum_i c_i a_i <= budget_t
a_i <= (1-x_i)(1-d_i)
```

Do not set infeasible actions to Q=0. Mask/exclude them.

---

## Nature heuristic candidate generation

Nature wants to destroy high value while spending little likelihood budget.

Nature does not pay parcel cost unless an explicit development cost is added. Its constraint is likelihood feasibility.

For fixed probabilities, developing parcel `i` changes log-likelihood by:

```text
Delta_i = log(p_i) - log(1-p_i)
```

Usually `p_i < 1/2`, so this is negative. Define likelihood penalty:

```text
L_i = log((1-p_i) / p_i)
```

A useful nature score is:

```text
S_i^N = v_i * (1 + alpha_N * p_i) * (1 + beta_N * q_i) / (L_i + eps)
```

Generate diverse nature actions by:

- varying `(alpha_N, beta_N)`;
- adding noise;
- stochastic greedy selection while maintaining likelihood feasibility;
- high-value variants;
- high-probability variants;
- frontier-focused variants;
- random feasible variants;
- always include the zero action;
- deduplicate actions.

The stochastic greedy selection enforces feasibility with **exact immediate-p
recomputation** (the repo default, matching the feasibility rule above): as each
parcel is tentatively developed, the cumulative log-likelihood-ratio slack is
updated using probabilities recomputed from the growing `d_plus`. Developing a
parcel raises its same-cluster neighbours' `p`, so this is done incrementally —
only the touched neighbours' contributions are updated (O(deg) per parcel) rather
than recomputing over all `n`. The relative-slack form sums **only over developed
parcels** (the `(1-d) log(1-p)` terms cancel against the all-zero baseline), and
developing can only raise that slack, so every generated candidate is guaranteed
feasible under the exact rule. Setting `fixed_p_feasibility = True` switches to the
older frozen-p fast path (probabilities from the current `d`, never updated) for
ablations; it is an approximation and is no longer the default.

Nature feasibility:

```text
k_i <= (1-x_plus_i)(1-d_i)
loglik(d_plus) >= t * log(lambda_uncertainty)
```

---

## Q approximation

The Q-function is approximated as:

```text
Q_hat(s, a, k) = theta^T phi(s, a, k)
```

where `phi` may be either:

1. engineered features; or
2. a frozen neural encoder output.

The robust MDP algorithm assumes a **fixed feature map**. Therefore, if using a neural encoder:

- pretrain encoder first;
- freeze encoder weights;
- start a separate robust-MDP run;
- update only the linear head `theta`.

Do not update encoder weights inside Algorithm 1/2 if claiming compatibility with the fixed-feature linear-Q setup.

Normalize or clip features so that feature norms are controlled.

---

## Engineered features `phi(s,a,k)`

Use normalized features.

Recommended feature groups:

### Global state features

- bias `1`;
- `t / T`;
- `(T-t) / T`;
- current budget / total cost;
- remaining budget / total cost;
- protected value / total value;
- developed value / total value;
- free vulnerable value / total value;
- risk-weighted free value: `sum_i v_i p_i free_i / total_value`;
- frontier vulnerable value: `sum_i v_i q_i free_i / total_value`.

### Controller action features

- newly protected value / total value;
- newly protected cost / total cost;
- safe/clipped protected value per cost;
- risk-weighted protected value / total value;
- safe/clipped risk-weighted protected value per cost;
- frontier protected value: `sum_i v_i q_i a_i / total_value`;
- number of protected parcels normalized;
- isolated protected value;
- value adjacent to already protected parcels.

### Nature action features

- newly developed value / total value;
- risk-weighted developed value / total value;
- frontier developed value / total value;
- likelihood slack:

```text
slack = (loglik(d_plus) - t * log(lambda_uncertainty)) / n
```

- number of developed parcels normalized.

### Post-action features

After applying both actions:

- protected value after action / total value;
- developed value after action / total value;
- free vulnerable value after action / total value;
- risk-weighted free value after action / total value;
- frontier vulnerable value after action / total value.

### Cluster features

For each cluster:

- protected value in cluster / total value;
- developed value in cluster / total value;
- free value in cluster / total value;
- risk-weighted free value in cluster / total value;
- newly protected value in cluster / total value;
- newly developed value in cluster / total value.

Cluster features matter because neighbor effects are cluster-local.

### Parcel-level identity features

For each parcel:

- `v_i x_i / total_value`;
- `v_i d_i / total_value`;
- `v_i a_i / total_value`;
- `v_i k_i / total_value`;
- `v_i p_i free_i / total_value`;
- `v_i p_i a_i / total_value`;
- `v_i p_i k_i / total_value`;
- `v_i q_i free_i / total_value`.

For `n=100`, this is small. For `n=692`, this is still manageable for linear/ridge models.

---

## Frozen neural encoder option

If using an encoder, input the full Markov information. Do not hide anything.

For a grid encoder, use a tensor:

```text
H x W x C
```

with channels:

- protected `x`;
- developed `d`;
- controller action `a`;
- nature action `k`;
- value `v`;
- cost `c`;
- threat index `TI`;
- current development probability `p`;
- frontier pressure `q`;
- cluster one-hot channels;
- row coordinate;
- column coordinate.

Append global scalars:

- `t/T`;
- current budget / total cost;
- remaining budget / total cost;
- `lambda_uncertainty`;
- `gamma`.

A CNN is natural for the `10x10` toy grid. A GNN is cleaner for arbitrary parcel graphs.

Once pretrained, freeze the encoder and use:

```text
phi(s,a,k) = frozen_encoder(s,a,k)
Q_hat = theta^T phi
```

---

## Policy evaluation and policy improvement

### Local matrix game

For a queried state `s`:

1. generate `A_cand(s)`;
2. generate `K_cand(s)`;
3. compute matrix:

```text
Q[j,l] = theta^T phi(s, A_cand[j], K_cand[l])
```

4. solve:

```text
max_pi min_omega pi^T Q omega
```

where:

- `pi` is a simplex over controller candidates;
- `omega` is a simplex over nature candidates.

Use `scipy.optimize.linprog` or another stable simplex/matrix-game solver.

### Infinite state space convention

Do not precompute policies for all states.

Policies are generated on demand:

```text
s -> candidate sets -> Q matrix -> pi(.|s), omega(.|s)
```

But during evaluation, freeze the policy rule/oracle.

### Freezing rule

At outer iteration `n`, old robust Q parameters define the current controller policy:

```text
pi_n(.|s) = matrix_game_policy(s; old_theta)
```

During evaluation of `pi_n`:

- compute `pi_n(.|s)` on the fly;
- but do not update it using the new theta currently being learned;
- nature policy `omega_m(.|s)` is updated between Algorithm-1 iterations, not inside the inner TD fit.

Algorithm structure:

```text
outer iteration n:
    freeze controller policy oracle pi_n

    for m in nature dual-averaging/evaluation loop:
        freeze omega_m
        learn theta_m for Q_{pi_n}^{omega_m}
        update omega_{m+1} using theta_0,...,theta_m and fixed pi_n

    average theta_m -> robust Q estimate
    use robust Q estimate to define pi_{n+1}
```

---

## Meaning of H and omega update

Nature policy at state `s` is a vector over candidate kernels/actions:

```text
omega(k | s)
```

In the dual-averaging update, `v` is a candidate new nature simplex vector.

`H(s,v)` is scalar:

```text
H(s,v) = sum_a sum_k v_k * pi(a|s) * Q(s,a,k)
```

Equivalently:

```text
g_k(s) = sum_a pi(a|s) Q(s,a,k)
H(s,v) = v^T g(s)
```

So `v` has dimension `|K_cand|`, but `H` returns one number.

---

## Span / action-reduction diagnostics

The mentor asked about comparing the reduced action set to the original action space.

Do **not** compare raw binary spans only. The raw binary action span is usually uninformative: a few hundred binary vectors can already span all of `R^n`, even if the actions are strategically bad.

Instead compare **Q-relevant action-effect embeddings**.

For controller actions, define `z_C(s,a)` using value/cost/threat/frontier/cluster/action-effect features.

For nature actions, define `z_N(s,k)` using value/threat/frontier/likelihood/cluster/action-effect features.

For each state:

1. generate a large diverse reference pool, e.g. 50k actions;
2. generate the heuristic candidate pool, e.g. 1000 actions;
3. embed both pools into action-effect space;
4. compute an orthonormal basis for the candidate span;
5. compute projection errors of reference vectors onto candidate span:

```text
err(z) = ||z - U_small U_small^T z|| / ||z||
```

Report median, 90th percentile, and 95th percentile errors.

Also compute principal angles between reference and candidate subspaces if useful.

Use this diagnostic separately for controller and nature.

Also evaluate an optimality-gap diagnostic:

```text
max over reference actions Q - max over candidate actions Q
```

or the corresponding min for nature.

Span coverage is not the same as optimal-action coverage. Track both.

---

## Notebook requirements

The main notebook should run end-to-end.

Required initial plots:

1. value heatmap;
2. threat-index heatmap;
3. cost heatmap.

Each plot must:

- show the `10x10` grid;
- include a colorbar;
- draw bold black cluster borders.

Required interactive episode cell:

- runs one episode from `t=1` to `T`;
- at each step:
  - display current grid;
  - print protected value, lost value, free value, budget;
  - wait for Enter;
  - generate controller candidates;
  - solve/select controller action;
  - apply protection;
  - display grid after protection;
  - wait for Enter;
  - generate nature candidates;
  - solve/select nature action;
  - apply development;
  - display grid after nature;
  - print likelihood slack, newly protected value, newly developed value;
  - wait for Enter.

Grid status colors:

- free;
- protected;
- developed/lost;
- newly protected;
- newly developed.

Always draw cluster borders.

---

## Training outputs

Save:

- config;
- trained theta;
- optionally frozen encoder metadata;
- training curves;
- rollout summaries;
- protected/lost/free value plots;
- action-coverage diagnostics.

Suggested files:

- `biodiv_robust_mdp.ipynb`;
- `run_biodiv_train.py`;
- `run_clariden.sh`;
- `src/` package with clean modules.

---

## Cluster run guidance

Provide a non-interactive script similar to:

```bash
#!/usr/bin/env bash
set -euo pipefail

cd ~/qa-gym
uenv start pytorch/v2.9.1:v2 --view=default
source .venv/bin/activate

which python
python --version
nvidia-smi || true

python run_biodiv_train.py \
  --lambda-uncertainty 0.75 \
  --samples-per-iter 2000 \
  --outer-iters 10 \
  --n-controller-candidates 500 \
  --n-nature-candidates 500
```

Do not include vLLM serving commands. This is not an LLM-serving task.

---

## Sanity checks

Add tests/assertions for:

- cluster layout is correct;
- neighbors are same-cluster only;
- probabilities are clipped and finite;
- log-likelihood is finite;
- controller actions respect budget;
- controller cannot protect developed parcels;
- nature cannot develop protected parcels;
- nature actions satisfy likelihood feasibility unless explicitly marked infeasible;
- candidate sets include zero action;
- candidate actions are deduplicated;
- Q matrix has expected shape;
- matrix-game policies sum to 1 and are nonnegative;
- encoder output is frozen and normalized if using encoder.

---

## Things to avoid

Do not:

- enumerate `2^n` actions;
- use raw likelihood products instead of logs;
- assign infeasible actions Q=0 instead of masking/excluding them;
- update a neural encoder inside the linear-head robust-MDP run;
- recompute controller policy from the theta currently being trained in the same evaluation run;
- rely only on aggregate features if parcel-level identity matters;
- claim theoretical guarantees for nonlinear end-to-end deep Q training under the linear-feature paper assumptions.

---

## Coding style

- Keep code simple and readable.
- Prefer `numpy`, `scipy`, `matplotlib`, `sklearn.linear_model.Ridge`.
- `torch` is optional, mainly for encoder experiments.
- Use dataclasses for `Config` and `State`.
- Keep functions small.
- Use clear names:
  - `compute_development_probabilities`
  - `log_likelihood`
  - `generate_controller_candidates`
  - `generate_nature_candidates`
  - `features_phi`
  - `solve_matrix_game`
  - `rollout_episode`
  - `fit_linear_q`
  - `action_span_diagnostics`

