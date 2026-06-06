#!/bin/bash
#SBATCH --job-name=biodiv-692
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=outputs/slurm/biodiv-%j/slurm.out
#SBATCH --error=outputs/slurm/biodiv-%j/slurm.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail

cd ~/robust-mdp
RUN_DIR="outputs/slurm/biodiv-${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

source ~/qa-gym/.venv/bin/activate

which python
python --version
nvidia-smi || true

# === 692-parcel run (exact-nature-feasibility branch) ========================
# Matches the paper's parcel COUNT (692). We do not have the paper's real land
# dataset, so the map is synthetic: a 4x173 grid (=exactly 692 cells, no masking)
# tiled into 1x4 = 4 rectangular clusters. Budget scaled ~7x from the n=100 toy
# (2.0 -> 14.0) to keep the protected fraction comparable. lambda kept at 0.20
# for consistency with the n=100 runs (see game_balance memory).
#
# Nature feasibility is EXACT immediate-p (default on this branch) -> fully
# coherent with the paper / CLAUDE.md. Encoder: pretrained-then-frozen CNN; only
# the linear head theta is trained inside the robust-MDP loop.
#
# MEASURED timing @692 (job 2441645, 200x200 cand, exact nature, 1 GPU):
#   pretrain ~39 min; ONE outer iter (2000 samples) ~4.3 h. The first 8h run
#   only finished outer 1/5 before TIMEOUT. Account only has QOS=normal (no low
#   partition), so cap is the 12h normal partition. Halve samples to 1000
#   (~2.15 h/outer) and run 4 outers (theta converged by iter 2 at n=100).
#   Projected: 0.65h pretrain + 4*2.15h ~= 9.3h core (+~1h overhead) < 12h.
python -u run_biodiv_train.py \
  --grid-h 4 --grid-w 173 \
  --n-cluster-rows 3 --n-cluster-cols 3 \
  --lambda-uncertainty 0.20 \
  --budget-per-year 14.0 \
  --samples-per-iter 1000 \
  --outer-iters 4 \
  --inner-M 5 \
  --n-controller-candidates 200 \
  --n-nature-candidates 200 \
  --horizon-T 10 \
  --rollout-episodes 20 \
  --use-encoder \
  --encoder-embed-dim 64 \
  --encoder-pretrain-episodes 800 \
  --encoder-pretrain-epochs 40 \
  --out-dir "$RUN_DIR" \
  --run-diagnostics
