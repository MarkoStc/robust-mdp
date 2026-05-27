#!/bin/bash
#SBATCH --job-name=biodiv-robust
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
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

# New defaults (chosen 2026-05-19):
#   * Gaussian-clustered values: two high-value hotspots so the game has
#     spatial structure the controller should learn to defend.
#   * budget_per_year=2.0 (was 5.0): controller can no longer trivially blanket
#     the grid — it must pick *where* to spend.
#   * lambda=0.20 (was 0.75): the multiplicative neighbor probability formula
#     keeps p small until a cluster of developments seeds neighbor pressure;
#     0.20 lets nature bootstrap and develop ~60 parcels worst-case, so the
#     controller faces a meaningful adversary and should learn to cluster
#     protections around developed parcels.
#   * --use-encoder: phi(s,a,k) comes from a pretrained-then-frozen CNN; the
#     robust-MDP linear head theta is the only weight Algorithm 1 updates.
python -u run_biodiv_train.py \
  --lambda-uncertainty 0.20 \
  --budget-per-year 2.0 \
  --samples-per-iter 2000 \
  --outer-iters 5 \
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
