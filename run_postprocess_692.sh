#!/bin/bash
#SBATCH --job-name=biodiv-pp692
#SBATCH --account=infra01
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=outputs/slurm/pp692-%j.out
#SBATCH --error=outputs/slurm/pp692-%j.err
#SBATCH --uenv=pytorch/v2.9.1:v2
#SBATCH --view=default

set -euo pipefail
cd ~/robust-mdp
source ~/qa-gym/.venv/bin/activate
which python; python --version; nvidia-smi || true

RUN=outputs/slurm/biodiv-2483604

# Paper comparison (ours vs StaticApprox vs Knapsack) at 692 parcels. The greedy
# adversary is O(n^2) per step, so this is heavy -> compute node, not login node.
echo "===== run_compare ($RUN) ====="
python run_compare.py --run-dir "$RUN" --n-stochastic 40 --n-worstcase 20

# Encoder feature interpretation (Q-fit vs engineered + effective rank)
echo "===== analyze_features ($RUN) ====="
PYTHONPATH=. python tools/analyze_features.py --run-dir "$RUN"

echo "===== postprocess done ====="
