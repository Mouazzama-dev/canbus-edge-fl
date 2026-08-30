#!/usr/bin/env bash
# Multi-seed sweep for the contribution matrix (4 methods x 2 scenarios x 3 seeds).
cd "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch"
for seed in 0 1 2; do
  for m in fedavg fedprox fedmedian filterhybrid; do
    echo "=== $m | extreme | seed $seed ==="
    flwr run . local-deployment --run-config "strategy=\"$m\" partition-mode=\"extreme\" seed=$seed run-tag=\"${m}_extreme_s${seed}\"" --stream
    echo "=== $m | poison | seed $seed ==="
    flwr run . local-deployment --run-config "strategy=\"$m\" partition-mode=\"dirichlet\" malicious-id=3 seed=$seed run-tag=\"${m}_poison_s${seed}\"" --stream
  done
done
echo "Done. results/contrib/"