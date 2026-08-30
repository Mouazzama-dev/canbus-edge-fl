#!/usr/bin/env bash
# DP privacy-utility sweep on the moderate (dirichlet) regime.
cd "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch"
for nm in 0.0 0.25 0.5 1.0 2.0 4.0; do
  echo "=================== DP noise_multiplier=$nm ==================="
  flwr run . local-deployment --run-config "strategy=\"dp\" partition-mode=\"dirichlet\" noise-multiplier=$nm clipping-norm=1.0" --stream
done
echo "Done. Results in results/dp/"