#!/usr/bin/env bash
# Usage: bash run_strategies.sh [extreme|dirichlet]
MODE="${1:-extreme}"
cd "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch"

for s in fedavg fedprox fedadam fedyogi fedavgm fedmedian krum; do
  echo "=================== Running: $s ($MODE) ==================="
  flwr run . local-deployment --run-config "strategy=\"$s\" partition-mode=\"$MODE\"" --stream
done

echo "Done. Results in results/$MODE/"