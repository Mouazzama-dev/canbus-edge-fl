#!/usr/bin/env bash
# Run the remaining strategies against the running local-deployment.
cd "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch"

for s in fedyogi fedavgm krum; do
  echo "=================== Running: $s ==================="
  flwr run . local-deployment --run-config "strategy=\"$s\"" --stream
done

echo "Done. Results in results/"