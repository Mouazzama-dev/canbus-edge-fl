#!/usr/bin/env bash
# Poisoning experiment: 1 malicious client (id 3) on the moderate regime.
# FedAvg should collapse; robust aggregators (krum, fedmedian) should resist.
cd "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch"
for s in fedavg krum fedmedian; do
  echo "=================== $s under attack ==================="
  flwr run . local-deployment --run-config "strategy=\"$s\" partition-mode=\"dirichlet\" malicious-id=3" --stream
done
echo "Done. Results in results/poison/"