#!/usr/bin/env bash
# Validate the hybrid under both stress tests, plus FedProx under attack (baseline).
cd "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch"

echo "=================== hybrid: extreme non-IID (no attack) ==================="
flwr run . local-deployment --run-config "strategy=\"hybrid\" partition-mode=\"extreme\"" --stream

echo "=================== hybrid: moderate + poisoning ==================="
flwr run . local-deployment --run-config "strategy=\"hybrid\" partition-mode=\"dirichlet\" malicious-id=3" --stream

echo "=================== fedprox: moderate + poisoning (expected to fail) ==================="
flwr run . local-deployment --run-config "strategy=\"fedprox\" partition-mode=\"dirichlet\" malicious-id=3" --stream

echo "Done."