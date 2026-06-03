# CAN Bus FL Application

Flower + PyTorch federated learning app for HCRL CAN bus intrusion detection.

**Full documentation** (dataset setup, architecture, configuration, deployment): see the [repository README](../README.md).

## Quick start

```bash
# Place canbus_data.csv in this directory (see parent README)
pip install -e .
flwr run . --stream
```
