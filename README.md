# CAN Bus Edge Federated Learning

Federated learning (FL) for **in-vehicle CAN bus intrusion detection** on resource-constrained edge nodes. The project trains a lightweight PyTorch MLP with **[Flower](https://flower.ai/)** using **FedAvg**, on tabular frames from the **[HCRL Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)** (Korea University).

Each simulated client holds a disjoint slice of CAN traffic; the server aggregates model weights without centralizing raw bus logs—suited to edge and privacy-sensitive automotive scenarios.

## Features

- **Binary classification**: normal driving (`R`) vs attack payload (`T`)
- **10-dimensional features**: `CAN_ID`, `DLC`, and data bytes `D0`–`D7` (hex parsed safely)
- **Small MLP** (32 → 16 → 2) for edge-friendly inference
- **FedAvg** with configurable rounds, learning rate, and local epochs
- **Global evaluation** on a held-out server subset
- **`plot_results.py`** for accuracy/loss curves after a run

## Architecture

```text
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Client 0       │     │  Client 1       │     │  Client N       │
│  Local CAN slice│     │  Local CAN slice│     │  Local CAN slice│
│  Train MLP      │     │  Train MLP      │     │  Train MLP      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌────────────────────────┐
                    │  Flower Server (FedAvg) │
                    │  Aggregate weights      │
                    │  Global eval on subset  │
                    └────────────────────────┘
```

## Repository layout

```text
canbus-edge-fl/
├── archive/                      # Original HCRL CSV/TXT sources (reference)
│   ├── DoS_dataset.csv
│   ├── Fuzzy_dataset.csv
│   ├── gear_dataset.csv
│   ├── RPM_dataset.csv
│   └── normal_run_data.txt
└── quickstart-pytorch/             # Flower + PyTorch FL application
    ├── pyproject.toml              # Dependencies and Flower app config
    ├── canbus_data.csv             # Training CSV (not in git; see below)
    ├── plot_results.py             # Plot global metrics after training
    └── pytorchexample/
        ├── task.py                 # Model, data loading, train/test
        ├── client_app.py           # Flower ClientApp (train/evaluate)
        └── server_app.py           # Flower ServerApp (FedAvg + global eval)
```

## Dataset

### Format

`canbus_data.csv` is comma-separated with **no header**, 12 columns:

| Column     | Description                         |
|------------|-------------------------------------|
| Timestamp  | Capture time                        |
| CAN_ID     | Frame ID (hex string)               |
| DLC        | Data length                         |
| D0–D7      | Payload bytes (hex)                 |
| Label      | `R` = normal, `T` = attack (binary) |

Example row:

```text
1478198376.389427,0316,8,05,21,68,09,21,21,00,6f,R
```

### Obtaining `canbus_data.csv`

1. Download the [Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset) from HCRL.
2. Copy or merge attack/normal CSVs into `quickstart-pytorch/canbus_data.csv` using the layout above (files under `archive/` follow the same format, e.g. `DoS_dataset.csv`).
3. By default, `task.py` loads the **first 50,000 rows** for memory stability on ~8 GB RAM.

The file is listed in `.gitignore` and is **not** committed to the repository.

### Citation

If you use the HCRL dataset, cite the original work from the [dataset page](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset).

## Requirements

- Python 3.10+ (recommended)
- pip
- Optional: NVIDIA GPU (CUDA) for faster local training
- ~8 GB RAM for the default 50k-row cap

## Installation

```bash
cd quickstart-pytorch
pip install -e .
```

Dependencies are declared in `pyproject.toml` (Flower simulation, PyTorch, scikit-learn, pandas, etc.).

## Running federated training

From `quickstart-pytorch/`:

```bash
flwr run . --stream
```

Override Flower run config (see `pyproject.toml` → `[tool.flwr.app.config]`):

```bash
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05 local-epochs=2" --stream
```

### Default configuration

| Parameter            | Default | Description                     |
|----------------------|---------|---------------------------------|
| `num-server-rounds`  | 3       | Federated aggregation rounds    |
| `local-epochs`       | 1       | Local training epochs per round |
| `learning-rate`      | 0.1     | Adam learning rate              |
| `batch-size`         | 32      | Minibatch size                  |
| `fraction-evaluate`  | 0.5     | Fraction of clients evaluated   |
| `save-model`         | false   | Save `final_model.pt` if true   |

### Save the global model

```bash
flwr run . --run-config "save-model=true" --stream
```

## Visualizing results

After a run, update `plot_results.py` with your round-wise `accuracy` and `loss`, then:

```bash
cd quickstart-pytorch
python plot_results.py
```

This writes `federated_performance.png`.

Example metrics from a 3-round simulation (illustrative):

| Round | Global accuracy | Global loss |
|-------|-----------------|-------------|
| 0     | 0.577           | 0.691       |
| 1     | 0.902           | 0.226       |
| 2     | 0.953           | 0.118       |
| 3     | 0.962           | 0.122       |

## How it works

| Component       | Role |
|-----------------|------|
| `task.py`       | Loads and preprocesses CSV, partitions data per client, defines `Net`, `train()`, `test()` |
| `client_app.py` | Flower `@app.train` / `@app.evaluate` on each client's partition |
| `server_app.py` | FedAvg orchestration and `global_evaluate()` on the last 2000 samples |

Preprocessing steps in `task.py`:

- Parse hex fields (`CAN_ID`, `D0`–`D7`) with a safe fallback for non-hex values
- Map labels: `T` → 1 (attack), anything else (e.g. `R`) → 0 (normal)
- `StandardScaler` on the 10 feature columns
- Sequential partitioning by `partition_id` across simulation clients; 80/20 train/test split per partition

## Known limitations

- **Dataset path**: `task.py` expects `canbus_data.csv` next to `pyproject.toml` under `quickstart-pytorch/`. If you see a `FileNotFoundError`, place the file there or adjust the path in `task.py`.
- **50k row cap** in `_lazy_load_csv()` trades full-dataset training for RAM use on laptops.
- **`flwr-datasets[vision]`** remains in `pyproject.toml` from the upstream Flower quickstart but is unused by the CAN bus pipeline.

## Deployment

For multi-machine FL (beyond simulation), see Flower's guides:

- [Deployment Engine](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)
- [TLS connections](https://flower.ai/docs/framework/how-to-enable-tls-connections.html)
- [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html)

## License

See `quickstart-pytorch/LICENSE` (upstream Flower quickstart license). Respect HCRL dataset terms separately.

## Acknowledgments

- [Flower](https://flower.ai/) federated learning framework
- [HCRL Car-Hacking Dataset](https://ocslab.hksecurity.net/Datasets/car-hacking-dataset)
- Adapted from [flwrlabs/quickstart-pytorch](https://github.com/flwrlabs/quickstart-pytorch) for automotive CAN intrusion detection
