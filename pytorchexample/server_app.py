"""server_app.py: Aggregation-strategy comparison plus differential privacy
for CAN-bus federated IDS. Same model and data across runs; only the
aggregation (or the DP wrapper) changes. Per-round metrics saved to results/.
"""

import json
import os

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import (
    FedAvg,
    FedProx,
    FedAdam,
    FedYogi,
    FedAvgM,
    FedMedian,
    Krum,
    DifferentialPrivacyServerSideFixedClipping,
)

from pytorchexample.task import Net, load_centralized_dataset, test

app = ServerApp()

RESULTS_DIR = os.environ.get(
    "CANBUS_RESULTS",
    "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch/results",
)

HISTORY = []

NUM_CLIENTS = 4  # our fixed deployment (4 SuperNodes)


def make_strategy(name, fraction_evaluate, proximal_mu, server_lr,
                  noise_multiplier, clipping_norm):
    """Build a Flower strategy by name. 'dp' wraps FedAvg with central DP."""
    name = name.lower()
    if name == "fedavg":
        return FedAvg(fraction_evaluate=fraction_evaluate)
    if name == "fedprox":
        return FedProx(fraction_evaluate=fraction_evaluate, proximal_mu=proximal_mu)
    if name == "fedadam":
        return FedAdam(fraction_evaluate=fraction_evaluate, eta=server_lr)
    if name == "fedyogi":
        return FedYogi(fraction_evaluate=fraction_evaluate, eta=server_lr)
    if name == "fedavgm":
        return FedAvgM(fraction_evaluate=fraction_evaluate, server_momentum=0.9)
    if name == "fedmedian":
        return FedMedian(fraction_evaluate=fraction_evaluate)
    if name == "krum":
        return Krum(fraction_evaluate=fraction_evaluate)
    if name == "dp":
        base = FedAvg(fraction_evaluate=fraction_evaluate)
        return DifferentialPrivacyServerSideFixedClipping(
            base,
            noise_multiplier=noise_multiplier,
            clipping_norm=clipping_norm,
            num_sampled_clients=NUM_CLIENTS,
        )
    raise ValueError(f"Unknown strategy: {name}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    strategy_name = context.run_config["strategy"]
    proximal_mu = context.run_config["proximal-mu"]
    server_lr = context.run_config["server-lr"]
    partition_mode = context.run_config["partition-mode"]
    noise_multiplier = context.run_config["noise-multiplier"]
    clipping_norm = context.run_config["clipping-norm"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = make_strategy(
        strategy_name, fraction_evaluate, proximal_mu, server_lr,
        noise_multiplier, clipping_norm,
    )

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # DP results -> results/dp/ keyed by noise; strategies -> results/<mode>/
    if strategy_name == "dp":
        out_dir = os.path.join(RESULTS_DIR, "dp")
        tag = f"noise_{noise_multiplier}"
    else:
        out_dir = os.path.join(RESULTS_DIR, partition_mode)
        tag = strategy_name
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{tag}.json")
    with open(out_path, "w") as f:
        json.dump({"strategy": strategy_name,
                   "noise_multiplier": noise_multiplier,
                   "history": HISTORY}, f, indent=2)
    print(f"\nSaved results to {out_path}")

    if context.run_config["save-model"]:
        print("Saving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"final_model_{strategy_name}.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)
    HISTORY.append({"round": server_round, "accuracy": test_acc, "loss": test_loss})
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})