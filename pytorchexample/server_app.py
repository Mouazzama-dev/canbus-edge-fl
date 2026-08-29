"""server_app.py: Aggregation-strategy comparison for CAN-bus federated IDS.

Same model and data across all strategies; only the aggregation strategy
changes. Per-round global metrics are saved to results/<strategy>.json.
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
)

from pytorchexample.task import Net, load_centralized_dataset, test

app = ServerApp()

# Where to save results (server runs from ~/.flwr/apps/..., so use an absolute
# path back into the project; override with CANBUS_RESULTS if the project moves).
RESULTS_DIR = os.environ.get(
    "CANBUS_RESULTS",
    "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch/results",
)

# Accumulates {round, accuracy, loss} across this run for later plotting.
HISTORY = []


def make_strategy(name, fraction_evaluate, proximal_mu, server_lr):
    """Build a Flower strategy by name (fair comparison: only this changes)."""
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
    raise ValueError(f"Unknown strategy: {name}")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run the selected strategy for the configured number of rounds."""
    fraction_evaluate = context.run_config["fraction-evaluate"]
    num_rounds = context.run_config["num-server-rounds"]
    lr = context.run_config["learning-rate"]
    strategy_name = context.run_config["strategy"]
    proximal_mu = context.run_config["proximal-mu"]
    server_lr = context.run_config["server-lr"]

    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = make_strategy(strategy_name, fraction_evaluate, proximal_mu, server_lr)

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Save per-round history for the comparison plot.
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, f"{strategy_name}.json")
    with open(out_path, "w") as f:
        json.dump({"strategy": strategy_name, "history": HISTORY}, f, indent=2)
    print(f"\nSaved results to {out_path}")

    if context.run_config["save-model"]:
        print("Saving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, f"final_model_{strategy_name}.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate the global model on the held-out 5-class test set."""
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    test_dataloader = load_centralized_dataset()
    test_loss, test_acc = test(model, test_dataloader, device)

    HISTORY.append({"round": server_round, "accuracy": test_acc, "loss": test_loss})
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})