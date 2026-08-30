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
    FedTrimmedAvg,
)
import numpy as np

from pytorchexample.task import Net, load_centralized_dataset, test
from flwr.serverapp.strategy.strategy_utils import aggregate_arrayrecords

app = ServerApp()

RESULTS_DIR = os.environ.get(
    "CANBUS_RESULTS",
    "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch/results",
)

HISTORY = []

NUM_CLIENTS = 4  # our fixed deployment (4 SuperNodes)

class FedProxMedian(FedMedian):
    """Hybrid (our contribution): coordinate-median aggregation (robust to
    poisoning) plus a FedProx proximal term sent to clients (robust to non-IID).
    No single existing strategy handles both; this composes the two.
    """

    def __init__(self, *args, proximal_mu=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu

    def configure_train(self, server_round, arrays, config, grid):
        config["proximal-mu"] = self.proximal_mu
        return super().configure_train(server_round, arrays, config, grid)

class FedProxTrimmedAvg(FedTrimmedAvg):
    """Hybrid v2 (our contribution): trimmed-mean aggregation (drops the
    extreme/poisoned values per coordinate but still averages the rest) plus a
    FedProx proximal term. Aims to keep FedProx's non-IID recovery while being
    robust to poisoning, which the median-based hybrid could not do.
    """

    def __init__(self, *args, proximal_mu=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu

    def configure_train(self, server_round, arrays, config, grid):
        config["proximal-mu"] = self.proximal_mu
        return super().configure_train(server_round, arrays, config, grid)

class FedProxFilterMean(FedAvg):
    """Hybrid (our contribution): reject abnormal-norm client updates (a poisoned
    client sends a huge-norm update), then FedProx-mean the survivors. Because it
    AVERAGES (not medians) the surviving honest updates, it keeps FedProx's non-IID
    recovery while rejecting the attacker - the only strategy in our study robust
    to BOTH extreme non-IID and poisoning.
    """

    def __init__(self, *args, proximal_mu=0.0, norm_factor=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.proximal_mu = proximal_mu
        self.norm_factor = norm_factor

    def configure_train(self, server_round, arrays, config, grid):
        config["proximal-mu"] = self.proximal_mu
        return super().configure_train(server_round, arrays, config, grid)

    def aggregate_train(self, server_round, replies):
        valid_replies, _ = self._check_and_log_replies(replies, is_train=True)
        valid_replies = list(valid_replies)
        if not valid_replies:
            return None, None

        # L2 norm of each client's update; reject those far above the median norm
        norms = []
        for msg in valid_replies:
            sd = msg.content["arrays"].to_torch_state_dict()
            norms.append(torch.cat([v.flatten().float() for v in sd.values()]).norm(2).item())
        median_norm = float(np.median(norms))
        kept = [msg for msg, n in zip(valid_replies, norms)
                if n <= self.norm_factor * median_norm] or valid_replies

        reply_contents = [msg.content for msg in kept]
        arrays = aggregate_arrayrecords(reply_contents, self.weighted_by_key)
        metrics = self.train_metrics_aggr_fn(reply_contents, self.weighted_by_key)
        return arrays, metrics

def make_strategy(name, fraction_evaluate, proximal_mu, server_lr,
                  noise_multiplier, clipping_norm, trim_beta, norm_factor):
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
    if name == "hybrid":
        return FedProxMedian(fraction_evaluate=fraction_evaluate, proximal_mu=proximal_mu)
    if name == "trimhybrid":
        return FedProxTrimmedAvg(
            fraction_evaluate=fraction_evaluate,
            proximal_mu=proximal_mu,
            beta=trim_beta,)
    if name == "filterhybrid":
        return FedProxFilterMean(
            fraction_evaluate=fraction_evaluate,
            proximal_mu=proximal_mu,
            norm_factor=norm_factor,
        )
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
    malicious_id = context.run_config["malicious-id"]
    trim_beta = context.run_config["trim-beta"]
    seed = context.run_config["seed"]
    norm_factor = context.run_config["norm-factor"]

    if seed >= 0:
        torch.manual_seed(seed)
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    strategy = make_strategy(
        strategy_name, fraction_evaluate, proximal_mu, server_lr,
        noise_multiplier, clipping_norm, trim_beta, norm_factor
    )
    

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # DP results -> results/dp/ keyed by noise; strategies -> results/<mode>/
        # route results: poison run -> results/poison/, DP -> results/dp/, else results/<mode>/
    run_tag = context.run_config["run-tag"]
    if run_tag:
        out_dir = os.path.join(RESULTS_DIR, "contrib")
        tag = run_tag
    elif malicious_id >= 0:
        out_dir = os.path.join(RESULTS_DIR, "poison")
        tag = strategy_name
    elif strategy_name == "dp":
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