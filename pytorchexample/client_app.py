"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from pytorchexample.task import Net, load_data
from pytorchexample.task import test as test_fn
from pytorchexample.task import train as train_fn

app = ClientApp()


def _partition_settings(context):
    """Read the non-IID setup from run config (mode + Dirichlet alpha)."""
    mode = context.run_config["partition-mode"]
    alpha = context.run_config["dirichlet-alpha"]
    return mode, alpha


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data. A malicious client instead sends a large
    garbage update (model-poisoning attack)."""
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    mode, alpha = _partition_settings(context)
    trainloader, _ = load_data(partition_id, num_partitions, batch_size, mode, alpha)

    malicious_id = context.run_config["malicious-id"]
    if partition_id == malicious_id:
        # Poisoning: replace the update with large random garbage weights.
        poisoned = {k: torch.randn_like(v.float()) * 5.0
                    for k, v in model.state_dict().items()}
        model.load_state_dict(poisoned)
        train_loss = 0.0
    else:
        config = msg.content["config"]
        try:
            proximal_mu = float(config["proximal-mu"])
        except KeyError:
            proximal_mu = 0.0
        train_loss = train_fn(
            model,
            trainloader,
            context.run_config["local-epochs"],
            config["lr"],
            device,
            proximal_mu=proximal_mu,
        )

    model_record = ArrayRecord(model.state_dict())
    metrics = {"train_loss": train_loss, "num-examples": len(trainloader.dataset)}
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    """Evaluate the model on local data."""
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    batch_size = context.run_config["batch-size"]
    mode, alpha = _partition_settings(context)
    _, valloader = load_data(partition_id, num_partitions, batch_size, mode, alpha)

    eval_loss, eval_acc = test_fn(model, valloader, device)

    metrics = {
        "eval_loss": eval_loss,
        "eval_acc": eval_acc,
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)