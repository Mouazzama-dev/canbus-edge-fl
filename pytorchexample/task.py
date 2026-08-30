"""task.py: Flower / PyTorch app for 5-class CAN-bus intrusion detection.

Classes: 0=Normal, 1=DoS, 2=Fuzzy, 3=Gear-spoof, 4=RPM-spoof.
Data is produced by prepare_data.py (canbus_5class.npz).
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

NUM_CLASSES = 5
CLASS_NAMES = ["Normal", "DoS", "Fuzzy", "Gear", "RPM"]


# --- 1. LIGHTWEIGHT MLP MODEL FOR EDGE NODES ---
class Net(nn.Module):
    """Small tabular MLP for resource-constrained CAN-bus classification."""

    def __init__(self, input_dim=10, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# --- 2. DATA LOADING (cached once per process) ---
_TRAIN_X = None
_TRAIN_Y = None
_TEST = None


def _load_all():
    """Load the prepared 5-class dataset and carve out a global test set."""
    global _TRAIN_X, _TRAIN_Y, _TEST
    if _TRAIN_X is not None:
        return

    path = os.environ.get(
        "CANBUS_NPZ",
        "/home/user/Documents/Masters/Edge Computing/canbus-edge-fl/quickstart-pytorch/canbus_5class.npz",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run 'python prepare_data.py' first.")

    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.int64)

    # Hold out a stratified global test set: 2000 rows per class (all 5 classes).
    gen = torch.Generator().manual_seed(42)
    test_idx, train_idx = [], []
    for c in range(NUM_CLASSES):
        idx_c = (y == c).nonzero(as_tuple=True)[0]
        perm = idx_c[torch.randperm(len(idx_c), generator=gen)]
        test_idx.append(perm[:2000])
        train_idx.append(perm[2000:])

    _TEST = (X[torch.cat(test_idx)], y[torch.cat(test_idx)])
    _TRAIN_X = X[torch.cat(train_idx)]
    _TRAIN_Y = y[torch.cat(train_idx)]


def _extreme_indices(partition_id, num_partitions):
    """Extreme non-IID: client gets Normal + exactly ONE attack type."""
    attack_class = 1 + (partition_id % (NUM_CLASSES - 1))
    normal_idx = (_TRAIN_Y == 0).nonzero(as_tuple=True)[0]
    normal_share = normal_idx[partition_id::num_partitions]
    attack_idx = (_TRAIN_Y == attack_class).nonzero(as_tuple=True)[0]
    return torch.cat([normal_share, attack_idx])


def _dirichlet_indices(partition_id, num_partitions, alpha):
    """Moderate non-IID: split each class across clients via Dirichlet(alpha).

    Uses a fixed seed so every client process computes the same partition and
    then takes its own slice (deterministic, no overlap, full coverage).
    """
    rng = np.random.default_rng(1234)
    parts = [[] for _ in range(num_partitions)]
    for c in range(NUM_CLASSES):
        idx_c = (_TRAIN_Y == c).nonzero(as_tuple=True)[0].numpy()
        rng.shuffle(idx_c)
        props = rng.dirichlet([alpha] * num_partitions)
        cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
        for i, chunk in enumerate(np.split(idx_c, cuts)):
            parts[i].append(torch.tensor(chunk, dtype=torch.long))
    return torch.cat(parts[partition_id])


def load_data(partition_id, num_partitions, batch_size, partition_mode="extreme", alpha=0.3):
    """Return (trainloader, valloader) for one client under the chosen skew."""
    _load_all()

    if partition_mode == "dirichlet":
        idx = _dirichlet_indices(partition_id, num_partitions, alpha)
    else:
        idx = _extreme_indices(partition_id, num_partitions)

    X_c, y_c = _TRAIN_X[idx], _TRAIN_Y[idx]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_c, y_c, test_size=0.2, random_state=42, stratify=y_c
        )
    except ValueError:
        # a class is too small to stratify -> plain split
        X_train, X_test, y_train, y_test = train_test_split(
            X_c, y_c, test_size=0.2, random_state=42
        )

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size),
    )


def load_centralized_dataset():
    """Global test set for server-side evaluation (all 5 classes)."""
    _load_all()
    X_test, y_test = _TEST
    return DataLoader(TensorDataset(X_test, y_test), batch_size=256)


# --- 3. TRAIN / TEST LOOPS ---
def train(net, trainloader, epochs, lr, device, proximal_mu=0.0):
    """Train on one client's local data.

    If proximal_mu > 0 (FedProx), add (mu/2)*||w - w_global||^2 to keep the
    local model close to the global model received this round.
    """
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    global_params = [p.detach().clone() for p in net.parameters()]

    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for features, labels in trainloader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), labels)
            if proximal_mu > 0.0:
                prox = 0.0
                for p, gp in zip(net.parameters(), global_params):
                    prox = prox + (p - gp).norm(2) ** 2
                loss = loss + (proximal_mu / 2.0) * prox
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    gc.collect()
    return running_loss / (epochs * len(trainloader))


def test(net, testloader, device):
    """Evaluate the model: returns (average loss, accuracy)."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for features, labels in testloader:
            features, labels = features.to(device), labels.to(device)
            outputs = net(features)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    return loss / len(testloader), accuracy