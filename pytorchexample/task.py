"""task.py: Flower / PyTorch app for Connected Cars (CAN-Bus) Cybersecurity."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from datasets import load_dataset
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# --- 1. LIGHTWEIGHT MLP MODEL FOR EDGE NODES ---
class Net(nn.Module):
    """Optimized Tabular MLP for resource-constrained Vehicle Telemetry classification."""
    def __init__(self, input_dim=8, num_classes=5):
        super(Net, self).__init__()
        # Tight layers to prevent laptop hanging/OOM
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset Reference

# --- 2. DATA LOAD & RE-FORMATTING ENGINE ---
def process_hf_batch(partition):
    """Helper to convert HuggingFace Arrow dataset format to clean PyTorch Tensors safely."""
    # CAN-Bus packets fields mapping (Arbitration ID + Data Bytes)
    features = ['can_id', 'data_b0', 'data_b1', 'data_b2', 'data_b3', 'data_b4', 'data_b5', 'data_b6']
    
    # Missing columns handle karne ka bulletproof logic
    X_list = []
    for col in features:
        if col in partition.column_names:
            X_list.append(np.array(partition[col], dtype=np.float32))
        else:
            X_list.append(np.zeros(len(partition), dtype=np.float32))
            
    X = np.stack(X_list, axis=1)
    
    # Target label extraction (0: Normal, 1: DoS, 2: Spoofing, etc.)
    if 'label' in partition.column_names:
        y = np.array(partition['label'], dtype=np.int64)
    else:
        y = np.zeros(len(partition), dtype=np.int64)
        
    return torch.tensor(X), torch.tensor(y)


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load and partition customized CAN-Bus automotive cybersecurity data stream."""
    global fds
    if fds is None:
        # IidPartitioner automatically maps network boundaries across edge devices
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="bwandowando/can-bus-intrusion-dataset",
            partitioners={"train": partitioner},
        )
    
    # Load specific isolated partition for the client node
    partition = fds.load_partition(partition_id)
    
    # Limit rows to 15,000 per partition to keep dual-core CPU processing makhkhan!
    max_rows = min(len(partition), 15000)
    partition = partition.select(range(max_rows))
    
    # 80% Local Train, 20% Local Validation
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # PyTorch Tensors generation
    X_train, y_train = process_hf_batch(partition_train_test["train"])
    X_test, y_test = process_hf_batch(partition_train_test["test"])
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader


def load_centralized_dataset():
    """Server-side Global Evaluation dataset generator."""
    # Pull sample of the main stream dataset for the central SuperLink audit
    global_test = load_dataset("bwandowando/can-bus-intrusion-dataset", split="train")
    # Small validation slice to avoid high RAM spike on the central node
    global_test = global_test.select(range(min(len(global_test), 3000)))
    
    X_val, y_val = process_hf_batch(global_test)
    val_dataset = TensorDataset(X_val, y_val)
    
    return DataLoader(val_dataset, batch_size=128)


# --- 3. OPTIMIZED TRAINING LOOP WITH RAM REFRESH ---
def train(net, trainloader, epochs, lr, device):
    """Train the automotive model with explicit execution controls to avoid laptop freezing."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr) # Adam for fast categorical convergence
    
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
    # RAM Footprint Clean-up Tactic (Crucial for 8GB hardware layout)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        
    avg_trainloss = running_loss / (epochs * len(trainloader))
    return avg_trainloss


def test(net, testloader, device):
    """Validate vehicular communication security compliance matrix."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy