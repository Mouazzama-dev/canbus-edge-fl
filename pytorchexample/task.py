"""task.py: Flower / PyTorch app using Real HCRL Car-Hacking Dataset."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
import os
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- 1. LIGHTWEIGHT MLP MODEL FOR EDGE NODES ---
class Net(nn.Module):
    """Optimized Tabular MLP for resource-constrained Vehicle Telemetry classification."""
    def __init__(self, input_dim=10, num_classes=2):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Global variables to cache data allocation safely in system memory
_GLOBAL_X = None
_GLOBAL_Y = None
_INPUT_DIM = 10
_NUM_CLASSES = 2

# --- Safe Hex to Decimal Parsing Block ---
def safe_hex_parse(val):
    """Converts hex strings to integers safely. Returns 0 if invalid or string matches flags like 'R'."""
    if pd.isnull(val):
        return 0
    clean_str = str(val).strip()
    try:
        return int(clean_str, 16)
    except ValueError:
        # Fallback if character strings like 'R' or 'T' bleed into data payload metrics
        return 0


def _lazy_load_csv():
    """Helper function to load and preprocess HCRL CSV with error-tolerant parsers."""
    global _GLOBAL_X, _GLOBAL_Y, _INPUT_DIM, _NUM_CLASSES
    if _GLOBAL_X is not None:
        return

    csv_path = os.path.join(os.path.dirname(__file__), "..", "canbus_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"⚠️ Please place HCRL dataset as '{csv_path}' in your directory!")

    print(f"📖 Loading real HCRL CAN-Bus data from {csv_path}...")
    
    # Mapping complete 12 column layout of Korea University framework
    col_names = ['Timestamp', 'CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'Label']
    
    # 50k rows allocation for maximum stability on 8GB RAM architecture
    df = pd.read_csv(csv_path, names=col_names, nrows=50000, header=None)
    
    # Processing metrics safely via functional map pipelines
    df['CAN_ID'] = df['CAN_ID'].apply(safe_hex_parse)
    
    for col in ['D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']:
        df[col] = df[col].apply(safe_hex_parse)
        
    # Standardize Labels: 'T' (Attack Payload) -> 1, Anything else ('R' / Normal driving) -> 0
    df['Label'] = df['Label'].apply(lambda x: 1 if str(x).strip() == 'T' else 0)

    feature_cols = ['CAN_ID', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
    
    X = df[feature_cols].values.astype(np.float32)
    y = df['Label'].values.astype(np.int64)
    
    # Scale features for neural network stability
    X = StandardScaler().fit_transform(X)
    
    _GLOBAL_X = torch.tensor(X, dtype=torch.float32)
    _GLOBAL_Y = torch.tensor(y, dtype=torch.int64)
    _INPUT_DIM = X.shape[1]
    _NUM_CLASSES = 2
    print(f"✅ HCRL Data Loaded Successfully. Features: {_INPUT_DIM}, Classes: 2")


# --- 2. FLOWER DATA LOAD INTERFACE ---
def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Loads a unique chunk of the preprocessed dataset for a specific SuperNode."""
    _lazy_load_csv()
    
    total_samples = len(_GLOBAL_X)
    chunk_size = total_samples // num_partitions
    
    start_idx = partition_id * chunk_size
    end_idx = start_idx + chunk_size
    
    X_chunk = _GLOBAL_X[start_idx:end_idx]
    y_chunk = _GLOBAL_Y[start_idx:end_idx]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_chunk, y_chunk, test_size=0.2, random_state=42
    )
    
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True), DataLoader(test_dataset, batch_size=batch_size)


def load_centralized_dataset():
    """Server-side Global Evaluation dataset generator without network leaks."""
    _lazy_load_csv()
    # Centralized validation subset tracked directly via ServerApp
    val_dataset = TensorDataset(_GLOBAL_X[-2000:], _GLOBAL_Y[-2000:])
    return DataLoader(val_dataset, batch_size=128)


# --- 3. OPTIMIZED TRAINING LOOP WITH RAM REFRESH ---
def train(net, trainloader, epochs, lr, device):
    """Train the automotive model with explicit execution controls to avoid laptop freezing."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
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
            
    # Garbage collection block to maintain low RAM signature
    gc.collect()
    return running_loss / (epochs * len(trainloader))


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
    return loss / len(testloader), accuracy