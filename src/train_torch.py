import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Paths to ML-ready data
ML_DIR = "../data/ml_ready"
X = np.load(f"{ML_DIR}/ml_light_curves.npy")  # shape: (num_windows, window_size, num_features)
y = np.load(f"{ML_DIR}/ml_labels.npy")        # shape: (num_windows, window_size)

class LightCurveDataset(Dataset):
    """PyTorch Dataset for light curve windows."""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)   # Convert features to float32
        self.y = torch.tensor(y, dtype=torch.float32)   # Convert labels to float32 (for BCEWithLogitsLoss)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Create Dataset
dataset = LightCurveDataset(X, y)

# Create DataLoader for batch training
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Example: iterate over the first batch
for batch_X, batch_y in dataloader:
    print("Batch X shape:", batch_X.shape)  # (batch_size, window_size, num_features)
    print("Batch y shape:", batch_y.shape)  # (batch_size, window_size)
    break
