"""
Dataset module for the SME project.
Defines SimulationDataset for managing simulation data and provides a helper function
to create parallel DataLoaders for improved training throughput.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class SimulationDataset(Dataset):
    def __init__(self, phi_data: np.ndarray, Y_data: np.ndarray):
        """
        Args:
            phi_data: NumPy array of parameter samples.
            Y_data: NumPy array of corresponding simulated data.
        """
        self.phi_data = torch.tensor(phi_data, dtype=torch.float32)
        self.Y_data = torch.tensor(Y_data, dtype=torch.float32)

    def __len__(self):
        return self.Y_data.shape[0]

    def __getitem__(self, idx):
        return self.phi_data[idx], self.Y_data[idx]


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4):
    """
    Create a DataLoader with parallel workers to improve data loading performance.

    Args:
        dataset: A PyTorch Dataset.
        batch_size: Batch size for loading.
        shuffle: Whether to shuffle the data.
        num_workers: Number of subprocesses to use for data loading.

    Returns:
        A DataLoader instance.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)