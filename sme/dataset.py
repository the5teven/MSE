import torch
from torch.utils.data import Dataset

class SimulationDataset(Dataset):
    """
    PyTorch Dataset for simulated time-series data.
    """
    def __init__(self, phi: torch.Tensor, Y: torch.Tensor, device=None):
        """
        Args:
            phi: Parameter tensor (shape: [N, param_dim]).
            Y: Time-series data tensor (shape: [N, seq_len, features]).
            device: The device where the tensors should be stored.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # ✅ Convert to tensors and store efficiently
        self.phi = phi.to(self.device, non_blocking=True) if isinstance(phi, torch.Tensor) else torch.tensor(phi, dtype=torch.float32, device=self.device)
        self.Y = Y.to(self.device, non_blocking=True) if isinstance(Y, torch.Tensor) else torch.tensor(Y, dtype=torch.float32, device=self.device)

    def __len__(self):
        """Returns the number of samples."""
        return len(self.phi)

    def __getitem__(self, idx):
        """Returns a single sample (phi, Y) efficiently."""
        return self.phi[idx], self.Y[idx]

def create_dataloader(dataset: SimulationDataset, batch_size: int, shuffle=True, num_workers=4):
    """
    Creates an optimized DataLoader.

    Args:
        dataset: The dataset object.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle the dataset.
        num_workers: Number of CPU workers for data loading.

    Returns:
        A PyTorch DataLoader optimized for GPU training.
    """
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if dataset.device == "cuda" else False  # ✅ Faster GPU memory transfers
    )
