"""
A simplified dataset module for SME.
Generates simulation data on the fly using the GeneralSimulator.
"""
import torch
from torch.utils.data import Dataset

class SimulationDataset(Dataset):
    def __init__(self, num_samples, simulator):
        self.num_samples = num_samples
        self.simulator = simulator

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Generate one sample using the simulator.
        return self.simulator.simulate()