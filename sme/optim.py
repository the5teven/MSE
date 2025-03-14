import torch
from torch.utils.data.sampler import Sampler
import numpy as np

class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Maintains a shadow copy with smoother updates.
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_params = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        self.original_params = {}

    @torch.no_grad()
    def update(self):
        """Update EMA parameters using current model state."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_params[name].lerp_(param, 1 - self.decay)

    def apply(self):
        """Backup current parameters and apply EMA weights."""
        self.original_params = {name: param.clone() for name, param in self.model.named_parameters() if param.requires_grad}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.ema_params[name])

    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.original_params:
                param.data.copy_(self.original_params[name])

class CurriculumSampler(Sampler):
    """
    Samples training examples based on difficulty scores for curriculum learning.
    """
    def __init__(self, dataset, difficulty_fn, increasing=True):
        """
        Args:
            dataset: The dataset for sampling.
            difficulty_fn: Function to assign difficulty.
            increasing: If True, easier samples come first.
        """
        self.dataset = dataset
        self.difficulty_scores = np.array([difficulty_fn(sample) for sample in dataset])
        self.sorted_indices = np.argsort(self.difficulty_scores)
        if not increasing:
            self.sorted_indices = self.sorted_indices[::-1]

    def __iter__(self):
        yield from self.sorted_indices

    def __len__(self):
        return len(self.dataset)