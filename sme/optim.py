import torch
from torch.utils.data.sampler import Sampler
import numpy as np


class EMA:
    """
    Exponential Moving Average (EMA) for model parameters.
    Maintains a shadow copy of the model with smoother updates.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.ema_params = {name: param.clone().detach() for name, param in model.named_parameters() if
                           param.requires_grad}

    @torch.no_grad()
    def update(self):
        """Update EMA parameters using the current model state."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.ema_params[name].lerp_(param, 1 - self.decay)
    def apply(self):
        """Applies the EMA weights to the model (useful for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.copy_(self.ema_params[name])

    def restore(self):
        """Restores the original model parameters (useful after evaluation)."""
        self.apply()

class CurriculumSampler(Sampler):
    """
    Samples training examples based on difficulty scores for curriculum learning.
    """

    def __init__(self, dataset, difficulty_fn, increasing=True):
        """
        Args:
            dataset: The dataset from which to sample.
            difficulty_fn: Function that assigns difficulty to each sample.
            increasing: If True, easier samples are prioritized first.
        """
        self.dataset = dataset
        self.difficulty_scores = np.array([difficulty_fn(sample) for sample in dataset])
        self.sorted_indices = np.argsort(self.difficulty_scores)
        if not increasing:
            self.sorted_indices = self.sorted_indices[::-1]

    def __iter__(self):
        """Yield samples in order of difficulty."""
        yield from self.sorted_indices

    def __len__(self):
        """Returns dataset length."""
        return len(self.dataset)
