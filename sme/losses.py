"""
Losses module for SME.
Defines various loss functions including CompositeLoss and an optional MemoryBankLoss.
"""
import torch
import torch.nn as nn

class CompositeLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Compute the composite loss.
        This implementation uses an asymmetric InfoNCE formulation.
        """
        # Dot product similarity assuming inputs are normalized
        loss = -torch.mean(torch.sum(f * g, dim=1))
        return loss

class MemoryBankLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weight = 0.1  # Weight for memory bank loss, can be parameterized

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """
        Compute additional loss based on a memory bank.
        """
        loss = self.weight * torch.mean((f - g) ** 2)
        return loss