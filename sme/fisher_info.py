"""
A minimal implementation for computing Fisher Information and standard errors.
This module can be kept if statistical inference is a core part of your workflow,
otherwise it can be removed or merged into stats.py.
"""
import torch

def calculate_fisher_information(grad_phi: torch.Tensor):
    return torch.outer(grad_phi, grad_phi)

def calculate_standard_errors(fisher_information: torch.Tensor):
    jitter = 1e-6
    fisher_information += jitter * torch.eye(fisher_information.size(0))
    fisher_inv = torch.inverse(fisher_information)
    return torch.sqrt(torch.diag(fisher_inv))