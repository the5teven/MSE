"""
SME: Simulated Method of Embeddings Package

This package provides functionality for parameter estimation using machine learning
techniques, including simulation, training, parameter estimation, statistical analysis,
and optimization tools.

Modules:
- config: Configuration and hyperparameters for the package.
- dataset: Data loading and simulation dataset management.
- simulator: Tools for synthetic data simulation.
- models: Definitions of the core models (encoder, emulator) and training loops.
- losses: Loss functions used during training.
- optim: Optimization utilities (e.g., EMA).
- stats: Statistical reporting for estimated parameters, standard errors, and confidence intervals.
- fisher_info: Methods for computing Fisher information and standard errors.

For further documentation and examples, please refer to the repository README.md.
"""

from .config import SMEConfig
from .dataset import SimulationDataset, create_dataloader
from .simulator import SimulatorConfig, GeneralSimulator
from .models import SMEModel, BaseEncoder, BaseEmulator
from .losses import CompositeLoss, MemoryBankLoss
from .optim import EMA
from .stats import generate_stats_table
from .fisher_info import calculate_fisher_information, calculate_standard_errors
from .sme import SME  # Ensure that SME is imported from sme.py

__all__ = [
    "SMEConfig",
    "SimulationDataset",
    "create_dataloader",
    "SimulatorConfig",
    "GeneralSimulator",
    "SMEModel",
    "BaseEncoder",
    "BaseEmulator",
    "CompositeLoss",
    "MemoryBankLoss",
    "EMA",
    "generate_stats_table",
    "calculate_fisher_information",
    "calculate_standard_errors",
    "SME",
]