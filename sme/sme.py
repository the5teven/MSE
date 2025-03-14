"""
SME Module - Updated

This module provides a simplified interface to the core simulation-based parameter estimation
pipeline. It integrates the streamlined configuration, core model (SMEModel), and advanced 
simulator functionalities. This version removes experimental pretraining and extra utilities,
focusing on the essential operations:

- Configuration: Set up core parameters
- Model: Train SMEModel using streamlined InfoNCE losses
- Simulator: Generate synthetic data based on advanced simulation settings
- Estimation: Estimate model parameters from simulated (or observed) data

Usage:
    1. Configure the SME module with model components (encoder and emulator classes).
    2. Configure training and simulation settings via SMEConfig and simulation config.
    3. Train the model using train_model().
    4. Simulate data via simulate_data().
    5. Estimate parameters using estimate_parameters().
"""
import logging
import torch
import numpy as np

from .config import SMEConfig
from .models import SMEModel, BaseEncoder, BaseEmulator
from .simulator import GeneralSimulator
# Uncomment the following if using DataLoader integration or stats module:
# from .dataset import SimulationDataset
# from .stats import generate_stats_table

class SME:
    def __init__(self):
        # Initialize with a default configuration
        self.config = SMEConfig()
        self.model = None
        self.simulator = None
        self.last_observed_series = None
        self.last_estimated_params = None
        self.training_loss_history = []
        self.internal_logs = []
        self.logger = logging.getLogger("SME")
        logging.basicConfig(level=logging.INFO)

    def configure_device(self, device: torch.device):
        self.config.device_config.device = device
        self.logger.info(f"Device set to {device}")

    def configure_model(self, encoder_class: type, emulator_class: type):
        """
        Configure the model components with the provided encoder and emulator classes.
        """
        self.config.model_components.encoder_class = encoder_class
        self.config.model_components.emulator_class = emulator_class
        self.logger.info("Model components configured.")

    def configure_training(self, **kwargs):
        """
        Update training configuration parameters.
        Example keyword arguments: batch_size, num_epochs, learning_rate, tau, etc.
        """
        for key, value in kwargs.items():
            if hasattr(self.config.training_config, key):
                setattr(self.config.training_config, key, value)
                self.logger.info(f"Training config: {key} set to {value}")

    def configure_simulation(self, simulation_config: dict):
        """
        Configure the simulator with simulation-specific parameters.
        simulation_config should include:
            - model_type: e.g. "var", "nonlinear", "regime_switch"
            - T: Length of the time series
            - n_vars: Number of variables
            - params: Model-specific parameters (e.g. phi for VAR)
        """
        self.simulator = GeneralSimulator(simulation_config)
        self.logger.info(f"Simulator configured for model type {simulation_config.get('model_type', 'default')}")

    def train_model(self):
        """
        Initialize the SMEModel and train it.
        """
        if self.config.model_components.encoder_class is None or self.config.model_components.emulator_class is None:
            self.logger.error("Model components are not configured. Call configure_model() first.")
            return

        self.model = SMEModel(self.config)
        self.logger.info("Training started...")
        self.model.train()
        self.training_loss_history = self.model.training_losses
        self.logger.info("Training completed.")

    def simulate_data(self):
        """
        Generate synthetic data using the configured simulator.
        """
        if self.simulator is None:
            self.logger.error("Simulator not configured. Call configure_simulation() first.")
            return None

        simulated_series = self.simulator.simulate()
        self.last_observed_series = simulated_series
        self.logger.info("Data simulation completed.")
        return simulated_series

    def estimate_parameters(self, observed_data: np.ndarray, candidate_pool_size: int = 1000):
        """
        Estimate parameters from the provided observed data.
        Steps:
            1. Generate a candidate pool.
            2. Estimate parameters using the SMEModel's estimation procedure.
            3. Optionally, compute standard errors.
        """
        if self.model is None:
            self.logger.error("Model is not trained. Call train_model() first.")
            return None

        candidate_pool = self.model.generate_candidate_pool(candidate_pool_size)
        self.logger.info(f"Candidate pool of size {candidate_pool_size} generated.")

        estimated_params = self.model.estimate_phi(observed_data, candidate_pool)
        self.last_estimated_params = estimated_params

        # Compute standard errors (if needed)
        se = self.model.compute_standard_errors(observed_data, estimated_params)
        self.logger.info(f"Parameter estimation completed. Estimated parameters: {estimated_params}")
        self.logger.info(f"Standard errors: {se}")

        # Uncomment the lines below to generate an ASCII stats table for reporting
        # stats = generate_stats_table(estimated_params, se)
        # self.logger.info("Stats table generated.")
        # return stats

        return estimated_params  # or return (estimated_params, se)

    def run_analysis(self, simulation_config: dict, candidate_pool_size: int = 1000):
        """
        Full analysis pipeline:
            1. Configure simulation.
            2. Simulate observed data.
            3. Train the model.
            4. Estimate parameters on the observed data.
        """
        self.configure_simulation(simulation_config)
        observed_data = self.simulate_data()
        if observed_data is None:
            self.logger.error("Simulation failed, analysis aborted.")
            return

        if isinstance(observed_data, torch.Tensor):
            observed_data = observed_data.cpu().numpy()

        self.train_model()
        estimated_params = self.estimate_parameters(observed_data, candidate_pool_size)
        return estimated_params

    def get_training_loss_history(self):
        return self.training_loss_history

    def get_internal_logs(self):
        return self.internal_logs

    def get_last_observed_series(self):
        return self.last_observed_series

    def get_estimated_parameters(self):
        return self.last_estimated_params