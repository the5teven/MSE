from typing import Any, Dict, Type, Callable, List
import torch
import matplotlib.pyplot as plt
import numpy as np
from .config import SMEConfig
from .models import SMEModel
from .simulator import GeneralSimulator, SimulatorConfig, register_custom_simulator
from .stats import generate_stats_table
import pandas as pd
import json


class StructuralModel:
    def __init__(self):
        self.sim_config = None

    def configure_simulation(self, model_type: str, params: Dict[str, Any], T: int, n_vars: int, **kwargs):
        self.sim_config = SimulatorConfig(
            model_type=model_type,
            params=params,
            T=T,
            n_vars=n_vars,
            **kwargs
        )

    def run_simulation(self):
        if self.sim_config is None:
            raise ValueError("Simulation configuration is not set.")
        simulator = GeneralSimulator(self.sim_config)
        return simulator.simulate()

    def set_custom_noise_function(self, noise_func: Callable):
        if self.sim_config is not None:
            self.sim_config.noise_dist = noise_func

    def register_custom_model(self, model_name: str, model_func: Callable):
        register_custom_simulator(model_name)(model_func)

    def add_break_points(self, break_points: List[int]):
        if self.sim_config is not None:
            self.sim_config.break_points = break_points


class SME:
    def __init__(self):
        self.config = SMEConfig()
        self.model = None
        self.last_observed_series = None
        self.last_estimated_params = None
        self.training_loss_history = []  # Advanced: capture training loss history
        self.internal_logs = []  # Advanced: capture internal logs for low-level inspection
        self.custom_simulation_function = None  # Allow custom simulation strategies
        self.structural_model = StructuralModel()  # Initialize StructuralModel instance

    def configure_device(self, device: torch.device):
        self.config.device_config.device = device

    def configure_model(self, encoder_class: Type, emulator_class: Type):
        self.config.model_components.encoder_class = encoder_class
        self.config.model_components.emulator_class = emulator_class
        # Set number of parameters in the model components if not yet defined.
        if not hasattr(self.config.model_components, "n_params"):
            self.config.model_components.n_params = self.config.training_config.param_dim

    def configure_training(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.training_config, key):
                setattr(self.config.training_config, key, value)

    def configure_optimization(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.optimization_config, key):
                setattr(self.config.optimization_config, key, value)

    def configure_regularization(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.regularization_config, key):
                setattr(self.config.regularization_config, key, value)

    def configure_learning_strategies(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self.config.learning_strategies, key):
                setattr(self.config.learning_strategies, key, value)

    def configure_logging(self, verbose: bool, logging_level: str):
        self.config.logging_config.verbose = verbose
        self.config.logging_config.logging_level = logging_level

    def set_custom_simulation_function(self, func: Callable):
        """Allow users to set a custom simulation function."""
        self.custom_simulation_function = func

    def configure_structural_model(self, model_type: str, params: Dict[str, Any], T: int, n_vars: int, **kwargs):
        """Configure the StructuralModel for simulations."""
        self.structural_model.configure_simulation(model_type, params, T, n_vars, **kwargs)

    def train_model(self):
        """
        Trains the model using an active-learning simulation selection strategy.
        Tracks training losses and internal logs for advanced low-level inspection.
        """
        self.model = SMEModel(self.config)
        self.model.train()
        # Capture training history from the model for advanced users.
        self.training_loss_history = self.model.training_losses
        self.internal_logs.append("Training complete with {} epochs.".format(len(self.training_loss_history)))

    def simulate_data(self, model_type: str, params: Dict[str, Any], T: int, n_vars: int = 1, sigma: float = 1.0):
        if self.custom_simulation_function:
            simulated_series = self.custom_simulation_function(model_type, params, T, n_vars, sigma)
        else:
            if self.structural_model:
                self.structural_model.configure_simulation(model_type, params, T, n_vars)
                simulated_series = self.structural_model.run_simulation()
            else:
                sim_config = SimulatorConfig(
                    model_type=model_type,
                    params=params,
                    T=T,
                    n_vars=n_vars,
                    device=self.config.device_config.device,
                    noise_kwargs={"std": sigma}  # Pass sigma as part of noise_kwargs
                )
                simulator = GeneralSimulator(sim_config)
                simulated_series = simulator.simulate()
        self.last_observed_series = simulated_series
        self.internal_logs.append("Simulated observed series with T={} and n_vars={}".format(T, n_vars))
        return simulated_series

    def estimate_parameters(self, Y_star, candidate_size: int = 1000) -> str:
        """
        Estimate parameters using internally generated candidate simulations and
        compute standard errors using a Jacobian-based method.
        Returns an ASCII formatted stats table with estimates and standard errors.
        """
        candidate_pool = self.model.generate_candidate_pool(candidate_size)
        self.internal_logs.append("Generated candidate pool of size {}".format(candidate_size))
        estimated_params = self.model.estimate_phi(Y_star, candidate_pool)
        self.last_estimated_params = estimated_params
        se = self.model.compute_standard_errors(Y_star, estimated_params)
        stats_table = generate_stats_table(estimated_params, se)
        self.internal_logs.append("Parameter estimation complete with SE computed.")
        return stats_table

    def run_analysis(self, model_type: str, params: Dict[str, Any], T: int, n_vars: int, sigma: float = 1.0) -> str:
        """
        Full analysis pipeline:
         - Simulate observed data.
         - Train the model with active-learning based simulation selection.
         - Estimate parameters (including standard error computation).
         - Plot performance graphs.
         - Return an ASCII formatted statistics table.
        """
        Y_star = self.simulate_data(model_type, params, T, n_vars, sigma)
        if self.model is None:
            self.train_model()
        stats_table = self.estimate_parameters(Y_star)
        self.show_performance_graphs(Y_star, params, T, n_vars)
        return stats_table

    def get_estimated_parameters(self):
        """Return the last estimated parameter vector."""
        return self.last_estimated_params

    def get_last_observed_series(self):
        """Return the last simulated observed series."""
        return self.last_observed_series

    def get_training_loss_history(self):
        """Return the full training loss history (advanced info)."""
        return self.training_loss_history

    def get_internal_logs(self):
        """Return internal log details for debugging and advanced inspection."""
        return self.internal_logs

    def show_performance_graphs(self, Y_observed, true_params: Dict[str, Any], T: int, n_vars: int):
        """
        Plot graphs for model performance:
         - Observed vs. fitted time series.
         - Training loss curve.
         Advanced users can extend these plots or extract low-level metrics.
        """
        if self.last_estimated_params is None:
            print("No estimated parameters available for plotting.")
            return

        # Extract estimated parameters.
        n = int((len(self.last_estimated_params) - 1) / 2)
        phi1_est = self.last_estimated_params[:n * n].reshape(n, n)
        phi2_est = self.last_estimated_params[n * n:2 * n * n].reshape(n, n)
        tau_est = int(self.last_estimated_params[-1] * T)

        fitted_series = self.simulate_data("var_break", {"phi1": phi1_est, "phi2": phi2_est, "tau": tau_est}, T, n_vars,
                                           sigma=0)

        plt.figure(figsize=(10, 6))
        plt.plot(Y_observed[:, 0], label="Observed Y[0]", alpha=0.7)
        plt.plot(fitted_series[:, 0], label="Fitted Y[0]", linestyle="--")
        true_tau = true_params.get("tau", tau_est)
        plt.axvline(true_tau, color="r", linestyle="--", label="True Break")
        plt.axvline(tau_est, color="g", linestyle="--", label="Estimated Break")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Observed vs. Fitted Time Series with Regime Break")
        plt.legend()
        plt.show()

        # Plot training loss history if available.
        if self.training_loss_history:
            plt.figure(figsize=(8, 4))
            plt.plot(self.training_loss_history, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss History")
            plt.legend()
            plt.show()

    def export_stats_to_json(self, stats_table: str, file_name: str):
        """Export the parameter estimates and SEs to a JSON file for advanced analysis."""
        stats_dict = {}
        for line in stats_table.split('\n'):
            if line.strip():
                parts = line.split(':')
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    stats_dict[key] = value
        with open(file_name, 'w') as f:
            json.dump(stats_dict, f, indent=4)

    def export_stats_to_dataframe(self, stats_table: str) -> pd.DataFrame:
        """Convert the parameter estimates and SEs to a pandas DataFrame for advanced analysis."""
        data = []
        for line in stats_table.split('\n'):
            if line.strip():
                parts = line.split(':')
                if len(parts) == 2:
                    key, value = parts[0].strip(), parts[1].strip()
                    data.append((key, value))
        df = pd.DataFrame(data, columns=['Parameter', 'Value'])
        return df