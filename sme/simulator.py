"""
An advanced simulator for SME.

This module generates synthetic data for training and testing.
It supports multiple simulation models by selecting a simulation type
and providing the corresponding parameters.

Usage:
    config = {
        "model_type": "VAR",  # or "nonlinear", "regime_switch", etc.
        "T": 100,             # time series length
        "n_vars": 2,          # number of variables
        "params": {           # parameters for the simulation model
            "phi": [[0.5, 0.1], [0.2, 0.3]]
        }
    }
    simulator = GeneralSimulator(config)
    simulated_data = simulator.simulate()
"""
import torch
import numpy as np

class GeneralSimulator:
    def __init__(self, config):
        """
        Args:
            config (dict): A dictionary with simulation configuration.
              Expected keys:
                  - "model_type": A string identifier for the simulation model.
                  - "T": (int) length of the time series.
                  - "n_vars": (int) number of variables.
                  - "params": (dict) parameters needed for the chosen model.
                  - Additional keys can be defined based on model_type.
        """
        self.config = config
        self.model_type = config.get("model_type", "default")
        self.T = config.get("T", 100)
        self.n_vars = config.get("n_vars", 2)
        self.params = config.get("params", {})

    def simulate(self):
        """
        Dispatches simulation to the appropriate model based on `model_type`.
        """
        if self.model_type.lower() == "var":
            return self.simulate_var()
        elif self.model_type.lower() == "nonlinear":
            return self.simulate_nonlinear()
        elif self.model_type.lower() == "regime_switch":
            return self.simulate_regime_switch()
        else:
            # Default simulation: generate random noise
            return torch.rand(self.T, self.n_vars)

    def simulate_var(self):
        """
        Simulate a Vector Autoregression (VAR) process.
        Requires "phi" parameter in config.params.
        """
        phi = self.params.get("phi")
        if phi is None:
            raise ValueError("VAR simulation requires 'phi' in params.")
        phi = np.array(phi)
        T, n_vars = self.T, self.n_vars
        data = np.zeros((T, n_vars))
        data[0] = np.random.randn(n_vars)
        for t in range(1, T):
            data[t] = phi.dot(data[t-1]) + np.random.randn(n_vars) * 0.1
        return torch.tensor(data, dtype=torch.float32)

    def simulate_nonlinear(self):
        """
        Simulate a nonlinear time series process.
        User should provide necessary parameters such as nonlinearity scale.
        """
        T, n_vars = self.T, self.n_vars
        scale = self.params.get("scale", 1.0)
        time = np.linspace(0, 2 * np.pi, T)
        data = np.zeros((T, n_vars))
        data[:, 0] = scale * np.sin(time)
        if n_vars > 1:
            for i in range(1, n_vars):
                data[:, i] = np.random.randn(T) * 0.5
        return torch.tensor(data, dtype=torch.float32)

    def simulate_regime_switch(self):
        """
        Simulate a regime switching process.
        Requires additional parameters like break points and regime-specific parameters.
        """
        T, n_vars = self.T, self.n_vars
        break_point = self.params.get("break_point", T // 2)
        phi_regime1 = np.array(self.params.get("phi_regime1", np.eye(n_vars) * 0.5))
        phi_regime2 = np.array(self.params.get("phi_regime2", np.eye(n_vars) * -0.5))
        data = np.zeros((T, n_vars))
        data[0] = np.random.randn(n_vars)
        for t in range(1, T):
            if t < break_point:
                data[t] = phi_regime1.dot(data[t-1]) + np.random.randn(n_vars) * 0.1
            else:
                data[t] = phi_regime2.dot(data[t-1]) + np.random.randn(n_vars) * 0.1
        return torch.tensor(data, dtype=torch.float32)