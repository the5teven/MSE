"""
Demonstrates:
- Data simulation
- Model training
- Parameter estimation
- Confidence intervals
"""

import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F

from sme import SMEConfig, SMEModel, simulate_time_series, SimulatorConfig, SimulationDataset

from sme import register_custom_simulator


@register_custom_simulator("Lorenz")
def lorenz_simulator(config: SimulatorConfig) -> np.ndarray:
    """
    Simulate the Lorenz system:
    dx/dt = σ(y - x)
    dy/dt = x(ρ - z) - y
    dz/dt = xy - βz
    """
    # Extract parameters
    sigma = config.params.get("sigma", 10.0)
    rho = config.params.get("rho", 28.0)
    beta = config.params.get("beta", 8 / 3)

    # Time steps
    T = config.T
    dt = 0.01  # Time step size
    steps = int(T / dt)

    # Initialize state
    x = np.zeros(steps)
    y = np.zeros(steps)
    z = np.zeros(steps)
    x[0], y[0], z[0] = 1.0, 1.0, 1.0  # Initial conditions

    # Simulate
    for t in range(steps - 1):
        dx = sigma * (y[t] - x[t])
        dy = x[t] * (rho - z[t]) - y[t]
        dz = x[t] * y[t] - beta * z[t]

        x[t + 1] = x[t] + dx * dt
        y[t + 1] = y[t] + dy * dt
        z[t + 1] = z[t] + dz * dt

    # Add noise if specified
    noise = config.noise_dist(size=(steps, 3), **config.noise_kwargs)
    x += noise[:, 0]
    y += noise[:, 1]
    z += noise[:, 2]

    # Downsample to match T
    indices = np.linspace(0, steps - 1, T, dtype=int)
    return np.stack([x[indices], y[indices], z[indices]], axis=1)

class CNNEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Input shape: (batch, T, n_vars)
        self.conv1 = nn.Conv1d(
            in_channels=config.input_dim[1],  # Number of variables
            out_channels=32,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(64 * (config.input_dim[0] // 4), 128)
        self.fc2 = nn.Linear(128, config.param_dim)

    def forward(self, x):
        # x shape: (batch, T, n_vars)
        x = x.permute(0, 2, 1)  # (batch, n_vars, T)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MLPEmulator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.fc1 = nn.Linear(config.param_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, config.param_dim)

    def forward(self, phi):
        phi = F.relu(self.fc1(phi))
        phi = F.relu(self.fc2(phi))
        phi = self.fc3(phi)
        return F.normalize(phi, p=2, dim=1)  # L2 normalization


# Configuration
sim_config = SimulatorConfig(
    model_type="Lorenz",
    params={"sigma": 10.0, "rho": 28.0, "beta": 2.67},
    T=1000,
    n_vars=3,
    noise_kwargs={"scale": 0.1},
    seed=42
)

sme_config = SMEConfig(
    encoder_class=CNNEncoder,
    emulator_class=MLPEmulator,
    input_dim=(1000, 3),
    param_dim=3,
    batch_size=128,
    num_epochs=200,
    use_amp=True,
    compute_fisher=True,
    nn_method='faiss'
)

# Simulate training data
phi_pool = np.random.uniform(
    low=[8.0, 20.0, 2.0],
    high=[12.0, 30.0, 3.0],
    size=(5000, 3)
)
Y_pool = np.array([simulate_time_series(sim_config) for _ in range(5000)])

# Create and train model
model = SMEModel(sme_config)
model.train(SimulationDataset(phi_pool, Y_pool))

# Estimation with new observation
true_params = np.array([10.0, 28.0, 2.67])
Y_star = simulate_time_series(SimulatorConfig(**{**sim_config.__dict__, "seed": 123}))

phi_hat, (ci_low, ci_high) = model.estimate(Y_star, phi_pool)

print(f"""
True parameters: {true_params}
Estimated:       {phi_hat}
Confidence (95%):
Lower bound:    {ci_low}
Upper bound:    {ci_high}
""")

# Visualization
plt.figure(figsize=(10, 6))
plt.errorbar(range(3), phi_hat, yerr=[phi_hat - ci_low, ci_high - phi_hat],
             fmt='o', capsize=5, label='Estimate')
plt.plot(true_params, 'rx', markersize=10, label='True Values')
plt.xticks([0, 1, 2], ['σ', 'ρ', 'β'])
plt.title("Lorenz System Parameter Estimation")
plt.legend()
plt.grid(True)
plt.show()