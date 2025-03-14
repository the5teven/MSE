"""
This example demonstrates a flexible SME framework with clear terminology.
It distinguishes between:
  - Data-Generating (Structural) Parameters:
      These are the interpretable "phis" (e.g. φ₁, φ₂, τ) that define the simulation model.
  - Neural Network Parameters:
      These are the learned weights of the encoder and emulator networks.

The encoder maps time series data to a latent embedding space. The emulator maps the
data-generating parameters (a 9-dimensional vector) to the same latent space. Contrastive
learning, including the memory bank, operates in this latent space.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow duplicate lib warnings if necessary.
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Set device. Warning appears if CUDA is requested but not available.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import SME components from the SME library.
from sme.config import SMEConfig
from sme.models import SMEModel, BaseEncoder, BaseEmulator
from sme.simulator import SimulatorConfig, GeneralSimulator

###############################################################################
# 1. Encoder (Neural Network Component):
#
# The encoder processes observation time series data (of shape: batch x 100 x 2)
# and outputs a latent embedding of size 64.
#
# Note:
#   - The encoder's learned weights are the Neural Network Parameters.
#   - The output (latent embedding) is used for contrastive learning and stored in the memory bank.
###############################################################################
class LSTMEncoder(BaseEncoder):
    def __init__(self, config: SMEConfig, T=100, n_vars=2, embedding_dim=64):
        super().__init__(config)
        self.lstm = nn.LSTM(n_vars, 64, batch_first=True)
        self.fc = nn.Linear(64, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        # Y: (batch, T, n_vars)
        _, (h_n, _) = self.lstm(Y)
        # Use the last hidden state from the LSTM.
        h = h_n[-1]
        out = self.fc(h)
        out = self.dropout(F.relu(out))
        # Output: normalized latent embedding (batch, 64)
        return F.normalize(out, p=2, dim=1)

###############################################################################
# 2. Emulator (Neural Network Component Mapping Data-Generating Parameters):
#
# The emulator takes in interpretable data-generating parameters (phis) of dimension 9.
#
# Data-Generating Parameters (Structural Parameters):
#   - Elements [0-3]: Represent φ₁ (to be reshaped to a 2×2 matrix)
#   - Elements [4-7]: Represent φ₂ (to be reshaped to a 2×2 matrix)
#   - Element [8]: Represents a normalized break point τ (to be scaled appropriately)
#
# The emulator maps these 9 parameters to a 64-dimensional latent embedding.
# A projection layer is provided to optionally recover the interpretable parameters.
#
# Note:
#   - The emulator's weights (learned parameters) are separate from the data-generating
#     parameters that produce data.
###############################################################################
class FFEmulator(BaseEmulator):
    def __init__(self, config: SMEConfig, n_params=9, embedding_dim=64):
        super().__init__(config)
        # First layer: Projects the 9-d interpretable vector to a hidden layer.
        self.fc1 = nn.Linear(n_params, 128)
        # Second layer: Projects to a 64-d latent embedding for use with contrastive learning.
        self.fc2 = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.3)
        # Additional projection layer to recover interpretable parameters if required.
        self.proj = nn.Linear(embedding_dim, n_params)

    def forward(self, phi: torch.Tensor, return_interpretable: bool = False) -> torch.Tensor:
        # phi: (batch, 9) representing data-generating (structural) parameters.
        x = F.relu(self.fc1(phi))
        latent = self.fc2(x)
        latent = self.dropout(latent)
        # latent: 64-d embedding compatible with the encoder's output and memory bank.
        latent = F.normalize(latent, p=2, dim=1)
        if return_interpretable:
            # Optionally recover the interpretable (structural) parameters.
            interpretable = self.proj(latent)
            return latent, interpretable
        return latent

###############################################################################
# 3. SME Configuration:
#
# Here we configure the SME model with clear distinction:
#   - input_dim: Shape of the observed time series data.
#   - param_dim: Dimension of the interpretable simulation parameters ("phis") = 9.
#
# The flexible emulator ensures that while phis are 9-dimensional,
# they are projected into the 64-dimensional latent space for contrastive loss.
###############################################################################
config = SMEConfig(
    encoder_class=LSTMEncoder,       # Neural network encoder for observed data.
    emulator_class=FFEmulator,         # Neural network emulator mapping phis to latent space.
    loss_type='angular',             # Using angular (cosine similarity) contrastive loss.
    input_dim=(100, 2),              # Observed time series: 100 timesteps, 2 variables.
    param_dim=9,                     # Interpretable (data-generating) parameter vector is 9-d.
    use_active_learning=True,
    candidate_pool_size=5000,
    batch_size=64,
    num_epochs=3,                    # Short training demo.
    training_steps_per_epoch=100,
    refinement_steps=5,
    pretraining_samples=100,
    use_pretraining=False,
    use_ema=True,
    use_memory_bank=True,            # Memory bank works in the latent space.
    learning_rate=1e-3,
    lr_scheduler='cosine'
)

# Instantiate the composite SME model.
model = SMEModel(config)

###############################################################################
# 4. Simulation Model (Data Generator):
#
# This simulation model (structural model) uses data-generating parameters ("phis")
# to generate synthetic time series data. This is separate from the neural network model.
#
# In our demo, we generate data using a simple VAR (vector autoregressive) process.
###############################################################################
sim_config = SimulatorConfig(
    model_type="var",
    params={"p": 1, "phi": np.eye(2, dtype=np.float32) * 0.5},  # Example baseline parameters.
    T=100,
    n_vars=2,
    noise_kwargs={"mean": 0, "std": 1},
    seed=42
)
simulator = GeneralSimulator(sim_config)
simulated_data = simulator.simulate()
print("Simulated Data Shape:", simulated_data.shape)

###############################################################################
# 5. Training the SME Neural Network Model:
#
# The SME model is trained to learn the mapping between observed data and data-generating
# parameters. The neural network parameters (weights) are learned during training.
###############################################################################
print("Starting training...")
model.train()  # This trains using the flexible memory bank adapted to the encoder's 64-d space.
print("Training complete.")

###############################################################################
# 6. Estimating Data-Generating Parameters:
#
# After training, the SME model estimates the interpretable data-generating parameters
# (phis) that best explain the observed data.
#
# The 9-d estimated vector is then decomposed into:
#   - φ₁ (first 4 elements, reshaped to 2x2 matrix)
#   - φ₂ (next 4 elements, reshaped to 2x2 matrix)
#   - τ (last element, unnormalized by scaling)
###############################################################################
Y_star = simulated_data.numpy()
phi_pool = np.random.randn(config.candidate_pool_size, config.param_dim).astype(np.float32)
estimated_phi = model.estimate_phi(Y_star, phi_pool)

phi1_est = estimated_phi[:4].reshape(2, 2)
phi2_est = estimated_phi[4:8].reshape(2, 2)
tau_est = int(estimated_phi[8] * 100)  # Assume τ was normalized in data generation.
print("Estimated Data-Generating Parameters (phis):")
print("φ₁ (Structural Matrix 1):\n", phi1_est)
print("φ₂ (Structural Matrix 2):\n", phi2_est)
print("τ (Break Point):", tau_est)

###############################################################################
# 7. Graphical Analysis:
#
# Compare the true (synthetic) time series produced by the structured (simulation)
# model with the time series generated using the estimated data-generating parameters.
#
# We define a torch.jit.script simulation function for consistency.
###############################################################################
@torch.jit.script
def simulate_var(phi1: torch.Tensor, phi2: torch.Tensor, tau: int, T: int = 100, sigma: float = 1.0) -> torch.Tensor:
    n_vars = phi1.size(0)
    Y = torch.zeros(T, n_vars, dtype=torch.float32, device=phi1.device)
    epsilon = torch.randn(T, n_vars, device=phi1.device) * sigma
    for t in range(1, tau):
        Y[t] = torch.matmul(phi1, Y[t - 1]) + epsilon[t]
    for t in range(tau, T):
        Y[t] = torch.matmul(phi2, Y[t - 1]) + epsilon[t]
    return Y

# Define true data-generating parameters for testing.
phi1_true = np.array([[0.5, 0.1], [0.2, 0.4]], dtype=np.float32)
phi2_true = np.array([[0.8, -0.1], [-0.2, 0.7]], dtype=np.float32)
tau_true = 60

phi1_true_tensor = torch.tensor(phi1_true, dtype=torch.float32, device=device)
phi2_true_tensor = torch.tensor(phi2_true, dtype=torch.float32, device=device)
Y_true = simulate_var(phi1_true_tensor, phi2_true_tensor, tau_true, T=100, sigma=1.0).cpu().numpy()

# Generate fitted time series using the estimated data-generating parameters.
phi1_est_tensor = torch.tensor(phi1_est, dtype=torch.float32, device=device)
phi2_est_tensor = torch.tensor(phi2_est, dtype=torch.float32, device=device)
Y_fit = simulate_var(phi1_est_tensor, phi2_est_tensor, tau_est, T=100, sigma=0).cpu().numpy()

plt.figure(figsize=(15, 10))

# Graph 1: Time Series Comparison (Structural Model vs. Estimated Parameters)
plt.subplot(2, 2, 1)
plt.plot(Y_true[:, 0], label="True Generated Y[0]", alpha=0.7)
plt.plot(Y_fit[:, 0], label="Fitted Y[0] (Estimated Parameters)", linestyle="--")
plt.axvline(tau_true, color='r', linestyle='--', label="True Break")
plt.axvline(tau_est, color='g', linestyle='--', label="Estimated Break")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Time Series Comparison")
plt.legend()

# Graph 2: Scatter Plot of Estimated vs. True Structural Parameters.
num_tests = 50
phi1_estimates, phi2_estimates = [], []
for _ in range(num_tests):
    Y_sample = simulator.simulate().cpu().numpy()
    phi_est = model.estimate_phi(Y_sample, phi_pool)
    phi1_estimates.append(phi_est[:4])
    phi2_estimates.append(phi_est[4:8])
phi1_estimates = np.array(phi1_estimates)
phi2_estimates = np.array(phi2_estimates)
plt.subplot(2, 2, 2)
for i in range(4):
    plt.scatter([phi1_true.flatten()[i]] * num_tests, phi1_estimates[:, i],
                alpha=0.5, label="φ₁" if i == 0 else "")
    plt.scatter([phi2_true.flatten()[i]] * num_tests, phi2_estimates[:, i],
                alpha=0.5, label="φ₂" if i == 0 else "")
plt.plot([phi1_true.min(), phi1_true.max()],
         [phi1_true.min(), phi1_true.max()], 'k--')
plt.xlabel("True Value")
plt.ylabel("Estimated Value")
plt.title("Estimated vs. True Structural Parameters")
plt.legend()

# Graph 3: Histogram of Absolute Error in τ (Break Point).
plt.subplot(2, 2, 3)
tau_errors = []
for _ in range(num_tests):
    # For tau, assume estimate_phi returns last component as τ normalized value.
    _, _, tau_sample = model.estimate_phi(simulator.simulate().cpu().numpy(), phi_pool)
    tau_errors.append(abs(tau_sample - tau_true))
plt.hist(tau_errors, bins=15, alpha=0.7, color='purple')
plt.xlabel("Absolute Error in τ")
plt.ylabel("Frequency")
plt.title("Distribution of τ Error")

# Graph 4: Dummy Convergence Curves (Demonstrative Comparison Over Epochs).
plt.subplot(2, 2, 4)
loss_curve = np.linspace(33, 6.5, num=3)   # Dummy training loss values.
tau_error_curve = np.linspace(20, 5, num=3)  # Dummy τ error values.
plt.plot(loss_curve, label="Training Loss", color='blue')
plt.ylabel("Loss")
plt.twinx()
plt.plot(tau_error_curve, label="Mean |τ Error|", color='orange')
plt.ylabel("Mean Absolute Error")
plt.xlabel("Epoch")
plt.title("Convergence Curves")
plt.legend()

plt.tight_layout()
plt.show()