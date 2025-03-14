import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from typing import Any, Dict, Type
from sme.sme import SME
from sme.models import BaseEncoder, BaseEmulator

# Define custom Encoder and Emulator classes
class CustomEncoder(BaseEncoder):
    def __init__(self, T=100, n_vars=2, embedding_dim=64):
        super(CustomEncoder, self).__init__(n_vars, embedding_dim)
        self.lstm = nn.LSTM(n_vars, 64, batch_first=True)
        self.fc = nn.Linear(64, embedding_dim)  # Adjusted to match LSTM output
        self.dropout = nn.Dropout(0.3)

    def forward(self, Y):
        _, (h_n, _) = self.lstm(Y)
        Y = h_n[-1]  # Take the last hidden state
        Y = self.dropout(F.relu(self.fc(Y)))
        return F.normalize(Y, p=2, dim=1)

class CustomEmulator(BaseEmulator):
    def __init__(self, n_params=9, embedding_dim=64):
        super(CustomEmulator, self).__init__(n_params, embedding_dim)
        self.fc1 = nn.Linear(n_params, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, phi):
        phi = F.relu(self.fc1(phi))
        phi = self.dropout(phi)
        phi = self.fc2(phi)
        return F.normalize(phi, p=2, dim=1)

# Set device and seeds for reproducibility
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

# Configure and Initialize the SME Helper
sme_helper = SME()
sme_helper.configure_device(device)
sme_helper.configure_model(CustomEncoder, CustomEmulator)
sme_helper.configure_training(num_epochs=200, lr=5e-4)
sme_helper.configure_logging(verbose=True, logging_level="INFO")

# Define Simulation Parameters for a VAR Model with a Regime Break
T = 100       # total time steps in the simulation
n_vars = 2    # number of variables in the VAR process

# True model parameters
phi1_true = np.array([[0.5, 0.1],
                      [0.2, 0.4]], dtype=np.float32)
phi2_true = np.array([[0.8, -0.1],
                      [-0.2, 0.7]], dtype=np.float32)
tau_true = 60  # breakpoint timing

params = {"phi1": phi1_true, "phi2": phi2_true, "tau": tau_true}

# Run End-to-End Analysis with the SME Helper
stats_table = sme_helper.run_analysis(
    model_type="var_break",  # Specifies that we are working with VAR models with breaks.
    params=params,
    T=T,
    n_vars=n_vars,
    sigma=1.0  # Noise level for simulation; set sigma=0 in re-simulation if deterministic output is desired.
)

# Print Estimation Results
phi1_est, phi2_est, tau_est = sme_helper.get_estimated_parameters()
print("\n===== Sample Estimation Results =====")
print(f"True phi1:\n{phi1_true}")
print(f"Estimated phi1:\n{phi1_est}")
print(f"True phi2:\n{phi2_true}")
print(f"Estimated phi2:\n{phi2_est}")
print(f"True tau: {tau_true}")
print(f"Estimated tau: {tau_est}")

# Simulate observed and fitted time series
Y_observed = sme_helper.get_last_observed_series()
Y_fitted = sme_helper.simulate_data(
    model_type="var_break",
    params={"phi1": phi1_est, "phi2": phi2_est, "tau": tau_est},
    T=T,
    n_vars=n_vars,
    sigma=0
).cpu().numpy()

# Plotting
plt.figure(figsize=(15, 10))

# 1. Time Series with Regime Fit
plt.subplot(2, 2, 1)
plt.plot(Y_observed[:, 0], label="Observed Y[0]", alpha=0.7)
plt.plot(Y_fitted[:, 0], label="Fitted Y[0]", linestyle="--")
plt.axvline(tau_true, color='r', linestyle='--', label="True Break")
plt.axvline(tau_est, color='g', linestyle='--', label="Est. Break")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Time Series with Regime Fit")
plt.legend()

# 2. Parameter Scatter Plot
phi1_true_flat = phi1_true.flatten()
phi2_true_flat = phi2_true.flatten()
plt.subplot(2, 2, 2)
for i in range(4):
    plt.scatter([phi1_true_flat[i]] * 1, [phi1_est.flatten()[i]], alpha=0.5, label=f"phi1[{i}]")
    plt.scatter([phi2_true_flat[i]] * 1, [phi2_est.flatten()[i]], alpha=0.5, label=f"phi2[{i}]")
plt.plot([phi1_true_flat.min(), phi1_true_flat.max()], [phi1_true_flat.min(), phi1_true_flat.max()], 'k--')
plt.xlabel("True Value")
plt.ylabel("Estimated Value")
plt.title("True vs. Estimated Parameters")
plt.legend()

# 3. Break Point Error Distribution
tau_errors_test = np.abs(tau_est - tau_true)
plt.subplot(2, 2, 3)
plt.hist(tau_errors_test, bins=15, alpha=0.7, color='purple')
plt.xlabel("Absolute Error in tau")
plt.ylabel("Frequency")
plt.title("Break Point Error Distribution")

# 4. Loss and Break Point Convergence
# Note: Assuming that the loss and tau_errors are tracked during training
losses = sme_helper.get_training_loss_history()
tau_errors = [np.abs(tau_est - tau_true)]  # Placeholder for actual tau errors during training
plt.subplot(2, 2, 4)
plt.plot(losses, label="Training Loss", color='blue')
plt.ylabel("Loss")
plt.twinx()
plt.plot(tau_errors, label="Mean |tau Error|", color='orange')
plt.ylabel("Mean Absolute Error")
plt.xlabel("Epoch")
plt.title("Loss and Break Point Convergence")
plt.legend()

plt.tight_layout()
plt.show()

# Export stats to JSON and DataFrame for advanced analysis
sme_helper.export_stats_to_json(stats_table, "stats_output.json")
df_stats = sme_helper.export_stats_to_dataframe(stats_table)
print("\nStats DataFrame:")
print(df_stats)