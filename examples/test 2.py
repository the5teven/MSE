import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sme import SME  # Ensure this imports your SME class correctly


class Encoder(nn.Module):
    def __init__(self, n_vars=2, embedding_dim=64):
        super(Encoder, self).__init__()
        self.lstm = nn.LSTM(n_vars, 64, batch_first=True)
        self.fc = nn.Linear(64, embedding_dim)  # Adjusted to match LSTM output
        self.dropout = nn.Dropout(0.3)

    def forward(self, Y):
        # Y is expected to be of shape (batch, T, n_vars)
        _, (h_n, _) = self.lstm(Y)  # h_n: (num_layers, batch, hidden_size)
        Y = h_n[-1]  # Take the last hidden state
        Y = self.dropout(F.relu(self.fc(Y)))
        return F.normalize(Y, p=2, dim=1)


class Emulator(nn.Module):
    def __init__(self, n_params=10, embedding_dim=64):
        super(Emulator, self).__init__()
        self.fc1 = nn.Linear(n_params, 128)  # input: n_params, output: 128
        self.fc2 = nn.Linear(128, embedding_dim)  # input: 128, output: embedding_dim
        self.dropout = nn.Dropout(0.3)

    def forward(self, phi):
        # phi is expected to be of shape (batch, n_params)
        phi = self.dropout(F.relu(self.fc1(phi)))  # First layer with dropout
        phi = self.fc2(phi)  # Second layer
        return F.normalize(phi, p=2, dim=1)  # L2 normalization

# A dummy implementation of generate_stats_table.
def generate_stats_table(estimated_params, standard_errors):
    import pandas as pd
    z_value = 1.96  # For 95% CI
    estimated_params = np.array(estimated_params)
    standard_errors = np.array(standard_errors)
    lower_ci = estimated_params - z_value * standard_errors
    upper_ci = estimated_params + z_value * standard_errors
    df = pd.DataFrame({
        "Parameter": estimated_params,
        "Std_Error": standard_errors,
        "Lower_CI": lower_ci,
        "Upper_CI": upper_ci,
    })
    return df


def compute_fisher_information(model, data_loader, device):
    model.eval()
    fisher_information = None
    batch_count = 0

    for batch in data_loader:
        phi_batch, Y_batch = batch
        phi_batch = phi_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Zero out gradients.
        model.encoder.zero_grad()
        model.emulator.zero_grad()

        # Forward pass.
        f_output = model.encoder(Y_batch)
        g_output = model.emulator(phi_batch)

        # Compute loss using the model's loss function.
        loss = model.loss_fn(f_output, g_output)
        loss.backward()

        # Aggregate gradients from both encoder and emulator.
        gradients = []
        for param in model.encoder.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        for param in model.emulator.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        if gradients:
            grad_flat = torch.cat(gradients)
            batch_fisher = torch.outer(grad_flat, grad_flat)
            if fisher_information is None:
                fisher_information = batch_fisher
            else:
                fisher_information += batch_fisher
        batch_count += 1

    if fisher_information is not None and batch_count > 0:
        fisher_information /= batch_count

    # Regularize Fisher information to avoid singularity.
    jitter = 1e-6
    fisher_information = fisher_information + jitter * torch.eye(fisher_information.size(0), device=device)

    return fisher_information


def compute_standard_errors(fisher_information):
    try:
        fisher_information_inv = torch.inverse(fisher_information)
        standard_errors = torch.sqrt(torch.diag(fisher_information_inv))
        return standard_errors
    except Exception as e:
        print("Error inverting Fisher information matrix:", e)
        return torch.tensor([])


def main():
    # Instantiate the SME class.
    sme_instance = SME()

    # Configure device.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sme_instance.configure_device(device)

    # Configure the model with our custom Encoder and Emulator.
    sme_instance.configure_model(Encoder, Emulator)

    # Configure training parameters for a longer training period.
    sme_instance.configure_training(num_epochs=50, training_steps_per_epoch=100, batch_size=32)
    sme_instance.configure_optimization(learning_rate=0.001)
    sme_instance.configure_logging(verbose=True, logging_level="DEBUG")

    # Train the model.
    sme_instance.train_model()

    # Simulate some observed data.
    Y_star = np.random.rand(100, 2).astype(np.float32)  # Shape: (100, 2)

    # Create a candidate parameter pool with 500 candidates (dimension 10).
    phi_pool = (np.random.rand(500, 10).astype(np.float32) * 2) - 1

    # Create an improved dataloader for aggregated Fisher information.
    # Simulate a dataset with 1000 samples.
    n_samples = 1000
    n_params = 10
    T, n_vars = 100, 2  # Dimensions for Y data.
    # For simplicity, generate random data.
    phi_data = torch.rand(n_samples, n_params) * 2 - 1
    Y_data = torch.rand(n_samples, T, n_vars)
    dataset = TensorDataset(phi_data, Y_data)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- Aggregated Fisher Information Computation Across Batches ---
    fisher_information = compute_fisher_information(sme_instance.model, dataloader, device)
    standard_errors = compute_standard_errors(fisher_information)

    # Estimate parameters using the SME method.
    estimated_params = sme_instance.estimate_phi(Y_star, phi_pool)

    # Generate a statistical summary using the estimated parameters and derived standard errors.
    stats_table = generate_stats_table(estimated_params, standard_errors.cpu().numpy())
    print("Statistical Summary of the Estimated Parameters:")
    print(stats_table)


if __name__ == '__main__':
    main()