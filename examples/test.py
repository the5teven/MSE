import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


# Define the custom Encoder and Emulator classes inline.
class Encoder(nn.Module):
    def __init__(self, T=100, n_vars=2, embedding_dim=64):
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
        self.fc1 = nn.Linear(n_params, 128)
        self.fc2 = nn.Linear(128, embedding_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, phi):
        # phi is expected to be of shape (batch, n_params)
        phi = F.relu(self.fc1(phi))
        phi = self.dropout(phi)
        phi = self.fc2(phi)
        return F.normalize(phi, p=2, dim=1)


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


# Dummy SME implementation for testing purposes.
class SME:
    def __init__(self):
        from collections import namedtuple
        # Create a dummy configuration using a simple Namespace-like approach.
        Config = namedtuple("Config", ["device_config", "model_components", "training_config", "optimization_config",
                                       "regularization_config", "learning_strategies", "logging_config"])
        DeviceConfig = namedtuple("DeviceConfig", ["device"])
        ModelComponents = namedtuple("ModelComponents", ["encoder_class", "emulator_class"])
        TrainingConfig = namedtuple("TrainingConfig", ["num_epochs", "training_steps_per_epoch", "batch_size"])
        OptimizationConfig = namedtuple("OptimizationConfig", ["learning_rate"])
        RegularizationConfig = namedtuple("RegularizationConfig", [])
        LearningStrategies = namedtuple("LearningStrategies", [])
        LoggingConfig = namedtuple("LoggingConfig", ["verbose", "logging_level"])

        # Set default configurations.
        self.config = Config(
            device_config=DeviceConfig(device=torch.device("cpu")),
            model_components=ModelComponents(encoder_class=None, emulator_class=None),
            training_config=TrainingConfig(num_epochs=10, training_steps_per_epoch=10, batch_size=32),
            optimization_config=OptimizationConfig(learning_rate=0.001),
            regularization_config=RegularizationConfig(),
            learning_strategies=LearningStrategies(),
            logging_config=LoggingConfig(verbose=False, logging_level="INFO"),
        )
        self.model = None

    def configure_device(self, device: torch.device):
        self.config = self.config._replace(
            device_config=self.config.device_config._replace(device=device)
        )

    def configure_model(self, encoder_class, emulator_class):
        self.config = self.config._replace(
            model_components=self.config.model_components._replace(encoder_class=encoder_class,
                                                                   emulator_class=emulator_class)
        )

    def configure_training(self, **kwargs):
        config_dict = self.config.training_config._asdict()
        for key, value in kwargs.items():
            if key in config_dict:
                config_dict[key] = value
        TrainingConfig = type(self.config.training_config)
        self.config = self.config._replace(training_config=TrainingConfig(**config_dict))

    def configure_optimization(self, **kwargs):
        config_dict = self.config.optimization_config._asdict()
        for key, value in kwargs.items():
            if key in config_dict:
                config_dict[key] = value
        OptimizationConfig = type(self.config.optimization_config)
        self.config = self.config._replace(optimization_config=OptimizationConfig(**config_dict))

    def configure_regularization(self, **kwargs):
        # For this dummy implementation, assume no regularization parameters.
        pass

    def configure_learning_strategies(self, **kwargs):
        # For this dummy implementation, assume no learning strategies.
        pass

    def configure_logging(self, verbose: bool, logging_level: str):
        self.config = self.config._replace(
            logging_config=self.config.logging_config._replace(verbose=verbose, logging_level=logging_level)
        )

    def train_model(self):
        # Instantiate and "train" the model using the configuration.
        # For simplicity, we'll set up the encoder and emulator without an actual training loop.
        device = self.config.device_config.device
        encoder_cls = self.config.model_components.encoder_class
        emulator_cls = self.config.model_components.emulator_class

        if encoder_cls is None or emulator_cls is None:
            raise ValueError("Encoder and Emulator classes must be configured before training.")

        self.model = type("SMEModel", (object,), {})()  # A dummy model object.
        # Instantiate encoder and emulator.
        self.model.encoder = encoder_cls().to(device)
        self.model.emulator = emulator_cls().to(device)
        # Dummy loss function for the model.
        self.model.loss_fn = lambda f, g: torch.mean(f - g)
        # Set training mode.
        self.model.encoder.train()
        self.model.emulator.train()
        # Print a log message.
        if self.config.logging_config.verbose:
            print("Training SMEModel with configuration:")
            print(self.config.training_config)
        # Dummy training loop.
        for epoch in range(self.config.training_config.num_epochs):
            if self.config.logging_config.verbose:
                print(f"Epoch {epoch + 1}/{self.config.training_config.num_epochs}")
            # In actual training, you'd iterate over your training data.
        if self.config.logging_config.verbose:
            print("SMEModel training completed.")

    def eval(self):
        # Set the underlying model to evaluation mode.
        self.model.encoder.eval()
        self.model.emulator.eval()

    def estimate_phi(self, Y_star, phi_pool):
        # For demonstration, select the candidate with the smallest L2 difference between encoder output and emulator output.
        device = self.config.device_config.device
        Y_tensor = torch.tensor(Y_star, dtype=torch.float32, device=device).unsqueeze(0)  # shape: (1, T, n_vars)
        with torch.no_grad():
            f_star = self.model.encoder(Y_tensor)  # shape: (1, embedding_dim)

        # Evaluate candidates from phi_pool.
        phi_candidates = torch.tensor(phi_pool, dtype=torch.float32, device=device)  # shape: (num_candidates, n_params)
        # Pass candidates through emulator.
        with torch.no_grad():
            g_candidates = self.model.emulator(phi_candidates)  # shape: (num_candidates, embedding_dim)
        # Compute distances.
        differences = torch.norm(f_star - g_candidates, dim=1)
        best_idx = torch.argmin(differences)
        best_phi = phi_candidates[best_idx].cpu().numpy()
        if self.config.logging_config.verbose:
            print("Estimated parameters:", best_phi)
        return best_phi


def main():
    # Instantiate the SME interface.
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
    sme_instance.eval()  # Set model to evaluation mode.
    aggregated_fisher = None
    batch_count = 0

    for batch in dataloader:
        phi_batch, Y_batch = batch
        phi_batch = phi_batch.to(device)
        Y_batch = Y_batch.to(device)

        # Zero out gradients.
        sme_instance.model.encoder.zero_grad()
        sme_instance.model.emulator.zero_grad()

        # Forward pass.
        f_output = sme_instance.model.encoder(Y_batch)
        g_output = sme_instance.model.emulator(phi_batch)

        # Compute loss using the model's loss function.
        loss = sme_instance.model.loss_fn(f_output, g_output)
        loss.backward()

        # Aggregate gradients from both encoder and emulator.
        gradients = []
        for param in sme_instance.model.encoder.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        for param in sme_instance.model.emulator.parameters():
            if param.grad is not None:
                gradients.append(param.grad.view(-1))
        if gradients:
            grad_flat = torch.cat(gradients)
            batch_fisher = torch.outer(grad_flat, grad_flat)
            if aggregated_fisher is None:
                aggregated_fisher = batch_fisher
            else:
                aggregated_fisher += batch_fisher
        batch_count += 1

    if aggregated_fisher is not None and batch_count > 0:
        aggregated_fisher /= batch_count

    # Regularize Fisher information to avoid singularity.
    jitter = 1e-6
    aggregated_fisher = aggregated_fisher + jitter * torch.eye(aggregated_fisher.size(0), device=device)

    try:
        aggregated_fisher_inv = torch.inverse(aggregated_fisher)
        standard_errors = torch.sqrt(torch.diag(aggregated_fisher_inv))
    except Exception as e:
        print("Error inverting aggregated Fisher information matrix:", e)
        standard_errors = torch.tensor([])

    # Estimate parameters using the SME method.
    estimated_params = sme_instance.estimate_phi(Y_star, phi_pool)

    # Generate a statistical summary using the estimated parameters and derived standard errors.
    stats_table = generate_stats_table(estimated_params, standard_errors.cpu().numpy())
    print("Statistical Summary of the Estimated Parameters:")
    print(stats_table)


if __name__ == '__main__':
    main()