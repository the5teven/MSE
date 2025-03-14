import torch
from sme.sme import SME
from sme.models import BaseEncoder, BaseEmulator
from sme.config import SMEConfig
from sme.simulator import register_custom_simulator, SimulatorConfig


# Register the custom AR(1) model simulator
@register_custom_simulator("ar1")
def simulate_ar1(config: SimulatorConfig) -> torch.Tensor:
    phi = config.params["phi"]
    intercept = config.params["intercept"]
    T = config.T
    n_vars = config.n_vars

    # Ensure noise_kwargs is not None and has default values
    noise_kwargs = config.noise_kwargs if config.noise_kwargs is not None else {"mean": 0, "std": 1}

    Y = torch.zeros((T, n_vars), dtype=torch.float32, device=config.device)
    Y[0, :] = intercept  # Initialize the first value

    for t in range(1, T):
        noise = torch.normal(mean=noise_kwargs["mean"], std=noise_kwargs["std"], size=(n_vars,), device=config.device)
        Y[t, :] = phi * Y[t - 1, :] + intercept + noise

    return Y


# Define SimpleEncoder class
class SimpleEncoder(BaseEncoder):
    def __init__(self, n_vars, embedding_dim):
        super(SimpleEncoder, self).__init__(n_vars, embedding_dim)
        self.linear = torch.nn.Linear(n_vars * 100, embedding_dim)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        Y_flat = Y.view(Y.size(0), -1)
        return self.linear(Y_flat)


# Define SimpleEmulator class
class SimpleEmulator(BaseEmulator):
    def __init__(self, n_params, embedding_dim):
        super(SimpleEmulator, self).__init__(n_params, embedding_dim)
        self.linear = torch.nn.Linear(n_params, embedding_dim)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        return self.linear(phi)


# Test function
def test_ar1_model():
    config = SMEConfig()
    config.model_components.encoder_class = SimpleEncoder
    config.model_components.emulator_class = SimpleEmulator
    config.training_config.use_pretraining = True
    config.training_config.pretraining_epochs = 10
    config.training_config.num_epochs = 20
    config.training_config.input_dim = (100, 2)
    config.training_config.param_dim = 3
    config.training_config.embedding_dim = 5
    config.training_config.batch_size = 16

    sme = SME()
    sme.configure_model(SimpleEncoder, SimpleEmulator)
    sme.configure_training(
        batch_size=config.training_config.batch_size,
        num_epochs=config.training_config.num_epochs,
        use_pretraining=config.training_config.use_pretraining,
        pretraining_epochs=config.training_config.pretraining_epochs,
        input_dim=config.training_config.input_dim,
        param_dim=config.training_config.param_dim,
        embedding_dim=config.training_config.embedding_dim,
    )

    # Generate training data with varying theta values
    T = 100
    n_vars = 2
    num_samples = 1000
    for _ in range(num_samples):
        params = {
            "phi": torch.rand(1).item(),
            "intercept": torch.rand(1).item()
        }
        simulated_data = sme.simulate_data("ar1", params, T, n_vars)
        sme.train_model()

    # Evaluate model with a fixed theta
    eval_params = {
        "phi": 0.5,
        "intercept": 1.0
    }
    Y_star = sme.simulate_data("ar1", eval_params, T, n_vars)
    stats_table = sme.estimate_parameters(Y_star, candidate_size=1000)

    print(stats_table)


if __name__ == "__main__":
    test_ar1_model()