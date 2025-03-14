from typing import Any, Dict, Type
import torch
from .config import SMEConfig
from .models import SMEModel
from .simulator import GeneralSimulator, SimulatorConfig
from .stats import generate_stats_table
class SME:
    def __init__(self):
        self.config = SMEConfig()
        self.model = None

    def configure_device(self, device: torch.device):
        self.config.device_config.device = device

    def configure_model(self, encoder_class: Type, emulator_class: Type):
        self.config.model_components.encoder_class = encoder_class
        self.config.model_components.emulator_class = emulator_class

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

    def train_model(self):
        # Instantiate the model using the current configuration.
        self.model = SMEModel(self.config)
        self.model.train()

    def simulate_data(self, model_type: str, params: Dict[str, Any], T: int, n_vars: int = 1):
        sim_config = SimulatorConfig(
            model_type=model_type,
            params=params,
            T=T,
            n_vars=n_vars,
            device=self.config.device_config.device
        )
        simulator = GeneralSimulator(sim_config)
        return simulator.simulate()

    def estimate_phi(self, Y_star, phi_pool):
        # Delegate to the internal model's estimate_phi method.
        return self.model.estimate_phi(Y_star, phi_pool)

    def eval(self):
        # Delegate evaluation to the internal model.
        self.model.eval()

    def estimate_parameters(self, Y_star, phi_pool, dataloader):
        """
        Estimate parameters and generate a statistical summary comparing the estimated parameters
        with their standard errors computed from the Fisher information with respect to the refined
        parameter vector.
        """
        # First estimate φ using the established refinement process.
        estimated_params = self.estimate_phi(Y_star, phi_pool)

        # Compute Fisher information for φ.
        # Re-run the forward pass for φ with gradient tracking.
        self.eval()
        device = self.config.device_config.device
        Y_tensor = torch.tensor(Y_star, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            f_star = self.model.encoder(Y_tensor)

        # Turn the estimated parameters into a tensor for gradient computation.
        phi_opt = torch.tensor(estimated_params, dtype=torch.float32, device=device, requires_grad=True)
        # Forward pass through the emulator using the refined φ.
        G_opt = self.model.emulator(phi_opt.unsqueeze(0))
        # Compute a loss whose gradient with respect to φ will be used for Fisher information.
        loss = 1 - torch.dot(f_star.squeeze(), G_opt.squeeze())

        # Compute gradient of the loss with respect to φ.
        grad_phi = torch.autograd.grad(loss, phi_opt)[0]
        # Compute Fisher information as the outer product of grad_phi.
        fisher_information_phi = torch.outer(grad_phi, grad_phi)
        # Regularize Fisher information matrix to avoid singularity.
        jitter = 1e-6
        fisher_information_phi = fisher_information_phi + jitter * torch.eye(fisher_information_phi.size(0), device=device)

        # Invert Fisher information to obtain the covariance matrix and compute standard errors.
        fisher_information_inv = torch.inverse(fisher_information_phi)
        standard_errors = torch.sqrt(torch.diag(fisher_information_inv))

        # Generate and return the statistics table.
        return generate_stats_table(estimated_params, standard_errors.cpu().numpy())