import concurrent.futures
import logging

import torch
from sme import SME
from sme.models import BaseEncoder, BaseEmulator
import torch.nn as nn

# Set up logging for detailed debug information.
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# Dummy encoder with input flattening from (batch, 100, 2) to (batch, 200) and output dimension 10.
class DummyEncoder(BaseEncoder):
    def __init__(self, config):
        super().__init__(config)
        # Input is 100*2=200, output is 10.
        self.linear = nn.Linear(200, 10)

    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        # Flatten the input and apply the linear layer.
        Y_flat = Y.view(Y.size(0), -1)
        out = self.linear(Y_flat)
        # Normalize the output onto the unit sphere.
        return out / (out.norm(dim=1, keepdim=True) + 1e-6)


# Dummy emulator now expects phi input with dimension specified by training_config.param_dim.
# For our test, we will set param_dim to 10.
class DummyEmulator(BaseEmulator):
    def __init__(self, config):
        super().__init__(config)
        # Input dimension should match training_config.param_dim which we will set to 10.
        self.linear = nn.Linear(10, 10)

    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        out = self.linear(phi)
        # Normalize the output onto the unit sphere.
        return out / (out.norm(dim=1, keepdim=True) + 1e-6)


def simulate_task(task_id: int, model_type: str, params: dict, T: int = 100):
    logger.debug(f"Task {task_id}: Starting simulation using model type '{model_type}' with T={T}")
    sme_instance = SME()
    # Using the simulation part of SME to generate synthetic data.
    data = sme_instance.simulate_data(model_type=model_type, params=params, T=T, n_vars=2)
    logger.debug(f"Task {task_id}: Simulation complete with result shape {data.shape}")
    return f"Task {task_id}: Simulation result shape: {data.shape}"


def train_model_task(task_id: int):
    logger.debug(f"Task {task_id}: Starting model training")
    sme_instance = SME()
    # Configure the dummy model for testing.
    sme_instance.configure_model(encoder_class=DummyEncoder, emulator_class=DummyEmulator)
    # Override training configuration so that param_dim matches DummyEmulator input dimension.
    sme_instance.configure_training(num_epochs=2, training_steps_per_epoch=5, param_dim=10)
    sme_instance.train_model()
    logger.debug(f"Task {task_id}: Model training complete")
    return f"Task {task_id}: Training complete"


def main():
    tasks = []
    # Use ThreadPoolExecutor to run simulation and training tasks in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Schedule simulation tasks.
        for i in range(3):
            tasks.append(executor.submit(simulate_task, i, "var", {"p": 1, "phi": [[0.5, 0], [0, 0.5]]}, 100))
        # Schedule training tasks.
        for i in range(3, 6):
            tasks.append(executor.submit(train_model_task, i))

        # Wait for tasks to complete and log results.
        for future in concurrent.futures.as_completed(tasks):
            try:
                result = future.result()
                logger.info(result)
            except Exception as e:
                logger.error(f"Task generated an exception: {e}")


if __name__ == "__main__":
    main()