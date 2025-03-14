"""
Demonstrates:
- Active learning loop
- Uncertainty sampling
- Progressive model improvement
"""

import numpy as np
from tqdm import trange
from sme import SMEConfig, SMEModel, SimulationDataset, simulate_time_series, SimulatorConfig

# Active learning config
config = SMEConfig(
    active_learning_batch_size=50,
    uncertainty_metric='variance',
    memory_bank_size=10000,
    use_curriculum_learning=True,
    num_epochs=100,
    param_dim=5
)


class ActiveLearningAgent:
    def __init__(self, param_space):
        self.param_space = param_space
        self.labeled_pool = []
        self.unlabeled_pool = np.random.uniform(
            low=param_space['low'],
            high=param_space['high'],
            size=(10000, 5)
        )

    def query(self, model, n_samples):
        uncertain_indices = model.select_uncertain_samples(self.unlabeled_pool, n_samples)
        return self.unlabeled_pool[uncertain_indices]

    def label(self, params):
        # Simulator acts as oracle
        return np.array([simulate_time_series(
            SimulatorConfig(model_type="VAR", params=p)) for p in params]
        )


# Initialize components
agent = ActiveLearningAgent({'low': [0] * 5, 'high': [1] * 5})
model = SMEModel(config)

# Active learning loop
for cycle in trange(10, desc="Active Learning Cycles"):
    # Query uncertain parameters
    query_params = agent.query(model, config.active_learning_batch_size)

    # Get labels from oracle (simulator)
    query_Y = agent.label(query_params)

    # Update dataset
    dataset = SimulationDataset(query_params, query_Y)
    model.train(dataset, incremental=True)

    # Evaluate on test set
    test_loss = model.evaluate(test_dataset)
    print(f"Cycle {cycle + 1}: Test Loss = {test_loss:.4f}")