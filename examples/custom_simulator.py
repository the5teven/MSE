"""
Demonstrates:
- Custom simulator registration
- Structural breaks
- Complex time series generation
"""

import numpy as np
import matplotlib.pyplot as plt
from sme import register_custom_simulator, SimulatorConfig, simulate_time_series

@register_custom_simulator("RegimeSwitchingVAR")
def regime_var_simulator(config: SimulatorConfig) -> np.ndarray:
    """VAR model with regime switches at break points"""
    T = config.T
    n_vars = config.n_vars
    breaks = config.break_points or []
    params = config.params  # List of regime parameters

    # Debug: Print params structure
    print(f"Params structure in custom simulator: {params}")

    # Validate params structure
    if not isinstance(params, list):
        params = [params]  # Convert single dict to list of one dict
    assert all(isinstance(p, dict) for p in params), "Each regime must be a dictionary"
    assert all('phi' in p for p in params), "Each regime must have a 'phi' key"

    Y = np.zeros((T, n_vars))
    current_regime = 0
    break_idx = 0

    for t in range(1, T):
        # Check for regime switch
        if break_idx < len(breaks) and t >= breaks[break_idx]:
            current_regime = (current_regime + 1) % len(params)  # Cycle through regimes
            break_idx += 1

        # Debug: Print current regime and parameters
        print(f"Time {t}: Current regime = {current_regime}")
        print(f"Regime params: {params[current_regime]}")

        # Get parameters for current regime
        regime_params = params[current_regime]
        phi = regime_params['phi']  # Transition matrix for current regime
        noise = config.noise_dist(size=n_vars, **config.noise_kwargs)

        # VAR(1) process
        Y[t] = phi @ Y[t-1] + noise

    return Y

# Usage example
sim_config = SimulatorConfig(
    model_type="RegimeSwitchingVAR",
    params=[
        {'phi': np.array([[0.5, 0.2], [-0.1, 0.6]])},  # Regime 0
        {'phi': np.array([[0.8, -0.3], [0.4, 0.5]])}   # Regime 1
    ],
    T=500,
    n_vars=2,
    break_points=[200, 350],  # Switch regimes at t=200 and t=350
    noise_kwargs={'scale': 0.1},
    seed=42
)

# Debug: Print the params structure
print(f"Params structure in config: {sim_config.params}")

# Generate and plot sample
Y = simulate_time_series(sim_config)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(Y[:, 0], label='Variable 1')
plt.plot(Y[:, 1], label='Variable 2')
for bp in sim_config.break_points:
    plt.axvline(bp, color='r', linestyle='--', alpha=0.5, label=f'Break at t={bp}')
plt.title("Regime-Switching VAR Process")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()