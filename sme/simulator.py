"""
Simulator module for the SME project.
Generates synthetic data using various simulation models (e.g., ARIMA, VAR, etc.).
Efficient implementations and vectorization are used where possible.
"""
import torch
from dataclasses import dataclass
from typing import Dict, Callable, Optional, List, Union, Any

# Registry for custom simulation functions.
CUSTOM_SIMULATORS: Dict[str, Callable] = {}

def register_custom_simulator(name: str):
    """Decorator to register a custom simulator function."""
    def decorator(func: Callable):
        CUSTOM_SIMULATORS[name.lower()] = func
        return func
    return decorator

@dataclass
class SimulatorConfig:
    model_type: Union[str, List[str]]
    params: Union[Dict[str, Any], List[Dict[str, Any]]]
    T: int
    n_vars: int = 1
    noise_dist: Callable = staticmethod(torch.randn)
    noise_kwargs: Optional[Dict[str, Any]] = None
    exogenous_vars: Optional[torch.Tensor] = None
    time_varying_params: Optional[Dict[str, torch.Tensor]] = None
    break_points: Optional[List[int]] = None
    seed: Optional[int] = None
    use_gpu: bool = True
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GeneralSimulator:
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.device = self.config.device
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
        self.noise_kwargs = self.config.noise_kwargs or {"mean": 0, "std": 1}
        self.simulation_methods = {
            "arima": self._simulate_arima,
            "garch": self._simulate_garch,
            "var": self._simulate_var,
            "statespace": self._simulate_state_space,
            "stochasticvolatility": self._simulate_stochastic_volatility,
            "regimeswitching": self._simulate_regime_switching,
        }

    def _simulate_single_model(self) -> torch.Tensor:
        model_type = (self.config.model_type[0].lower() if isinstance(self.config.model_type, list) 
                      else self.config.model_type.lower())
        if model_type in CUSTOM_SIMULATORS:
            return CUSTOM_SIMULATORS[model_type](self.config).to(self.device)
        if model_type in self.simulation_methods:
            return self.simulation_methods[model_type]()
        raise ValueError(f"Unsupported model type: {self.config.model_type}")

    def simulate(self) -> torch.Tensor:
        if self.config.break_points:
            return self._simulate_with_breaks()
        else:
            return self._simulate_single_model()

    def _simulate_with_breaks(self) -> torch.Tensor:
        break_points = sorted(self.config.break_points)
        if break_points[0] != 0:
            break_points.insert(0, 0)
        if break_points[-1] != self.config.T:
            break_points.append(self.config.T)
        Y = torch.zeros((self.config.T, self.config.n_vars), device=self.device)
        for start, end in zip(break_points[:-1], break_points[1:]):
            segment_config = SimulatorConfig(
                model_type=self.config.model_type,
                params=self.config.params,
                T=end - start,
                n_vars=self.config.n_vars,
                noise_dist=self.config.noise_dist,
                noise_kwargs=self.config.noise_kwargs,
                exogenous_vars=self.config.exogenous_vars,
                time_varying_params=self.config.time_varying_params,
                break_points=None,
                seed=self.config.seed,
                use_gpu=self.config.use_gpu,
                device=self.device
            )
            segment_simulator = GeneralSimulator(segment_config)
            Y[start:end] = segment_simulator.simulate()
        return Y

    def _simulate_arima(self) -> torch.Tensor:
        p, d, q = (self.config.params.get(k, 1) for k in ["p", "d", "q"])
        ar_params = torch.tensor(self.config.params.get("ar_params", [0.5] * p), device=self.device)
        ma_params = torch.tensor(self.config.params.get("ma_params", [0.5] * q), device=self.device)
        noise = torch.randn(self.config.T + d + q, device=self.device, **self.noise_kwargs)
        series = torch.zeros(self.config.T + d + q, device=self.device)
        for t in range(max(p, q), self.config.T + d + q):
            ar_term = torch.dot(ar_params, series[t-p:t].flip(0))
            ma_term = torch.dot(ma_params, noise[t-q:t].flip(0))
            series[t] = ar_term + ma_term + noise[t]
        if d > 0:
            for _ in range(d):
                series = torch.diff(series)
        return series[-self.config.T:].reshape(-1, self.config.n_vars)

    def _simulate_garch(self) -> torch.Tensor:
        p = self.config.params.get("p", 1)
        q = self.config.params.get("q", 1)
        alpha = torch.tensor(self.config.params.get("alpha", [0.1] * p), device=self.device)
        beta = torch.tensor(self.config.params.get("beta", [0.8] * q), device=self.device)
        volatility = torch.zeros(self.config.T + max(p, q), device=self.device)
        returns = torch.zeros(self.config.T + max(p, q), device=self.device)
        for t in range(max(p, q), volatility.shape[0]):
            volatility[t] = (alpha @ (volatility[t-p:t].flip(0)**2)) + (beta @ (returns[t-q:t].flip(0)**2))
            returns[t] = torch.randn(1, device=self.device) * torch.sqrt(volatility[t])
        return returns[-self.config.T:].reshape(-1, self.config.n_vars)

    def _simulate_var(self) -> torch.Tensor:
        p = self.config.params.get("p", 1)
        phi = self.config.params.get("phi", torch.eye(self.config.n_vars) * 0.5)
        if not isinstance(phi, torch.Tensor):
            phi = torch.tensor(phi, device=self.device)
        Y = torch.zeros((self.config.T + p, self.config.n_vars), device=self.device, dtype=torch.float32)
        mean = float(self.noise_kwargs.get("mean", 0.0))
        std = float(self.noise_kwargs.get("std", 1.0))
        for t in range(p, self.config.T + p):
            for lag in range(1, p + 1):
                Y[t] += phi @ Y[t - lag]
            Y[t] += torch.normal(
                mean=torch.full((self.config.n_vars,), mean, dtype=torch.float32, device=self.device),
                std=torch.full((self.config.n_vars,), std, dtype=torch.float32, device=self.device)
            )
        return Y[-self.config.T:]

    def _simulate_state_space(self) -> torch.Tensor:
        A = torch.tensor(self.config.params.get("A", torch.eye(self.config.n_vars)), device=self.device)
        C = torch.tensor(self.config.params.get("C", torch.eye(self.config.n_vars)), device=self.device)
        Q = torch.tensor(self.config.params.get("Q", torch.eye(self.config.n_vars)), device=self.device)
        R = torch.tensor(self.config.params.get("R", torch.eye(self.config.n_vars)), device=self.device)
        states = torch.zeros((self.config.T, self.config.n_vars), device=self.device)
        observations = torch.zeros((self.config.T, self.config.n_vars), device=self.device)
        for t in range(1, self.config.T):
            process_noise = torch.randn(self.config.n_vars, device=self.device) @ Q
            observation_noise = torch.randn(self.config.n_vars, device=self.device) @ R
            states[t] = A @ states[t - 1] + process_noise
            observations[t] = C @ states[t] + observation_noise
        return observations

    def _simulate_stochastic_volatility(self) -> torch.Tensor:
        mu = self.config.params.get("mu", 0.0)
        phi = self.config.params.get("phi", 0.9)
        sigma = self.config.params.get("sigma", 0.1)
        log_volatility = torch.zeros(self.config.T, device=self.device)
        returns = torch.zeros(self.config.T, device=self.device)
        for t in range(1, self.config.T):
            log_volatility[t] = mu + phi * (log_volatility[t - 1] - mu) + torch.randn(1, device=self.device) * sigma
            returns[t] = torch.randn(1, device=self.device) * torch.exp(log_volatility[t] / 2)
        return returns.reshape(-1, self.config.n_vars)

    def _simulate_regime_switching(self) -> torch.Tensor:
        n_regimes = self.config.params.get("n_regimes", 2)
        transition_matrix = torch.tensor(self.config.params.get("transition_matrix", 
                                                                   torch.ones((n_regimes, n_regimes)) / n_regimes),
                                           device=self.device)
        regime_params = self.config.params.get("regime_params", [{"mu": 0.0, "sigma": 1.0} for _ in range(n_regimes)])
        regimes = torch.zeros(self.config.T, dtype=torch.long, device=self.device)
        returns = torch.zeros(self.config.T, device=self.device)
        for t in range(1, self.config.T):
            regimes[t] = torch.multinomial(transition_matrix[regimes[t - 1]], 1).item()
            params = regime_params[regimes[t]]
            returns[t] = torch.randn(1, device=self.device) * params["sigma"] + params["mu"]
        return returns.reshape(-1, self.config.n_vars)