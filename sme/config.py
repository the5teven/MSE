"""
Simplified configuration for SME.
This file now defines only the essential parameters:
- Device configurations,
- Model dimensions (param_dim and embedding_dim),
- Basic training hyperparameters,
- And fundamental optimization options.
"""
from dataclasses import dataclass, field
from typing import Tuple, Type, Dict, Any
import torch

@dataclass
class DeviceConfig:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@dataclass
class ModelComponents:
    encoder_class: Type = None
    emulator_class: Type = None

@dataclass
class TrainingConfig:
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    tau: float = 0.07
    param_dim: int = 128
    embedding_dim: int = 64
    training_steps_per_epoch: int = 100
    nn_method: str = 'faiss'
    # Remove pretraining, candidate pool size, even refinement details if not needed.
    
@dataclass
class OptimizationConfig:
    use_amp: bool = True
    use_ema: bool = False

@dataclass
class LoggingConfig:
    verbose: bool = True
    logging_level: str = "INFO"

@dataclass
class SMEConfig:
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    model_components: ModelComponents = field(default_factory=ModelComponents)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    optimization_config: OptimizationConfig = field(default_factory=OptimizationConfig)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    extra_params: Dict[str, Any] = field(default_factory=dict)