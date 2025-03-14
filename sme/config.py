"""
Configuration file for the SME project.
Holds various configuration parameters, including those for device, model components,
training hyperparameters, parallel data loading, logging, etc.
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
    embedding_dim: int = 64  # Added embedding_dim attribute
    memory_bank_size: int = 1024
    lr_scheduler: str = 'cosine'
    loss_type: str = 'symmetric'
    input_dim: Tuple[int, int] = (100, 2)
    refinement_steps: int = 10
    refinement_lr: float = 0.001
    training_steps_per_epoch: int = 100
    pretraining_samples: int = 1000
    pretraining_model: str = "VAR"
    candidate_pool_size: int = 500
    use_active_learning: bool = False
    use_pretraining: bool = False
    pretraining_epochs: int = 10
    use_early_stopping: bool = False
    early_stopping_patience: int = 5
    nn_method: str = 'faiss'
    faiss_index_type: str = 'FlatIP'

@dataclass
class OptimizationConfig:
    use_amp: bool = True
    use_ema: bool = False
    use_memory_bank: bool = False
    compute_fisher: bool = True
    adversarial_training: bool = False  # Enable adversarial robustness if True

@dataclass
class RegularizationConfig:
    use_weight_norm: bool = False
    spectral_normalization: bool = False
    use_gradient_clip_norm: bool = False
    grad_clip_value: float = 1.0
    moment_matching_weight: float = 0.0  # Regularization weight for moment matching

@dataclass
class LearningStrategies:
    use_curriculum_learning: bool = False
    label_smoothing: float = 0.1
    alpha: float = 0.5

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
    regularization_config: RegularizationConfig = field(default_factory=RegularizationConfig)
    learning_strategies: LearningStrategies = field(default_factory=LearningStrategies)
    logging_config: LoggingConfig = field(default_factory=LoggingConfig)
    extra_params: Dict[str, Any] = field(default_factory=dict)