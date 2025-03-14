from dataclasses import dataclass
from typing import Tuple
@dataclass
class SMEConfig:
    # Core Model Components
    encoder_class: type
    emulator_class: type
    loss_type: str = 'symmetric'
    input_dim: Tuple[int, int] = (100, 2)

    # Training Hyperparameters
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    tau: float = 0.07
    param_dim: int = 128
    memory_bank_size: int = 1024
    lr_scheduler: str = 'cosine'

    # Mixed Precision & Optimization
    use_amp: bool = True
    use_ema: bool = False
    use_memory_bank: bool = False
    compute_fisher: bool = True

    # Regularization
    use_weight_norm: bool = False
    spectral_normalization: bool = False
    use_gradient_clip_norm: bool = False
    grad_clip_value: float = 1.0

    # Learning Strategies
    use_curriculum_learning: bool = False
    label_smoothing: float = 0.1
    alpha: float = 0.5  # Used in AngularLoss

    # FAISS Nearest Neighbor Search
    nn_method: str = 'faiss'
    faiss_index_type: str = 'FlatIP'  # FAISS index type for nearest neighbor search

    # Breakpoint Estimation
    use_breakpoint_estimation: bool = False

    # Transfer Learning (Pretraining)
    use_pretraining: bool = False
    pretraining_epochs: int = 10

    # Early Stopping
    use_early_stopping: bool = False
    early_stopping_patience: int = 5

    # Refinement
    use_refinement: bool = False
    refinement_steps: int = 10
    refinement_lr: float = 0.001

    # Memory Bank Updates
    use_memory_bank_updates: bool = False