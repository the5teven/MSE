import numpy as np
import torch
import torch.nn as nn
import faiss
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from abc import ABC, abstractmethod
from .config import SMEConfig
from .losses import CompositeLoss, MemoryBankLoss
from .optim import EMA


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for encoder models."""
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        pass


class BaseEmulator(nn.Module, ABC):
    """Abstract base class for emulator models."""
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        pass


class SMEModel:
    def __init__(self, config: SMEConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()
        self._init_optimizations()
        self.loss_fn = CompositeLoss(self.config)  # ✅ Main contrastive loss
        self.memory_bank_loss = MemoryBankLoss(
            self.config) if self.config.use_memory_bank else None  # ✅ Optional memory bank loss


    def _init_models(self):
        """Initialize encoder and emulator models."""
        self.encoder = self.config.encoder_class(self.config).to(self.device)
        self.emulator = self.config.emulator_class(self.config).to(self.device)

    def _init_optimizations(self):
        """Initialize optimization components."""
        self.scaler = GradScaler(enabled=self.config.use_amp and torch.cuda.is_available())
        self.ema = EMA(self) if self.config.use_ema else None

        # Memory Bank
        self.memory_bank = None
        if self.config.use_memory_bank:
            self.memory_bank = torch.randn(
                self.config.memory_bank_size, self.config.param_dim, device=self.device, requires_grad=False
            )

        # FAISS Index for Efficient Nearest Neighbor Search
        self.index = None
        if self.config.nn_method == 'faiss':
            self.index = faiss.IndexFlatL2(self.config.param_dim)

    def estimate_phi(self, Y_star: np.ndarray, phi_pool: np.ndarray):
        """Estimate parameters for a given Y_star using FAISS and refinement."""
        self.encoder.eval()
        self.emulator.eval()

        Y_tensor = torch.tensor(Y_star, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            f_star = self.encoder(Y_tensor)

            if self.config.nn_method == 'faiss' and self.index is not None:
                self.index.reset()
                self.index.add(phi_pool.astype("float32"))
                _, I = self.index.search(f_star.cpu().numpy(), 1)
                phi_init = phi_pool[I[0][0]]
            else:
                scores = torch.matmul(f_star, self.emulator(torch.tensor(phi_pool, dtype=torch.float32, device=self.device)).T)
                phi_init = phi_pool[torch.argmax(scores).item()]

        # Refinement with gradient optimization
        phi_opt = torch.tensor(phi_init, dtype=torch.float32, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([phi_opt], lr=self.config.refinement_lr)

        for _ in range(self.config.refinement_steps):
            optimizer.zero_grad()
            G_opt = self.emulator(phi_opt.unsqueeze(0))
            loss = 1 - torch.dot(f_star.squeeze(), G_opt.squeeze())  # Maximize similarity
            loss.backward()
            optimizer.step()

        return phi_opt.detach().cpu().numpy()

    def pretrain(self):
        """Pretraining phase using dynamically generated data."""
        print("Starting Pretraining...")
        self._run_training_loop(self.config.pretraining_epochs, pretraining_mode=True)
        print("Pretraining Complete.")

    def train(self):
        """Train the SME model dynamically generating training data."""
        if self.config.use_pretraining:
            self.pretrain()

        self._run_training_loop(self.config.num_epochs, pretraining_mode=False)
        print("Training Complete.")

    def _run_training_loop(self, num_epochs, pretraining_mode):
        """Unified training loop for both pretraining and full training."""
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(num_epochs),
                          desc="Training SME Model" if not pretraining_mode else "Pretraining SME Model"):
            self.encoder.train()
            self.emulator.train()
            total_loss = 0

            for _ in range(
                    self.config.training_steps_per_epoch if not pretraining_mode else self.config.pretraining_samples // self.config.batch_size):
                phi_batch, Y_batch = self._generate_training_batch(pretraining_mode=pretraining_mode)
                phi_batch, Y_batch = phi_batch.to(self.device), Y_batch.to(self.device)

                optimizer.zero_grad()
                with autocast(enabled=self.config.use_amp):
                    f_output = self.encoder(Y_batch)
                    g_output = self.emulator(phi_batch)
                    loss = self.loss_fn(f=f_output, g=g_output)

                    if self.memory_bank_loss:
                        loss += self.memory_bank_loss(f_output, g_output)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / (self.config.training_steps_per_epoch if not pretraining_mode else (
                        self.config.pretraining_samples // self.config.batch_size))

            if self.config.use_early_stopping and not pretraining_mode:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        print("Early stopping triggered.")
                        break

            if scheduler and not pretraining_mode:
                scheduler.step()

    def _generate_training_batch(self):
        """Dynamically generates a batch of training data, with optional active learning."""
        batch_size = self.config.batch_size
        t, n_vars = self.config.input_dim

        if self.config.use_active_learning:
            phi_candidates = torch.rand((self.config.candidate_pool_size, n_vars, n_vars), device=self.device) * 2 - 1
            tau_candidates = torch.randint(10, t - 10, (self.config.candidate_pool_size,), device=self.device).float()

            with torch.no_grad():
                phi_flat = phi_candidates.flatten(1)
                tau_scaled = tau_candidates / t
                phi_input = torch.cat([phi_flat, tau_scaled.unsqueeze(1)], dim=1)
                g_emb = self.emulator(phi_input)

            uncertainty = -torch.var(g_emb, dim=1)
            selected_indices = torch.argsort(uncertainty)[:batch_size]
            selected_phi = phi_candidates[selected_indices]
        else:
            selected_phi = torch.rand((batch_size, n_vars, n_vars), device=self.device) * 2 - 1

        phi_batch = selected_phi.flatten(1)
        y_batch = torch.rand(batch_size, t, n_vars, device=self.device)  # Simulated output

        return phi_batch, y_batch



    def _create_optimizer(self):
        params = list(self.encoder.parameters()) + list(self.emulator.parameters())
        return torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _create_scheduler(self, optimizer):
        if self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        return None
