import numpy as np
import torch
import torch.nn as nn
import faiss
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from abc import ABC, abstractmethod
from .config import SMEConfig
from .losses import CompositeLoss, MemoryBankLoss
from .optim import EMA, CurriculumSampler
from .dataset import SimulationDataset
from .simulator import GeneralSimulator, SimulatorConfig


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for encoder models."""
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.use_weight_norm:
            self.apply(lambda m: nn.utils.weight_norm(m) if isinstance(m, nn.Linear) else m)

        if self.config.spectral_normalization:
            self.apply(lambda m: nn.utils.spectral_norm(m) if isinstance(m, nn.Linear) else m)

    @abstractmethod
    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Forward pass for the encoder."""
        pass


class BaseEmulator(nn.Module, ABC):
    """Abstract base class for emulator models."""
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.config.use_weight_norm:
            self.apply(lambda m: nn.utils.weight_norm(m) if isinstance(m, nn.Linear) else m)

        if self.config.spectral_normalization:
            self.apply(lambda m: nn.utils.spectral_norm(m) if isinstance(m, nn.Linear) else m)

    @abstractmethod
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Forward pass for the emulator."""
        pass


class SMEModel:
    def __init__(self, config: SMEConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_models()
        self._init_optimizations()
        self._init_loss()

    def _init_models(self):
        """Initialize encoder and emulator models with optional normalization."""
        encoder = self.config.encoder_class(self.config)
        emulator = self.config.emulator_class(self.config)

        self.encoder = encoder.to(self.device)
        self.emulator = emulator.to(self.device)

    def _init_optimizations(self):
        """Initialize optimization components."""
        use_amp = self.config.use_amp and torch.cuda.is_available()  # ✅ Only enable AMP if CUDA is available
        self.scaler = GradScaler(enabled=use_amp)

        self.ema = EMA(self) if self.config.use_ema else None

        self.memory_bank = None
        if self.config.use_memory_bank:
            self.memory_bank = torch.randn(
                self.config.memory_bank_size, self.config.param_dim, device=self.device, requires_grad=False
            )

        self.index = None  # Start with no index
        if self.config.nn_method == 'faiss':
            if torch.cuda.is_available():
                res = faiss.StandardGpuResources()
                self.index = faiss.GpuIndexFlatL2(res, self.config.param_dim)
            else:
                self.index = faiss.IndexFlatL2(self.config.param_dim)

    def _update_faiss_index(self, phi_pool: np.ndarray):
        """Updates FAISS index with the latest parameter pool."""
        if self.config.nn_method == 'faiss' and self.index is not None:
            self.index.reset()
            self.index.add(phi_pool.astype("float32"))

    def estimate_phi(self, Y_star: np.ndarray, phi_pool: np.ndarray):
        """Efficiently estimates parameters for a given Y_star using FAISS & refinement."""
        self.encoder.eval()
        self.emulator.eval()

        Y_tensor = torch.tensor(Y_star, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            f_star = self.encoder(Y_tensor)

            self._update_faiss_index(phi_pool)

            _, I = self.index.search(f_star.cpu().numpy(), 1)
            phi_init = phi_pool[I[0][0]]

        phi_opt = torch.tensor(phi_init, dtype=torch.float32, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([phi_opt], lr=1e-2)

        for _ in range(self.config.refinement_steps):
            optimizer.zero_grad()
            G_opt = self.emulator(phi_opt.unsqueeze(0))
            loss = 1 - torch.dot(f_star.squeeze(), G_opt.squeeze())  # Maximize similarity
            loss.backward()
            optimizer.step()

        phi_opt = phi_opt.detach().cpu().numpy()
        return phi_opt[:4].reshape(2, 2), phi_opt[4:8].reshape(2, 2), int(phi_opt[8] * 100)

    def pretrain(self, dataset: Dataset):
        """Pretraining phase using a simpler dataset."""
        print("Starting Pretraining...")
        dataloader = self._create_dataloader(dataset)
        optimizer = self._create_optimizer()

        for epoch in range(self.config.pretraining_epochs):
            self.encoder.train()
            self.emulator.train()

            for phi_batch, Y_batch in dataloader:
                phi_batch, Y_batch = phi_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                with autocast(enabled=self.config.use_amp):
                    f_output = self.encoder(Y_batch)
                    g_output = self.emulator(phi_batch)
                    loss = self.loss_fn(f=f_output, g=g_output)

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            for epoch in tqdm(range(self.config.num_epochs), desc="Pre-Training SME Model"):
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")

        print("Pretraining Complete.")

    def _init_loss(self):
        self.loss_fn = CompositeLoss(self.config)
        self.memory_bank_loss = MemoryBankLoss(self.config) if self.config.use_memory_bank else None

    def train(self, dataset: Dataset):
        """Train the SME model with optional pretraining, early stopping, and memory bank updates."""
        if self.config.use_pretraining:
            pretraining_data = self._generate_pretraining_data()
            self.pretrain(pretraining_data)

        dataloader = self._create_dataloader(dataset)
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)

        best_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(self.config.num_epochs), desc="Training SME Model"):
            self.encoder.train()
            self.emulator.train()
            total_loss = 0

            for phi_batch, Y_batch in dataloader:
                phi_batch, Y_batch = phi_batch.to(self.device, non_blocking=True), Y_batch.to(self.device, non_blocking=True)
                optimizer.zero_grad()

                with autocast(enabled=self.config.use_amp):
                    f_output = self.encoder(Y_batch)
                    g_output = self.emulator(phi_batch)
                    loss = self.loss_fn(f=f_output, g=g_output)

                    if self.memory_bank_loss:
                        loss += self.memory_bank_loss(f_output, g_output)

                self.scaler.scale(loss).backward()

                if self.config.use_gradient_clip_norm:
                    self.scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.encoder.parameters()) + list(self.emulator.parameters()),
                        self.config.grad_clip_value
                    )

                self.scaler.step(optimizer)
                self.scaler.update()
                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)

            if self.config.use_early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.early_stopping_patience:
                        print("Early stopping triggered.")
                        break
            for epoch in tqdm(range(self.config.num_epochs), desc="Training SME Model"):
                print(f"Epoch {epoch + 1}/{self.config.num_epochs}")

            if scheduler:
                scheduler.step()

        print("Training Complete.")

    def _create_dataloader(self, dataset: Dataset) -> DataLoader:
        """Create a DataLoader for training, with optional curriculum learning."""
        if self.config.use_curriculum_learning:
            sampler = CurriculumSampler(dataset, self._difficulty_fn)
            return DataLoader(dataset, batch_size=self.config.batch_size, sampler=sampler, pin_memory=True)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create an optimizer."""
        params = list(self.encoder.parameters()) + list(self.emulator.parameters())
        return torch.optim.AdamW(params, lr=self.config.learning_rate, weight_decay=self.config.weight_decay)

    def _create_scheduler(self, optimizer: torch.optim.Optimizer):
        """Create a learning rate scheduler."""
        if self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)
        return None

    def _difficulty_fn(self, sample):
        """Compute a difficulty score for curriculum learning."""
        return np.random.rand()

