import numpy as np
import torch
import torch.nn as nn
import faiss
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from abc import ABC, abstractmethod
import logging
from .config import SMEConfig
from .losses import TotalInfoNCELoss
from .optim import EMA

class BaseEncoder(nn.Module, ABC):
    def __init__(self, n_vars, embedding_dim):
        super().__init__()
        self.n_vars = n_vars
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        pass

class BaseEmulator(nn.Module, ABC):
    def __init__(self, n_params, embedding_dim):
        super().__init__()
        self.n_params = n_params
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        pass

class SMEModel:
    def __init__(self, config: SMEConfig):
        self.logger = logging.getLogger(__name__)
        logging_level = getattr(logging, config.logging_config.logging_level.upper(), logging.INFO)
        logging.basicConfig(level=logging_level)
        self.config = config
        self.device = self.config.device_config.device
        self._init_models()
        self._init_optimizations()
        self.loss_fn = TotalInfoNCELoss(self.config)
        self.training_losses = []
        self.logger.info("SMEModel initialized successfully.")

    def _init_models(self):
        self.encoder = self.config.model_components.encoder_class(
            self.config.training_config.input_dim[1],
            self.config.training_config.embedding_dim
        ).to(self.device)
        self.emulator = self.config.model_components.emulator_class(
            self.config.training_config.param_dim,
            self.config.training_config.embedding_dim
        ).to(self.device)
        self.logger.info("Models (encoder/emulator) initialized.")

    def _init_optimizations(self):
        self.scaler = GradScaler(enabled=self.config.optimization_config.use_amp and torch.cuda.is_available())
        self.ema = EMA(self) if self.config.optimization_config.use_ema else None
        if self.config.training_config.nn_method == 'faiss':
            self.index = faiss.IndexFlatL2(self.config.training_config.param_dim)
        else:
            self.index = None
        self.logger.info("Optimization components set up.")

    def eval(self):
        self.encoder.eval()
        self.emulator.eval()
        self.logger.info("SMEModel set to evaluation mode.")

    def estimate_phi(self, Y_star: np.ndarray, phi_pool: np.ndarray):
        self.eval()
        Y_tensor = torch.tensor(Y_star, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            f_star = self.encoder(Y_tensor)
            if self.config.training_config.nn_method == 'faiss' and self.index is not None:
                self.index.reset()
                self.index.add(phi_pool.astype("float32"), phi_pool.shape[0])
                _, I = self.index.search(f_star.cpu().numpy(), 1)
                phi_init = phi_pool[I[0][0]]
            else:
                emulator_embeddings = self.emulator(
                    torch.tensor(phi_pool, dtype=torch.float32, device=self.device)
                )
                scores = torch.matmul(f_star, emulator_embeddings.T)
                phi_init = phi_pool[torch.argmax(scores).item()]
        phi_opt = torch.tensor(phi_init, dtype=torch.float32, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([phi_opt], lr=self.config.training_config.refinement_lr)
        for _ in range(self.config.training_config.refinement_steps):
            optimizer.zero_grad()
            G_opt = self.emulator(phi_opt.unsqueeze(0))
            loss = 1 - torch.dot(f_star.squeeze(), G_opt.squeeze())
            loss.backward()
            optimizer.step()
        self.logger.info(f"Estimated parameters: {phi_opt.detach().cpu().numpy()}")
        return phi_opt.detach().cpu().numpy()

    def compute_standard_errors(self, Y_star: np.ndarray, estimated_params: np.ndarray):
        device = self.device
        Y_tensor = torch.tensor(Y_star, dtype=torch.float32, device=device).unsqueeze(0)
        f_star = self.encoder(Y_tensor)
        phi_opt = torch.tensor(estimated_params, dtype=torch.float32, device=device, requires_grad=True)
        G_opt = self.emulator(phi_opt.unsqueeze(0))
        loss = 1 - torch.dot(f_star.squeeze(), G_opt.squeeze())
        grad_phi = torch.autograd.grad(loss, phi_opt)[0]
        fisher_information = torch.outer(grad_phi, grad_phi)
        jitter = 1e-6
        fisher_information += jitter * torch.eye(len(estimated_params), device=device)
        fisher_inv = torch.inverse(fisher_information)
        se = torch.sqrt(torch.diag(fisher_inv))
        return se.cpu().numpy()

    def train(self):
        total_epochs = self.config.training_config.num_epochs
        max_iterations = total_epochs * self.config.training_config.training_steps_per_epoch
        with tqdm(total=max_iterations, desc="Overall Training Progress", unit="iteration") as pbar:
            self._run_training_loop(self.config.training_config.num_epochs, pbar=pbar)
        self.logger.info("Training completed.")

    def _run_training_loop(self, num_epochs, pbar=None):
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        best_loss = float("inf")
        patience_counter = 0
        for epoch in range(num_epochs):
            self.encoder.train()
            self.emulator.train()
            total_loss = 0
            for _ in range(self.config.training_config.training_steps_per_epoch):
                phi_batch, Y_batch = self._generate_training_batch()
                phi_batch, Y_batch = phi_batch.to(self.device), Y_batch.to(self.device)
                optimizer.zero_grad()
                with autocast():
                    f_output = self.encoder(Y_batch)
                    g_output = self.emulator(phi_batch)
                    loss = self.loss_fn(f=f_output, g=g_output)
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                total_loss += loss.item()
                if pbar:
                    pbar.update(1)
            avg_loss = total_loss / self.config.training_config.training_steps_per_epoch
            self.training_losses.append(avg_loss)
            if self.config.training_config.use_early_stopping:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.config.training_config.early_stopping_patience:
                        self.logger.info("Early stopping triggered.")
                        break
            if scheduler:
                scheduler.step()
            if pbar:
                pbar.set_postfix({"epoch": epoch + 1, "avg_loss": avg_loss})

    def _generate_training_batch(self):
        batch_size = self.config.training_config.batch_size
        t, n_vars = self.config.training_config.input_dim
        phi_batch = torch.rand((batch_size, self.config.training_config.param_dim), device=self.device) * 2 - 1
        Y_batch = torch.rand(batch_size, t, n_vars, device=self.device)
        return phi_batch, Y_batch

    def generate_candidate_pool(self, candidate_size: int):
        n_params = self.config.training_config.param_dim
        return np.random.uniform(-1, 1, size=(candidate_size, n_params)).astype(np.float32)

    def _create_optimizer(self):
        params = list(self.encoder.parameters()) + list(self.emulator.parameters())
        return torch.optim.AdamW(params, lr=self.config.training_config.learning_rate,
                                 weight_decay=self.config.training_config.weight_decay)

    def _create_scheduler(self, optimizer):
        if self.config.training_config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.training_config.num_epochs)
        return None