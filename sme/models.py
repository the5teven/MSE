import numpy as np
import torch
import torch.nn as nn
import faiss
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler  # Updated import for deprecation warnings
from abc import ABC, abstractmethod
import logging
from .config import SMEConfig
from .losses import CompositeLoss, MemoryBankLoss
from .optim import EMA

class BaseEncoder(nn.Module, ABC):
    """Abstract base class for encoder models."""
    def __init__(self, n_vars, embedding_dim):
        super().__init__()
        self.n_vars = n_vars
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, Y: torch.Tensor) -> torch.Tensor:
        """Encode observations into an embedding space."""
        pass

class BaseEmulator(nn.Module, ABC):
    """Abstract base class for emulator models."""
    def __init__(self, n_params, embedding_dim):
        super().__init__()
        self.n_params = n_params
        self.embedding_dim = embedding_dim

    @abstractmethod
    def forward(self, phi: torch.Tensor) -> torch.Tensor:
        """Map parameter space to an embedding."""
        pass

class SMEModel:
    def __init__(self, config: SMEConfig):
        # Initialize logger.
        self.logger = logging.getLogger(__name__)
        logging_level = getattr(logging, config.logging_config.logging_level.upper(), logging.INFO)
        logging.basicConfig(level=logging_level)

        self.config = config
        self.device = self.config.device_config.device
        self._init_models()
        self._init_optimizations()
        self.loss_fn = CompositeLoss(self.config)
        self.memory_bank_loss = MemoryBankLoss(self.config) if self.config.optimization_config.use_memory_bank else None

        self.logger.info("SMEModel initialized successfully.")

    def _init_models(self):
        """Instantiate encoder and emulator models."""
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
        """Initialize optimizers, AMP scaler, EMA, memory bank and FAISS index if applicable."""
        self.scaler = GradScaler(enabled=self.config.optimization_config.use_amp and torch.cuda.is_available())
        self.ema = EMA(self) if self.config.optimization_config.use_ema else None

        self.memory_bank = None
        if self.config.optimization_config.use_memory_bank:
            self.memory_bank = torch.randn(
                self.config.training_config.memory_bank_size,
                self.config.training_config.param_dim,
                device=self.device,
                requires_grad=False
            )
        if self.config.training_config.nn_method == 'faiss':
            self.index = faiss.IndexFlatL2(self.config.training_config.param_dim)
        else:
            self.index = None
        self.logger.info("Optimization components set up.")

    def eval(self):
        """
        Set the encoder and emulator models to evaluation mode.
        This method allows SMEModel to be used in contexts where model.eval() is expected.
        """
        self.encoder.eval()
        self.emulator.eval()
        self.logger.info("SMEModel set to evaluation mode.")

    def estimate_phi(self, Y_star: np.ndarray, phi_pool: np.ndarray):
        """
        Estimate parameters for observed data Y_star using ANN to retrieve a candidate
        and then refining via gradient-based optimization.
        """
        self.eval()
        Y_tensor = torch.tensor(Y_star, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            f_star = self.encoder(Y_tensor)
            if self.config.training_config.nn_method == 'faiss' and self.index is not None:
                self.index.reset()
                self.index.add(phi_pool.astype("float32"), phi_pool.shape[0])
                distances, I = self.index.search(f_star.cpu().numpy(), 1)
                phi_init = phi_pool[I[0][0]]
            else:
                # Compute similarity scores in embedding space.
                emulator_embeddings = self.emulator(torch.tensor(phi_pool, dtype=torch.float32, device=self.device))
                scores = torch.matmul(f_star, emulator_embeddings.T)
                phi_init = phi_pool[torch.argmax(scores).item()]

        # Refine candidate using gradient based optimization.
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
        """
        Compute standard errors using the Jacobian from the emulator.
        1. Pass Y_star through encoder to get f_star.
        2. Run emulator on estimated_params to get G_opt.
        3. Compute the gradient (Jacobian) of the emulator output with respect to parameters.
        4. Approximate the Fisher information matrix; invert to get parameter covariance and SE.
        """
        device = self.device
        Y_tensor = torch.tensor(Y_star, dtype=torch.float32, device=device).unsqueeze(0)
        f_star = self.encoder(Y_tensor)  # [1, embedding_dim]
        phi_opt = torch.tensor(estimated_params, dtype=torch.float32, device=device, requires_grad=True)
        G_opt = self.emulator(phi_opt.unsqueeze(0))  # [1, embedding_dim]
        loss = 1 - torch.dot(f_star.squeeze(), G_opt.squeeze())
        grad_phi = torch.autograd.grad(loss, phi_opt)[0]
        fisher_information = torch.outer(grad_phi, grad_phi)
        jitter = 1e-6
        fisher_information += jitter * torch.eye(len(estimated_params), device=device)
        fisher_inv = torch.inverse(fisher_information)
        se = torch.sqrt(torch.diag(fisher_inv))
        return se.cpu().numpy()

    def pretrain(self):
        """Perform pretraining with dynamically generated data."""
        self.logger.info("Pretraining started...")
        self._run_training_loop(self.config.training_config.pretraining_epochs, pretraining_mode=True)
        self.logger.info("Pretraining completed.")

    def train(self):
        """Train the model (with pretraining if configured) using a parallelized and efficient training loop."""
        if self.config.training_config.use_pretraining:
            self.pretrain()
        self._run_training_loop(self.config.training_config.num_epochs, pretraining_mode=False)
        self.logger.info("Training completed.")

    def _run_training_loop(self, num_epochs, pretraining_mode):
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer)
        best_loss = float("inf")
        patience_counter = 0

        for epoch in tqdm(range(num_epochs), desc="Training SME Model" if not pretraining_mode else "Pretraining SME Model"):
            self.encoder.train()
            self.emulator.train()
            total_loss = 0
            iterations = (self.config.training_config.training_steps_per_epoch
                          if not pretraining_mode
                          else self.config.training_config.pretraining_samples // self.config.training_config.batch_size)
            for iteration in range(iterations):
                phi_batch, Y_batch = self._generate_training_batch(pretraining_mode=pretraining_mode)
                phi_batch, Y_batch = phi_batch.to(self.device), Y_batch.to(self.device)
                optimizer.zero_grad()
                with autocast():  # Corrected usage for deprecation warning
                    f_output = self.encoder(Y_batch)
                    g_output = self.emulator(phi_batch)
                    loss = self.loss_fn(f=f_output, g=g_output)

                    # Moment matching regularization if configured.
                    if self.config.regularization_config.moment_matching_weight > 0 and "economic_moments" in self.config.extra_params:
                        mY = self.config.extra_params["economic_moments"](Y_batch)
                        loss += self.config.regularization_config.moment_matching_weight * torch.norm(f_output - mY, p=2)

                    # Memory bank loss if active.
                    if self.memory_bank_loss:
                        loss += self.memory_bank_loss(f_output, g_output)

                    # Optional adversarial training.
                    if self.config.optimization_config.adversarial_training:
                        epsilon = 0.01
                        delta = epsilon * torch.sign(torch.autograd.grad(loss, Y_batch, retain_graph=True)[0])
                        adversarial_loss = self.loss_fn(f=self.encoder(Y_batch + delta), g=g_output)
                        loss += adversarial_loss

                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                total_loss += loss.item()

                if self.config.logging_config.verbose and iteration % 10 == 0:
                    self.logger.debug(f"Epoch {epoch} Iteration {iteration}: Loss = {loss.item():.4f}")

            avg_loss = total_loss / iterations
            self.logger.info(f"Epoch {epoch} average loss: {avg_loss:.4f}")

            if self.config.training_config.use_early_stopping and not pretraining_mode:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    self.logger.info(f"Epoch {epoch}: Early stopping counter = {patience_counter}")
                    if patience_counter >= self.config.training_config.early_stopping_patience:
                        self.logger.info("Early stopping triggered.")
                        break

            if scheduler and not pretraining_mode:
                scheduler.step()

    def _generate_training_batch(self, pretraining_mode=False):
        """
        Generate a training batch.
        If active learning is enabled, select candidates based on an uncertainty metric.
        Otherwise, generate random batches.
        """
        batch_size = self.config.training_config.batch_size
        t, n_vars = self.config.training_config.input_dim
        if self.config.training_config.use_active_learning:
            phi_candidates = torch.rand((self.config.training_config.candidate_pool_size, self.config.training_config.param_dim),
                                        device=self.device) * 2 - 1
            tau_candidates = torch.randint(10, t - 10, (self.config.training_config.candidate_pool_size,), device=self.device).float()
            with torch.no_grad():
                tau_scaled = tau_candidates / t
                phi_input = torch.cat([phi_candidates, tau_scaled.unsqueeze(1)], dim=1)
                g_emb = self.emulator(phi_input)
            uncertainty = -torch.var(g_emb, dim=1)
            selected_indices = torch.argsort(uncertainty)[:batch_size]
            selected_phi = phi_candidates[selected_indices]
        else:
            selected_phi = torch.rand((batch_size, self.config.training_config.param_dim), device=self.device) * 2 - 1

        Y_batch = torch.rand(batch_size, t, n_vars, device=self.device)
        return selected_phi, Y_batch

    def generate_candidate_pool(self, candidate_size: int):
        """
        Generate a candidate pool for parameter estimation using an active-learning strategy.
        This method creates candidate parameter vectors uniformly in the parameter space.
        """
        n_params = self.config.training_config.param_dim
        # Generate uniform candidates in [-1, 1] for all parameters.
        candidate_pool = np.random.uniform(-1, 1, size=(candidate_size, n_params)).astype(np.float32)
        return candidate_pool

    def _create_optimizer(self):
        params = list(self.encoder.parameters()) + list(self.emulator.parameters())
        return torch.optim.AdamW(params, lr=self.config.training_config.learning_rate,
                                 weight_decay=self.config.training_config.weight_decay)

    def _create_scheduler(self, optimizer):
        if self.config.training_config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.training_config.num_epochs)
        return None
