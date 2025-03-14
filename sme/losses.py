import torch
import torch.nn.functional as F
from torch import nn
from .config import SMEConfig


class CompositeLoss(nn.Module):
    def __init__(self, config: SMEConfig):
        """Composite loss that selects the appropriate loss based on configuration.

        If config.use_memory_bank is True, it instantiates a MemoryBankLoss that automatically
        adapts to the loss type (e.g., "angular" or "dot"). Otherwise, it uses one of the standard
        loss implementations defined in this file.
        """
        super().__init__()
        self.config = config
        self.device = self.config.device

        # Use embedding_dim defined in the configuration (replace with latent_dim if that's what you use)
        if not hasattr(config, "embedding_dim"):
            raise AttributeError(
                "SMEConfig must have an attribute 'embedding_dim' representing the encoder's output dimension.")

        if getattr(config, "use_memory_bank", False):
            self.loss_fn = get_memory_bank_loss(
                embedding_dim=self.config.embedding_dim,
                config_loss_type=self.config.loss_type,
                memory_bank_size=getattr(self.config, "memory_bank_size", 1024),
                temperature=self.config.tau
            ).to(self.device)
        else:
            loss_functions = {
                "symmetric": SymmetricInfoNCE,
                "angular": AngularLoss,
                "protonce": ProtoNCE
            }
            if config.loss_type not in loss_functions:
                raise ValueError(f"Invalid loss_type: {config.loss_type}")
            self.loss_fn = loss_functions[config.loss_type](config).to(self.device)

    def forward(self, f, g):
        """Compute loss using mixed precision if enabled."""
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            return self.loss_fn(f, g)


class SymmetricInfoNCE(nn.Module):
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.temperature = config.tau

    def forward(self, f, g):
        """Compute Symmetric InfoNCE loss."""
        f, g = F.normalize(f, dim=-1), F.normalize(g, dim=-1)
        logits = torch.matmul(f, g.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)
        loss_f = F.cross_entropy(logits, labels)
        loss_g = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_f + loss_g)


# Updated AngularLoss using a mean squared error on cosine similarity.
class AngularLoss(nn.Module):
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.alpha = config.alpha
        self.temperature = config.tau

    def forward(self, f, g):
        """Compute Angular loss to encourage maximum cosine similarity."""
        cos_sim = F.cosine_similarity(f, g, dim=-1)
        loss = F.mse_loss(cos_sim, torch.ones_like(cos_sim))
        return self.alpha * loss


class ProtoNCE(nn.Module):
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.temperature = config.tau

    def forward(self, f, g):
        """Compute prototype-based contrastive loss."""
        prototypes = torch.mean(g, dim=0, keepdim=True)
        f, prototypes = F.normalize(f, dim=-1), F.normalize(prototypes, dim=-1)
        logits = torch.matmul(f, prototypes.T) / self.temperature
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, labels)


class MemoryBankLoss(nn.Module):
    def __init__(self, embedding_dim, loss_type="angular", memory_bank_size=1024, temperature=0.07):
        """
        Initializes the Memory Bank Loss which automatically adapts to the specified loss type.

        Args:
            embedding_dim (int): Dimension of the latent embeddings (should match encoder output).
            loss_type (str): Type of loss to use. Supported:
                - "angular": Uses cosine (angular) similarity.
                - "dot": Uses raw dot product similarity.
            memory_bank_size (int): Number of negative samples stored in the memory bank.
            temperature (float): Temperature parameter to scale logits.
        """
        super(MemoryBankLoss, self).__init__()
        self.loss_type = loss_type.lower()
        self.memory_bank_size = memory_bank_size
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        # Initialize memory bank with shape: (memory_bank_size, embedding_dim)
        self.register_buffer("memory_bank", torch.randn(memory_bank_size, embedding_dim))
        # For angular loss, normalize the memory bank entries; "dot" loss may not require it.
        if self.loss_type == "angular":
            self.memory_bank = F.normalize(self.memory_bank, p=2, dim=1)

    def forward(self, f, g):
        """
        Computes the loss using the memory bank and adapts the computation to the chosen loss type.

        Args:
            f (torch.Tensor): Encoder latent embeddings of shape (batch_size, embedding_dim).
            g (torch.Tensor): Emulator latent embeddings of shape (batch_size, embedding_dim).

        Returns:
            torch.Tensor: Computed loss.
        """
        if self.loss_type == "angular":
            # For angular loss, ensure embeddings are normalized.
            f = F.normalize(f, p=2, dim=1)
            g = F.normalize(g, p=2, dim=1)
            # Positive logits: dot product between corresponding f and g.
            pos_logits = torch.sum(f * g, dim=1, keepdim=True) / self.temperature
            # Negative logits: use all entries in memory bank.
            neg_logits = torch.matmul(f, self.memory_bank.t()) / self.temperature
        elif self.loss_type == "dot":
            # For dot loss, use raw dot product (assume normalization is handled externally if needed).
            pos_logits = torch.sum(f * g, dim=1, keepdim=True) / self.temperature
            neg_logits = torch.matmul(f, self.memory_bank.t()) / self.temperature
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

        # Concatenate logits (positive sample at index 0, negatives follow).
        logits = torch.cat([pos_logits, neg_logits], dim=1)
        labels = torch.zeros(f.size(0), dtype=torch.long, device=f.device)
        loss = F.cross_entropy(logits, labels)

        # Update the memory bank using a momentum update.
        m = 0.5
        with torch.no_grad():
            batch_size = g.size(0)
            indices = torch.randperm(self.memory_bank_size)[:batch_size]
            updated = m * self.memory_bank[indices] + (1 - m) * g
            if self.loss_type == "angular":
                updated = F.normalize(updated, p=2, dim=1)
            self.memory_bank[indices] = updated

        return loss


def get_memory_bank_loss(embedding_dim, config_loss_type, memory_bank_size=1024, temperature=0.07):
    """
    Factory function to automatically instantiate the memory bank loss matching the chosen loss type.

    Args:
        embedding_dim (int): Dimension of latent embeddings.
        config_loss_type (str): The loss type specified in configuration ("angular" or "dot").
        memory_bank_size (int): Number of negative samples to store.
        temperature (float): Temperature parameter for scaling logits.

    Returns:
        MemoryBankLoss: An instance of MemoryBankLoss configured as per the config_loss_type.
    """
    return MemoryBankLoss(
        embedding_dim=embedding_dim,
        loss_type=config_loss_type,
        memory_bank_size=memory_bank_size,
        temperature=temperature
    )