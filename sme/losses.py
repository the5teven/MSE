import torch
import torch.nn.functional as F
from torch import nn
from .config import SMEConfig

class CompositeLoss(nn.Module):
    def __init__(self, config: SMEConfig):
        """Composite loss that selects the appropriate loss based on configuration."""
        super().__init__()
        self.config = config
        self.device = self.config.device

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

# Updated AngularLoss using a mean squared error on cosine similarity
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
    """
    Contrastive loss with a memory bank for negative sampling.
    """
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.memory_bank_size = config.memory_bank_size
        self.temperature = config.tau
        self.memory_bank = torch.randn(self.memory_bank_size, config.param_dim).to(config.device)

    def forward(self, f, g):
        """
        Compute contrastive loss using both positive pairs and negatives from the memory bank.
        """
        f, g = F.normalize(f, dim=-1), F.normalize(g, dim=-1)
        pos_logits = torch.sum(f * g, dim=-1) / self.temperature
        neg_logits = torch.matmul(f, self.memory_bank.T) / self.temperature
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        self._update_memory_bank(g)
        return F.cross_entropy(logits, labels)

    def _update_memory_bank(self, g):
        """Update memory bank with new embeddings."""
        self.memory_bank = torch.cat([g.detach(), self.memory_bank[:-g.shape[0]]], dim=0)
