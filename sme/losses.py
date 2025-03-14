import torch
import torch.nn.functional as F
from torch import nn
from .config import SMEConfig

class CompositeLoss(nn.Module):
    def __init__(self, config: SMEConfig):
        """
        Composite loss function that selects the appropriate loss function based on the configuration.
        """
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        loss_functions = {
            "symmetric": SymmetricInfoNCE,
            "angular": AngularLoss,
            "protonce": ProtoNCE
        }

        if config.loss_type not in loss_functions:
            raise ValueError(f"Invalid loss_type: {config.loss_type}")

        self.loss_fn = loss_functions[config.loss_type](config).to(self.device)

    def forward(self, f, g):
        """
        Compute the loss in mixed precision (AMP) mode for better performance.
        """
        with torch.cuda.amp.autocast(enabled=self.config.use_amp):
            return self.loss_fn(f, g)

# ✅ Optimized Symmetric InfoNCE Loss
class SymmetricInfoNCE(nn.Module):
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.temperature = config.tau

    def forward(self, f, g):
        """
        Compute the Symmetric InfoNCE loss.
        """
        f, g = F.normalize(f, dim=-1), F.normalize(g, dim=-1)
        logits = torch.matmul(f, g.T) / self.temperature
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss_f = F.cross_entropy(logits, labels)
        loss_g = F.cross_entropy(logits.T, labels)
        return 0.5 * (loss_f + loss_g)

class AngularLoss(nn.Module):
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.alpha = config.alpha
        self.temperature = config.tau

    def forward(self, f, g):
        """
        Compute the Angular loss, which encourages angular similarity.
        """
        cos_sim = F.cosine_similarity(f, g, dim=-1)
        angular_distance = 1 - cos_sim

        loss = F.cross_entropy(angular_distance / self.temperature, torch.zeros_like(angular_distance))
        return self.alpha * loss

# ProtoNCE Loss (Prototype Contrastive Learning)
class ProtoNCE(nn.Module):
    def __init__(self, config: SMEConfig):
        super().__init__()
        self.temperature = config.tau

    def forward(self, f, g):
        """
        Prototype-NCE loss.
        """
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
        Computes contrastive loss using memory bank.
        Args:
            f: Encoded features from current batch.
            g: Encoded parameters from emulator.
        Returns:
            Contrastive loss with memory bank negatives.
        """
        f, g = F.normalize(f, dim=-1), F.normalize(g, dim=-1)

        # Compute similarity scores
        pos_logits = torch.sum(f * g, dim=-1) / self.temperature  # Positive pairs
        neg_logits = torch.matmul(f, self.memory_bank.T) / self.temperature  # Negative samples

        # Create labels (positives at index 0)
        logits = torch.cat([pos_logits.unsqueeze(1), neg_logits], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)

        # Update memory bank
        self._update_memory_bank(g)

        return F.cross_entropy(logits, labels)

    def _update_memory_bank(self, g):
        """Updates memory bank with new embeddings."""
        self.memory_bank = torch.cat([g.detach(), self.memory_bank[:-g.shape[0]]], dim=0)
