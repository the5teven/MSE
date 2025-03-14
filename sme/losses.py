"""
Losses module for SME.
Defines the core contrastive losses:
- InterDomainInfoNCELoss for aligning embeddings across domains,
- IntraDomainInfoNCELoss for regularizing within a domain,
- Combined via TotalInfoNCELoss.
"""
import torch
import torch.nn as nn

class InterDomainInfoNCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tau = config.training_config.tau

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        sim_matrix_fg = torch.exp(torch.mm(f, g.t()) / self.tau)
        sim_matrix_gf = torch.exp(torch.mm(g, f.t()) / self.tau)
        pos_samples_fg = torch.diag(sim_matrix_fg)
        pos_samples_gf = torch.diag(sim_matrix_gf)
        neg_samples_fg = sim_matrix_fg.sum(dim=1) - pos_samples_fg
        neg_samples_gf = sim_matrix_gf.sum(dim=1) - pos_samples_gf
        loss_fg = -torch.log(pos_samples_fg / (pos_samples_fg + neg_samples_fg)).mean()
        loss_gf = -torch.log(pos_samples_gf / (pos_samples_gf + neg_samples_gf)).mean()
        return loss_fg + loss_gf

class IntraDomainInfoNCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tau = config.training_config.tau

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        sim_matrix = torch.exp(torch.mm(f, f.t()) / self.tau)
        pos_samples = torch.diag(sim_matrix)
        neg_samples = sim_matrix.sum(dim=1) - pos_samples
        return -torch.log(pos_samples / (pos_samples + neg_samples)).mean()

class TotalInfoNCELoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.inter_loss = InterDomainInfoNCELoss(config)
        self.intra_loss = IntraDomainInfoNCELoss(config)

    def forward(self, f: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        return self.inter_loss(f, g) + self.intra_loss(f)