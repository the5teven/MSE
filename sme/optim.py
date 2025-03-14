"""
Optimization utilities for the SME project.
Defines an Exponential Moving Average (EMA) helper to stabilize training.
"""
import torch
import copy

class EMA:
    def __init__(self, model, decay=0.999):
        """
        Args:
            model: The model (SMEModel) to apply EMA on.
            decay: Decay rate for the exponential moving average.
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights from model parameters
        self.register()

    def register(self):
        for name, param in self.model.encoder.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        for name, param in self.model.emulator.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.encoder.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
        for name, param in self.model.emulator.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]

    def apply_shadow(self):
        for name, param in self.model.encoder.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
        for name, param in self.model.emulator.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.encoder.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        for name, param in self.model.emulator.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}