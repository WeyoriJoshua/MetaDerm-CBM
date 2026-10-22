from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureScaler(nn.Module):
    def __init__(self, init_temperature=1.0):
        super().__init__()
        self.log_temperature = nn.Parameter(torch.tensor(float(init_temperature)).log())

    @property
    def temperature(self):
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits):
        return logits / self.temperature

    def fit_multiclass(self, logits, targets, max_iter=100):
        logits = logits.detach()
        targets = targets.detach()
        opt = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)
        def closure():
            opt.zero_grad()
            loss = F.cross_entropy(self(logits), targets)
            loss.backward()
            return loss
        opt.step(closure)
        return float(self.temperature.detach().cpu())

    def fit_binary(self, logits, targets, max_iter=100):
        logits = logits.detach()
        targets = targets.detach().float()
        opt = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=max_iter)
        def closure():
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(self(logits), targets)
            loss.backward()
            return loss
        opt.step(closure)
        return float(self.temperature.detach().cpu())


def apply_temperature(logits, temperature):
    return logits / float(temperature)
