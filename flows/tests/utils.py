from __future__ import annotations

import torch
import pytorch_lightning as pl

from flows.utils import Flow, metropolis_acceptance
from flows.distributions import Prior, FreeScalarDistribution
from flows.phi_four import PhiFourAction

Tensor: TypeAlias = torch.Tensor
Module: TypeAlias = torch.nn.Module
Distribution: TypeAlias = torch.distributions.Distribution
Optimizer: TypeAlias = torch.optim.Optimizer


def basic_train_loop(
    prior: Prior,
    target: PhiFourAction | Distribution,
    flow: Flow,
    optimizer: Optimizer,
    n_iter: PositiveInt,
):
    for _ in range(n_iter):
        z, log_prob_z = next(prior)
        x, log_det_jacob = flow(z)
        log_prob_x = target.log_prob(x)
        weights = log_prob_z - log_det_jacob - log_prob_x
        loss = weights.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss


def check_acceptance(
    prior: Prior,
    target: PhiFourAction | Distribution,
    flow: Flow,
):
    z, log_prob_z = next(prior)
    x, log_det_jacob = flow(z)
    log_prob_x = target.log_prob(x)
    weights = log_prob_z - log_det_jacob - log_prob_x
    return metropolis_acceptance(weights)


class BasicRevKLModule(pl.LightningModule):
    def __init__(self, prior: Prior, flow: Flow, target: PhiFourAction | Distribution):
        super().__init__()
        self.prior = prior
        self.flow = flow
        self.target = target

    def forward(self, batch):
        z, log_prob_z = batch
        x, log_det_jacob = self.flow(z)
        log_prob_x = self.target.log_prob(x)
        weights = log_prob_z - log_det_jacob - log_prob_x
        return x, weights

    def training_step(self, batch, batch_idx):
        _, weights = self.forward(batch)
        loss = weights.mean()
        return loss

    def train_dataloader(self):
        return self.prior

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.001)
        return optimizer

    @torch.no_grad()
    def sample(self, n_iter: int = 1):
        x, weights = self.forward(next(self.prior))
        for _ in range(n_iter - 1):
            _x, _weights = self.forward(next(prior))
            x = torch.cat((x, _x), dim=0)
            weights = torch.cat((weights, _weights), dim=0)
        return x, weights


class UnconditionalLayer(torch.nn.Module):
    def __init__(self, transform, init_params: Tensor | None = None):
        super().__init__()
        self.transform = transform
        if init_params is None:
            init_params = torch.empty(transform.params_dof).uniform_(-1, 1)
        self.params = torch.nn.Parameter(init_params.view(1, -1, 1, 1))

    def forward(self, x: tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        n_batch, _, l1, l2 = x.shape
        y, log_det_jacob_this = self.transform.forward(
            x, self.params.expand(n_batch, -1, l1, l2)
        )
        log_det_jacob.add_(log_det_jacob_this)
        return y, log_det_jacob

    def inverse(self, y: tensor, log_det_jacob: Tensor) -> tuple[Tensor]:
        n_batch, _, l1, l2 = x.shape
        x, log_det_jacob_this = self.transform.inverse(
            y, self.params.expand(n_batch, -1, l1, l2)
        )
        log_det_jacob.add_(log_det_jacob_this)
        return x, log_det_jacob
