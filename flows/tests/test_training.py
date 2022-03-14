from __future__ import annotations

import math

import torch
import pytorch_lightning as pl
import pytest

import flows.transforms
from flows.utils import Flow
from flows.distributions import Prior, FreeScalarDistribution
from flows.phi_four import PhiFourAction
from flows.tests.utils import basic_train_loop, check_acceptance, UnconditionalLayer

Tensor: TypeAlias = torch.Tensor
Module: TypeAlias = torch.nn.Module
Distribution: TypeAlias = torch.distributions.Distribution

LATTICE_LENGTH = 8
M_SQ = 1
LATTICE_SHAPE = (LATTICE_LENGTH, LATTICE_LENGTH)
BATCH_SIZE = 1000
GAUSSIAN = torch.distributions.Normal(
    loc=torch.zeros(LATTICE_SHAPE), scale=torch.ones(LATTICE_SHAPE)
)
FREE_SCALAR = FreeScalarDistribution(LATTICE_LENGTH, M_SQ)

TRANSFORMS = [
    flows.transforms.PointwiseAdditiveTransform(),
    flows.transforms.PointwiseAffineTransform(),
    flows.transforms.PointwiseRationalQuadraticSplineTransform(8, (-5, 5)),
]


@pytest.mark.parametrize("distribution", [GAUSSIAN, FREE_SCALAR])
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_do_nothing(distribution, transform):
    prior = Prior(distribution, [BATCH_SIZE, 1])
    target = Prior(distribution, [1, 1])  # so that log prob is sum over config
    flow = Flow(UnconditionalLayer(transform, init_params=transform.identity_params))
    optimizer = torch.optim.SGD(flow.parameters(), lr=1e-3)

    acceptance = check_acceptance(prior, target, flow)
    assert math.isclose(acceptance, 1.0)

    _ = basic_train_loop(prior, target, flow, optimizer, n_iter=100)
    acceptance = check_acceptance(prior, target, flow)
    assert acceptance > 0.95


@pytest.mark.parametrize("distribution", [GAUSSIAN, FREE_SCALAR])
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_learns_identity(distribution, transform):
    prior = Prior(distribution, [BATCH_SIZE, 1])
    target = Prior(distribution, [1, 1])
    init_params = transform.identity_params.add(
        torch.empty_like(transform.identity_params).uniform_(-0.1, 0.1)
    )
    flow = Flow(UnconditionalLayer(transform, init_params=init_params))
    optimizer = torch.optim.SGD(flow.parameters(), lr=1e-3)
    
    acceptance_before_training = check_acceptance(prior, target, flow)

    _ = basic_train_loop(prior, target, flow, optimizer, n_iter=100)
    acceptance_after_training = check_acceptance(prior, target, flow)
    
    assert acceptance_after_training > acceptance_before_training
