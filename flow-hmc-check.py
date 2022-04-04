from __future__ import annotations
import sys

sys.path.append("../../")

import math
import logging

import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

# from jsonargparse.typing import PositiveInt, PositiveFloat, NonNegativeFloat

import flows.phi_four as phi_four
import flows.transforms as transforms
import flows.utils as utils
from flows.flow_hmc import *
from flows.models import MultilevelFlow
from flows.layers import GlobalRescalingLayer
from flows.distributions import Prior, FreeScalarDistribution

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor
Module: TypeAlias = torch.nn.Module
IterableDataset: TypeAlias = torch.utils.data.IterableDataset

logging.getLogger().setLevel("WARNING")

ADDITIVE_BLOCK = {
    "transform": transforms.PointwiseAdditiveTransform,
    "transform_spec": {},
    "net_spec": {
        "hidden_shape": [4, 4],
        "activation": torch.nn.Tanh(),
        "final_activation": torch.nn.Identity(),
        "use_bias": False,
    },
}
AFFINE_BLOCK = {
    "transform": transforms.PointwiseAffineTransform,
    "transform_spec": {},
    "net_spec": {
        "hidden_shape": [4, 4, 4, 4],
        "activation": torch.nn.Tanh(),
        "final_activation": torch.nn.Tanh(),
        "use_bias": False,
    },
}
SPLINE_BLOCK = {
    "transform": transforms.PointwiseRationalQuadraticSplineTransform,
    "transform_spec": {"n_segments": 8, "interval": (-4, 4)},
    "net_spec": {
        "hidden_shape": [4],
        "activation": torch.nn.Tanh(),
        "final_activation": torch.nn.Identity(),
        "use_bias": True,
    },
}

# Target theory
LATTICE_LENGTH = 8
BETA = 0.7
LAM = 0.5

MODEL_SPEC = [
    AFFINE_BLOCK,
    AFFINE_BLOCK,
    "rescaling",
]

N_TRAIN = 1000
N_BATCH = 1000
N_BATCH_VAL = 1000

model = MultilevelFlow(
    beta=BETA,
    lam=LAM,
    model_spec=MODEL_SPEC,
)

dist = torch.distributions.Normal(
    loc=torch.zeros((LATTICE_LENGTH, LATTICE_LENGTH)),
    scale=torch.ones((LATTICE_LENGTH, LATTICE_LENGTH)),
)


# Load model
model.load_state_dict(torch.load("models/L8_b0.7_l0.5_model.pth"))

tau = 1.0
n_steps = 10

val_dataloader = Prior(dist, sample_shape=[1, 1])
phi = val_dataloader.sample()

for i in (range(1000)):
    flow_hmc(phi, model, tau=tau, n_steps=n_steps, reversibility=False)
    mag_i = phi.mean(axis=(1,2,3)).item()

    with open("results/L8_b0.7_l0.5-history.txt", "a") as file_object:
        file_object.write(str(mag_i)+"\n")
