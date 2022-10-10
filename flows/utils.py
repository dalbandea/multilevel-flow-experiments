from __future__ import annotations

import itertools
import math
import logging
from random import random
from typing import Iterable

import torch
import pytorch_lightning as pl
import tqdm
import scipy.signal
import numpy as np

Tensor: TypeAlias = torch.Tensor
BoolTensor: TypeAlias = torch.BoolTensor


class Composition(torch.nn.Sequential):
    """Compose multiple layers."""

    def forward(self, x: Tensor, log_det_jacob: Tensor, *args) -> tuple[Tensor]:
        for layer in self:
            x, log_det_jacob = layer(x, log_det_jacob, *args)
        return x, log_det_jacob

    def inverse(self, y: Tensor, log_det_jacob: Tensor, *args) -> tuple[Tensor]:
        for layer in reversed(self):
            y, log_det_jacob = layer.inverse(y, log_det_jacob, *args)
        return y, log_det_jacob


class Flow(Composition):
    """Wraps around Composition, starting with zero log det Jacobian."""

    def forward(self, x: Tensor, *args) -> tuple[Tensor]:
        return super().forward(x, torch.zeros(x.shape[0]).to(x.device), *args)

    def inverse(self, y: Tensor, *args) -> tuple[Tensor]:
        return super().inverse(y, torch.zeros(y.shape[0]).to(y.device), *args)


def laplacian_2d(lattice_length: int) -> torch.Tensor:
    """Creates a 2d Laplacian matrix.

    This works by taking the kronecker product of the one-dimensional
    Laplacian matrix with the identity.

    Notes
    -----
    For now, assume a square lattice. Periodic BCs are also assumed.
    """
    identity = torch.eye(lattice_length)
    lapl_1d = (
        2 * identity  # main diagonal
        - torch.diag(torch.ones(lattice_length - 1), diagonal=1)  # upper
        - torch.diag(torch.ones(lattice_length - 1), diagonal=-1)  # lower
    )
    lapl_1d[0, -1] = lapl_1d[-1, 0] = -1  # periodicity
    lapl_2d = torch.kron(lapl_1d, identity) + torch.kron(identity, lapl_1d)
    return lapl_2d


def make_checkerboard(lattice_shape: list[int]) -> BoolTensor:
    """Return a boolean mask that selects 'even' lattice sites."""
    assert all(
        [n % 2 == 0 for n in lattice_shape]
    ), "each lattice dimension should be even"
    checkerboard = torch.full(lattice_shape, False)
    if len(lattice_shape) == 1:
        checkerboard[::2] = True
    elif len(lattice_shape) == 2:
        checkerboard[::2, ::2] = True
        checkerboard[1::2, 1::2] = True
    else:
        raise NotImplementedError("d > 2 currently not supported")
    return checkerboard


def prod(iterable: Iterable):
    """Return product of elements of iterable."""
    out = 1
    for el in iterable:
        out *= el
    return out


def metropolis_acceptance(weights: Tensor) -> float:
    """Returns the fraction of configs that pass the Metropolis test."""
    weights = weights.tolist()
    curr_weight = weights.pop(0)
    history = []

    for prop_weight in weights:
        if math.log(random()) < min(0, (curr_weight - prop_weight)):
            curr_weight = prop_weight
            history.append(1)
        else:
            history.append(0)

    return sum(history) / len(history)


def autocorrelation(chain: Tensor):
    signal = chain.sub(chain.mean()).numpy()
    autocorr = scipy.signal.correlate(signal, signal, mode="same")
    t0 = autocorr.size // 2
    autocorr = autocorr[t0:] / autocorr[t0]
    return autocorr


def integrated_autocorrelation(chain: Tensor):
    autocorr = autocorrelation(chain)
    integrated = np.cumsum(autocorr)

    with np.errstate(invalid="ignore", divide="ignore"):
        exponential = np.clip(
            np.nan_to_num(2.0 / np.log((2 * integrated + 1) / (2 * integrated - 1))),
            a_min=1e-6,
            a_max=None,
        )

    # Infer ensemble size. Assumes correlation mode was 'same'
    n_t = integrated.shape[-1]
    ensemble_size = n_t * 2

    # g_func is the derivative of the sum of errors wrt window size
    window = np.arange(1, n_t + 1)
    g_func = np.exp(-window / exponential) - exponential / np.sqrt(
        window * ensemble_size
    )

    # Return first occurrence of g_func changing sign
    w = np.argmax((g_func[..., 1:] < 0), axis=-1)

    return integrated[w]


class JlabProgBar(pl.callbacks.TQDMProgressBar):
    """Disable validation progress bar since it's broken in Jupyter Lab."""

    def init_validation_tqdm(self):
        return tqdm.tqdm(disable=True)


def nearest_neighbour_kernel(lattice_dim):
    identity_kernel = torch.zeros([3 for _ in range(lattice_dim)])
    identity_kernel.view(-1)[pow(3, lattice_dim) // 2] = 1

    nn_kernel = torch.zeros([3 for _ in range(lattice_dim)])
    for shift, dim in itertools.product([+1, -1], range(lattice_dim)):
        nn_kernel.add_(identity_kernel.roll(shift, dim))

    return nn_kernel.view(1, 1, *nn_kernel.shape)
