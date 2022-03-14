from __future__ import annotations

import torch

from flows.utils import laplacian_2d

Tensor: TypeAlias = torch.Tensor
IterableDataset: TypeAlias = torch.utils.data.IterableDataset
Distribution: TypeAlias = torch.distributions.Distribution


class Prior(torch.utils.data.IterableDataset):
    """Wraps around torch.distributions.Distribution to make it iterable."""

    def __init__(self, distribution: Distribution, sample_shape: list[int]):
        super().__init__()
        self.distribution = distribution
        self.sample_shape = sample_shape

    def __iter__(self):
        return self

    def __next__(self) -> tuple[Tensor]:
        sample = self.sample()
        return sample, self.log_prob(sample)

    def sample(self) -> Tensor:
        return self.distribution.sample(self.sample_shape)

    def log_prob(self, sample: Tensor) -> Tensor:
        return self.distribution.log_prob(sample).flatten(start_dim=1).sum(dim=1)


class FreeScalarDistribution(torch.distributions.MultivariateNormal):
    r"""A distribution representing a non-interacting scalar field.

    This is a subclass of torch.distributions.MultivariateNormal in which the
    covariance matrix is specified by the bare mass of the scalar field.

    Parameters
    ----------
    lattice_length
        Number of nodes on one side of the square 2-dimensional lattice.
    m_sq
        Bare mass, squared.
    """

    def __init__(self, lattice_length: int, m_sq: float):
        sigma_inv = laplacian_2d(lattice_length) + torch.eye(lattice_length ** 2).mul(
            m_sq
        )
        super().__init__(
            loc=torch.zeros(lattice_length ** 2), precision_matrix=sigma_inv
        )
        self.lattice_length = lattice_length

    def log_prob(self, value):
        """Flattens 2d configurations and calls superclass log_prob."""
        return super().log_prob(value.flatten(start_dim=-2))

    def rsample(self, sample_shape=torch.Size()):
        """Calls superclass rsample and restores 2d geometry."""
        return (
            super()
            .rsample(sample_shape)
            .view(*sample_shape, self.lattice_length, self.lattice_length)
        )
