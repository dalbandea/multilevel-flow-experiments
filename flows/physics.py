from __future__ import annotations

import torch

Tensor: TypeAlias = torch.Tensor


class PhiFourAction:
    """Action for a scalar field on a lattice with quartic interaction.

    The following action

        S(\phi) = \sum_{x\in\Lambda} \left[
            -\sum_{\mu=1}^d \phi(x + \hat\mu) \phi(x)
            + (4 + m^2) / 2 \phi(x)^2
            + \lambda \phi(x)^4

    defines an interacting theory of a scalar field \phi with bare
    mass m_sq and quartic interaction strength \lambda, living on a
    d-dimensional square lattice with constant lattice spacing a=1.
    """

    def __init__(self, m_sq: float, lam: float, scale: float = 1):
        self.m_sq = m_sq
        self.lam = lam

    def __call__(self, config: Tensor) -> Tensor:
        """Computes action for a sample of field configurations.

        Parameters
        ----------
        config: Tensor
            The sample of configurations, where the first dimension
            (``config[i]``) runs over the configurations.

        Returns
        -------
        Tensor
            1-dimensional tensor containing the action for each
            configuration in the sample.
        """
        action = torch.zeros_like(config)

        # Interaction with nearest neighbours
        for dim in range(1, config.dim() + 1):
            action.sub_(config.mul(config.roll(-1, dim)))

        # phi^2 term
        config_sq = config.pow(2)
        action.add_(config_sq.mul((4 + self.m_sq) / 2))

        # phi^4 term
        action.add_(config_sq.pow(2).mul(self.lam))

        # Sum over lattice sites
        action = action.flatten(start_dim=1).sum(dim=1)

        return action

class FreeScalarDistribution(torch.distributions.MultivariateNormal):
    r"""A distribution representing a non-interacting scalar field.

    This is a subclass of torch.distributions.MultivariateNormal in which the
    covariance matrix is specified by the bare mass of the scalar field.

    The action
        
        S(\phi) = \frac{1}{2} \sum_{x\in\Lambda} \sum_{y\in\Lambda}
        \phi(x) \Sigma^{-1}(x, y) \phi(y)
        
    where the precision matrix (inverse of the covariance matrix) is

        \Sigma^{-1}(x, y) = (2 d + m_0^2) \delta(x, y)
        - \sum_{\mu=1}^d \big( \delta(x + e_\mu, y) + \delta(x - e_\mu, y) \big)
    
    describes a free (non-interacting) scalar field \phi living on a
    square lattice.

    Parameters
    ----------
    lattice_length
        Number of nodes on one side of the lattice. The lattice is assumed to be
        square.
    m_sq
        Bare mass, squared.

    """

    def __init__(self, lattice_length: int, m_sq: float):
        sigma_inv = gelato.utils.laplacian_2d(lattice_length) + torch.eye(
            lattice_length ** 2
        ).mul(m_sq)
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
