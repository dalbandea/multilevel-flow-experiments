from __future__ import annotations

import torch

Tensor: TypeAlias = torch.Tensor


class PhiFourAction:
    """Action for a scalar field on a lattice with quartic interaction.

    Parameters
    ----------
    m_sq: float
        Bare mass, squared
    lam: float
        Coupling for quartic interaction
    """

    def __init__(self, m_sq: float, lam: float):
        self.m_sq = m_sq
        self.lam = lam

    def __call__(self, phi: Tensor) -> Tensor:
        """Computes action for a sample of field configurations.

        Parameters
        ----------
        phi: Tensor
            The sample of configurations, where the first dimension
            (``phi[i]``) runs over the configurations.

        Returns
        -------
        Tensor
            1-dimensional tensor containing the action for each
            configuration in the sample.
        """
        action = torch.zeros_like(phi)

        # Interaction with nearest neighbours
        for dim in range(2, phi.dim()):
            action.sub_(phi.mul(phi.roll(-1, dim)))

        # phi^2 term
        phi_sq = phi.pow(2)
        action.add_(phi_sq.mul((4 + self.m_sq) / 2))

        # phi^4 term
        action.add_(phi_sq.pow(2).mul(self.lam))

        # Sum over lattice sites
        action = action.flatten(start_dim=1).sum(dim=1)

        return action

    def log_prob(self, phi: Tensor) -> Tensor:
        return self.action(phi).neg()


class PhiFourActionBeta:
    """Action for a scalar field on a lattice with quartic interaction.

    Parameters
    ----------
    beta: float
        Next-neighbour interaction
    lam: float
        Coupling for quartic interaction
    """

    def __init__(self, beta: float, lam: float):
        self.beta = beta
        self.lam = lam

    def __call__(self, phi: Tensor) -> Tensor:
        """Computes action for a sample of field configurations.

        Parameters
        ----------
        phi: Tensor
            The sample of configurations, where the first dimension
            (``phi[i]``) runs over the configurations.

        Returns
        -------
        Tensor
            1-dimensional tensor containing the action for each
            configuration in the sample.
        """
        action = torch.zeros_like(phi)

        # Interaction with nearest neighbours
        for dim in range(2, phi.dim()):
            action.sub_(phi.mul(phi.roll(-1, dim)).mul(self.beta))

        # phi^2 term
        phi_sq = phi.pow(2)
        action.add_(phi_sq.mul(1.0 - 2.0 * self.lam))

        # phi^4 term
        action.add_(phi_sq.pow(2).mul(self.lam))

        # Sum over lattice sites
        action = action.flatten(start_dim=1).sum(dim=1)

        return action

    def log_prob(self, phi: Tensor) -> Tensor:
        return self.action(phi).neg()
