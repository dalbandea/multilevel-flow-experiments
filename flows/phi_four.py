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


class PhiUpInterpFourActionBeta:
    """Action for a scalar field on a lattice with quartic interaction with
    interpolated variables.

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

        # # Interaction with nearest neighbours
        for dim in range(2, phi.dim()):
            action.sub_(
                    phi.add(phi.roll(-1, dim)).pow(2).mul(0.5).mul(self.beta)
                        )
        action.sub_(
                self.plaquette(phi).pow(2).mul(self.beta).mul(0.25)
                )

        # # phi^2 term
        action.add_(phi.pow(2).add((2*phi+phi.roll(-1,-1)+phi.roll(-1,-2)).mul(0.25)).mul(1-2*self.lam)
                + 1/16 * self.plaquette(phi).pow(2) * (1-2*self.lam))

        # # phi^4 term
        action.add_(self.lam * (phi.pow(4) + 1/16 * ((phi +
            phi.roll(-1,-1)).pow(4) + (phi + phi.roll(-1,-2)).pow(4)) 
            + 1/256 * self.plaquette(phi).pow(4))
            )

        # Sum over lattice sites
        action = action.flatten(start_dim=1).sum(dim=1)

        return action

    def plaquette(self, phi):
        return phi.add(phi.roll(-1,-1)).add(phi.roll(-1,-2)).add(phi.roll((-1,-1),(-2,-1)))

    def log_prob(self, phi: Tensor) -> Tensor:
        return self.action(phi).neg()
