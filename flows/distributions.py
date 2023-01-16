from __future__ import annotations

import torch

from flows.utils import laplacian_2d
from flows.phi_four import PhiFourActionBeta, PhiUpInterpFourActionBeta
from flows.layers import UpsamplingLayer

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


class Phi4Dist(torch.distributions.Distribution):
    def __init__(self, beta, lam, lsize, *, thermalization, discard, upscaling = 0):
        self.beta, self.lam, self.lsize = beta, lam, lsize
        self.phi = torch.zeros([lsize, lsize])
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.thermalization, self.discard = thermalization, discard
        self.thermalized = False
        self.action = lambda state: PhiFourActionBeta(self.beta,
                self.lam)(state.view([-1, 1, self.lsize, self.lsize])) # hacky, PhiFourActionBeta needs the fields to have that shape...
        self.upaction = lambda state: PhiFourActionBeta(self.beta,
                self.lam)(state.view([-1, 1, 2*self.lsize, 2*self.lsize])) # upscaled action
        super(Phi4Dist, self).__init__(torch.Size(), validate_args=False)
        self.upscaling = upscaling
        
    def sample(self, sample_shape=torch.Size()):
        samples = torch.zeros(sample_shape + [self.lsize, self.lsize])
        
        if not self.thermalized:
            for i in range(self.thermalization):
                self.hmc(tau = 1.0, n_steps = 10)
            self.thermalized = True
            
        for i in range(sample_shape[0]):
            for j in range(self.discard):
                self.hmc(tau = 1.0, n_steps = 10)
            self.phi.requires_grad = False
            samples[i,:,:] = self.phi[:]
    
        if self.upscaling:
            upsamples = torch.zeros(sample_shape + [2*self.lsize,2*self.lsize])
            upsamples[..., ::2, ::2] = samples[...,:,:]
            upsamples[..., ::2, 1::2] = 1.0/2.0 * (samples + samples.roll(-1,-1))
            upsamples[..., 1::2, ::2] = 1.0/2.0 * (samples + samples.roll(-1,-2))
            upsamples[..., 1::2, 1::2] = 1.0/4.0 * (samples + samples.roll(-1,-2) +
                    samples.roll(-1,-1) + samples.roll((-1,-1),(-2,-1)) )
            return upsamples
        else:
            return samples
        
    def log_prob(self, phis: Tensor) -> Tensor:
        if self.upscaling:
            # return self.upaction(phis).neg().view([-1, 1, 1])
            return 4*self.action(phis[..., ::2, ::2]).neg().view([-1, 1, 1])
        else:
            return self.action(phis).neg().view([-1, 1, 1]) # hacky, to get Prior to compute the log_prob correctly...
    
    def hmc(self, *, tau: float, n_steps: int) -> bool:
        phi_cp = torch.clone(self.phi).detach()

        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape) # initialize gradient

        # Initialize momenta
        mom = torch.randn(self.phi.shape)

        # Initial Hamiltonian
        H_0 = self.hamiltonian(mom)

        # Leapfrog integrator
        self.leapfrog(mom, tau = tau, n_steps = n_steps)

        # Final Hamiltonian
        dH = self.hamiltonian(mom) - H_0

        if dH > 0:
            if torch.rand(1).item() >= torch.exp(-torch.Tensor([dH])).item():
                with torch.no_grad():
                    self.phi[:] = phi_cp # element-wise assignment
                return False

        return True
    
    def hamiltonian(self, mom):
        """
        Computes the Hamiltonian of `hmc` function.
        """
        H = 0.5 * torch.sum(mom**2) + self.action(self.phi)

        return H.item()

    def leapfrog_AD(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        self.load_action_gradient()
        mom -= 0.5 * dt * self.phi.grad

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                self.load_action_gradient()
                mom -= 0.5 * dt * self.phi.grad
            else:
                self.load_action_gradient()
                mom -= dt * self.phi.grad
                

    def leapfrog(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        mom -= 0.5 * dt * self.get_force(self.phi)

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                mom -= 0.5 * dt * self.get_force(self.phi)
            else:
                mom -= dt * self.get_force(self.phi)

    def get_force(self, state):
        frc = self.beta * (torch.roll(state, -1, 0) + torch.roll(state, 1, 0) + torch.roll(state, -1, 1) + torch.roll(state, 1, 1)) + 2 * state * (2 * self.lam * (1 - state**2) - 1)

        return -frc
    
    def check_force(self, epsilon):
        
        phi_check = torch.normal(0, 1, size=(self.lsize, self.lsize))
        
        phi_check_2 = torch.clone(phi_check)
        phi_check_2[2,2] += epsilon
        
        F_num = (self.action(phi_check_2) - self.action(phi_check))/epsilon
        F_ana = self.get_force(phi_check)[2,2]
        
        self.phi = phi_check
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.load_action_gradient()
        F_AD = self.phi.grad[2,2]
        
        print(F_num)
        print(F_ana)
        print(F_AD)


    # Does not seem to work to train the network...
    def load_action_gradient(self):
        """
        Passes `phi` through fucntion `action`and loads the gradient with respect to
        the initial fields into `phi.grad`, without overwriting `phi`.
        """
        self.phi.grad.zero_()
        
        S = self.action(self.phi)
        print(S.requires_grad)

        external_grad_S = torch.ones(S.shape)

        S.backward(gradient=external_grad_S)
        
        #state.requires_grad_()
        #state.grad = None
        #with torch.enable_grad():
        #    self.potential(state).backward()
        #force = state.grad
        #state.requires_grad_(False)
        #state.grad = None
        #return force


class Phi4DistUpscaledInterpolated(torch.distributions.Distribution):
    def __init__(self, beta, lam, lsize, *, thermalization, discard, upscaling = 0):
        self.beta, self.lam, self.lsize = beta, lam, lsize
        self.phi = torch.zeros([lsize, lsize])
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.action = lambda state: PhiUpInterpFourActionBeta(self.beta,
                self.lam)(state.view([-1, 1, self.lsize, self.lsize])) # hacky, PhiFourActionBeta needs the fields to have that shape...
        self.thermalization, self.discard = thermalization, discard
        self.thermalized = False
        super(Phi4DistUpscaledInterpolated, self).__init__(torch.Size(), validate_args=False)
        
    def sample(self, sample_shape=torch.Size()):
        samples = torch.zeros(sample_shape + [self.lsize, self.lsize])
        
        if not self.thermalized:
            for i in range(self.thermalization):
                self.hmc(tau = 1.0, n_steps = 10)
            self.thermalized = True
            
        for i in range(sample_shape[0]):
            for j in range(self.discard):
                self.hmc(tau = 1.0, n_steps = 10)
            self.phi.requires_grad = False
            samples[i,:,:] = self.phi[:]
    
        # return samples
        upsamples = self.upscale_interp(samples)
        return upsamples

    def upscale_interp(self, samples):
        sample_size = samples.size()
        upsamples = torch.zeros([*sample_size[:-2]] + [2*self.lsize, 2*self.lsize])
        upsamples[..., ::2, ::2] = samples[...,:,:]
        upsamples[..., ::2, 1::2] = 1.0/2.0 * (samples + samples.roll(-1,-1))
        upsamples[..., 1::2, ::2] = 1.0/2.0 * (samples + samples.roll(-1,-2))
        upsamples[..., 1::2, 1::2] = 1.0/4.0 * (samples + samples.roll(-1,-2) +
                samples.roll(-1,-1) + samples.roll((-1,-1),(-2,-1)) )
        return upsamples

    def upaction(self, phis: Tensor) -> Tensor:
        return PhiFourActionBeta(self.beta, self.lam)(self.upscale_interp(phis).view([-1, 1, 2*self.lsize, 2*self.lsize]))

    def log_prob(self, phis: Tensor) -> Tensor:
        return PhiFourActionBeta(self.beta, self.lam)(phis.view([-1, 1,
            2*self.lsize, 2*self.lsize])).neg().view([-1, 1, 1]) # this acts on the (up)samples
        # return self.action(phis).neg().view([-1,1,1])
    
    def hmc(self, *, tau: float, n_steps: int) -> bool:
        phi_cp = torch.clone(self.phi).detach()

        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape) # initialize gradient

        # Initialize momenta
        mom = torch.randn(self.phi.shape)

        # Initial Hamiltonian
        H_0 = self.hamiltonian(mom)

        # Leapfrog integrator
        self.leapfrog(mom, tau = tau, n_steps = n_steps)

        # Final Hamiltonian
        dH = self.hamiltonian(mom) - H_0

        if dH > 0:
            if torch.rand(1).item() >= torch.exp(-torch.Tensor([dH])).item():
                with torch.no_grad():
                    self.phi[:] = phi_cp # element-wise assignment
                return False

        return True
    
    def hamiltonian(self, mom):
        """
        Computes the Hamiltonian of `hmc` function.
        """
        H = 0.5 * torch.sum(mom**2) + self.action(self.phi)

        return H.item()

    def leapfrog_AD(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        self.load_action_gradient()
        mom -= 0.5 * dt * self.phi.grad

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                self.load_action_gradient()
                mom -= 0.5 * dt * self.phi.grad
            else:
                self.load_action_gradient()
                mom -= dt * self.phi.grad
                

    def leapfrog(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        mom -= 0.5 * dt * self.get_force(self.phi)

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                mom -= 0.5 * dt * self.get_force(self.phi)
            else:
                mom -= dt * self.get_force(self.phi)

    def get_force(self, state):
        plaquettes = PhiUpInterpFourActionBeta(self.beta, self.lam).plaquette(state)
        
        frc = self.beta *   (
                4 * state +
                torch.roll(state, -1, 0) +
                torch.roll(state, 1, 0) + torch.roll(state, -1, 1) +
                torch.roll(state, 1, 1) 
                + 1/2 * (plaquettes+plaquettes.roll(1,0)+plaquettes.roll(1,1)+plaquettes.roll((1,1),(0,1)))
                ) - (1 - 2*self.lam) * (
                        2 * state + 1/2 * (4 * state + torch.roll(state, -1, 0)
                            + torch.roll(state, 1, 0) + torch.roll(state, -1, 1)
                            + torch.roll(state, 1, 1))
                        + 1/8
                        *(plaquettes+plaquettes.roll(1,0)+plaquettes.roll(1,1)+plaquettes.roll((1,1),(0,1)))) - self.lam * (
                                4 * state.pow(3) + 1/4 *
                                ((state + state.roll(-1,0)).pow(3)+(state +
                                    state.roll(1,0)).pow(3)+(state +
                                        state.roll(-1,1)).pow(3)+(state +
                                        state.roll(1,1)).pow(3))
                        + 1/64
                        *
                        (plaquettes.pow(3)+plaquettes.roll(1,0).pow(3)+plaquettes.roll(1,1).pow(3)+plaquettes.roll((1,1),(0,1)).pow(3))
                        ) 
        

        return -frc
    
    def check_force(self, epsilon):
        
        phi_check = torch.normal(0, 1, size=(self.lsize, self.lsize))
        
        phi_check_2 = torch.clone(phi_check)
        phi_check_2[2,2] += epsilon
        
        F_num = (self.action(phi_check_2) - self.action(phi_check))/epsilon
        F_ana = self.get_force(phi_check)[2,2]
        
        self.phi = phi_check
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.load_action_gradient()
        F_AD = self.phi.grad[2,2]
        
        print(F_num)
        print(F_ana)
        print(F_AD)

    def check_action(self):
        
        phi_check = torch.normal(0, 1, size=(self.lsize, self.lsize))
        
        action1 = self.action(phi_check)
        action2 = self.upaction(phi_check)

        print(action1)
        print(action2)

    # Does not seem to work to train the network...
    def load_action_gradient(self):
        """
        Passes `phi` through fucntion `action`and loads the gradient with respect to
        the initial fields into `phi.grad`, without overwriting `phi`.
        """
        self.phi.grad.zero_()
        
        S = self.action(self.phi)
        print(S.requires_grad)

        external_grad_S = torch.ones(S.shape)

        S.backward(gradient=external_grad_S)



class Phi4DistUpscaledGaussian(torch.distributions.Distribution):
    def __init__(self, beta, lam, lsize, *, thermalization, discard):
        self.beta, self.lam, self.lsize = beta, lam, lsize
        self.phi = torch.zeros([lsize, lsize])
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.action = lambda state: PhiFourActionBeta(self.beta,
                self.lam)(state.view([-1, 1, self.lsize, self.lsize])) # hacky, PhiFourActionBeta needs the fields to have that shape...
        # self.action = lambda state: PhiUpInterpFourActionBeta(self.beta,
        #         self.lam)(state.view([-1, 1, self.lsize, self.lsize])) # hacky, PhiFourActionBeta needs the fields to have that shape...
        self.thermalization, self.discard = thermalization, discard
        self.thermalized = False
        self.dist = torch.distributions.Normal(
            loc=torch.zeros((lsize, lsize)),
            scale=torch.ones((lsize, lsize)),
        )
        self.upsampling_layer = UpsamplingLayer(use_batch_dimension=True)
        super(Phi4DistUpscaledGaussian, self).__init__(torch.Size(), validate_args=False)
        
    def sample(self, sample_shape=torch.Size()):
        samples = torch.zeros(sample_shape + [self.lsize, self.lsize])
        
        if not self.thermalized:
            for i in range(self.thermalization):
                self.hmc(tau = 1.0, n_steps = 10)
            self.thermalized = True
            
        for i in range(sample_shape[0]):
            if i%4==0:
                for j in range(self.discard):
                    self.hmc(tau = 1.0, n_steps = 10)
                self.phi.requires_grad = False
                samples[i,:,:] = self.phi[:]
            else:
                samples[i,...] = self.dist.sample([1,1])

        upsamples, _ = self.upsampling_layer(samples, torch.zeros(samples.size()))
        return upsamples

    def log_prob(self, phi_gaussian):
        phi_gaussian_down, _ = self.upsampling_layer.inverse(phi_gaussian,
                torch.zeros([1]))
        tbool = torch.full([len(phi_gaussian_down)], False).to(phi_gaussian.device)
        tbool[::4, ...] = True
        phi = phi_gaussian_down[tbool]
        gaussian = phi_gaussian_down[~tbool]
        return self.action(phi).neg().view([-1,1,1]) + self.dist.log_prob(gaussian.view(-1,3,self.lsize,self.lsize)).flatten(start_dim=1).sum(dim=1).view([-1,1,1])

    def hmc(self, *, tau: float, n_steps: int) -> bool:
        phi_cp = torch.clone(self.phi).detach()

        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape) # initialize gradient

        # Initialize momenta
        mom = torch.randn(self.phi.shape)

        # Initial Hamiltonian
        H_0 = self.hamiltonian(mom)

        # Leapfrog integrator
        self.leapfrog(mom, tau = tau, n_steps = n_steps)

        # Final Hamiltonian
        dH = self.hamiltonian(mom) - H_0

        if dH > 0:
            if torch.rand(1).item() >= torch.exp(-torch.Tensor([dH])).item():
                with torch.no_grad():
                    self.phi[:] = phi_cp # element-wise assignment
                return False

        return True
    
    def hamiltonian(self, mom):
        """
        Computes the Hamiltonian of `hmc` function.
        """
        H = 0.5 * torch.sum(mom**2) + self.action(self.phi)

        return H.item()

    def leapfrog_AD(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        self.load_action_gradient()
        mom -= 0.5 * dt * self.phi.grad

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                self.load_action_gradient()
                mom -= 0.5 * dt * self.phi.grad
            else:
                self.load_action_gradient()
                mom -= dt * self.phi.grad
                

    def leapfrog(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        mom -= 0.5 * dt * self.get_force(self.phi)

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                mom -= 0.5 * dt * self.get_force(self.phi)
            else:
                mom -= dt * self.get_force(self.phi)

    def get_force(self, state):
        frc = self.beta * (torch.roll(state, -1, 0) + torch.roll(state, 1, 0) + torch.roll(state, -1, 1) + torch.roll(state, 1, 1)) + 2 * state * (2 * self.lam * (1 - state**2) - 1)

        return -frc
    
    def check_force(self, epsilon):
        
        phi_check = torch.normal(0, 1, size=(self.lsize, self.lsize))
        
        phi_check_2 = torch.clone(phi_check)
        phi_check_2[2,2] += epsilon
        
        F_num = (self.action(phi_check_2) - self.action(phi_check))/epsilon
        F_ana = self.get_force(phi_check)[2,2]
        
        self.phi = phi_check
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.load_action_gradient()
        F_AD = self.phi.grad[2,2]
        
        print(F_num)
        print(F_ana)
        print(F_AD)

    # Does not seem to work to train the network...
    def load_action_gradient(self):
        """
        Passes `phi` through fucntion `action`and loads the gradient with respect to
        the initial fields into `phi.grad`, without overwriting `phi`.
        """
        self.phi.grad.zero_()
        
        S = self.action(self.phi)
        print(S.requires_grad)

        external_grad_S = torch.ones(S.shape)

        S.backward(gradient=external_grad_S)



class Phi4DistUpscaledGaussianOnSameValue(torch.distributions.Distribution):
    def __init__(self, beta, lam, lsize, *, thermalization, discard):
        self.beta, self.lam, self.lsize = beta, lam, lsize
        self.phi = torch.zeros([lsize, lsize])
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.action = lambda state: PhiFourActionBeta(self.beta,
                self.lam)(state.view([-1, 1, self.lsize, self.lsize])) # hacky, PhiFourActionBeta needs the fields to have that shape...
        # self.action = lambda state: PhiUpInterpFourActionBeta(self.beta,
        #         self.lam)(state.view([-1, 1, self.lsize, self.lsize])) # hacky, PhiFourActionBeta needs the fields to have that shape...
        self.thermalization, self.discard = thermalization, discard
        self.thermalized = False
        self.dist = torch.distributions.Normal(
            loc=torch.zeros((lsize, lsize)),
            scale=torch.ones((lsize, lsize)),
        )
        self.upsampling_layer = UpsamplingLayer(use_batch_dimension=True)
        super(Phi4DistUpscaledGaussianOnSameValue, self).__init__(torch.Size(), validate_args=False)
        
    def sample(self, sample_shape=torch.Size()):
        samples = torch.zeros(sample_shape + [self.lsize, self.lsize])
        
        if not self.thermalized:
            for i in range(self.thermalization):
                self.hmc(tau = 1.0, n_steps = 10)
            self.thermalized = True
            
        for i in range(sample_shape[0]):
            if i%4==0:
                for j in range(self.discard):
                    self.hmc(tau = 1.0, n_steps = 10)
                self.phi.requires_grad = False
                samples[i,:,:] = self.phi[:]
            else:
                self.dist = torch.distributions.Normal(
                    loc=self.phi,
                    scale=0.1*torch.ones((self.lsize, self.lsize)),
                )
                samples[i,...] = self.dist.sample([1,1])

        upsamples, _ = self.upsampling_layer(samples, torch.zeros(samples.size()))
        return upsamples

    def log_prob(self, phi_gaussian):
        phi_gaussian_down, _ = self.upsampling_layer.inverse(phi_gaussian,
                torch.zeros([1]))
        tbool = torch.full([len(phi_gaussian_down)], False).to(phi_gaussian.device)
        tbool[::4, ...] = True
        phi = phi_gaussian_down[tbool]
        self.dist = torch.distributions.Normal(
            loc=phi,
            scale=0.1*torch.ones(phi.shape),
        )
        gaussian = phi_gaussian_down[~tbool]
        return self.action(phi).neg().view([-1,1,1]) + self.dist.log_prob(gaussian.view(-1,3,self.lsize,self.lsize)).flatten(start_dim=1).sum(dim=1).view([-1,1,1])

    def hmc(self, *, tau: float, n_steps: int) -> bool:
        phi_cp = torch.clone(self.phi).detach()

        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape) # initialize gradient

        # Initialize momenta
        mom = torch.randn(self.phi.shape)

        # Initial Hamiltonian
        H_0 = self.hamiltonian(mom)

        # Leapfrog integrator
        self.leapfrog(mom, tau = tau, n_steps = n_steps)

        # Final Hamiltonian
        dH = self.hamiltonian(mom) - H_0

        if dH > 0:
            if torch.rand(1).item() >= torch.exp(-torch.Tensor([dH])).item():
                with torch.no_grad():
                    self.phi[:] = phi_cp # element-wise assignment
                return False

        return True
    
    def hamiltonian(self, mom):
        """
        Computes the Hamiltonian of `hmc` function.
        """
        H = 0.5 * torch.sum(mom**2) + self.action(self.phi)

        return H.item()

    def leapfrog_AD(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        self.load_action_gradient()
        mom -= 0.5 * dt * self.phi.grad

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                self.load_action_gradient()
                mom -= 0.5 * dt * self.phi.grad
            else:
                self.load_action_gradient()
                mom -= dt * self.phi.grad
                

    def leapfrog(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        mom -= 0.5 * dt * self.get_force(self.phi)

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                mom -= 0.5 * dt * self.get_force(self.phi)
            else:
                mom -= dt * self.get_force(self.phi)

    def get_force(self, state):
        frc = self.beta * (torch.roll(state, -1, 0) + torch.roll(state, 1, 0) + torch.roll(state, -1, 1) + torch.roll(state, 1, 1)) + 2 * state * (2 * self.lam * (1 - state**2) - 1)

        return -frc
    
    def check_force(self, epsilon):
        
        phi_check = torch.normal(0, 1, size=(self.lsize, self.lsize))
        
        phi_check_2 = torch.clone(phi_check)
        phi_check_2[2,2] += epsilon
        
        F_num = (self.action(phi_check_2) - self.action(phi_check))/epsilon
        F_ana = self.get_force(phi_check)[2,2]
        
        self.phi = phi_check
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.load_action_gradient()
        F_AD = self.phi.grad[2,2]
        
        print(F_num)
        print(F_ana)
        print(F_AD)

    # Does not seem to work to train the network...
    def load_action_gradient(self):
        """
        Passes `phi` through fucntion `action`and loads the gradient with respect to
        the initial fields into `phi.grad`, without overwriting `phi`.
        """
        self.phi.grad.zero_()
        
        S = self.action(self.phi)
        print(S.requires_grad)

        external_grad_S = torch.ones(S.shape)

        S.backward(gradient=external_grad_S)



class Phi4DistUpscaledGaussianOnAverage(torch.distributions.Distribution):
    def __init__(self, beta, lam, lsize, *, thermalization, discard):
        self.beta, self.lam, self.lsize = beta, lam, lsize
        self.phi = torch.zeros([lsize, lsize])
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.action = lambda state: PhiFourActionBeta(self.beta,
                self.lam)(state.view([-1, 1, self.lsize, self.lsize])) # hacky, PhiFourActionBeta needs the fields to have that shape...
        # self.action = lambda state: PhiUpInterpFourActionBeta(self.beta,
        #         self.lam)(state.view([-1, 1, self.lsize, self.lsize])) # hacky, PhiFourActionBeta needs the fields to have that shape...
        self.thermalization, self.discard = thermalization, discard
        self.thermalized = False
        self.dist = torch.distributions.Normal(
            loc=torch.zeros((lsize, lsize)),
            scale=torch.ones((lsize, lsize)),
        )
        self.upsampling_layer = UpsamplingLayer(use_batch_dimension=True)
        super(Phi4DistUpscaledGaussianOnAverage, self).__init__(torch.Size(), validate_args=False)
        
    def sample(self, sample_shape=torch.Size()):
        samples = torch.zeros(sample_shape + [self.lsize, self.lsize])
        
        if not self.thermalized:
            for i in range(self.thermalization):
                self.hmc(tau = 1.0, n_steps = 10)
            self.thermalized = True
            
        for i in range(sample_shape[0]):
            if i%4==0:
                for j in range(self.discard):
                    self.hmc(tau = 1.0, n_steps = 10)
                self.phi.requires_grad = False
                samples[i,:,:] = self.phi[:]
            elif i%4==1:
                phi_h = 1.0/2.0 * (self.phi + self.phi.roll(-1,-1))
                self.dist = torch.distributions.Normal(
                    loc=phi_h,
                    scale=0.1*torch.ones((self.lsize, self.lsize)),
                )
                samples[i,...] = self.dist.sample([1,1])
            elif i%4==2:
                phi_v = 1.0/2.0 * (self.phi + self.phi.roll(-1,-2))
                self.dist = torch.distributions.Normal(
                    loc=phi_v,
                    scale=0.1*torch.ones((self.lsize, self.lsize)),
                )
                samples[i,...] = self.dist.sample([1,1])
            else:
                phi_hv = 1.0/4.0 * (self.phi + self.phi.roll(-1,-2) +
                    self.phi.roll(-1,-1) + self.phi.roll((-1,-1),(-2,-1)) )
                self.dist = torch.distributions.Normal(
                    loc=phi_hv,
                    scale=0.1*torch.ones((self.lsize, self.lsize)),
                )
                samples[i,...] = self.dist.sample([1,1])

        upsamples, _ = self.upsampling_layer(samples, torch.zeros(samples.size()))
        return upsamples

    def log_prob(self, phi_gaussian):
        phi_gaussian_down, _ = self.upsampling_layer.inverse(phi_gaussian,
                torch.zeros([1]))
        tbool = torch.full([len(phi_gaussian_down)], False).to(phi_gaussian.device)
        tbool[::4, ...] = True

        phi = phi_gaussian_down[tbool]

        phi_h = 1.0/2.0 * (phi + phi.roll(-1,-1))
        phi_v = 1.0/2.0 * (phi + phi.roll(-1,-2))
        phi_hv = 1.0/4.0 * (phi + phi.roll(-1,-2) +
            phi.roll(-1,-1) + phi.roll((-1,-1),(-2,-1)) )

        phi_averages = torch.cat((phi_h, phi_v, phi_hv), 1)

        self.dist = torch.distributions.Normal(
            loc=phi_averages,
            scale=0.1*torch.ones(phi_averages.shape),
        )
        gaussian = phi_gaussian_down[~tbool]
        return self.action(phi).neg().view([-1,1,1]) + self.dist.log_prob(gaussian.view(-1,3,self.lsize,self.lsize)).flatten(start_dim=1).sum(dim=1).view([-1,1,1])

    def hmc(self, *, tau: float, n_steps: int) -> bool:
        phi_cp = torch.clone(self.phi).detach()

        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape) # initialize gradient

        # Initialize momenta
        mom = torch.randn(self.phi.shape)

        # Initial Hamiltonian
        H_0 = self.hamiltonian(mom)

        # Leapfrog integrator
        self.leapfrog(mom, tau = tau, n_steps = n_steps)

        # Final Hamiltonian
        dH = self.hamiltonian(mom) - H_0

        if dH > 0:
            if torch.rand(1).item() >= torch.exp(-torch.Tensor([dH])).item():
                with torch.no_grad():
                    self.phi[:] = phi_cp # element-wise assignment
                return False

        return True
    
    def hamiltonian(self, mom):
        """
        Computes the Hamiltonian of `hmc` function.
        """
        H = 0.5 * torch.sum(mom**2) + self.action(self.phi)

        return H.item()

    def leapfrog_AD(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        self.load_action_gradient()
        mom -= 0.5 * dt * self.phi.grad

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                self.load_action_gradient()
                mom -= 0.5 * dt * self.phi.grad
            else:
                self.load_action_gradient()
                mom -= dt * self.phi.grad
                

    def leapfrog(self, mom, *, tau, n_steps):
        dt = tau / n_steps

        mom -= 0.5 * dt * self.get_force(self.phi)

        for i in range(n_steps):
            with torch.no_grad():
                self.phi += dt * mom

            if i == n_steps-1:
                mom -= 0.5 * dt * self.get_force(self.phi)
            else:
                mom -= dt * self.get_force(self.phi)

    def get_force(self, state):
        frc = self.beta * (torch.roll(state, -1, 0) + torch.roll(state, 1, 0) + torch.roll(state, -1, 1) + torch.roll(state, 1, 1)) + 2 * state * (2 * self.lam * (1 - state**2) - 1)

        return -frc
    
    def check_force(self, epsilon):
        
        phi_check = torch.normal(0, 1, size=(self.lsize, self.lsize))
        
        phi_check_2 = torch.clone(phi_check)
        phi_check_2[2,2] += epsilon
        
        F_num = (self.action(phi_check_2) - self.action(phi_check))/epsilon
        F_ana = self.get_force(phi_check)[2,2]
        
        self.phi = phi_check
        self.phi.requires_grad = True
        self.phi.grad = torch.zeros(self.phi.shape)
        self.load_action_gradient()
        F_AD = self.phi.grad[2,2]
        
        print(F_num)
        print(F_ana)
        print(F_AD)

    # Does not seem to work to train the network...
    def load_action_gradient(self):
        """
        Passes `phi` through fucntion `action`and loads the gradient with respect to
        the initial fields into `phi.grad`, without overwriting `phi`.
        """
        self.phi.grad.zero_()
        
        S = self.action(self.phi)
        print(S.requires_grad)

        external_grad_S = torch.ones(S.shape)

        S.backward(gradient=external_grad_S)
