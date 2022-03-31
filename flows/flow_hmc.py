import numpy as np
import torch

def apply_reverse_flow_to_fields(phis, model):
    """
    Given a set of fields `phis` and a function f given by the layers in
    `model`, performs `f^(-1) (phis)`. Does not overwrite `phis`
    """
    phis_tilde, logq = model.flow.inverse(phis)
    S = model.action(phis_tilde)
    return phis_tilde, S, logq

def apply_flow_to_fields(phis, model):
    """
    Given a set of fields `phis` and a function f given by the layers in
    `model`, performs `f(phis)` on each of the fields in `phis`. Does
    not overwrite `phis`.
    """
    phis_tilde, logq = model.flow(phis)
    S = model.action(phis_tilde)
    return phis_tilde, S, logq

def load_flow_action_gradient(phis, model):
    """
    Passes `phis` through layers in `model`, computing its new action
    and log det J of the transformation, and loads the gradient with respect
    to the initial fields into `phis.grad`, without overwriting `phis`.
    """
    phis.grad.zero_()
    
    flowed_phis, S, logJ = apply_flow_to_fields(phis, model)
    
    external_grad_S = torch.ones(S.shape)
    external_grad_logJ = torch.ones(logJ.shape)
    
    logJ.backward(gradient=external_grad_logJ, retain_graph=True)
    S.backward(gradient=external_grad_S)


def flow_hmc(phi, model, *, tau, n_steps, reversibility = False):
    """
    Applies HMC evolution on `phi` with a coordinate transformation given by
    `model`. Input must be a torch tensor of the form `(1,1,L,L)`, with
    `L` the lattice size. This function overwrites `phi`, if configuration is
    accepted.

    `phi` is supposed to be from the target probability distribution. The
    inverse of the layers in `model` map `phi` to a configuration
    `phi_tilde` from a probability distribution easier to sample from.
    """

    # Get phi_tilde = f^{-1}(phi), where f is the trained NN
    phi_tilde = apply_reverse_flow_to_fields(phi, model)[0].detach()
    phi_tilde.requires_grad = True
    phi_tilde.grad = torch.zeros(phi_tilde.shape) # initialize gradient
    
    # Initialize momenta
    mom = torch.randn(phi.shape)
    
    # Initial Hamiltonian
    H_0 = flow_Hamiltonian(mom, phi_tilde, model)
    
    # Leapfrog integrator
    flow_leapfrog(mom, phi_tilde, model, tau = tau, n_steps = n_steps)

    # Final Hamiltonian
    dH = flow_Hamiltonian(mom, phi_tilde, model) - H_0
    
    if reversibility:
        phi_tilde_cp = torch.clone(phi_tilde).detach()
        phi_tilde_cp.requires_grad = True
        phi_tilde_cp.grad = torch.zeros(phi_tilde_cp.shape)

        flow_leapfrog(-mom, phi_tilde_cp, model, tau = tau, n_steps = n_steps)
        phi_tilde_cp.requires_grad = False
        phi_aux = apply_flow_to_fields(phi_tilde_cp, model)[0].detach()

        phi_tilde_in_cp = apply_reverse_flow_to_fields(phi, model)[0].detach()
        print("preReversibility: \sum (\Delta \phi)^2 = ", torch.sum((phi_tilde_cp - phi_tilde_in_cp)**2))
        print("Reversibility: \sum (\Delta \phi)^2 = ", torch.sum((phi_aux - phi)**2))

    if dH > 0:
        if np.random.rand() >= np.exp(-dH):
            return False
    phi[:] = apply_flow_to_fields(phi_tilde, model)[0].detach() # element-wise assignment
    return True


def flow_Hamiltonian(mom, phi_tilde, model):
    """
    Computes the Hamiltonian of `flow_hmc` function.
    """
    phi_aux, S, logJ = apply_flow_to_fields(phi_tilde, model)
    H = 0.5 * torch.sum(mom**2) + S.detach() + logJ.detach()
    return H.item()


def flow_leapfrog(mom, phi, model, *, tau, n_steps):
    dt = tau / n_steps
    
    load_flow_action_gradient(phi, model)
    mom -= 0.5 * dt * phi.grad

    for i in range(n_steps):
        with torch.no_grad():
            phi += dt * mom
        
        if i == n_steps-1:
            load_flow_action_gradient(phi, model)
            mom -= 0.5 * dt * phi.grad
        else:
            load_flow_action_gradient(phi, model)
            mom -= dt * phi.grad

def check_force(phi, model, epsilon):
    phi_eps = torch.clone(phi).detach()
    phi_eps[0,0,6,4] += epsilon
    
    phi_aux, S_i, log_i = apply_flow_to_fields(phi, model)
    phi_aux, S_eps, log_eps = apply_flow_to_fields(phi_eps, model)
    
    return ((S_i+log_i-S_eps-log_eps)/epsilon)

