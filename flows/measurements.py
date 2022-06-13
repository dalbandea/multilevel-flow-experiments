import torch
import numpy as np

def magnetization(phi):
    return phi.mean(axis=(1,2,3)).item()

def susceptibility(phi):
    return (phi**2).mean(axis=(1,2,3)).item()

def he_flow(phi, t):
    """
    Apply the heat equation to a configuration `phi` (with dimensions [L,L] or
    [1,1,L,L]) integrating it up to a time `t`, returning the new configuration.
    """
    shape_0 = phi.shape # get input shape
    phi_cp = phi.view(phi.shape[-2:]).detach() # pick last 2 dimensions
    phi_p0 = torch.fft.fftn(phi_cp)
    phi_pt = he_flow_p(phi_p0, t)
    phi_t = torch.fft.ifftn(phi_pt)

    return torch.real(phi_t.view(shape_0))

def he_flow_p(phi, t):
    """
    Apply the kernel of the heat equation in momentum space given a
    configuration `phi` and a flow time `t`. `phi` must have dimensions [L,L].
    """
    L = phi.shape[-1]
    phi_pt = torch.clone(phi).detach()

    for i in range(L):
        for j in range(L):
            p2 =  4.0 * (np.sin(np.pi * i / L)**2.0 + np.sin(np.pi * j / L)**2.0)
            phi_pt[i,j] = np.exp(-p2 * t) * phi[i,j]

    return phi_pt
