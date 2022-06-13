import torch
import numpy as np

def he_flow(phi, t):
    phi_p0 = torch.fft.fftn(phi)
    phi_pt = he_flow_p(phi_p0, t)
    phi_t = torch.fft.ifftn(phi_pt)

    return phi_t

def he_flow_p(phi, t):
    L = phi.shape[0]
    phi_pt = torch.clone(phi).detach()

    for i in range(L):
        for j in range(L):
            p2 =  4.0 * (np.sin(np.pi * i / L)**2.0 + np.sin(np.pi * j / L)**2.0)
            phi_pt[i,j] = np.exp(-p2 * t) * phi[i,j]

    return phi_pt


X = torch.randn(10,10, dtype=torch.float64)

## Checks

torch.max(torch.abs(he_flow(he_flow(X, 1), -1) - X))
torch.max(torch.abs(he_flow(X, 1000) - X.mean()))
