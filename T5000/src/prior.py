import torch

from pbc_config import BOX, wrap

def uniform_prior(x1):
     x0 = torch.rand_like(x1) 
     return wrap(x0, **BOX)

def wrapped_gaussian_prior(x1):
    x0 = torch.randn_like(x1)
    return wrap(x0, **BOX)

def scaled_log10_gaussian_mass_prior(m1):
     m0 = torch.normal(mean=0.3550, std=0.0980, size=m1.shape, device=m1.device)
     return m0